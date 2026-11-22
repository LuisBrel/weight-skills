#!/usr/bin/env python3
"""Universal Weight-Skills Compiler & Multi-Harness Proxy (weight-skills/universal-bundle@1).

Compile local SKILL.md directories (Claude Code, Codex, Antigravity, Cursor, or
any agent harness) into a Two-Regime Skill Bundle and serve it over OpenAI
(/v1/chat/completions, /v1/models) and Anthropic (/v1/messages) endpoints in front
of any local model server (OMLX, LM Studio, Ollama, vLLM, llama.cpp).

Compile and serve use Python stdlib only. Optional `export-omlx` requires
`numpy` and `safetensors`.

Usage:
    python3 universal_adapter.py compile --skills-dir ~/.agents/skills --out bundle.json
    python3 universal_adapter.py serve --bundle bundle.json --upstream http://127.0.0.1:8199 --port 8788
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BUNDLE_SCHEMA = "weight-skills/universal-bundle@1"
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
})
_BLOCK_SCALARS = frozenset({">", ">-", ">+", "|", "|-", "|+"})


def _parse_skill_md(path: Path, root: Path) -> Dict[str, Any]:
    """Parse a SKILL.md file into name, description, keywords, and body without leaking host paths."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    name = path.parent.name
    description = ""
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            fm, body = parts[1], parts[2].strip()
            for line in fm.splitlines():
                s = line.strip()
                if s.startswith("name:"):
                    val = s.split(":", 1)[1].strip().strip("\"'")
                    if val:
                        name = val
                elif s.startswith("description:"):
                    val = s.split(":", 1)[1].strip().strip("\"'")
                    if val and val not in _BLOCK_SCALARS:
                        description = val
            if not description:
                m = re.search(r"description:\s*[>|][+-]?\s*\n((?:\s+.+\n?)+)", fm)
                if m:
                    description = " ".join(ln.strip() for ln in m.group(1).splitlines())
    if not description:
        for ln in body.splitlines():
            s = ln.strip().lstrip("#").strip()
            if s:
                description = s
                break
    compact_desc = (description[:117].rstrip() + "...") if len(description) > 120 else description
    kw_text = f"{name} {description}".lower()
    keywords = sorted(t for t in set(_TOKEN_RE.findall(kw_text)) if t not in _STOPWORDS)
    try:
        rel_path = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel_path = f"{path.parent.name}/{path.name}"
    return {
        "name": name,
        "path": rel_path,
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "bytes": len(raw.encode("utf-8")),
        "description": description,
        "compact_line": f"- **{name}**: {compact_desc}",
        "keywords": keywords,
        "excerpt": body[:2400],
    }


def compile_bundle(
    skills_dirs: List[Path],
    cartridges_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compile SKILL.md files and optional weight cartridges into a Two-Regime bundle."""
    skills: Dict[str, Dict[str, Any]] = {}
    visited_dirs: Set[Tuple[int, int]] = set()
    for sdir in skills_dirs:
        root = sdir.expanduser().resolve()
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            try:
                st = os.stat(dirpath)
            except OSError:
                dirnames[:] = []
                continue
            key = (st.st_dev, st.st_ino)
            if key in visited_dirs:
                dirnames[:] = []
                continue
            visited_dirs.add(key)
            # Prune circular symlinked subdirectories deterministically
            kept: List[str] = []
            for dn in sorted(dirnames):
                try:
                    dst = os.stat(os.path.join(dirpath, dn))
                    if (dst.st_dev, dst.st_ino) not in visited_dirs:
                        kept.append(dn)
                except OSError:
                    continue
            dirnames[:] = kept
            if "SKILL.md" in filenames:
                parsed = _parse_skill_md(Path(dirpath) / "SKILL.md", root)
                skills[parsed["name"]] = parsed

    cartridges: List[Dict[str, Any]] = []
    if cartridges_dir is not None and cartridges_dir.expanduser().exists():
        croot = cartridges_dir.expanduser().resolve()
        for cf in sorted(list(croot.rglob("*.safetensors")) + list(croot.rglob("*.npz"))):
            cartridges.append({
                "id": cf.stem,
                "path": cf.relative_to(croot).as_posix(),
                "bytes": cf.stat().st_size,
                "sha256": hashlib.sha256(cf.read_bytes()).hexdigest(),
            })

    index_lines = [
        "# Weight-Skills Regime-1 Index (Compact Static Prefix)",
        "Use the matching skill's instructions when triggered; request full SKILL.md only on demand.",
    ] + [skills[k]["compact_line"] for k in sorted(skills)]
    static_prefix = "\n".join(index_lines)

    return {
        "schema": BUNDLE_SCHEMA,
        "skill_count": len(skills),
        "cartridge_count": len(cartridges),
        "raw_bytes_total": sum(s["bytes"] for s in skills.values()),
        "compact_prefix_bytes": len(static_prefix.encode("utf-8")),
        "static_prefix": static_prefix,
        "skills": [skills[k] for k in sorted(skills)],
        "cartridges": cartridges,
    }


def route_query(bundle: Dict[str, Any], query: str, min_score: float = 0.12) -> Optional[Dict[str, Any]]:
    """Deterministic query-recall keyword router with word-boundary name boost and abstention."""
    q_tokens = {t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOPWORDS}
    if not q_tokens:
        return None
    q_lower = query.lower()
    best_skill: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for sk in bundle.get("skills", []):
        k_tokens = set(sk.get("keywords", []))
        if not k_tokens:
            continue
        overlap = len(q_tokens & k_tokens)
        if overlap == 0:
            continue
        # Query-recall ratio over non-stopword tokens + exact word-boundary skill name bonus
        score = overlap / max(1, len(q_tokens))
        if re.search(rf"(?<![a-z0-9_-]){re.escape(sk['name'].lower())}(?![a-z0-9_-])", q_lower):
            score += 0.5
        if score > best_score:
            best_score = score
            best_skill = sk
    if best_skill is None or best_score < min_score:
        return None
    return {"skill": best_skill, "score": round(best_score, 4)}


def prepare_openai_payload(
    payload: Dict[str, Any],
    bundle: Dict[str, Any],
    *,
    inject_matched_excerpt: bool = True,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Hoist Regime-1 static skill index into system[0] and optionally inject top-1 excerpt."""
    out = dict(payload)
    msgs = list(out.get("messages") or [])
    static_prefix = bundle.get("static_prefix", "")

    user_query = ""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                user_query = c
            elif isinstance(c, list):
                user_query = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            break

    matched = route_query(bundle, user_query)
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        existing_sys = str(msgs[0].get("content") or "")
        if static_prefix and static_prefix not in existing_sys:
            msgs[0] = {"role": "system", "content": static_prefix + "\n\n" + existing_sys}
    elif static_prefix:
        msgs.insert(0, {"role": "system", "content": static_prefix})

    if inject_matched_excerpt and matched is not None:
        sk = matched["skill"]
        excerpt_note = f"\n\n[Matched Skill: {sk['name']} (score={matched['score']})]\n{sk['excerpt']}"
        for idx in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[idx], dict) and msgs[idx].get("role") == "user":
                c = msgs[idx].get("content")
                if isinstance(c, str):
                    msgs[idx] = {**msgs[idx], "content": c + excerpt_note}
                elif isinstance(c, list):
                    msgs[idx] = {**msgs[idx], "content": [*c, {"type": "text", "text": excerpt_note}]}
                break

    out["messages"] = msgs
    return out, matched


def anthropic_to_openai(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic /v1/messages (including tools, tool_use, tool_result) to OpenAI format."""
    out: Dict[str, Any] = {
        "model": payload.get("model", ""),
        "messages": [],
        "stream": bool(payload.get("stream", False)),
    }
    if payload.get("max_tokens") is not None:
        out["max_tokens"] = int(payload["max_tokens"])
    if payload.get("temperature") is not None:
        out["temperature"] = float(payload["temperature"])
    if payload.get("stop_sequences"):
        out["stop"] = payload["stop_sequences"]

    sys_val = payload.get("system")
    if isinstance(sys_val, str) and sys_val.strip():
        out["messages"].append({"role": "system", "content": sys_val})
    elif isinstance(sys_val, list):
        txt = "".join(b.get("text", "") for b in sys_val if isinstance(b, dict) and b.get("type") == "text")
        if txt.strip():
            out["messages"].append({"role": "system", "content": txt})

    for m in payload.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, str) or c is None:
            out["messages"].append({"role": role, "content": c or ""})
            continue
        if not isinstance(c, list):
            out["messages"].append({"role": role, "content": str(c)})
            continue

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in c:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or f"toolu_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                })
            elif btype == "tool_result":
                res = block.get("content")
                if isinstance(res, list):
                    res_txt = "".join(x.get("text", "") for x in res if isinstance(x, dict) and x.get("type") == "text")
                else:
                    res_txt = "" if res is None else str(res)
                out["messages"].append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": res_txt,
                })
        if tool_calls:
            out["messages"].append({"role": "assistant", "content": "".join(text_parts) or None, "tool_calls": tool_calls})
        elif text_parts:
            out["messages"].append({"role": role, "content": "".join(text_parts)})

    tools = [t for t in (payload.get("tools") or []) if isinstance(t, dict)]
    if tools:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description") or "",
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]
    tc = payload.get("tool_choice")
    if isinstance(tc, dict):
        tt = tc.get("type")
        if tt == "any":
            out["tool_choice"] = "required"
        elif tt == "tool" and tc.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        else:
            out["tool_choice"] = "auto"
    return out


def openai_to_anthropic(resp: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Convert OpenAI response (including tool_calls) to Anthropic Messages response."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks: List[Dict[str, Any]] = []
    text = msg.get("content") or ""
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            inp = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or "toolu_0",
            "name": fn.get("name", ""),
            "input": inp,
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    finish = choice.get("finish_reason")
    stop_reason = "tool_use" if (finish == "tool_calls" or msg.get("tool_calls")) else ("max_tokens" if finish == "length" else "end_turn")
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id", "msg_wssc"),
        "type": "message",
        "role": "assistant",
        "model": model or resp.get("model", ""),
        "content": blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
    }


def make_handler(bundle: Dict[str, Any], upstream: str):
    upstream = upstream.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/healthz"):
                self._send_json(200, {
                    "ok": True,
                    "schema": bundle.get("schema"),
                    "skills": bundle.get("skill_count", 0),
                    "cartridges": bundle.get("cartridge_count", 0),
                    "upstream": upstream,
                })
                return
            if path == "/v1/models":
                try:
                    with urllib.request.urlopen(f"{upstream}/v1/models", timeout=15) as r:
                        self._send_json(200, json.loads(r.read().decode("utf-8")))
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    self._send_json(502, {"error": f"upstream /v1/models error: {exc}"})
                return
            self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/chat/completions", "/v1/messages"):
                self._send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length) if length > 0 else b""
                body = json.loads(raw_body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": f"invalid_json: {exc}"})
                return

            is_anthropic = path == "/v1/messages"
            oai_req = anthropic_to_openai(body) if is_anthropic else body
            wants_stream = bool(oai_req.get("stream"))
            prepared, matched = prepare_openai_payload(oai_req, bundle)
            headers = {"Content-Type": "application/json"}
            if matched:
                headers["X-WSSC-Skill"] = matched["skill"]["name"]

            # Non-streaming Anthropic translation executes a synchronous JSON upstream call;
            # OpenAI streaming clients receive direct SSE pass-through from upstream.
            if is_anthropic:
                prepared["stream"] = False
            req = urllib.request.Request(
                f"{upstream}/v1/chat/completions",
                data=json.dumps(prepared).encode("utf-8"),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    if not is_anthropic and wants_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        while True:
                            chunk = resp.read(4096)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        return
                    oai_resp = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self._send_json(exc.code, {"error": exc.read().decode("utf-8", errors="replace")})
                return
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                self._send_json(502, {"error": f"upstream error: {exc}"})
                return

            if is_anthropic and wants_stream:
                anth = openai_to_anthropic(oai_resp, str(body.get("model", "")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                msg_start = {"type": "message_start", "message": {**anth, "content": []}}
                events = [("message_start", msg_start)]
                for idx, blk in enumerate(anth["content"]):
                    if blk["type"] == "text":
                        events.append(("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}}))
                        events.append(("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": blk["text"]}}))
                        events.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
                    else:
                        events.append(("content_block_start", {"type": "content_block_start", "index": idx, "content_block": blk}))
                        events.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
                events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": anth["stop_reason"]}, "usage": {"output_tokens": anth["usage"]["output_tokens"]}}))
                events.append(("message_stop", {"type": "message_stop"}))
                for ev, data_obj in events:
                    self.wfile.write(f"event: {ev}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8"))
                self.wfile.flush()
                return

            out = openai_to_anthropic(oai_resp, str(body.get("model", ""))) if is_anthropic else oai_resp
            self._send_json(200, out)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Universal Weight-Skills Compiler & Multi-Harness Proxy")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compile", help="Compile SKILL.md directory into a Two-Regime bundle")
    pc.add_argument("--skills-dir", action="append", required=True, type=Path)
    pc.add_argument("--cartridges-dir", type=Path, default=None)
    pc.add_argument("--out", required=True, type=Path)

    ps = sub.add_parser("serve", help="Serve OpenAI + Anthropic proxy with compiled skill bundle")
    ps.add_argument("--bundle", required=True, type=Path)
    ps.add_argument("--upstream", default="http://127.0.0.1:8199")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8788)

    pe = sub.add_parser("export-omlx", help="Export matching-dimension mlx-lm/OMLX adapter from cartridge@1")
    pe.add_argument("--cartridge", required=True, type=Path)
    pe.add_argument("--out-dir", required=True, type=Path)
    pe.add_argument("--target-module", default="mlp.shared_expert.down_proj")
    pe.add_argument("--num-layers", type=int, default=40)

    args = p.parse_args(argv)
    if args.cmd == "export-omlx":
        import numpy as np
        from safetensors import safe_open
        from safetensors.numpy import save_file

        cpath = args.cartridge.expanduser().resolve()
        out_dir = args.out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        tensors: Dict[str, Any] = {}
        with safe_open(str(cpath), framework="np") as f:
            meta = json.loads(f.metadata().get("weight_skills", "{}"))
            for k in f.keys():
                if k.startswith("layer_") and (k.endswith(".a") or k.endswith(".b")):
                    lnum, ab = k[6:].split(".", 1)
                    tensors[f"model.layers.{lnum}.{args.target_module}.lora_{ab}"] = np.ascontiguousarray(f.get_tensor(k).T)
        save_file(tensors, str(out_dir / "adapters.safetensors"))
        cfg = {
            "fine_tune_type": "lora",
            "num_layers": int(args.num_layers),
            "lora_parameters": {
                "rank": int(meta.get("rank", 16)),
                "scale": float(meta.get("alpha", 1.0)),
                "dropout": 0.0,
                "keys": [args.target_module],
            },
        }
        (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"Exported OMLX adapter ({len(tensors)} tensors) -> {out_dir}")
        return 0

    if args.cmd == "compile":
        bundle = compile_bundle(args.skills_dir, args.cartridges_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        ratio = (1.0 - bundle["compact_prefix_bytes"] / max(1, bundle["raw_bytes_total"])) * 100.0
        print(
            f"Compiled {bundle['skill_count']} skills, {bundle['cartridge_count']} cartridges -> {args.out} "
            f"(raw={bundle['raw_bytes_total']} B -> compact_prefix={bundle['compact_prefix_bytes']} B, -{ratio:.1f}%)"
        )
        return 0

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(bundle, args.upstream))
    print(f"Serving universal weight-skills proxy on http://{args.host}:{args.port} -> {args.upstream}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
