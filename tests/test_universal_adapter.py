"""Portable tests for scripts/universal_adapter.py (compile, route, tools, streaming, Anthropic/OpenAI proxy)."""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import universal_adapter as ua  # noqa: E402


def _make_skills(root: Path) -> Path:
    s1 = root / "coder"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(
        "---\nname: python-coder\ndescription: >-\n  Fix Python bugs, refactor modules, and run pytest cleanly.\n---\n\nAlways run pytest after edits.\n",
        encoding="utf-8",
    )
    s2 = root / "research"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text(
        "---\nname: literature-synthesizer\ndescription: Synthesize arXiv papers and empirical benchmarks.\n---\n\nCite exact artifact SHAs.\n",
        encoding="utf-8",
    )
    return root


def test_compile_relative_paths_circular_symlinks_and_abstention(tmp_path: Path) -> None:
    skills_dir = _make_skills(tmp_path / "skills")
    # Circular symlink must not cause infinite recursion
    os.symlink(skills_dir, skills_dir / "loop")
    bundle = ua.compile_bundle([skills_dir])
    assert bundle["schema"] == ua.BUNDLE_SCHEMA
    assert bundle["skill_count"] == 2
    # Paths stored in bundle must be relative, never exposing host home directory
    for sk in bundle["skills"]:
        assert not Path(sk["path"]).is_absolute()

    hit = ua.route_query(bundle, "Fix python bugs and run pytest")
    assert hit is not None and hit["skill"]["name"] == "python-coder"
    # Stopwords alone must abstain
    assert ua.route_query(bundle, "the and of in") is None
    assert ua.route_query(bundle, "zzzzqqqqxxxx") is None


def test_anthropic_tools_streaming_and_openai_proxy(tmp_path: Path) -> None:
    skills_dir = _make_skills(tmp_path / "skills")
    bundle = ua.compile_bundle([skills_dir])

    class MockUpstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", "0") or "0")
            req_body = json.loads(self.rfile.read(n).decode("utf-8"))
            sys_msg = req_body["messages"][0]["content"]
            assert "# Weight-Skills Regime-1 Index" in sys_msg
            assert "tools" in req_body and req_body["tools"][0]["function"]["name"] == "run_tests"
            if req_body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n')
                return
            resp = {
                "id": "chatcmpl_mock",
                "model": req_body.get("model", "local"),
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "run_tests", "arguments": '{"path":"tests/"}'}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 42, "completion_tokens": 5},
            }
            raw = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args) -> None:
            return

    up = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstream)
    up_port = up.server_address[1]
    threading.Thread(target=up.serve_forever, daemon=True).start()

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ua.make_handler(bundle, f"http://127.0.0.1:{up_port}"))
    pr_port = proxy.server_address[1]
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        # 1. Both /health and /healthz work
        for ep in ("/health", "/healthz"):
            with urllib.request.urlopen(f"http://127.0.0.1:{pr_port}{ep}", timeout=5) as r:
                assert json.loads(r.read().decode())["ok"] is True

        # 2. Anthropic tool-calling round-trip
        r_anth = urllib.request.Request(
            f"http://127.0.0.1:{pr_port}/v1/messages",
            data=json.dumps({
                "model": "qwen-35b",
                "tools": [{"name": "run_tests", "description": "Run pytest", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
                "messages": [{"role": "user", "content": "Run pytest on python code"}],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r_anth, timeout=10) as resp:
            d_anth = json.loads(resp.read().decode())
        assert d_anth["stop_reason"] == "tool_use"
        assert d_anth["content"][0]["type"] == "tool_use"
        assert d_anth["content"][0]["name"] == "run_tests"

        # 3. Anthropic streaming SSE synthesis
        r_stream = urllib.request.Request(
            f"http://127.0.0.1:{pr_port}/v1/messages",
            data=json.dumps({
                "model": "qwen-35b",
                "stream": True,
                "tools": [{"name": "run_tests", "description": "Run pytest", "input_schema": {"type": "object", "properties": {}}}],
                "messages": [{"role": "user", "content": "Run pytest"}],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r_stream, timeout=10) as resp:
            sse = resp.read().decode()
        assert "event: message_start" in sse and "event: message_stop" in sse
    finally:
        proxy.shutdown()
        proxy.server_close()
        up.shutdown()
        up.server_close()
