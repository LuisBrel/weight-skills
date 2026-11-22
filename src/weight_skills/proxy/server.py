"""OpenAI-compatible reverse proxy on :8080.

Request path
------------
1. Parse the model alias (``base+skill@ver+skill2``).
2. Strip a recognised harness preamble (SimHash) and collect the skills it
   implies.
3. Mount the union of alias skills and stripped-prompt skills through the
   engine's push-through backend.
4. Render the chat prompt and stream the completion back in OpenAI format.

Every response carries telemetry headers so a client can see what happened:

* ``X-Weight-Skills-Stripped``   bytes of system prompt removed
* ``X-Weight-Skills-Skill``      comma-separated mounted skills
* ``X-Weight-Skills-Mount-Ms``   mount latency
"""

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..proxy.dsl import AliasSpec, parse_alias
from ..proxy.engine import Engine, MockEngine
from ..proxy.policy import PolicyGate
from ..proxy.prompt_stripper import PromptStripper
from ..serving.compiled_prompts import ProvenanceError, resolve_prompt_key
from .engine import resolve_agent_prompt_key


@dataclass
class ProxyConfig:
    """Runtime configuration for the proxy."""

    engine: Engine
    stripper: PromptStripper = field(default_factory=PromptStripper)
    mount_per_request: bool = True
    default_max_tokens: int = 512
    request_timeout_s: float = 600.0
    gate: PolicyGate | None = None
    # skill id -> literal seed appended after the chat template (format lock).
    # A 4-token marker cannot be composed by a low-rank weight edit; seeding it
    # deterministically is what makes a compiled style cartridge adherent.
    skill_prefixes: dict[str, str] = field(default_factory=dict)
    # optional compiled-prompt library (see weight_skills.serving)
    prompt_library: Any = None
    # optional AgentCacheRegistry mapping X-Weight-Skills-Agent ids to prompt keys
    agent_registry: Any = None


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _chunk(model: str, completion_id: str, created: int, delta: dict[str, Any], finish: str | None = None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def create_app(config: ProxyConfig):
    """Build the FastAPI application (import is local so the core stays light)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="weight-skills proxy", version="0.1.0")
    state: dict[str, Any] = {"mounted": [], "requests": 0, "stripped_tokens": 0}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "model": config.engine.model_id, "mounted": state["mounted"]}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        created = int(time.time())
        entries = [
            {
                "id": config.engine.model_id,
                "object": "model",
                "created": created,
                "owned_by": "weight-skills",
            }
        ]
        for skill in config.engine.skills():
            entries.append(
                {
                    "id": f"{config.engine.model_id}+{skill['id']}",
                    "object": "model",
                    "created": created,
                    "owned_by": "weight-skills",
                }
            )
        return {"object": "list", "data": entries}

    @app.get("/v1/skills")
    def skills() -> dict[str, Any]:
        return {
            "data": config.engine.skills(),
            "mounted": state["mounted"],
            "strip_stats": config.stripper.summary(),
        }

    @app.post("/v1/mount")
    async def mount(request: Request) -> Any:
        body = await request.json()
        names = body.get("skills", [])
        if isinstance(names, str):
            names = [names]
        try:
            report = config.engine.mount(names, scale=float(body.get("scale", 1.0)))
        except KeyError as exc:
            return JSONResponse(status_code=404, content={"error": {"message": str(exc)}})
        state["mounted"] = list(report.get("skills", names))
        state["kv"] = list(report.get("kv", []))
        state["kv_mb"] = report.get("kv_mb")
        return report

    @app.post("/v1/unmount")
    def unmount() -> Any:
        report = config.engine.unmount()
        state["mounted"] = []
        state["kv"] = []
        state["kv_mb"] = None
        return report

    def _prepare(body: dict[str, Any]) -> tuple[AliasSpec, list[dict[str, Any]], Any, list[str]]:
        alias: AliasSpec = parse_alias(str(body.get("model") or config.engine.model_id))
        messages = body.get("messages") or []
        strip = config.stripper.strip(messages)
        skills = [s.name for s in alias.skills]
        if strip.skill_id and strip.skill_id not in skills:
            skills.append(strip.skill_id)
        state["stripped_tokens"] += strip.removed_tokens_estimate
        prepared = strip.messages if strip.stripped else messages
        return alias, prepared, strip, skills

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        state["requests"] += 1
        body = await request.json()
        gate = config.gate
        if gate is not None:
            verdict = gate.inspect_request(body.get("messages") or [])
            if not verdict.allowed:
                payload = {
                    "id": _completion_id(),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": str(body.get("model", config.engine.model_id)),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": gate.block_message},
                            "finish_reason": "content_filter",
                        }
                    ],
                    "policy": verdict.as_dict(),
                }
                return JSONResponse(
                    content=payload,
                    headers={
                        "X-Weight-Skills-Policy": f"blocked:{verdict.rule_id}",
                        "X-Weight-Skills-Skill": "",
                    },
                )
        try:
            alias, messages, strip, skills = _prepare(body)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})

        mount_ms = 0.0
        if config.mount_per_request and skills:
            try:
                report = config.engine.mount(skills, scale=float(body.get("skill_scale", 1.0)))
                mount_ms = float(report.get("latency_ms", 0.0))
                state["mounted"] = list(report.get("skills", skills))
                state["kv"] = list(report.get("kv", []))
                state["kv_mb"] = report.get("kv_mb")
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"error": {"message": str(exc)}})

        prefix = "".join(config.skill_prefixes.get(s, "") for s in state["mounted"])
        call_messages = list(messages)
        if prefix:
            call_messages = call_messages[:-1] + [
                {"role": call_messages[-1]["role"], "content": call_messages[-1]["content"] + prefix}
            ]
        prompt_chars = sum(len(m.get("content", "")) for m in call_messages)
        prompt_key = resolve_prompt_key(headers=request.headers, messages=messages)
        agent_id = (request.headers.get("X-Weight-Skills-Agent") or "").strip()
        if agent_id:
            if config.agent_registry is None:
                return JSONResponse(status_code=400, content={
                    "error": {"type": "unknown_agent", "message": "no agent registry configured on this server"}})
            resolved = resolve_agent_prompt_key(config.agent_registry, agent_id)
            if not resolved:
                return JSONResponse(status_code=400, content={
                    "error": {"type": "unknown_agent", "message": f"unknown agent {agent_id!r}"}})
            prompt_key = resolved
        prompt_hit = bool(
            prompt_key and config.prompt_library is not None and prompt_key in config.prompt_library
        )
        equivalence_label = ""
        resolve_equivalence = getattr(config.engine, "resolve_equivalence", None)
        if prompt_key and callable(resolve_equivalence):
            equivalence_label = resolve_equivalence(prompt_key)
        prompt_provenance = ""
        if prompt_key and config.prompt_library is not None:
            has_provenance = getattr(config.prompt_library, "has_provenance", None)
            if callable(has_provenance) and has_provenance(prompt_key):
                prompt_provenance = "verified"
            elif prompt_hit:
                prompt_provenance = "unverified"
        verify_prompt = getattr(config.engine, "verify_prompt", None)
        if prompt_hit and callable(verify_prompt):
            try:
                verify_prompt(call_messages, prompt_key)
            except ProvenanceError as exc:
                return JSONResponse(
                    status_code=409,
                    content={"error": {"type": "provenance_mismatch", "message": str(exc)}},
                )
        kv_names = list(state.get("kv", []))
        kv_mb = state.get("kv_mb")
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or config.default_max_tokens)
        temperature = float(body.get("temperature", 0.0))
        completion_id = _completion_id()
        created = int(time.time())
        policy_state = "allow"
        headers = {
            "X-Weight-Skills-Stripped": str(strip.removed_chars),
            "X-Weight-Skills-Skill": ",".join(state["mounted"]),
            "X-Weight-Skills-Mount-Ms": f"{mount_ms:.3f}",
            "X-Weight-Skills-Policy": policy_state,
            "X-Weight-Skills-Prefix": prefix,
            "X-Weight-Skills-KV": ",".join(kv_names),
            "X-Weight-Skills-Prompt-Cache": ("hit" if prompt_hit else "miss") if prompt_key else "",
            "X-Weight-Skills-Prompt-Provenance": prompt_provenance,
            "X-Weight-Skills-Equivalence": equivalence_label,
            "X-Weight-Skills-Agent": agent_id,
            "X-Weight-Skills-KV-MB": "" if kv_mb is None else f"{kv_mb:.2f}",
        }
        # Aligned-exact telemetry (D28/D30): exact = this request was served from an aligned cache;
        # cold = the aligned contract refused it (reason exposed); off = the mode is not selected.
        engine_policy = getattr(config.engine, "equivalence_policy", "off")
        aligned_state = "off"
        aligned_reason = ""
        if engine_policy == "aligned_exact":
            refusal = getattr(config.engine, "last_alignment_refusal", None)
            if refusal:
                aligned_state, aligned_reason = "cold", str(refusal)
            else:
                aligned_state = "exact"
        headers["X-Weight-Skills-Prompt-Aligned"] = aligned_state
        if aligned_reason:
            headers["X-Weight-Skills-Prompt-Aligned-Reason"] = aligned_reason

        if body.get("stream"):
            def event_stream() -> Iterator[str]:
                yield _chunk(alias.raw, completion_id, created, {"role": "assistant", "content": ""})
                buffer = ""
                for piece in config.engine.stream_messages(
                    call_messages, max_tokens=max_tokens, temperature=temperature, prompt_key=prompt_key
                ):
                    if not piece:
                        continue
                    buffer += piece
                    if gate is not None:
                        output_verdict = gate.inspect_output(buffer)
                        if not output_verdict.allowed:
                            yield _chunk(
                                alias.raw,
                                completion_id,
                                created,
                                {"content": gate.block_message},
                                finish="content_filter",
                            )
                            yield "data: [DONE]\n\n"
                            return
                    yield _chunk(alias.raw, completion_id, created, {"content": piece})
                yield _chunk(alias.raw, completion_id, created, {}, finish="stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(), media_type="text/event-stream", headers=headers
            )

        started = time.perf_counter()
        text = config.engine.complete_messages(
            call_messages, max_tokens=max_tokens, temperature=temperature, prompt_key=prompt_key
        )
        if gate is not None:
            text, output_verdict = gate.enforce_output(text)
            if not output_verdict.allowed:
                policy_state = f"blocked:{output_verdict.rule_id}"
                headers["X-Weight-Skills-Policy"] = policy_state
        elapsed = time.perf_counter() - started
        tokens = max(1, len(text) // 4)
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": alias.raw,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, prompt_chars // 4),
                "completion_tokens": tokens,
                "total_tokens": max(1, prompt_chars // 4) + tokens,
            },
            "weight_skills": {
                "stripped_chars": strip.removed_chars,
                "stripped_tokens_estimate": strip.removed_tokens_estimate,
                "skills": state["mounted"],
                "mount_ms": round(mount_ms, 4),
                "elapsed_s": round(elapsed, 4),
            },
        }
        return JSONResponse(content=payload, headers=headers)

    return app


def build_default_app(
    model_path: str | None = None,
    cartridge_dir: str = "cartridges",
    *,
    enable_policy: bool = False,
    skill_prefixes: dict[str, str] | None = None,
):
    """Entry point used by ``weight-skills serve``."""
    import os

    from ..serving.policy import decide_equivalence_policy

    raw_policy = os.environ.get("WEIGHT_SKILLS_EQUIVALENCE")
    if model_path:
        from ..proxy.engine import MLXEngine

        # The MLX engine carries the aligned serving path (D28/D30), so it may advertise it to the
        # policy resolver; the mock engine must stay fail-closed for aligned_exact.
        decision = decide_equivalence_policy(
            model_path, env_value=raw_policy, supports_aligned_exact=True
        )
        engine: Engine = MLXEngine(
            model_path=model_path, cartridge_dir=cartridge_dir, equivalence_policy=decision.mode
        )
    else:
        decision = decide_equivalence_policy("mock-base", env_value=raw_policy)
        engine = MockEngine(equivalence_policy=decision.mode)
    gate = PolicyGate() if enable_policy else None
    return create_app(
        ProxyConfig(engine=engine, gate=gate, skill_prefixes=dict(skill_prefixes or {}))
    )
