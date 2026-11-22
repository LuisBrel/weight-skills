"""Inference engines for the proxy.

``MLXEngine`` is the production engine (Apple Silicon). ``MockEngine`` is a
deterministic stand-in used by the test-suite and by ``weight-skills doctor``
on machines without MLX, so the proxy can be exercised end-to-end anywhere.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


def _validate_equivalence_policy(mode: str, *, supports_aligned_exact: bool = False) -> None:
    """Fail closed for modes this engine cannot honour.

    ``MLXEngine`` now carries the aligned serving path (D28/D30: suffix starts at the aligned cut and the
    serving chunk is checked against the recorded one) and passes ``supports_aligned_exact=True``.
    ``MockEngine`` has no such path, so selecting ``aligned_exact`` on it raises instead of silently
    serving a sampled cache.
    """
    from ..serving.policy import ALIGNED_EXACT_MODE
    if mode == ALIGNED_EXACT_MODE and not supports_aligned_exact:
        raise ValueError(
            "equivalence_policy='aligned_exact' requires the aligned serving path (D28/D30), which is "
            "not enabled for this engine; use 'off' or 'fallback'"
        )


def resolve_equivalence_label(prompt_library: Any, prompt_key: str | None, policy: str) -> str:
    """Return the serving label for a prompt key.

    ``""`` when there is no cache; the recorded label (``"certified"`` / ``"unverified"``) when the
    policy is ``"off"``; and ``"cache_certified"`` / ``"cold_fallback"`` when ``policy == "fallback"``,
    which is the path the engine actually serves.
    """
    if not prompt_key or prompt_library is None or prompt_key not in prompt_library:
        return ""
    label = None
    equivalence_of = getattr(prompt_library, "equivalence_of", None)
    if callable(equivalence_of):
        label = equivalence_of(prompt_key)
    label = label or "unverified"
    if policy == "fallback":
        return "cache_certified" if label == "certified" else "cold_fallback"
    return label


def physical_ram_bytes() -> int:
    """Total physical RAM in bytes (POSIX sysconf; 0 if unavailable)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def ram_budget_gb(fraction: float = 0.75, override: float | None = None) -> float:
    """RAM budget for a prefill plan (explicit override wins; else a fraction of physical RAM)."""
    if override is not None:
        if override <= 0:
            raise ValueError("override budget must be positive")
        return float(override)
    total = physical_ram_bytes()
    if total <= 0:
        raise ValueError("cannot determine physical RAM; pass budget_gb explicitly")
    return round(total / 1e9 * fraction, 1)


def model_memory_plan(args: Any, weights_gb: float, context_tokens: int, *, budget_gb: float | None = None,
                      dtype_bytes: int = 2, overhead_gb: float = 2.0) -> dict[str, Any]:
    """Plan the prefill chunk for a model config (pure; no model load, no generation).

    Derives the attention heads, full-attention layers and KV bytes/token from ``args`` and calls
    :func:`weight_skills.serving.plan_prefill_chunk`. Returns the plan plus the estimate. ``chunk`` is
    ``None`` when the budget cannot hold even the smallest chunk (caller should raise a clear error).
    """
    from ..serving.memory import estimate_peak, plan_prefill_chunk

    # Multimodal wrappers (e.g. qwen3_5_moe) keep the text architecture in ``args.text_config``.
    nested = getattr(args, "text_config", None)
    cfg = nested if nested is not None else args

    def field(name: str, default: Any = None) -> Any:
        source = cfg
        if isinstance(source, dict):
            return source.get(name, default)
        value = getattr(source, name, default)
        if value is None and nested is not None:
            value = args.get(name, default) if isinstance(args, dict) else getattr(args, name, default)
        return value

    heads = int(field("num_attention_heads", 0) or 0)
    if heads <= 0:
        raise ValueError("model args carry no num_attention_heads; cannot estimate the attention buffer")
    layers = int(field("num_hidden_layers", 0) or 0)
    layer_types = field("layer_types")
    if layer_types:
        full_layers = sum(1 for t in layer_types if str(t) == "full_attention")
    else:
        full_layers = layers
    kv_heads = int(field("num_key_value_heads", 0) or heads)
    head_dim = int(field("head_dim", 0) or 0)
    if head_dim <= 0:
        hidden = int(field("hidden_size", 0) or 0)
        head_dim = hidden // heads if heads else 0
    if head_dim <= 0:
        raise ValueError("model args carry no head_dim/hidden_size; cannot estimate KV bytes")
    budget = ram_budget_gb(override=budget_gb) if budget_gb is None else float(budget_gb)
    kv_bytes_per_token = 2 * full_layers * kv_heads * head_dim * dtype_bytes
    chunk = plan_prefill_chunk(weights_gb=weights_gb, context_tokens=context_tokens,
                              num_attention_heads=heads, kv_bytes_per_token=kv_bytes_per_token,
                              budget_gb=budget, dtype_bytes=dtype_bytes, overhead_gb=overhead_gb)
    estimate = estimate_peak(weights_gb=weights_gb, context_tokens=context_tokens,
                             chunk_tokens=chunk or 256, num_attention_heads=heads,
                             kv_bytes_per_token=kv_bytes_per_token, dtype_bytes=dtype_bytes,
                             overhead_gb=overhead_gb)
    return {"chunk": chunk, "budget_gb": budget, "weights_gb": round(weights_gb, 2),
            "num_attention_heads": heads, "full_attention_layers": full_layers,
            "head_dim": head_dim, "kv_heads": kv_heads, "kv_bytes_per_token": kv_bytes_per_token,
            "estimate_gb": round(estimate.total_gb, 2), "context_tokens": context_tokens}


def resolve_agent_prompt_key(agent_registry: Any, agent_id: str | None) -> str | None:
    """Resolve a crew/agent id to a compiled prompt key via an ``AgentCacheRegistry``-like object.

    Returns ``None`` when the id is unknown or no registry is configured; callers use that to reject the
    request explicitly instead of silently serving a different cache.
    """
    if not agent_id or agent_registry is None:
        return None
    key_for = getattr(agent_registry, "key_for", None)
    return key_for(agent_id) if callable(key_for) else None


class Engine(Protocol):
    """What the proxy needs from a model runtime."""

    model_id: str

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str: ...

    def stream(self, prompt: str, *, max_tokens: int, temperature: float) -> Iterator[str]: ...

    def mount(self, skills: Sequence[str], *, scale: float = 1.0) -> dict[str, Any]: ...

    def unmount(self) -> dict[str, Any]: ...

    def skills(self) -> list[dict[str, Any]]: ...

    def render_chat(self, messages: list[dict[str, str]]) -> str: ...

    def complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        prompt_key: str | None = None,
    ) -> str: ...

    def stream_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        prompt_key: str | None = None,
    ) -> Iterator[str]: ...


@dataclass
class MockEngine:
    """Deterministic engine used for tests and dry runs."""

    model_id: str = "mock-base"
    prefix: str = "mock"
    calls: list[dict[str, Any]] = field(default_factory=list)
    mounted: list[str] = field(default_factory=list)
    registry: dict[str, str] = field(default_factory=dict)
    prompt_library: Any = None
    equivalence_policy: str = "off"

    def __post_init__(self) -> None:
        _validate_equivalence_policy(self.equivalence_policy)

    def resolve_equivalence(self, prompt_key: str | None) -> str:
        """Serving label for a prompt key (see :func:`resolve_equivalence_label`)."""
        return resolve_equivalence_label(self.prompt_library, prompt_key, self.equivalence_policy)

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
        skills = "+".join(self.mounted) if self.mounted else "none"
        return f"[{self.prefix}] skills={skills} :: {prompt.strip()[:120]}"

    def stream(self, prompt: str, *, max_tokens: int, temperature: float) -> Iterator[str]:
        text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        for word in text.split(" "):
            yield word + " "

    def mount(self, skills: Sequence[str], *, scale: float = 1.0) -> dict[str, Any]:
        if self.registry:
            missing = [s for s in skills if s not in self.registry]
            if missing:
                raise KeyError(f"unknown skill(s): {missing}; available: {sorted(self.registry)}")
        t0 = time.perf_counter()
        self.mounted = list(skills)
        return {
            "backend": "mock",
            "skills": self.mounted,
            "scale": scale,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 4),
            "restored": None,
        }

    def unmount(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        previous = self.mounted
        self.mounted = []
        return {
            "backend": "mock",
            "unmounted": previous,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 4),
            "restored": True,
        }

    def skills(self) -> list[dict[str, Any]]:
        return [{"id": skill_id, "path": path} for skill_id, path in self.registry.items()]

    def render_chat(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        prompt_key: str | None = None,
    ) -> str:
        return self.complete(self.render_chat(messages), max_tokens=max_tokens, temperature=temperature)

    def stream_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        prompt_key: str | None = None,
    ):
        yield from self.stream(self.render_chat(messages), max_tokens=max_tokens, temperature=temperature)


@dataclass
class MLXEngine:
    """Apple Silicon engine backed by mlx-lm and the push-through backend.

    ``equivalence_policy="aligned_exact"`` serves only *aligned caches* (D28/D30): a cache stored from
    ``rendered_prefix_ids[:cut]`` with provenance ``prefill_chunk`` == ``prefill_step_size`` and
    ``aligned_extra_tokens = n - cut``. The request suffix then starts at ``cut`` (the re-prefilled
    prefix tokens plus the user turn); any cache that cannot prove that contract is served cold and the
    reason is exposed as :attr:`last_alignment_refusal`.
    """

    model_path: str
    model_id: str = ""
    cartridge_dir: str = "cartridges"
    kv_dir: str = "cartridges/kv"
    prompt_library: Any = None
    equivalence_policy: str = "off"
    agent_registry: Any = None
    #: Serving prefill chunk (tokens) that aligned caches must have been compiled with.
    prefill_step_size: int = 2048
    #: Reason the last aligned attempt was refused (``None`` when it was served aligned or mode is off).
    last_alignment_refusal: str | None = None
    _model: Any = None
    _tokenizer: Any = None
    _backend: Any = None
    _cartridges: dict[str, Any] = field(default_factory=dict)
    _mounted: list[str] = field(default_factory=list)
    _kv: dict[str, Any] = field(default_factory=dict)
    _active_kv: Any = None

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = self.model_path.rstrip("/").split("/")[-1]
        _validate_equivalence_policy(self.equivalence_policy, supports_aligned_exact=True)

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._model is not None:
            return
        import mlx_lm

        from ..backends.mlx_backend import MLXBackend

        self._model, self._tokenizer = mlx_lm.load(self.model_path)
        self._backend = MLXBackend(self._model)

    def _discover_cartridges(self) -> dict[str, Any]:
        from pathlib import Path

        from ..core.cartridge import Cartridge

        base = Path(self.cartridge_dir)
        found: dict[str, Any] = {}
        if base.is_dir():
            for path in sorted(base.rglob("*.safetensors")):
                try:
                    cart = Cartridge.load(path)
                except Exception:  # noqa: BLE001 - skip foreign files
                    continue
                found[cart.skill_id] = cart
        return found

    def _discover_kv(self) -> dict[str, Any]:
        from pathlib import Path

        from ..core.kv_cartridge import KVCartridge

        found: dict[str, Any] = {}
        base = Path(self.kv_dir)
        if base.is_dir():
            for path in sorted(base.rglob("*.safetensors")):
                try:
                    cart = KVCartridge.load(path)
                except Exception:  # noqa: BLE001 - skip foreign files
                    continue
                found[cart.skill_id] = cart
        return found

    def kv_memory_mb(self, cart: Any) -> float | None:
        """Estimated KV residency of a prefilled demonstration prefix."""
        try:
            args = self._model.args
            layers = int(getattr(args, "num_hidden_layers", 0))
            heads = int(getattr(args, "num_key_value_heads", 0))
            head_dim = int(getattr(args, "head_dim", 0))
            if not head_dim:
                head_dim = int(getattr(args, "hidden_size", 0)) // max(
                    1, int(getattr(args, "num_attention_heads", 1))
                )
            if not (layers and heads and head_dim):
                return None
            return round(cart.n_tokens * layers * 2 * heads * head_dim * 2 / 1e6, 2)
        except Exception:  # noqa: BLE001
            return None

    def skills(self) -> list[dict[str, Any]]:
        if not self._cartridges:
            self._cartridges = self._discover_cartridges()
        if not self._kv:
            self._kv = self._discover_kv()
        weight = [
            {
                "id": cart.skill_id,
                "version": cart.version,
                "rank": cart.rank,
                "layers": list(cart.layer_indices),
                "nbytes": cart.nbytes,
                "path": str(cart.path) if cart.path else None,
                "mounted": cart.skill_id in self._mounted,
            }
            for cart in self._cartridges.values()
        ]
        weight += [
            {
                "id": cart.skill_id,
                "kind": "kv",
                "n_tokens": cart.n_tokens,
                "memory_mb": self.kv_memory_mb(cart),
                "path": str(cart.path) if cart.path else None,
                "mounted": self._active_kv is cart,
            }
            for cart in self._kv.values()
        ]
        return weight

    # ----------------------------------------------------------------- mount
    def mount(self, skills: Sequence[str], *, scale: float = 1.0) -> dict[str, Any]:
        self.load()
        if not self._cartridges:
            self._cartridges = self._discover_cartridges()
        if not self._kv:
            self._kv = self._discover_kv()
        weight, kv, missing = [], [], []
        for name in skills:
            if name in self._cartridges:
                weight.append(self._cartridges[name])
            elif name in self._kv:
                kv.append(self._kv[name])
            else:
                missing.append(name)
        if missing:
            raise KeyError(
                f"unknown skill(s): {missing}; available weight={sorted(self._cartridges)} "
                f"kv={sorted(self._kv)}"
            )
        base: dict[str, Any] = {"backend": "mlx", "latency_ms": 0.0, "layers": []}
        if weight:
            report = self._backend.mount(weight, scale=scale)
            base = report.as_dict()
        self._mounted = [c.skill_id for c in weight]
        self._active_kv = kv[0] if kv else None
        base["skills"] = list(skills)
        base["kv"] = [c.skill_id for c in kv]
        base["kv_mb"] = self.kv_memory_mb(self._active_kv) if self._active_kv else None
        return base

    def unmount(self) -> dict[str, Any]:
        previous, self._mounted = self._mounted, []
        self._active_kv = None
        if self._backend is None:
            return {"backend": "mlx", "unmounted": previous, "restored": True, "latency_ms": 0.0}
        report = self._backend.unmount()
        payload = report.as_dict()
        payload["unmounted"] = previous
        return payload

    # -------------------------------------------------------------- generate
    def render_chat(self, messages: list[dict[str, str]]) -> str:
        self.load()
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:  # noqa: BLE001
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:  # noqa: BLE001
                return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def _prefix_ids(self, messages: list[dict[str, str]]) -> list[int] | None:
        """Token ids of the rendered system prefix (the part a compiled cache stands for)."""
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system is None:
            return None
        self.load()
        try:
            prefix_text = self._tokenizer.apply_chat_template(
                [system], tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        except Exception:  # noqa: BLE001
            prefix_text = self._tokenizer.apply_chat_template(
                [system], tokenize=False, add_generation_prompt=False
            )
        return list(self._tokenizer.encode(prefix_text))

    def resolve_equivalence(self, prompt_key: str | None) -> str:
        """Serving label for a prompt key (see :func:`resolve_equivalence_label`)."""
        return resolve_equivalence_label(self.prompt_library, prompt_key, self.equivalence_policy)

    def _aligned_serve_plan(self, provenance: Any, serving_chunk: int) -> tuple[str, str]:
        """Enforce the aligned-exact contract for one compiled cache (D28/D30).

        Returns ``("aligned_exact", "")`` when the cache may be served from the compiled path, else
        ``("cold", reason)``; a caller must re-prefill cold on refusal and may expose ``reason`` as a
        header or log line. This is the enforcement primitive for
        ``equivalence_policy="aligned_exact"`` (D30): a non-aligned cache is never served as exact.
        """
        from ..serving.policy import decide_aligned_serving

        return decide_aligned_serving(provenance, serving_chunk)

    def _recorded_provenance(self, prompt_key: str | None) -> Any:
        """Recorded provenance for a compiled-prompt key, or ``None`` when unavailable."""
        if not prompt_key or self.prompt_library is None:
            return None
        provenance_of = getattr(self.prompt_library, "provenance_of", None)
        return provenance_of(prompt_key) if callable(provenance_of) else None

    def _aligned_prompt_plan(
        self, prompt_key: str | None, prefix_tokens_total: int
    ) -> tuple[str, int | None, str]:
        """Plan aligned serving for this request against ``prefill_step_size`` (pure, no generation)."""
        from ..serving.compiled_prompts import plan_aligned_prompt

        return plan_aligned_prompt(
            self._recorded_provenance(prompt_key), prefix_tokens_total, self.prefill_step_size
        )

    def plan_prefill(self, context_tokens: int, *, budget_gb: float | None = None,
                     dtype_bytes: int = 2, overhead_gb: float = 2.0) -> dict[str, Any]:
        """Memory plan for compiling/prefilling a prefix of ``context_tokens`` (no generation)."""
        self.load()
        from mlx.utils import tree_flatten

        weights_gb = sum(int(p.nbytes) for _, p in tree_flatten(self._model.parameters())) / 1e9
        return model_memory_plan(self._model.args, weights_gb, context_tokens, budget_gb=budget_gb,
                                 dtype_bytes=dtype_bytes, overhead_gb=overhead_gb)

    def compile_prefix_cache(self, prefix_ids: Sequence[int], *, budget_gb: float | None = None,
                             dtype_bytes: int = 2, overhead_gb: float = 2.0) -> dict[str, Any]:
        """Compile ``prefix_ids`` into a KV cache, auto-selecting the prefill chunk from the RAM budget.

        This is the D26 integration: the one-time compile pass is the memory peak (G.80), so the chunk
        is planned with the attention-buffer formula; an impossible budget raises ``ValueError`` with the
        estimate instead of crashing inside the Metal allocator.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        plan = self.plan_prefill(len(prefix_ids), budget_gb=budget_gb, dtype_bytes=dtype_bytes,
                                 overhead_gb=overhead_gb)
        if plan["chunk"] is None:
            raise ValueError(
                f"cannot fit a {len(prefix_ids)}-token prefill in budget {plan['budget_gb']} GB "
                f"(weights {plan['weights_gb']} GB); reduce context, raise budget_gb, or offload"
            )
        cache = make_prompt_cache(self._model)
        mx.reset_peak_memory()
        for _t, _lp in generate_step(mx.array(list(prefix_ids)), self._model, max_tokens=0,
                                     prompt_cache=cache, sampler=None, prefill_step_size=plan["chunk"]):
            pass
        plan["cache"] = cache
        plan["measured_peak_gb"] = round(float(mx.get_peak_memory()) / 1e9, 2)
        return plan

    def verify_prompt(self, messages: list[dict[str, str]], prompt_key: str | None) -> None:
        """Structurally verify a compiled cache against the request prefix (no generation).

        Raises :class:`~weight_skills.serving.ProvenanceError` when the library entry carries a prefix
        checksum that does not match this request, or when the model binding differs. Entries without a
        recorded checksum are accepted unless the library was built with ``require_provenance=True``.
        This is the level-1 gate measured in G.61 (16/16 misroutes rejected, ~0.07 ms).

        In ``aligned_exact`` mode the request is first planned (see
        :func:`~weight_skills.serving.plan_aligned_prompt`): the recorded cache covers ``full[:cut]``, so
        verification hashes exactly those ids. On refusal the request is left for the cold path and the
        reason is exposed through :attr:`last_alignment_refusal`; a cache is never served as exact here.
        """
        aligned = self.equivalence_policy == "aligned_exact"
        if aligned:
            self.last_alignment_refusal = None
        if not prompt_key or self.prompt_library is None or prompt_key not in self.prompt_library:
            return
        prefix = self._prefix_ids(messages)
        if prefix is None:
            return
        if aligned:
            mode, cut, reason = self._aligned_prompt_plan(prompt_key, len(prefix))
            if mode != "aligned_exact" or cut is None:
                self.last_alignment_refusal = reason
                return
            verify = getattr(self.prompt_library, "verify", None)
            if callable(verify):
                verify(prompt_key, prefix_token_ids=prefix[:cut],
                       model_id=self.model_id or self.model_path)
            return
        verify = getattr(self.prompt_library, "verify", None)
        if not callable(verify):
            return
        verify(prompt_key, prefix_token_ids=prefix, model_id=self.model_id or self.model_path)

    def _compiled_prompt(self, messages: list[dict[str, str]], prompt_key: str | None) -> Any:
        """Return (suffix_ids, cache) for a compiled system prompt, or None.

        ``off`` / ``fallback`` keep the original full-prefix behaviour byte-identically. In
        ``aligned_exact`` mode the cache must be an *aligned cache* (see
        :func:`~weight_skills.serving.plan_aligned_prompt`): the served suffix starts at ``cut`` (the
        ``W`` re-prefilled prefix tokens plus the user turn) and the cache is verified against
        ``prefix[:cut]``. Any refusal returns None so the caller re-prefills cold, records the reason in
        :attr:`last_alignment_refusal`, and never serves a sampled cache under this mode.
        """
        aligned = self.equivalence_policy == "aligned_exact"
        if not prompt_key or self.prompt_library is None or prompt_key not in self.prompt_library:
            if aligned:
                # a request without a compiled cache is served cold; say so instead of reporting "exact"
                self.last_alignment_refusal = "no compiled cache for this request (served cold)"
            return None
        self.load()
        prefix = self._prefix_ids(messages)
        if prefix is None:
            if aligned:
                self.last_alignment_refusal = "no rendered system prefix to align"
            return None
        ids = list(self._tokenizer.encode(self.render_chat(messages)))
        if ids[: len(prefix)] != prefix:
            if aligned:
                self.last_alignment_refusal = (
                    "rendered prompt does not begin with the rendered system prefix"
                )
            return None
        cut = len(prefix)
        if aligned:
            mode, planned_cut, reason = self._aligned_prompt_plan(prompt_key, len(prefix))
            if mode != "aligned_exact" or planned_cut is None:
                self.last_alignment_refusal = reason
                return None
            cut = planned_cut
            self.last_alignment_refusal = None
        get_verified = getattr(self.prompt_library, "get_verified", None)
        if callable(get_verified):
            cache = get_verified(
                prompt_key,
                prefix_token_ids=prefix[:cut],
                model_id=self.model_id or self.model_path,
            )
        else:
            cache = self.prompt_library.get(prompt_key)
        if cache is None:
            if aligned:
                self.last_alignment_refusal = "compiled cache is not resident"
            return None
        if self.equivalence_policy == "fallback" and self.resolve_equivalence(prompt_key) == "cold_fallback":
            return None
        return ids[cut:], cache

    def _stream_ids(self, ids: list[int], cache: Any, max_tokens: int, temperature: float):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        for response in stream_generate(
            self._model,
            self._tokenizer,
            ids,
            max_tokens=max_tokens,
            prompt_cache=cache,
            prefill_step_size=self.prefill_step_size,
            sampler=make_sampler(temp=temperature),
        ):
            yield response.text

    def complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        prompt_key: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        self.last_alignment_refusal = None
        if prompt_key is None and agent_id is not None:
            prompt_key = resolve_agent_prompt_key(self.agent_registry, agent_id)
        compiled = self._compiled_prompt(messages, prompt_key)
        if compiled is not None:
            suffix, cache = compiled
            return "".join(self._stream_ids(suffix, cache, max_tokens, temperature))
        if self.equivalence_policy == "aligned_exact" and self.last_alignment_refusal is None:
            self.last_alignment_refusal = "no aligned serving path for this request (served cold)"
        if self._active_kv is not None:
            prompt = [m["content"] for m in messages if m.get("role") == "user"][-1]
            self.load()
            return self._active_kv.generate(
                self._model, self._tokenizer, prompt, max_tokens=max_tokens, temperature=temperature
            )
        return self.complete(self.render_chat(messages), max_tokens=max_tokens, temperature=temperature)

    def stream_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        prompt_key: str | None = None,
        agent_id: str | None = None,
    ):
        if prompt_key is None and agent_id is not None:
            prompt_key = resolve_agent_prompt_key(self.agent_registry, agent_id)
        compiled = self._compiled_prompt(messages, prompt_key)
        if compiled is not None:
            suffix, cache = compiled
            yield from self._stream_ids(suffix, cache, max_tokens, temperature)
            return
        if self._active_kv is not None:
            prompt = [m["content"] for m in messages if m.get("role") == "user"][-1]
            self.load()
            yield from self._active_kv.stream(
                self._model, self._tokenizer, prompt, max_tokens=max_tokens, temperature=temperature
            )
            return
        yield from self.stream(self.render_chat(messages), max_tokens=max_tokens, temperature=temperature)

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        return "".join(self.stream(prompt, max_tokens=max_tokens, temperature=temperature))

    def stream(self, prompt: str, *, max_tokens: int, temperature: float) -> Iterator[str]:
        self.load()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            prefill_step_size=self.prefill_step_size,
            sampler=sampler,
        ):
            yield response.text
