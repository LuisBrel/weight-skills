"""Compiled prompt-cache library: prefill once, fork per request.

A *compiled prompt* is the attention KV cache produced by prefilling a fixed
system prompt through a model. Serving it from a library means those tokens are
encoded once and every request pays only for its own user turn -- the compiled
prompt never appears in the caller's prompt and never re-enters the encoder.

``CompiledPromptLibrary`` owns the resident caches and provides:

* ``store(name, cache, n_tokens)`` / ``get(name)``. ``get`` returns a fresh
  ``deepcopy`` (generation mutates a prompt cache) or ``None`` on a miss.
* LRU eviction against a byte budget. The size of an entry is read from the
  cache tensors themselves (``.nbytes`` attributes, ``.state`` mappings, lists
  of per-layer caches) or from a caller-supplied estimator.
* hit / miss / eviction / disk counters, exposed by :meth:`CompiledPromptLibrary.stats`.
* optional safetensors persistence through ``mlx_lm.models.cache``. The MLX
  import is lazy, so importing this module never loads MLX and CPU tests run
  without a GPU.
* prompt-hash routing: :func:`prompt_key` is the SHA-256 of the system text,
  and ``store_system`` / ``get_system`` use it as the cache key.

Usage
-----
>>> from weight_skills.serving import CompiledPromptLibrary, prompt_key
>>> library = CompiledPromptLibrary(max_bytes=2 * 1024**3)
>>> library.store("clu-persona", prompt_cache, n_tokens=1129)
>>> cache = library.get("clu-persona")        # fresh deepcopy, or None on a miss
>>> library.stats()["hit_rate"]

Routing by prompt hash (the key the serving layer puts on the wire)::

>>> key = prompt_key(system_message["content"])
>>> library.store_system(system_message["content"], prompt_cache, n_tokens=1129)
>>> key == prompt_key(system_message["content"])
True
>>> library.get_system(system_message["content"]) is not None
True

A library with a custom size estimator (useful for caches that do not expose
``.nbytes``)::

>>> library = CompiledPromptLibrary(max_bytes=512 * 1024**2, estimator=lambda c: c.size * 2)

Persistence (only this path touches MLX; it is imported lazily on first use)::

>>> library.save("clu-persona", "cartridges/compiled/clu-persona.safetensors")
>>> library.load("cartridges/compiled/clu-persona.safetensors")
'clu-persona'
>>> library.save_all("cartridges/compiled")   # every resident cache
>>> library.load_dir("cartridges/compiled")   # register everything on startup

Engine integration
------------------
The proxy engine resolves a key from an ``X-Weight-Skills-Prompt-Key`` header
or, failing that, from the SHA-256 of the first system message:

>>> from weight_skills.serving import resolve_prompt_key
>>> key = resolve_prompt_key(headers=request_headers, messages=chat_messages)
>>> cache = library.get(key) if key else None

When ``cache`` is not ``None`` generation continues from the compiled prefix and
only the user turn is encoded (``stream_generate(..., prompt_cache=cache)``).
When it is ``None`` the engine falls back to the normal render-and-prefill path
and can then compile the prompt once and ``store`` it.

Threading
---------
All bookkeeping is guarded by a re-entrant lock. Deepcopy of the returned cache
happens outside the lock, so concurrent requests fork independently. The byte
budget counts resident canonical caches only: transient per-request forks are
the caller's memory to manage.

Limitations
-----------
* No model or tokenizer is loaded here; the caller compiles the cache.
* Persistence stores whatever ``mlx_lm.models.cache`` serialises; the default
  serializer requires an MLX prompt cache (a list of per-layer cache objects).
* A single entry larger than ``max_bytes`` is rejected instead of silently
  evicting half the resident set.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from ..errors import WeightSkillsError

__all__ = [
    "DEFAULT_MAX_BYTES",
    "PROMPT_KEY_HEADER",
    "PROVENANCE_SCHEMA",
    "AGENT_KEY_SCHEMA",
    "agent_cache_key",
    "text_sha256",
    "CacheProvenance",
    "aligned_exact_contract",
    "plan_aligned_prompt",
    "CacheSerializer",
    "CacheSizeError",
    "ProvenanceError",
    "token_ids_sha256",
    "CompiledPromptLibrary",
    "MLXPromptCacheSerializer",
    "estimate_cache_nbytes",
    "key_from_messages",
    "prompt_key",
    "resolve_prompt_key",
]

#: Default memory budget for resident compiled caches (1 GiB).
DEFAULT_MAX_BYTES = 1024**3

#: Request header carrying a compiled-prompt cache key.
PROMPT_KEY_HEADER = "X-Weight-Skills-Prompt-Key"

#: On-disk schema tag written into safetensors metadata.
SCHEMA = "weight-skills/compiled-prompt@1"
# NB: dot-free metadata keys. mlx_lm's save_prompt_cache flattens the metadata dict with
# ``tree_flatten`` and restores it with ``tree_unflatten``, and dots are the tree separator: a key like
# "weight-skills.schema" comes back as {"weight-skills": {"schema": ...}} (observed on mlx-lm 0.31.3,
# 2026-11-13). Dot-free keys round-trip verbatim; the load path still normalises the old dotted shape.
META_SCHEMA = "weight_skills_schema"
META_CACHE_KEY = "weight_skills_cache_key"
META_N_TOKENS = "weight_skills_n_tokens"
META_SAVED_UTC = "weight_skills_saved_utc"
#: Provenance schema tag: binds a compiled cache to the token prefix it was built from.
PROVENANCE_SCHEMA = 1
#: Metadata key carrying the JSON-encoded :class:`CacheProvenance`.
META_PROVENANCE = "weight_skills_provenance"
#: Legacy (dotted) metadata keys written before 2026-11-13; normalised to the dot-free form on load so
#: caches persisted by an older build remain readable where the nesting can be reconstructed.
_LEGACY_META = {
    "weight-skills.schema": META_SCHEMA,
    "weight-skills.cache_key": META_CACHE_KEY,
    "weight-skills.n_tokens": META_N_TOKENS,
    "weight-skills.saved_utc": META_SAVED_UTC,
    "weight-skills.provenance": META_PROVENANCE,
}


def normalise_metadata(metadata: Any) -> dict[str, str]:
    """Return the flat metadata dict this library expects, tolerating old/rounded shapes.

    ``mlx_lm``'s ``tree_unflatten`` splits dotted keys into nested dicts, so an old artefact loads as
    ``{"weight-skills": {"schema": ..., "n_tokens": ...}}``. Re-flatten those one or two levels and map
    legacy dotted keys onto the current dot-free names.
    """
    out: dict[str, str] = {}
    stack: list[tuple[str, Any]] = [("", metadata or {})]
    while stack:
        prefix, value = stack.pop()
        if not isinstance(value, Mapping):
            if prefix:
                out[prefix] = str(value)
            continue
        for key, sub in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(sub, Mapping):
                stack.append((name, sub))
            elif name in _LEGACY_META:
                out[_LEGACY_META[name]] = str(sub)
            else:
                out[name] = str(sub)
    return out

#: Schema tag for agent-keyed cache keys (sha256 over agent id + agent text hash + skills text hash).
AGENT_KEY_SCHEMA = 1

_MAX_ESTIMATE_DEPTH = 8


class CacheSizeError(WeightSkillsError):
    """Raised when a compiled cache has no usable byte size or exceeds the budget."""


class ProvenanceError(WeightSkillsError):
    """Raised when a compiled cache cannot be proven to match the request prefix.

    The serving layer must never run a request against a cache whose token prefix differs from the
    request's prefix. This error is the structural reject (see the G.61 equivalence-gate result:
    sha256 over prefix token ids rejects a wrong cache 16/16 at ~0.07 ms, where a statistical
    token-margin check would have missed some).
    """


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Stable checksum of a token-id prefix, as stored in :class:`CacheProvenance`."""
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError(f"token_ids must be a sequence of ints, got {type(token_ids).__name__}")
    return hashlib.sha256(",".join(str(int(t)) for t in token_ids).encode("ascii")).hexdigest()


def text_sha256(text: str) -> str:
    """SHA-256 of a UTF-8 text (used for agent and skills definition hashes)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def agent_cache_key(agent_id: str, agent_text: str, skills_text: str) -> str:
    """Routing key for a per-agent compiled cache.

    KiroCrew loads agents on demand (one crew per agent), so the serving key is the agent's identity,
    not one system prompt: the key binds the agent id, a hash of the agent definition and a hash of the
    skills text mounted for it. Any change to the agent or its skills yields a different key, and the
    stored :class:`CacheProvenance` carries the same hashes so a wrong-agent cache is rejected with
    :class:`ProvenanceError` (see :meth:`CompiledPromptLibrary.store_agent` / ``get_agent_verified``).
    """
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("agent_id must be a non-empty string")
    payload = json.dumps(
        {
            "schema": AGENT_KEY_SCHEMA,
            "agent_id": agent_id,
            "agent_sha256": text_sha256(agent_text),
            "skills_sha256": text_sha256(skills_text),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheProvenance:
    """Identity of the token prefix a compiled cache was built from.

    ``text_key`` is the SHA-256 of the system text (the routing key). ``prefix_token_sha256`` is the
    checksum of the *token ids* the cache was prefilled with; the two differ when a tokenizer/template
    revision changes. ``model_id`` / ``tokenizer_id`` / ``dtype`` bind the artefact to the runtime that
    may use it, so a hash match cannot be reused across models.

    ``prefill_chunk`` records the prefill chunk size (in tokens) the cache was compiled with, and
    ``aligned_extra_tokens`` how many prefix tokens had to be re-prefilled with the suffix so the
    compiled cut landed on that chunk boundary (G.93/D28). :func:`aligned_exact_contract` uses
    ``prefill_chunk`` to decide whether a cache may be served as bit-exact rather than sampled.
    """

    schema: int = PROVENANCE_SCHEMA
    text_key: str | None = None
    prefix_token_sha256: str | None = None
    prefix_token_count: int | None = None
    model_id: str | None = None
    tokenizer_id: str | None = None
    dtype: str | None = None
    agent_id: str | None = None
    agent_sha256: str | None = None
    skills_sha256: str | None = None
    equivalence: str | None = None
    prefill_chunk: int | None = None
    aligned_extra_tokens: int | None = None

    _FIELDS = (
        "text_key",
        "prefix_token_sha256",
        "prefix_token_count",
        "model_id",
        "tokenizer_id",
        "dtype",
        "agent_id",
        "agent_sha256",
        "skills_sha256",
        "equivalence",
        "prefill_chunk",
        "aligned_extra_tokens",
    )

    def merged(self, **overrides: Any) -> CacheProvenance:
        """Return a copy with ``overrides`` filling fields that are currently ``None``."""
        values = {field: getattr(self, field) for field in self._FIELDS}
        for field, value in overrides.items():
            if field in values and values[field] is None and value is not None:
                values[field] = value
        return replace(self, **values)

    def to_metadata(self) -> str:
        payload = {field: getattr(self, field) for field in self._FIELDS}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_metadata(cls, raw: Any) -> CacheProvenance | None:
        if raw is None or raw == "":
            return None
        try:
            payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError) as exc:
            raise ProvenanceError(f"invalid provenance metadata: {raw!r}") from exc
        if not isinstance(payload, Mapping):
            raise ProvenanceError(f"invalid provenance metadata: {raw!r}")
        return cls(
            text_key=payload.get("text_key"),
            prefix_token_sha256=payload.get("prefix_token_sha256"),
            prefix_token_count=payload.get("prefix_token_count"),
            model_id=payload.get("model_id"),
            tokenizer_id=payload.get("tokenizer_id"),
            dtype=payload.get("dtype"),
            agent_id=payload.get("agent_id"),
            agent_sha256=payload.get("agent_sha256"),
            skills_sha256=payload.get("skills_sha256"),
            equivalence=payload.get("equivalence"),
            prefill_chunk=payload.get("prefill_chunk"),
            aligned_extra_tokens=payload.get("aligned_extra_tokens"),
        )


def aligned_exact_contract(
    provenance: CacheProvenance | None,
    serving_chunk: int,
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for serving a compiled cache as bit-identical to cold prefill.

    The G.93 measurement: a compiled prefix cache is bit-identical to a cold prefill when the suffix
    starts on the serving routine's prefill-chunk boundary, i.e. the cache was compiled with the same
    chunk size that serving uses (see :func:`weight_skills.serving.aligned_suffix_start`). The contract
    is only satisfiable when the artefact *recorded* its chunk; plain or sampled artefacts record none
    and must be served through the statistical-equivalence path instead.

    ``reason`` is ``""`` when ``ok`` is true; otherwise it is one of the two documented strings, so
    callers can surface it in serving headers/logs without rewriting the explanation.
    """
    recorded = None if provenance is None else provenance.prefill_chunk
    if recorded is None:
        return False, "no recorded prefill chunk (plain/sampled artifact)"
    if recorded != serving_chunk:
        return False, f"chunk mismatch: recorded {recorded} != serving {serving_chunk}"
    return True, ""


def plan_aligned_prompt(
    provenance: CacheProvenance | None,
    prefix_tokens_total: int,
    serving_chunk: int,
) -> tuple[str, int | None, str]:
    """Plan serving from an *aligned cache*, or refuse and fall back to cold (D28/D30).

    Convention (decided for the aligned serving path): an aligned cache is stored with the **cached
    prefix ids only** -- ``prefix = rendered_prefix_ids[:cut]`` with ``cut = n - W`` -- and its
    provenance records ``prefill_chunk=C`` and ``aligned_extra_tokens=W``. At request time the full
    rendered prompt ids are ``full``; the cached part is ``full[:cut]`` and the suffix to prefill is
    ``full[cut:]`` (it contains the ``W`` re-prefilled prefix tokens plus the user turn). No provenance
    schema change is needed: ``prefix_token_count`` / ``prefix_token_sha256`` describe the cached
    (cut) portion, so verification hashes ``full[:cut]``.

    Returns ``("aligned_exact", cut, "")`` when the cache may be served that way. Otherwise returns
    ``("cold", None, reason)`` with a distinct reason for every failed precondition: the chunk contract
    (no recorded chunk / chunk mismatch), no recorded ``aligned_extra_tokens``, ``W`` outside
    ``[0, prefix_tokens_total]``, or a recorded ``prefix_token_count`` that is not ``n - W``.
    """
    ok, contract_reason = aligned_exact_contract(provenance, serving_chunk)
    if not ok:
        return "cold", None, contract_reason
    if provenance is None:  # unreachable: aligned_exact_contract refuses None; keeps types honest
        return "cold", None, "no recorded prefill chunk (plain/sampled artifact)"
    extra = provenance.aligned_extra_tokens
    if extra is None:
        return (
            "cold",
            None,
            "no recorded aligned_extra_tokens (artifact was not compiled on a chunk boundary)",
        )
    if not 0 <= extra <= prefix_tokens_total:
        return (
            "cold",
            None,
            f"aligned_extra_tokens {extra} outside prefix length {prefix_tokens_total}",
        )
    cut = prefix_tokens_total - extra
    if provenance.prefix_token_count != cut:
        return (
            "cold",
            None,
            f"prefix_token_count mismatch: recorded {provenance.prefix_token_count} != "
            f"prefix_tokens_total {prefix_tokens_total} - aligned_extra_tokens {extra} = {cut}",
        )
    return "aligned_exact", cut, ""


class CacheSerializer(Protocol):
    """Persistence hook for compiled prompt caches (tests inject a fake one)."""

    def save(self, path: Path, cache: Any, metadata: dict[str, str]) -> None: ...

    def load(self, path: Path) -> tuple[Any, dict[str, str]]: ...


class MLXPromptCacheSerializer:
    """Default serializer: safetensors via ``mlx_lm.models.cache``.

    The MLX imports live inside the methods on purpose. Importing
    :mod:`weight_skills.serving` on a CPU-only machine must not import MLX.
    """

    def save(self, path: Path, cache: Any, metadata: dict[str, str]) -> None:
        from mlx_lm.models.cache import save_prompt_cache

        prompt_cache = cache if isinstance(cache, list) else [cache]
        path.parent.mkdir(parents=True, exist_ok=True)
        save_prompt_cache(str(path), prompt_cache, metadata=dict(metadata))

    def load(self, path: Path) -> tuple[Any, dict[str, str]]:
        from mlx_lm.models.cache import load_prompt_cache

        cache, metadata = load_prompt_cache(str(path), return_metadata=True)
        return cache, normalise_metadata(metadata)


# --------------------------------------------------------------------------- routing
def prompt_key(system_text: str) -> str:
    """Return the cache key for a system prompt: SHA-256 of its UTF-8 bytes.

    The text is hashed verbatim. Routing is exact by design: whitespace or
    wording changes produce a different compiled prompt, so they must miss
    rather than silently reuse a cache that no longer matches.
    """
    return hashlib.sha256(system_text.encode("utf-8")).hexdigest()


def key_from_messages(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the prompt key of the first system message, or ``None``."""
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            return prompt_key(content)
        return None
    return None


def resolve_prompt_key(
    *,
    headers: Mapping[str, str] | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    """Resolve the compiled-prompt key for a request.

    Priority: the ``X-Weight-Skills-Prompt-Key`` header (case-insensitive, an
    optional ``sha256:`` prefix is stripped), then the SHA-256 of the first
    system message. Returns ``None`` when neither is available.
    """
    if headers:
        wanted = PROMPT_KEY_HEADER.lower()
        for raw_name, raw_value in headers.items():
            if str(raw_name).lower() != wanted:
                continue
            value = str(raw_value).strip()
            if value.lower().startswith("sha256:"):
                value = value[len("sha256:") :].strip()
            if value:
                return value
    if messages:
        return key_from_messages(messages)
    return None


# --------------------------------------------------------------------------- size
def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - a broken property means "unknown size"
        return None


def _coerce_nbytes(value: Any) -> int | None:
    """Normalise an ``nbytes``-ish value to a non-negative int, else ``None``."""
    if value is None:
        return None
    if callable(value):
        try:
            value = value()
        except Exception:  # noqa: BLE001 - a broken hook means "unknown size"
            return None
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    return size if size >= 0 else None


def _estimate(obj: Any, depth: int) -> int | None:
    if depth > _MAX_ESTIMATE_DEPTH or obj is None:
        return None
    if isinstance(obj, (str, bytes, bytearray)):
        return 0
    size = _coerce_nbytes(_safe_attr(obj, "nbytes"))
    if size is not None:
        return size
    if isinstance(obj, Mapping):
        total = 0
        for value in obj.values():
            part = _estimate(value, depth + 1)
            if part is None:
                return None
            total += part
        return total
    if isinstance(obj, (list, tuple, set, frozenset)):
        total = 0
        for value in obj:
            if value is None:
                continue
            part = _estimate(value, depth + 1)
            if part is None:
                return None
            total += part
        return total
    state = _safe_attr(obj, "state")
    if state is not None and state is not obj:
        return _estimate(state, depth + 1)
    return None


def estimate_cache_nbytes(cache: Any) -> int | None:
    """Estimate the resident bytes of a prompt cache, or ``None`` if unknowable.

    Understands, in order: ``.nbytes`` on the cache (or on each per-layer
    cache), ``.nbytes`` on arrays inside a ``.state`` mapping/list, and nested
    containers of those. Unknown objects nested inside a container make the
    whole estimate unknown rather than silently under-counting the budget.
    """
    return _estimate(cache, 0)


# --------------------------------------------------------------------------- entries
@dataclass(slots=True)
class _Entry:
    cache: Any
    n_tokens: int
    nbytes: int
    created_at: float
    last_used: float
    hits: int = 0
    provenance: CacheProvenance | None = None


def _filename_for(name: str) -> str:
    """Deterministic, filesystem-safe filename for a cache name."""
    safe = "".join(char if (char.isalnum() or char in "._-") else "_" for char in name).strip("._")
    if safe == name and 0 < len(safe) <= 120:
        return f"{safe}.safetensors"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{(safe or 'cache')[:80]}-{digest}.safetensors"


# --------------------------------------------------------------------------- library
class CompiledPromptLibrary:
    """LRU cache of prefilled prompt KV caches, forked per request.

    Parameters
    ----------
    max_bytes:
        Memory budget for resident caches. Stores that would exceed it evict
        least-recently-used entries first.
    estimator:
        Optional ``cache -> bytes`` hook. Used instead of tensor inspection for
        sizing; a cache may also pass an explicit ``nbytes=`` to :meth:`store`.
    serializer:
        Optional persistence backend. Defaults to
        :class:`MLXPromptCacheSerializer` (lazy MLX import). Tests inject a
        fake serializer so no GPU is needed.
    persist_dir:
        Optional persistence directory for the **persistent LRU reload** path (D1/G.110): an entry
        evicted by the byte budget is serialised here first, and a later ``get`` / ``get_verified`` miss
        reloads the file for that key (deterministic filename, see :func:`_filename_for`) and re-inserts
        it, respecting the byte budget. Also the default directory for :meth:`save` / :meth:`save_all`.
        ``None`` (default) disables automatic persistence: no files are written or read and the library
        behaves exactly as before.
    clock:
        Monotonic clock used for LRU timestamps (injectable for tests).
    fork:
        Copy function applied on every :meth:`get`; defaults to ``deepcopy``.
    prefill_chunk / aligned_extra_tokens:
        Optional compile defaults recorded in the provenance of every stored cache when the per-store
        call does not pass its own values (see :func:`aligned_exact_contract`).
    """

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        estimator: Callable[[Any], int] | None = None,
        serializer: CacheSerializer | None = None,
        persist_dir: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        fork: Callable[[Any], Any] = copy.deepcopy,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        dtype: str | None = None,
        prefill_chunk: int | None = None,
        aligned_extra_tokens: int | None = None,
        require_provenance: bool = True,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError(f"max_bytes must be a non-negative int, got {max_bytes!r}")
        self._max_bytes = max_bytes
        self._estimator = estimator
        self._serializer: CacheSerializer = (
            serializer if serializer is not None else MLXPromptCacheSerializer()
        )
        self._persist_dir = Path(persist_dir) if persist_dir is not None else None
        self._clock = clock
        self._fork = fork
        self._model_id = model_id
        self._tokenizer_id = tokenizer_id
        self._dtype = dtype
        self._prefill_chunk = prefill_chunk
        self._aligned_extra_tokens = aligned_extra_tokens
        self._require_provenance = bool(require_provenance)
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._stores = 0
        self._overwrites = 0
        self._discards = 0
        self._disk_saves = 0
        self._disk_loads = 0
        self._persisted = 0
        self._reloaded = 0
        self._reload_seconds_total = 0.0
        self._persist_errors = 0
        self._reload_errors = 0

    # ------------------------------------------------------------ introspection
    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def persist_dir(self) -> Path | None:
        return self._persist_dir

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def evictions(self) -> int:
        with self._lock:
            return self._evictions

    def names(self) -> list[str]:
        """Cache names in LRU order (least recently used first)."""
        with self._lock:
            return list(self._entries)

    def stats(self) -> dict[str, Any]:
        """Return counters, occupancy and budget utilisation as a plain dict."""
        with self._lock:
            requests = self._hits + self._misses
            return {
                "entries": len(self._entries),
                "names": list(self._entries),
                "tokens": sum(entry.n_tokens for entry in self._entries.values()),
                "bytes": self._bytes,
                "budget_bytes": self._max_bytes,
                "utilization": (self._bytes / self._max_bytes) if self._max_bytes else 0.0,
                "peak_bytes": self._peak_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "requests": requests,
                "hit_rate": (self._hits / requests) if requests else 0.0,
                "evictions": self._evictions,
                "stores": self._stores,
                "overwrites": self._overwrites,
                "discards": self._discards,
                "disk_saves": self._disk_saves,
                "disk_loads": self._disk_loads,
                # persistent-LRU counters (D1/G.110); all zero when persist_dir is not configured
                "persisted": self._persisted,
                "reloaded": self._reloaded,
                "reload_seconds_total": self._reload_seconds_total,
                "persist_errors": self._persist_errors,
                "reload_errors": self._reload_errors,
                "provenance_entries": sum(
                    1
                    for entry in self._entries.values()
                    if entry.provenance is not None and entry.provenance.prefix_token_sha256
                ),
                "agent_entries": sum(
                    1
                    for entry in self._entries.values()
                    if entry.provenance is not None and entry.provenance.agent_id is not None
                ),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._entries

    def has_persisted(self, name: str) -> bool:
        """Whether ``name`` has a persisted file that ``get`` could reload (a ``stat`` only).

        ``__contains__`` stays resident-only to preserve its documented semantics; a proxy or router that
        wants the persistent-LRU hit path can consult this before classifying a request as a miss (D1/G.110).
        """
        if self._persist_dir is None or not isinstance(name, str):
            return False
        with self._lock:
            try:
                return (self._persist_dir / _filename_for(name)).is_file()
            except OSError:
                return False

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"CompiledPromptLibrary(entries={len(self._entries)}, "
                f"bytes={self._bytes}, budget={self._max_bytes})"
            )

    # ---------------------------------------------------------------- storage
    def store(
        self,
        name: str,
        cache: Any,
        n_tokens: int,
        *,
        nbytes: int | None = None,
        provenance: CacheProvenance | None = None,
    ) -> None:
        """Register ``cache`` under ``name``, evicting LRU entries if needed.

        ``n_tokens`` is the number of tokens the cache was prefilled with (kept
        for accounting and for persistence metadata). ``nbytes`` overrides size
        estimation for this entry. ``provenance`` records the token prefix the cache
        was built from (see :meth:`store_system` / :meth:`get_verified`).
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if cache is None:
            raise ValueError("cache must not be None (None would be indistinguishable from a miss)")
        tokens = self._validate_tokens(n_tokens)
        size = self._size_for(cache, nbytes)
        merged_provenance = self._merge_provenance(provenance)
        now = self._clock()
        with self._lock:
            if size > self._max_bytes:
                raise CacheSizeError(
                    f"cache {name!r} needs {size} bytes but the budget is {self._max_bytes}"
                )
            existing = self._entries.pop(name, None)
            if existing is not None:
                self._bytes -= existing.nbytes
                self._overwrites += 1
            while self._entries and self._bytes + size > self._max_bytes:
                self._evict_lru_locked()
            self._entries[name] = _Entry(
                cache=cache,
                n_tokens=tokens,
                nbytes=size,
                created_at=now,
                last_used=now,
                provenance=merged_provenance,
            )
            self._bytes += size
            self._stores += 1
            self._peak_bytes = max(self._peak_bytes, self._bytes)

    def get(self, name: str) -> Any | None:
        """Return a fresh deepcopy of the cache stored under ``name``.

        Returns ``None`` on a miss. A hit refreshes the entry's LRU position and
        increments the hit counter; the caller owns the returned copy. When ``persist_dir`` is
        configured, a miss first tries to reload the persisted file for ``name`` (D1/G.110): a
        successful reload is served as a hit and leaves the entry resident as most-recently-used.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None and self._persist_dir is not None:
                entry = self._reload_locked(name)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(name)
            entry.last_used = self._clock()
            entry.hits += 1
            self._hits += 1
            cache = entry.cache
        # Forking outside the lock keeps concurrent requests independent.
        return self._fork(cache)

    def store_system(
        self,
        system_text: str,
        cache: Any,
        n_tokens: int,
        *,
        nbytes: int | None = None,
        prefix_token_ids: Sequence[int] | None = None,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        dtype: str | None = None,
        equivalence: str | None = None,
        prefill_chunk: int | None = None,
        aligned_extra_tokens: int | None = None,
    ) -> str:
        """Store ``cache`` under ``prompt_key(system_text)`` and return the key.

        ``equivalence`` is the offline equality label measured for this prefix (G.60/G.61:
        ``"certified"`` when every divergence from a one-shot prefill is a near-tie, else
        ``"unverified"``). Serving with ``equivalence_policy="fallback"`` only serves ``"certified"``
        caches from the compiled path and re-prefills the rest cold.

        ``prefill_chunk`` is the prefill chunk size the cache was compiled with and
        ``aligned_extra_tokens`` the prefix tokens re-prefilled with the suffix to land on that
        boundary (G.93/D28); :func:`aligned_exact_contract` checks the former before serving the cache
        as bit-identical to cold.

        Pass ``prefix_token_ids`` (the exact ids the cache was prefilled with) to record a provenance
        checksum; :meth:`get_verified` then refuses to serve the cache for a different prefix.
        """
        key = prompt_key(system_text)
        provenance = CacheProvenance(
            text_key=key,
            prefix_token_sha256=token_ids_sha256(prefix_token_ids) if prefix_token_ids is not None else None,
            prefix_token_count=len(prefix_token_ids) if prefix_token_ids is not None else None,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            dtype=dtype,
            equivalence=equivalence,
            prefill_chunk=prefill_chunk,
            aligned_extra_tokens=aligned_extra_tokens,
        )
        self.store(key, cache, n_tokens, nbytes=nbytes, provenance=provenance)
        return key

    def store_agent(
        self,
        agent_id: str,
        agent_text: str,
        skills_text: str,
        cache: Any,
        n_tokens: int,
        *,
        nbytes: int | None = None,
        prefix_token_ids: Sequence[int] | None = None,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        dtype: str | None = None,
        equivalence: str | None = None,
        prefill_chunk: int | None = None,
        aligned_extra_tokens: int | None = None,
    ) -> str:
        """Store a per-agent compiled cache under :func:`agent_cache_key` and return the key.

        ``agent_text`` is the agent definition (prompt + description) and ``skills_text`` the mounted
        skills text for that agent; both are hashed into the key and recorded in the provenance so a
        request for a different agent or a changed skill set cannot reuse the cache. Pass
        ``prefix_token_ids`` to also bind the exact token prefix the cache was prefilled with.
        ``prefill_chunk`` / ``aligned_extra_tokens`` are recorded exactly as in :meth:`store_system`.
        """
        key = agent_cache_key(agent_id, agent_text, skills_text)
        provenance = CacheProvenance(
            text_key=key,
            prefix_token_sha256=token_ids_sha256(prefix_token_ids) if prefix_token_ids is not None else None,
            prefix_token_count=len(prefix_token_ids) if prefix_token_ids is not None else None,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            dtype=dtype,
            agent_id=agent_id,
            agent_sha256=text_sha256(agent_text),
            skills_sha256=text_sha256(skills_text),
            equivalence=equivalence,
            prefill_chunk=prefill_chunk,
            aligned_extra_tokens=aligned_extra_tokens,
        )
        self.store(key, cache, n_tokens, nbytes=nbytes, provenance=provenance)
        return key

    def get_system(self, system_text: str) -> Any | None:
        """Fork the cache compiled for ``system_text``, or ``None`` on a miss."""
        return self.get(prompt_key(system_text))

    def provenance_of(self, name: str) -> CacheProvenance | None:
        """Return the recorded provenance for ``name`` (``None`` if absent or unrecorded)."""
        with self._lock:
            entry = self._entries.get(name)
            return entry.provenance if entry is not None else None

    def has_provenance(self, name: str) -> bool:
        """Whether ``name`` carries a recorded prefix token checksum."""
        provenance = self.provenance_of(name)
        return bool(provenance is not None and provenance.prefix_token_sha256)

    def equivalence_of(self, name: str) -> str | None:
        """Return the recorded equality label (``"certified"``, ``"unverified"``) or ``None``."""
        provenance = self.provenance_of(name)
        return provenance.equivalence if provenance is not None else None

    def verify(
        self,
        name: str,
        *,
        prefix_token_ids: Sequence[int] | None,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        dtype: str | None = None,
        agent_id: str | None = None,
        agent_sha256: str | None = None,
        skills_sha256: str | None = None,
    ) -> CacheProvenance:
        """Raise :class:`ProvenanceError` unless ``name`` matches the request prefix.

        Structural checks only (no generation): token-id checksum, token count, and the optional
        model/tokenizer/dtype binding. A recorded checksum that does not match always raises; an entry
        with no recorded checksum raises only when the library was built with ``require_provenance=True``.
        """
        with self._lock:
            entry = self._entries.get(name)
        if entry is None:
            raise ProvenanceError(f"no compiled prompt named {name!r}")
        provenance = entry.provenance
        if provenance is None or provenance.prefix_token_sha256 is None:
            if self._require_provenance:
                raise ProvenanceError(f"compiled prompt {name!r} carries no prefix token checksum")
            return provenance or CacheProvenance()
        if prefix_token_ids is None:
            raise ProvenanceError(
                f"compiled prompt {name!r} is provenance-bound; pass prefix_token_ids to verify it"
            )
        actual = token_ids_sha256(prefix_token_ids)
        if actual != provenance.prefix_token_sha256:
            raise ProvenanceError(
                f"prefix token checksum mismatch for {name!r}: "
                f"cache {provenance.prefix_token_sha256[:12]}... request {actual[:12]}..."
            )
        if provenance.prefix_token_count is not None and len(prefix_token_ids) != provenance.prefix_token_count:
            raise ProvenanceError(
                f"prefix token count mismatch for {name!r}: "
                f"cache {provenance.prefix_token_count} request {len(prefix_token_ids)}"
            )
        for field, recorded, current in (
            ("model_id", provenance.model_id, model_id),
            ("tokenizer_id", provenance.tokenizer_id, tokenizer_id),
            ("dtype", provenance.dtype, dtype),
            ("agent_id", provenance.agent_id, agent_id),
            ("agent_sha256", provenance.agent_sha256, agent_sha256),
            ("skills_sha256", provenance.skills_sha256, skills_sha256),
        ):
            if recorded is not None and current is not None and recorded != current:
                raise ProvenanceError(
                    f"{field} mismatch for {name!r}: cache {recorded!r} request {current!r}"
                )
        return provenance

    def get_verified(
        self,
        name: str,
        *,
        prefix_token_ids: Sequence[int] | None,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        dtype: str | None = None,
    ) -> Any | None:
        """Like :meth:`get`, but verify provenance first (raises :class:`ProvenanceError`).

        With ``persist_dir`` configured, a non-resident key is first reloaded from its persisted file
        (D1/G.110) so the reloaded provenance is what gets verified.
        """
        self._ensure_resident(name)
        self.verify(
            name,
            prefix_token_ids=prefix_token_ids,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            dtype=dtype,
        )
        return self.get(name)

    def get_agent_verified(
        self,
        agent_id: str,
        agent_text: str,
        skills_text: str,
        *,
        prefix_token_ids: Sequence[int] | None,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        dtype: str | None = None,
    ) -> Any | None:
        """Like :meth:`get`, but route by agent identity and verify provenance first.

        Computes :func:`agent_cache_key`, checks the recorded agent/skills hashes and token prefix, and
        raises :class:`ProvenanceError` on any mismatch (wrong agent, changed skills, different prompt,
        model/tokenizer/dtype). Returns ``None`` on a cache miss. A non-resident key is reloaded from
        ``persist_dir`` before verification when persistence is configured (D1/G.110).
        """
        key = agent_cache_key(agent_id, agent_text, skills_text)
        self._ensure_resident(key)
        self.verify(
            key,
            prefix_token_ids=prefix_token_ids,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            dtype=dtype,
            agent_id=agent_id,
            agent_sha256=text_sha256(agent_text),
            skills_sha256=text_sha256(skills_text),
        )
        return self.get(key)

    def has_agent(self, agent_id: str, agent_text: str, skills_text: str) -> bool:
        """Whether a verified-capable cache exists for this exact agent + skills identity."""
        return agent_cache_key(agent_id, agent_text, skills_text) in self

    def _merge_provenance(self, provenance: CacheProvenance | None) -> CacheProvenance:
        if provenance is not None and not isinstance(provenance, CacheProvenance):
            raise TypeError(f"provenance must be a CacheProvenance or None, got {type(provenance).__name__}")
        base = provenance if provenance is not None else CacheProvenance()
        return base.merged(
            model_id=self._model_id,
            tokenizer_id=self._tokenizer_id,
            dtype=self._dtype,
            prefill_chunk=self._prefill_chunk,
            aligned_extra_tokens=self._aligned_extra_tokens,
        )

    def discard(self, name: str) -> bool:
        """Remove one entry explicitly; returns whether it existed."""
        with self._lock:
            entry = self._entries.pop(name, None)
            if entry is None:
                return False
            self._bytes -= entry.nbytes
            self._discards += 1
            return True

    def clear(self) -> int:
        """Drop every resident cache; counters are preserved. Returns entries removed."""
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._bytes = 0
            return removed

    # ------------------------------------------------------------ validation
    @staticmethod
    def _validate_tokens(n_tokens: Any) -> int:
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 0:
            raise ValueError(f"n_tokens must be a non-negative int, got {n_tokens!r}")
        return n_tokens

    @staticmethod
    def _validate_size(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CacheSizeError(f"nbytes must be a non-negative int, got {value!r}")
        if value < 0:
            raise CacheSizeError(f"nbytes must be non-negative, got {value}")
        return value

    def _size_for(self, cache: Any, nbytes: int | None) -> int:
        if nbytes is not None:
            return self._validate_size(nbytes)
        if self._estimator is not None:
            try:
                estimated = self._estimator(cache)
            except Exception as exc:  # noqa: BLE001 - surface bad hooks with context
                raise CacheSizeError(f"cache size estimator failed: {exc}") from exc
            size = _coerce_nbytes(estimated)
            if size is None:
                raise CacheSizeError(
                    "cache size estimator returned no usable size "
                    f"(expected a non-negative int, got {estimated!r})"
                )
            return size
        size = estimate_cache_nbytes(cache)
        if size is None:
            raise CacheSizeError(
                "cannot estimate cache size: cache exposes no .nbytes/.state; "
                "pass nbytes=... to store() or estimator=... to the library"
            )
        return size

    # ----------------------------------------------------------- persistence
    @staticmethod
    def _metadata_for(
        name: str, n_tokens: int, provenance: CacheProvenance | None
    ) -> dict[str, str]:
        """Serialized metadata for one cache (the contract ``load`` / reload rely on)."""
        metadata = {
            META_SCHEMA: SCHEMA,
            META_CACHE_KEY: name,
            META_N_TOKENS: str(n_tokens),
            META_SAVED_UTC: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if provenance is not None and provenance.prefix_token_sha256 is not None:
            metadata[META_PROVENANCE] = provenance.to_metadata()
        return metadata

    def _persist_entry(self, name: str, entry: _Entry) -> None:
        """Serialise an entry that is about to be evicted (D1/G.110).

        Persistence is best-effort: a serializer error increments ``persist_errors`` and the eviction
        still happens, so a broken disk can never break serving. Runs under the library lock.
        """
        if self._persist_dir is None:
            return
        destination = self._persist_dir / _filename_for(name)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._serializer.save(
                destination,
                entry.cache,
                self._metadata_for(name, entry.n_tokens, entry.provenance),
            )
        except Exception:  # noqa: BLE001 - persistence must never break eviction/serving
            self._persist_errors += 1
            return
        self._persisted += 1

    def _evict_lru_locked(self) -> None:
        """Evict the least-recently-used entry, persisting it first when ``persist_dir`` is set."""
        name, entry = next(iter(self._entries.items()))
        self._persist_entry(name, entry)
        del self._entries[name]
        self._bytes -= entry.nbytes
        self._evictions += 1

    def _reload_locked(self, name: str) -> _Entry | None:
        """Reload one persisted entry into the LRU (under the lock); ``None`` means no usable file.

        The reload honours the byte budget: least-recently-used residents are evicted (and persisted)
        first. A single entry that exceeds the whole budget raises :class:`CacheSizeError` rather than
        evicting the library for it. Serializer/metadata errors increment ``reload_errors`` and count as
        a miss; they never crash the caller.
        """
        if self._persist_dir is None:
            return None
        source = self._persist_dir / _filename_for(name)
        if not source.is_file():
            return None
        started = time.perf_counter()
        try:
            cache, metadata = self._serializer.load(source)
            schema = metadata.get(META_SCHEMA)
            if schema is not None and schema != SCHEMA:
                raise ValueError(f"{source}: unsupported compiled-prompt schema {schema!r}")
            recorded_key = metadata.get(META_CACHE_KEY)
            if recorded_key is not None and recorded_key != name:
                raise ValueError(f"{source}: cache key {recorded_key!r} does not match {name!r}")
            raw_tokens = metadata.get(META_N_TOKENS)
            if raw_tokens is None:
                raise ValueError(f"{source}: metadata has no n_tokens")
            tokens = self._validate_tokens(int(raw_tokens))
            provenance = CacheProvenance.from_metadata(metadata.get(META_PROVENANCE))
            nbytes = self._size_for(cache, None)
        except Exception:  # noqa: BLE001 - a corrupt/foreign file is a miss, not a crash
            self._reload_errors += 1
            return None
        if nbytes > self._max_bytes:
            raise CacheSizeError(
                f"cache {name!r} needs {nbytes} bytes but the budget is {self._max_bytes}"
            )
        merged_provenance = self._merge_provenance(provenance)
        while self._entries and self._bytes + nbytes > self._max_bytes:
            self._evict_lru_locked()
        now = self._clock()
        entry = _Entry(
            cache=cache,
            n_tokens=tokens,
            nbytes=nbytes,
            created_at=now,
            last_used=now,
            provenance=merged_provenance,
        )
        self._entries[name] = entry
        self._bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, self._bytes)
        self._reloaded += 1
        self._reload_seconds_total += time.perf_counter() - started
        return entry

    def _ensure_resident(self, name: str) -> None:
        """Reload ``name`` from ``persist_dir`` when it is not resident (miss is a no-op)."""
        if self._persist_dir is None:
            return
        with self._lock:
            if name not in self._entries:
                self._reload_locked(name)

    def _destination(self, name: str, path: str | Path | None) -> Path:
        if path is not None:
            return Path(path)
        if self._persist_dir is None:
            raise ValueError("save() needs a path or a library constructed with persist_dir=...")
        return self._persist_dir / _filename_for(name)

    def _write(
        self,
        name: str,
        cache: Any,
        n_tokens: int,
        destination: Path,
        provenance: CacheProvenance | None = None,
    ) -> Path:
        metadata = self._metadata_for(name, n_tokens, provenance)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._serializer.save(destination, cache, metadata)
        with self._lock:
            self._disk_saves += 1
        return destination

    def save(self, name: str, path: str | Path | None = None) -> Path:
        """Persist one resident cache to safetensors; returns the written path."""
        destination = self._destination(name, path)
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(f"no compiled prompt named {name!r}")
            cache, n_tokens, provenance = entry.cache, entry.n_tokens, entry.provenance
        return self._write(name, cache, n_tokens, destination, provenance)

    def save_all(self, directory: str | Path | None = None) -> list[Path]:
        """Persist every resident cache into ``directory`` (snapshot semantics)."""
        with self._lock:
            snapshot = [
                (name, entry.cache, entry.n_tokens, entry.provenance)
                for name, entry in self._entries.items()
            ]
        base = self._persist_dir if directory is None else Path(directory)
        if base is None:
            raise ValueError("save_all() needs a directory or a library constructed with persist_dir=...")
        return [
            self._write(name, cache, tokens, base / _filename_for(name), provenance)
            for name, cache, tokens, provenance in snapshot
        ]

    def load(self, path: str | Path, *, name: str | None = None, n_tokens: int | None = None, nbytes: int | None = None) -> str:
        """Load a persisted cache, register it, and return the name used.

        ``name`` / ``n_tokens`` override the metadata written by :meth:`save`.
        ``nbytes`` overrides size estimation for the loaded cache.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"compiled prompt cache not found: {source}")
        cache, metadata = self._serializer.load(source)
        schema = metadata.get(META_SCHEMA)
        if schema is not None and schema != SCHEMA:
            raise ValueError(f"{source}: unsupported compiled-prompt schema {schema!r}")
        key = name or metadata.get(META_CACHE_KEY) or source.stem
        tokens = n_tokens
        if tokens is None:
            raw = metadata.get(META_N_TOKENS)
            if raw is None:
                raise ValueError(f"{source}: metadata has no n_tokens; pass n_tokens=")
            try:
                tokens = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}: invalid n_tokens metadata {raw!r}") from exc
        provenance = CacheProvenance.from_metadata(metadata.get(META_PROVENANCE))
        self.store(key, cache, tokens, nbytes=nbytes, provenance=provenance)
        with self._lock:
            self._disk_loads += 1
        return key

    def load_dir(self, directory: str | Path) -> list[str]:
        """Load every ``*.safetensors`` cache in ``directory``; returns names loaded."""
        base = Path(directory)
        if not base.is_dir():
            raise FileNotFoundError(f"compiled prompt directory not found: {base}")
        return [self.load(path) for path in sorted(base.glob("*.safetensors"))]


class AgentCacheRegistry:
    """Maps agent ids (crews) to compiled-cache keys registered in a :class:`CompiledPromptLibrary`.

    A thin bookkeeping layer for on-demand agent hosts: register each agent once, then resolve a request's
    agent id to the prompt key the proxy/engine already knows how to route and verify (G.62/G.67/G.81).
    Unknown agents resolve to ``None`` so callers can reject them explicitly.
    """

    def __init__(self, library: CompiledPromptLibrary) -> None:
        self._library = library
        self._agents: dict[str, str] = {}

    @property
    def library(self) -> CompiledPromptLibrary:
        return self._library

    def register(
        self,
        agent_id: str,
        agent_text: str,
        skills_text: str,
        cache: Any,
        n_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Store an agent cache in the library and remember its key; returns the key."""
        key = self._library.store_agent(agent_id, agent_text, skills_text, cache, n_tokens, **kwargs)
        self._agents[agent_id] = key
        return key

    def key_for(self, agent_id: str) -> str | None:
        """Prompt key for ``agent_id``, or ``None`` if it was never registered."""
        return self._agents.get(agent_id)

    def ids(self) -> list[str]:
        return sorted(self._agents)

    def verify(self, agent_id: str, *, prefix_token_ids: Any, **kwargs: Any) -> None:
        """Verify the registered cache against the request prefix (raises on unknown agent)."""
        key = self.key_for(agent_id)
        if key is None:
            raise ProvenanceError(f"unknown agent {agent_id!r}; registered agents: {self.ids()}")
        self._library.verify(key, prefix_token_ids=prefix_token_ids, **kwargs)

    def get(self, agent_id: str) -> Any | None:
        key = self.key_for(agent_id)
        return self._library.get(key) if key is not None else None
