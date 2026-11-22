"""Memory accounting for compiled prompt caches and chunked prefill (G.80).

Why this exists: compiling a long prompt does not remove the one-time prefill cost. At 65k tokens a
monolithic prefill on Qwen3.6-35B-A3B needs a 16 x 65,536 x 65,536 x 2 B = 137.4 GB attention-score buffer
per full-attention layer — the observed OOM allocation was 137,573,203,968 B — while the KV cache itself is
only ~1.4 GB. Chunked prefill splits the query dimension and is what makes 65k feasible; this module gives the
planner the numbers instead of guessing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "bf16": 2, "f16": 2, "f32": 4}
DEFAULT_CANDIDATES = (256, 512, 1024, 2048, 4096, 8192, 16384)


def dtype_bytes(dtype: str) -> int:
    """Bytes per element for a supported dtype name."""
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {sorted(DTYPE_BYTES)}") from exc


def attention_score_bytes(
    num_attention_heads: int,
    query_tokens: int,
    kv_tokens: int,
    *,
    dtype_bytes: int = 2,
) -> int:
    """Bytes of one materialised attention-score matrix (heads x q x kv).

    This is the tensor that OOM'd the 65k monolithic prefill; chunking bounds ``query_tokens``.
    """
    for name, value in (("num_attention_heads", num_attention_heads), ("query_tokens", query_tokens),
                        ("kv_tokens", kv_tokens), ("dtype_bytes", dtype_bytes)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    return num_attention_heads * query_tokens * kv_tokens * dtype_bytes


def kv_cache_bytes(
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    tokens: int,
    *,
    dtype_bytes: int = 2,
    full_attention_fraction: float = 1.0,
) -> int:
    """Bytes for the K and V tensors of a transformer's full-attention layers.

    ``full_attention_fraction`` lets hybrid models pass the share of layers that keep a KV cache (the
    rest hold a fixed-size recurrent state); use the measured ``cache_mb`` per token when available.
    """
    for name, value in (("num_layers", num_layers), ("num_key_value_heads", num_key_value_heads),
                        ("head_dim", head_dim), ("tokens", tokens), ("dtype_bytes", dtype_bytes)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if not 0.0 <= full_attention_fraction <= 1.0:
        raise ValueError("full_attention_fraction must be in [0, 1]")
    effective_layers = max(1, math.ceil(num_layers * full_attention_fraction)) if full_attention_fraction else 0
    return 2 * effective_layers * num_key_value_heads * head_dim * tokens * dtype_bytes


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated peak memory for one prefill/serving configuration."""

    weights_bytes: int
    kv_cache_bytes: int
    attention_buffer_bytes: int
    overhead_bytes: int
    chunk_tokens: int
    context_tokens: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.attention_buffer_bytes + self.overhead_bytes

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1e9


def estimate_peak(
    *,
    weights_gb: float,
    context_tokens: int,
    chunk_tokens: int,
    num_attention_heads: int,
    kv_bytes_per_token: int,
    dtype_bytes: int = 2,
    overhead_gb: float = 2.0,
) -> MemoryEstimate:
    """Estimate peak memory for a chunked prefill of ``context_tokens`` into a cache."""
    if weights_gb < 0 or overhead_gb < 0:
        raise ValueError("weights_gb and overhead_gb must be non-negative")
    if not isinstance(context_tokens, int) or context_tokens <= 0:
        raise ValueError("context_tokens must be a positive int")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    kv_bytes = kv_bytes_per_token * context_tokens
    buffer_bytes = attention_score_bytes(num_attention_heads, min(chunk_tokens, context_tokens),
                                         context_tokens, dtype_bytes=dtype_bytes)
    return MemoryEstimate(
        weights_bytes=int(weights_gb * 1e9),
        kv_cache_bytes=kv_bytes,
        attention_buffer_bytes=buffer_bytes,
        overhead_bytes=int(overhead_gb * 1e9),
        chunk_tokens=chunk_tokens,
        context_tokens=context_tokens,
    )


def plan_prefill_chunk(
    *,
    weights_gb: float,
    context_tokens: int,
    num_attention_heads: int,
    kv_bytes_per_token: int,
    budget_gb: float,
    dtype_bytes: int = 2,
    overhead_gb: float = 2.0,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
) -> int | None:
    """Pick the largest chunk size whose estimated peak fits ``budget_gb`` (``None`` if none fits)."""
    if budget_gb <= 0:
        raise ValueError("budget_gb must be positive")
    for chunk in sorted(candidates, reverse=True):
        est = estimate_peak(weights_gb=weights_gb, context_tokens=context_tokens, chunk_tokens=chunk,
                            num_attention_heads=num_attention_heads, kv_bytes_per_token=kv_bytes_per_token,
                            dtype_bytes=dtype_bytes, overhead_gb=overhead_gb)
        if est.total_bytes <= budget_gb * 1e9:
            return int(chunk)
    return None


# --- chunk-boundary alignment for exact compiled serving (G.93 / D28) -----------------------------
@dataclass(frozen=True)
class AlignmentPlan:
    """Where to cut a compiled prefix so serving executes cold's chunk sequence exactly.

    Contract (measured, G.93): when the suffix begins on a prefill-chunk boundary, a compiled cache
    served through **the same chunked prefill routine** as the cold path reproduces cold prefill
    bit-identically (5/5 cells on two 35B MoE models; 4/4 after a safetensors round-trip). The contract is
    per serving routine: the same chunk size, dtype, process and engine must be used at compile and serve
    time. Cross-process/batch/version axes are D29.
    """

    prefix_tokens: int
    chunk_tokens: int
    cut_tokens: int
    extra_tokens: int
    no_compile_benefit: bool

    @property
    def cut_is_chunk_aligned(self) -> bool:
        return self.cut_tokens % self.chunk_tokens == 0

    def cost_fraction(self, question_tokens: int = 0) -> float:
        """Extra suffix prefill tokens as a fraction of the cold prompt (0.0-1.0)."""
        cold_tokens = self.prefix_tokens + max(0, question_tokens)
        if cold_tokens <= 0:
            return 1.0
        return self.extra_tokens / cold_tokens


def aligned_suffix_start(prefix_tokens: int, chunk_tokens: int) -> tuple[int, int]:
    """Return ``(cut_tokens, extra_tokens)`` for chunk-boundary-aligned compiled serving.

    Compile the first ``cut_tokens`` prefix tokens; re-prefill the remaining ``extra_tokens`` prefix tokens
    together with the question. ``cut_tokens`` is a multiple of ``chunk_tokens``, so the suffix starts on a
    boundary of the cold path's chunk sequence. ``extra_tokens`` is always ``< chunk_tokens``.
    """
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be >= 0")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    extra = prefix_tokens % chunk_tokens
    return prefix_tokens - extra, extra


def plan_aligned_compile(prefix_tokens: int, chunk_tokens: int) -> AlignmentPlan:
    """Alignment plan for one compiled prefix at one serving chunk size."""
    cut, extra = aligned_suffix_start(prefix_tokens, chunk_tokens)
    return AlignmentPlan(prefix_tokens=prefix_tokens, chunk_tokens=chunk_tokens, cut_tokens=cut,
                         extra_tokens=extra, no_compile_benefit=cut == 0)


def alignment_options(
    prefix_tokens: int,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
) -> list[AlignmentPlan]:
    """All chunk candidates ordered by alignment cost (fewest extra tokens first).

    Smaller chunks buy cheaper alignment (``extra < chunk``) at more prefill calls; the memory planner
    (``plan_prefill_chunk``) constrains which chunks are feasible, then this ordering exposes the
    alignment cost so the caller can trade the two.
    """
    plans = [plan_aligned_compile(prefix_tokens, int(c)) for c in candidates if int(c) > 0]
    return sorted(plans, key=lambda p: (p.extra_tokens, -p.chunk_tokens))
