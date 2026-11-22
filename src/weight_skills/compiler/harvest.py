"""Closed-form activation-distillation compiler.

Given a base model and a behavioural specification expressed as a system prompt,
the compiler measures the *behavioural shift* the prompt induces at a set of
decoder blocks and fits a low-rank push-through term that reproduces it without
the prompt:

1. **Harvest** — for each probe ``q_i`` run the model twice (clean / skill-
   conditioned) and record, at every target block, the block input ``x_i`` in the
   clean run and the block output delta ``d_i = y_i^skill - y_i^clean``.
2. **Shield** — project the inputs off an anchor subspace so the compiled edit
   cannot move activations the base model already relies on.
3. **Fit** — closed-form ridge regression ``M = D^T (X X^T + lam I)^-1 X``,
   then an exact rank-``r`` factorization obtained from a ``b x b`` SVD (never a
   ``d x d`` one).

The regression target is the prompt-induced residual delta at the final prompt
token, i.e. exactly the state the model decodes from.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.cartridge import Cartridge
from ..errors import CompilationError

DEFAULT_POSITION = -1


@dataclass
class CompileDiagnostics:
    """Everything measured during a compile, for reporting and reproducibility."""

    skill_id: str
    layers: tuple[int, ...]
    rank: int
    n_probes: int
    n_anchors: int
    ridge_lambda: float
    hidden_size: int
    layer_frobenius: dict[int, float] = field(default_factory=dict)
    residual_ratio: dict[int, float] = field(default_factory=dict)
    nullspace_energy_kept: float = 0.0
    harvest_seconds: float = 0.0
    compile_seconds: float = 0.0
    anchor_subspace_rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "layers": list(self.layers),
            "rank": self.rank,
            "n_probes": self.n_probes,
            "n_anchors": self.n_anchors,
            "ridge_lambda": self.ridge_lambda,
            "hidden_size": self.hidden_size,
            "layer_frobenius": {str(k): round(v, 6) for k, v in self.layer_frobenius.items()},
            "residual_ratio": {str(k): round(v, 6) for k, v in self.residual_ratio.items()},
            "nullspace_energy_kept": round(self.nullspace_energy_kept, 6),
            "anchor_subspace_rank": self.anchor_subspace_rank,
            "harvest_seconds": round(self.harvest_seconds, 3),
            "compile_seconds": round(self.compile_seconds, 3),
        }


# --------------------------------------------------------------------- capture
class _Capture:
    """Pass-through wrapper that records block inputs/outputs at one position."""

    def __init__(
        self,
        base: Any,
        store: dict[int, tuple[np.ndarray, np.ndarray]],
        layer: int,
        positions: Sequence[int],
    ):
        self.base = base
        self.store = store
        self.layer = layer
        self.positions = tuple(positions)

    def __call__(self, x):  # noqa: ANN001, ANN201 - mlx array in/out
        import mlx.core as mx

        y = self.base(x)
        if x.ndim == 3:
            index = mx.array(list(self.positions))
            xi = mx.take(x, index, axis=1)[0]
            yi = mx.take(y, index, axis=1)[0]
        else:
            xi = x[list(self.positions)]
            yi = y[list(self.positions)]
        # Cast on device, then evaluate: numpy cannot consume bfloat16 buffers.
        xi = xi.astype(mx.float32)
        yi = yi.astype(mx.float32)
        mx.eval(xi, yi)
        self.store[self.layer] = (np.array(xi), np.array(yi))
        return y


class _capture_blocks:
    """Context manager that temporarily wraps target blocks with recorders."""

    def __init__(self, backend: Any, layers: Sequence[int], positions: Sequence[int] = (DEFAULT_POSITION,)):
        self.backend = backend
        self.layers = list(layers)
        self.positions = tuple(positions)
        self.store: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._originals: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        for layer in self.layers:
            site = self.backend._sites[layer]
            original = getattr(site.parent, site.attr)
            self._originals.append((site.parent, site.attr, original))
            setattr(
                site.parent,
                site.attr,
                _Capture(original, self.store, layer, self.positions),
            )
        return self.store

    def __exit__(self, *exc: Any) -> None:
        for parent, attr, original in reversed(self._originals):
            setattr(parent, attr, original)


# ------------------------------------------------------------------- rendering
def render_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
    """Render chat messages with the model's template, falling back to plain text."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None:
        try:
            return apply(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except Exception:  # noqa: BLE001 - template may not support the flag
            pass
        try:
            return apply(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:  # noqa: BLE001 - some templates reject bare system turns
            pass
    parts = []
    for msg in messages:
        role = msg["role"]
        parts.append(f"<|{role}|>\n{msg['content']}\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


def encode_prompt(tokenizer: Any, text: str):
    """Tokenize and wrap as an MLX ``(1, T)`` uint32 batch.

    MLX's quantized embedding rejects numpy buffers at the index site, so the
    conversion to ``mx.array`` has to happen here rather than at the call site.
    """
    import mlx.core as mx

    ids = tokenizer.encode(text)
    if not ids:
        raise CompilationError("tokenizer produced an empty sequence")
    return mx.array(np.array(ids, dtype=np.uint32)[None, :])


# ------------------------------------------------------------------- nullspace
def anchor_nullspace(anchor_x: np.ndarray, variance_ratio: float = 0.992) -> tuple[np.ndarray, float]:
    """Orthonormal basis of the dominant anchor subspace, via the small Gram matrix.

    ``anchor_x`` is ``(n, d)`` with ``n << d``; the right singular vectors are
    recovered from the ``n x n`` Gram matrix, so no ``d x d`` decomposition is
    ever formed.
    """
    x = np.asarray(anchor_x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    gram = x @ x.T
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    u = evecs[:, order]
    total = float(evals.sum())
    if total <= 0:
        return np.zeros((x.shape[1], 0)), 0.0
    cumulative = np.cumsum(evals) / total
    r = int(np.searchsorted(cumulative, variance_ratio) + 1)
    r = max(1, min(r, n))
    singular = np.sqrt(evals[:r])
    v = (x.T @ u[:, :r]) / singular[None, :]
    q, _ = np.linalg.qr(v)
    kept = float(cumulative[r - 1])
    return q.astype(np.float32), kept


# --------------------------------------------------------------------- fitting
def factorize_ridge(
    x: np.ndarray,
    d: np.ndarray,
    *,
    rank: int,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit ``D ~ X M^T`` with ridge and return factors ``(A, B, residual_ratio)``.

    ``A`` is ``(rank, d_in)`` and ``B`` is ``(d_out, rank)``, so the push-through
    term ``(x @ A.T) @ B.T`` reproduces the fitted delta.
    """
    x64 = np.asarray(x, dtype=np.float64)
    d64 = np.asarray(d, dtype=np.float64)
    if x64.shape[0] != d64.shape[0]:
        raise CompilationError("X and D must have the same number of rows")
    n = x64.shape[0]
    gram = x64 @ x64.T + ridge_lambda * np.eye(n)
    # P = G D  (n, d_out);  M = P^T X
    p = np.linalg.solve(gram, d64)

    # X^T = Qx Rx  ->  X = Rx^T Qx^T      (Qx: d_in x n, Rx: n x n)
    # P   = Qp Rp  ->  P^T = Rp^T Qp^T    (Qp: n x n,    Rp: n x d_out)
    # M = P^T X = Rp^T (Qp^T Rx^T) Qx^T
    qx, rx = np.linalg.qr(x64.T)  # reduced: qx (d_in, n), rx (n, n)
    qp, rp = np.linalg.qr(p)      # reduced: qp (n, n), rp (n, d_out)
    core = qp.T @ rx.T            # (n, n) — exact, never d x d
    u, s, vt = np.linalg.svd(core, full_matrices=False)

    r = int(min(rank, s.shape[0]))
    root = np.sqrt(np.clip(s[:r], 0.0, None))
    b = (rp.T @ u[:, :r]) * root[None, :]        # (d_out, r)
    a = root[:, None] * (vt[:r, :] @ qx.T)       # (r, d_in)

    # Reconstruction quality on the fit set, in relative Frobenius terms.
    approx = (x64 @ a.T) @ b.T
    denom = float(np.linalg.norm(d64))
    residual = float(np.linalg.norm(d64 - approx)) / denom if denom > 0 else 0.0
    return a.astype(np.float32), b.astype(np.float32), residual


def choose_layers(candidates: Sequence[int], count: int = 3) -> list[int]:
    """Pick evenly spaced blocks from the *injectable* set, ~45-70% depth.

    Hybrid architectures (e.g. Nemotron-H) interleave Mamba, attention and MoE
    blocks, so the selection must be made over real feed-forward sites rather
    than over absolute layer indices.
    """
    pool = sorted(int(c) for c in candidates)
    if not pool:
        raise CompilationError("model exposes no injectable blocks")
    count = max(1, min(count, len(pool)))
    if count == 1:
        return [pool[len(pool) // 2]]
    lo = max(0, int(round(len(pool) * 0.45)))
    hi = max(lo + 1, int(round(len(pool) * 0.70)))
    hi = min(hi, len(pool) - 1)
    if hi <= lo:
        return pool[-count:]
    return [pool[int(round(lo + (hi - lo) * i / (count - 1)))] for i in range(count)]


def compile_skill(
    *,
    backend: Any,
    tokenizer: Any,
    skill_id: str,
    system_prompt: str,
    probes: Sequence[str],
    anchor_prompts: Sequence[str],
    layers: Sequence[int] | None = None,
    layer_count: int = 3,
    positions: int = 1,
    rank: int = 32,
    ridge_lambda: float = 1e-2,
    variance_ratio: float = 0.992,
    base_model: str = "",
    version: str = "0.1.0",
    notes: str = "",
    progress: Any = None,
    forward: Any = None,
) -> tuple[Cartridge, CompileDiagnostics]:
    """Compile a behavioural system prompt into a push-through cartridge."""
    import mlx.core as mx

    if not probes:
        raise CompilationError("at least one probe prompt is required")
    model = backend.model
    # Some runtimes (e.g. mlx-vlm) return a structured output object rather than
    # logits; ``forward`` lets the caller supply a logits-returning callable.
    run_forward = forward if forward is not None else model
    hidden = backend.hidden_size
    injectable = list(backend.injectable_layers)
    target_layers = list(layers) if layers is not None else choose_layers(injectable, layer_count)
    # Last `positions` tokens of the rendered prompt, aligned across conditions
    # because both prompts end with the same user turn + generation marker.
    offsets = tuple(range(-int(positions), 0))
    for layer in target_layers:
        if layer not in injectable:
            raise CompilationError(f"layer {layer} is not an injectable block")

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    clean_x: dict[int, list[np.ndarray]] = {layer: [] for layer in target_layers}
    clean_y: dict[int, list[np.ndarray]] = {layer: [] for layer in target_layers}
    skill_y: dict[int, list[np.ndarray]] = {layer: [] for layer in target_layers}
    anchor_x: dict[int, list[np.ndarray]] = {layer: [] for layer in target_layers}

    with _capture_blocks(backend, target_layers, offsets) as store:
        for i, prompt in enumerate(probes):
            _log(f"harvest clean {i + 1}/{len(probes)}")
            ids = encode_prompt(tokenizer, render_chat(tokenizer, [{"role": "user", "content": prompt}]))
            mx.eval(run_forward(ids))
            for layer in target_layers:
                x, y = store[layer]
                clean_x[layer].append(x)
                clean_y[layer].append(y)

            _log(f"harvest skill {i + 1}/{len(probes)}")
            ids_skill = encode_prompt(
                tokenizer,
                render_chat(
                    tokenizer,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                ),
            )
            mx.eval(run_forward(ids_skill))
            for layer in target_layers:
                _, y = store[layer]
                skill_y[layer].append(y)
            mx.clear_cache()

        for i, prompt in enumerate(anchor_prompts):
            _log(f"harvest anchor {i + 1}/{len(anchor_prompts)}")
            ids = encode_prompt(tokenizer, render_chat(tokenizer, [{"role": "user", "content": prompt}]))
            mx.eval(run_forward(ids))
            for layer in target_layers:
                x, _ = store[layer]
                anchor_x[layer].append(x)
            mx.clear_cache()

    harvest_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()

    # The fit lives in sample space: with multi-position harvesting each probe
    # contributes `positions` rows, so the rank ceiling is probes x positions.
    rank_eff = int(min(rank, len(probes) * int(positions)))
    diag = CompileDiagnostics(
        skill_id=skill_id,
        layers=tuple(target_layers),
        rank=rank_eff,
        n_probes=len(probes),
        n_anchors=len(anchor_prompts),
        ridge_lambda=ridge_lambda,
        hidden_size=hidden,
        harvest_seconds=harvest_seconds,
    )

    compiled: list[tuple[int, np.ndarray, np.ndarray]] = []
    for layer in target_layers:
        x = np.concatenate(clean_x[layer], axis=0)
        y_clean = np.concatenate(clean_y[layer], axis=0)
        y_skill = np.concatenate(skill_y[layer], axis=0)
        delta = y_skill - y_clean

        basis, kept = anchor_nullspace(np.concatenate(anchor_x[layer], axis=0), variance_ratio)
        diag.nullspace_energy_kept = kept
        diag.anchor_subspace_rank = int(basis.shape[1])
        if basis.shape[1]:
            # Project inputs off the anchor subspace: the compiled edit then has
            # no first-order effect on the directions the base model uses.
            x = x - (x @ basis) @ basis.T

        a, b, residual = factorize_ridge(x, delta, rank=rank_eff, ridge_lambda=ridge_lambda)
        compiled.append((layer, a, b))
        diag.layer_frobenius[layer] = float(np.linalg.norm(b.astype(np.float64) @ a.astype(np.float64)))
        diag.residual_ratio[layer] = residual
        _log(f"layer {layer}: rank={rank_eff} residual_ratio={residual:.4f}")

    diag.compile_seconds = time.perf_counter() - t1
    cart = Cartridge.from_arrays(
        skill_id=skill_id,
        layers=compiled,
        hidden_size=hidden,
        rank=rank_eff,
        version=version,
        base_model=base_model,
        compiler={
            "method": "closed_form_ridge_activation_distillation",
            "ridge_lambda": ridge_lambda,
            "variance_ratio": variance_ratio,
            "n_probes": len(probes),
            "n_anchors": len(anchor_prompts),
            "positions": int(positions),
            "layer_count": len(target_layers),
        },
        notes=notes,
    )
    return cart, diag
