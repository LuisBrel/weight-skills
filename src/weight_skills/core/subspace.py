"""Subspace algebra: Gram-Schmidt orthogonalization on the Stiefel manifold.

Two cartridges interfere when their input-side factors share direction: the
cross term ``A_j A_k^T`` is non-zero. Projecting each new cartridge onto the
orthogonal complement of the already-mounted ones drives that cross term to
zero, which is what makes multi-skill composition safe::

    A_k <- A_k (I - Q_prev Q_prev^T),   Q_prev = orthonormal rows of prior A_j

The projection is applied per decoder layer, because cartridges are only
comparable within the same residual stream.
"""

from __future__ import annotations

import numpy as np

from .cartridge import Cartridge, LayerDelta


def gram_schmidt(matrix: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, int]:
    """Modified Gram-Schmidt over the **rows** of ``matrix``.

    Returns ``(q, rank)`` where ``q`` has orthonormal rows spanning the same
    space. Rows that are numerically dependent (norm below ``eps``) are dropped,
    so ``q.shape[0] <= matrix.shape[0]``.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("gram_schmidt expects a 2-D matrix")
    rows: list[np.ndarray] = []
    for row in a:
        v = row.copy()
        for q in rows:
            v -= float(v @ q) * q
        norm = float(np.linalg.norm(v))
        if norm > eps:
            rows.append(v / norm)
    if not rows:
        return np.zeros((0, a.shape[1]), dtype=np.float64), 0
    return np.vstack(rows), len(rows)


def orthogonality_residual(cartridges: list[Cartridge]) -> float:
    """Max normalized cross-correlation between distinct cartridges.

    ``0.0`` means every pair is exactly orthogonal on every shared layer.
    """
    if len(cartridges) < 2:
        return 0.0
    worst = 0.0
    per_layer: dict[int, list[np.ndarray]] = {}
    for cart in cartridges:
        for delta in cart.layers:
            per_layer.setdefault(delta.layer, []).append(delta.a.astype(np.float64))
    for mats in per_layer.values():
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                a, b = mats[i], mats[j]
                cross = float(np.linalg.norm(a @ b.T))
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom > 0:
                    worst = max(worst, cross / denom)
    return worst


def _orthogonalize_layer(
    delta: LayerDelta, basis: np.ndarray, *, preserve_norm: bool
) -> tuple[LayerDelta, np.ndarray]:
    """Project ``delta.a`` off the row space of ``basis`` and refresh the basis."""
    a = delta.a.astype(np.float64)
    a_projected = a - (a @ basis.T) @ basis if basis.shape[0] else a

    before = float(np.linalg.norm(a))
    after = float(np.linalg.norm(a_projected))
    if after < 1e-12:
        # The cartridge is entirely inside the existing subspace: nothing can be
        # added without interference. Return a zeroed cartridge and let the
        # caller decide whether that is an error.
        a_projected = np.zeros_like(a_projected)
    elif preserve_norm and after > 0:
        a_projected *= before / after

    q_new, _ = gram_schmidt(a_projected)
    if basis.shape[0]:
        combined, _ = gram_schmidt(np.vstack([basis, q_new]))
    else:
        combined = q_new

    return LayerDelta(layer=delta.layer, a=a_projected.astype(np.float32), b=delta.b), combined


def orthogonalize_cartridges(
    cartridges: list[Cartridge], *, preserve_norm: bool = True
) -> list[Cartridge]:
    """Sequentially project cartridges so that ``A_j A_k^T = 0`` for ``j != k``.

    Order matters: the first cartridge is untouched, later ones are projected
    onto the complement of everything mounted before them.
    """
    if len(cartridges) < 2:
        return list(cartridges)

    bases: dict[int, np.ndarray] = {}
    out: list[Cartridge] = []
    for cart in cartridges:
        new_layers: list[LayerDelta] = []
        for delta in cart.layers:
            basis = bases.get(delta.layer, np.zeros((0, cart.hidden_size), dtype=np.float64))
            projected, new_basis = _orthogonalize_layer(
                delta, basis, preserve_norm=preserve_norm
            )
            bases[delta.layer] = new_basis
            new_layers.append(projected)
        out.append(
            Cartridge(
                skill_id=cart.skill_id,
                version=cart.version,
                hidden_size=cart.hidden_size,
                rank=cart.rank,
                layers=tuple(new_layers),
                base_model=cart.base_model,
                injection=cart.injection,
                alpha=cart.alpha,
                compiler=dict(cart.compiler),
                created_utc=cart.created_utc,
                notes=cart.notes,
                path=cart.path,
            )
        )
    return out


def fuse_factors(layer_deltas: list[LayerDelta], hidden_size: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate factors of several cartridges into one rank-``K*r`` pair.

    Used by the fused batched GEMM path: a single ``(x @ A_fused.T) @ B_fused.T``
    replaces ``K`` separate pushes.
    """
    if not layer_deltas:
        raise ValueError("fuse_factors requires at least one layer delta")
    for d in layer_deltas:
        d.validate(hidden_size, rank)
    a = np.concatenate([d.a for d in layer_deltas], axis=0)
    b = np.concatenate([d.b for d in layer_deltas], axis=1)
    return a.astype(np.float32), b.astype(np.float32)
