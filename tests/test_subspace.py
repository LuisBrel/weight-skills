from __future__ import annotations

import numpy as np

from weight_skills.core.subspace import (
    fuse_factors,
    gram_schmidt,
    orthogonality_residual,
    orthogonalize_cartridges,
)


def _cart(skill_id, hidden, rank, layers, seed):
    rng = np.random.default_rng(seed)
    deltas = [
        (layer, rng.normal(size=(rank, hidden)).astype(np.float32), rng.normal(size=(hidden, rank)).astype(np.float32))
        for layer in layers
    ]
    from weight_skills.core.cartridge import Cartridge

    return Cartridge.from_arrays(skill_id=skill_id, layers=deltas, hidden_size=hidden, rank=rank)


def test_gram_schmidt_produces_orthonormal_rows():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(6, 32))
    q, rank = gram_schmidt(matrix)
    assert rank == 6
    gram = q @ q.T
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-10)


def test_gram_schmidt_drops_dependent_rows():
    base = np.random.default_rng(1).normal(size=(1, 16))
    matrix = np.vstack([base, base * 2.0])
    q, rank = gram_schmidt(matrix)
    assert rank == 1


def test_orthogonalization_drives_cross_correlation_to_zero():
    carts = [_cart("a", 64, 8, (1, 2), 0), _cart("b", 64, 8, (1, 2), 1)]
    before = orthogonality_residual(carts)
    after_carts = orthogonalize_cartridges(carts)
    after = orthogonality_residual(after_carts)
    assert before > 0.05
    assert after < 1e-6


def test_orthogonalization_preserves_first_cartridge():
    carts = [_cart("a", 64, 8, (1,), 0), _cart("b", 64, 8, (1,), 1)]
    out = orthogonalize_cartridges(carts)
    np.testing.assert_array_equal(out[0].layers[0].a, carts[0].layers[0].a)


def test_fuse_factors_shapes():
    carts = [_cart("a", 32, 4, (5,), 0), _cart("b", 32, 4, (5,), 1)]
    a, b = fuse_factors([c.layers[0] for c in carts], hidden_size=32, rank=4)
    assert a.shape == (8, 32)
    assert b.shape == (32, 8)


def test_single_cartridge_is_untouched():
    cart = _cart("only", 16, 2, (0,), 3)
    out = orthogonalize_cartridges([cart])
    np.testing.assert_array_equal(out[0].layers[0].a, cart.layers[0].a)
