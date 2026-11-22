"""Shared fixtures for the weight-skills test-suite."""

from __future__ import annotations

import numpy as np
import pytest

from weight_skills.core.cartridge import Cartridge


def make_factors(
    rank: int, hidden: int, seed: int = 0, scale: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0.0, scale, (rank, hidden)).astype(np.float32),
        rng.normal(0.0, scale, (hidden, rank)).astype(np.float32),
    )


@pytest.fixture
def cartridge_factory():
    """Build a valid cartridge with deterministic factors."""

    def _make(skill_id: str = "test-skill", *, hidden: int = 32, rank: int = 4, layers=(1, 2), seed: int = 0) -> Cartridge:
        deltas = []
        for index, layer in enumerate(layers):
            a, b = make_factors(rank, hidden, seed=seed + index)
            deltas.append((layer, a, b))
        return Cartridge.from_arrays(
            skill_id=skill_id,
            layers=deltas,
            hidden_size=hidden,
            rank=rank,
            base_model="tiny",
        )

    return _make


@pytest.fixture
def sample_cartridge(cartridge_factory) -> Cartridge:
    return cartridge_factory()
