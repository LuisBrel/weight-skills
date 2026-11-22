from __future__ import annotations

import numpy as np
import pytest

from weight_skills.core.arena import ArenaExhaustedError, CartridgeArena
from weight_skills.errors import WeightSkillsError


def test_reserve_and_release_is_leak_free(cartridge_factory):
    arena = CartridgeArena(1 << 20)
    cart = cartridge_factory(layers=(1,))
    handle = arena.reserve(cart)
    assert arena.used_bytes == cart.nbytes
    assert arena.active_count == 1
    arena.release(handle)
    assert arena.used_bytes == 0
    assert arena.active_count == 0
    assert arena.fragmentation_bytes == arena.capacity_bytes


def test_many_cycles_do_not_fragment_permanently(cartridge_factory):
    arena = CartridgeArena(1 << 20)
    carts = [cartridge_factory(f"skill-{i}", layers=(1,), seed=i) for i in range(5)]
    for _ in range(100):
        handles = [arena.reserve(c) for c in carts]
        for handle in handles:
            arena.release(handle)
    assert arena.used_bytes == 0
    assert arena.fragmentation_bytes == arena.capacity_bytes


def test_arena_exhaustion_raises(cartridge_factory):
    cart = cartridge_factory(hidden=128, rank=8, layers=(1,))
    arena = CartridgeArena(cart.nbytes - 1)
    with pytest.raises(ArenaExhaustedError):
        arena.reserve(cart)


def test_view_copies_factor_bytes(cartridge_factory):
    cart = cartridge_factory(layers=(1,))
    arena = CartridgeArena(1 << 20)
    handle = arena.reserve(cart)
    view = arena.view(handle)
    expected = np.concatenate([cart.layers[0].a.ravel(), cart.layers[0].b.ravel()])
    np.testing.assert_array_equal(view, expected)
    arena.release(handle)


def test_release_unknown_handle_raises(cartridge_factory):
    from weight_skills.core.arena import SlabHandle

    arena = CartridgeArena(1 << 20)
    with pytest.raises(WeightSkillsError):
        arena.release(SlabHandle(0, 4, "ghost"))


def test_high_water_mark_tracks_peak(cartridge_factory):
    arena = CartridgeArena(1 << 20)
    carts = [cartridge_factory(f"s{i}", layers=(1,), seed=i) for i in range(3)]
    handles = [arena.reserve(c) for c in carts]
    peak = arena.high_water_bytes
    assert peak == sum(c.nbytes for c in carts)
    arena.release(handles[0])
    assert arena.high_water_bytes == peak
