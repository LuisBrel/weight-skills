"""Pre-allocated unified-memory slabs for cartridges.

The arena exists so that mounting a skill never triggers a large allocation on
the inference critical path. Capacity is reserved once; each cartridge gets a
view into the slab, and unmounting returns the range to a free list.

The allocator is intentionally simple (first-fit with coalescing) and is
validated in ``tests/test_arena.py`` for zero leakage across many cycles.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..errors import WeightSkillsError
from .cartridge import Cartridge


@dataclass(frozen=True)
class SlabHandle:
    """A reserved region inside the arena."""

    offset: int
    length: int
    skill_id: str

    @property
    def end(self) -> int:
        return self.offset + self.length


class CartridgeArena:
    """Bump/free-list allocator over one contiguous buffer.

    Parameters
    ----------
    capacity_bytes:
        Size of the backing buffer. Mounting beyond it raises
        :class:`ArenaExhaustedError` rather than growing.
    """

    def __init__(self, capacity_bytes: int, *, dtype: type = np.float32) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self._capacity = int(capacity_bytes)
        self._dtype = np.dtype(dtype)
        self._buffer = np.zeros(self._capacity // self._dtype.itemsize, dtype=self._dtype)
        self._free: list[tuple[int, int]] = [(0, self._capacity)]
        self._handles: list[SlabHandle] = []
        self._high_water = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ props
    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def used_bytes(self) -> int:
        return sum(h.length for h in self._handles)

    @property
    def high_water_bytes(self) -> int:
        return self._high_water

    @property
    def fragmentation_bytes(self) -> int:
        return sum(size for _, size in self._free)

    @property
    def active_count(self) -> int:
        return len(self._handles)

    # ------------------------------------------------------------------- ops
    def reserve(self, cartridge: Cartridge) -> SlabHandle:
        """Copy a cartridge's factors into the slab and return its handle."""
        need = cartridge.nbytes
        with self._lock:
            for idx, (offset, size) in enumerate(self._free):
                if size >= need:
                    handle = SlabHandle(offset, need, cartridge.skill_id)
                    self._write(cartridge, offset)
                    if size == need:
                        self._free.pop(idx)
                    else:
                        self._free[idx] = (offset + need, size - need)
                    self._handles.append(handle)
                    self._high_water = max(self._high_water, self.used_bytes)
                    return handle
        raise ArenaExhaustedError(
            f"arena exhausted: need {need} bytes, {self.fragmentation_bytes} free, "
            f"{self.used_bytes} in use of {self._capacity}"
        )

    def release(self, handle: SlabHandle) -> None:
        """Return a region to the free list and coalesce neighbours."""
        with self._lock:
            for index, existing in enumerate(self._handles):
                if existing is handle:
                    self._handles.pop(index)
                    break
            else:
                raise WeightSkillsError(f"handle for {handle.skill_id!r} is not reserved")
            self._free.append((handle.offset, handle.length))
            self._free.sort()
            merged: list[tuple[int, int]] = []
            for offset, size in self._free:
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    prev_off, prev_size = merged[-1]
                    merged[-1] = (prev_off, prev_size + size)
                else:
                    merged.append((offset, size))
            self._free = merged

    def reset(self) -> None:
        with self._lock:
            self._handles.clear()
            self._free = [(0, self._capacity)]

    def view(self, handle: SlabHandle) -> np.ndarray:
        """Return a 1-D view over the reserved bytes."""
        if handle.length % self._dtype.itemsize:
            raise WeightSkillsError("handle length is not a multiple of the arena dtype")
        start = handle.offset // self._dtype.itemsize
        stop = start + handle.length // self._dtype.itemsize
        return self._buffer[start:stop]

    def _write(self, cartridge: Cartridge, offset: int) -> None:
        cursor = offset
        for delta in sorted(cartridge.layers, key=lambda d: d.layer):
            for factor in (delta.a, delta.b):
                flat = factor.astype(self._dtype, copy=False).ravel()
                start = cursor // self._dtype.itemsize
                self._buffer[start : start + flat.size] = flat
                cursor += flat.nbytes


class ArenaExhaustedError(WeightSkillsError):
    """Raised when a cartridge does not fit in the reserved arena."""
