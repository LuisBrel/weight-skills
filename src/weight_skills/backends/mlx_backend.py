"""Apple Silicon MLX backend: frozen dual-path push-through projection.

The base model is never mutated. Each targeted block is replaced by a thin
wrapper that computes::

    y = FFN_base(x) + sum_k gamma_k * ((x @ A_k.T) @ B_k.T)

``FFN_base`` is the original module object, held as a child so the parameter
tree is unchanged. Unmounting restores the original object reference, which
makes restoration exact by construction; the backend additionally verifies a
fingerprint to catch accidental external mutation.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from ..core.cartridge import Cartridge
from ..errors import BackendError, GeometryMismatchError
from .base import MountReport, UnmountReport

_BLOCK_ATTRS = ("mlp", "mixer", "feed_forward")
_MOE_TOKENS = ("MoE", "MoEBlock", "SparseMoeBlock", "SwitchMLP")


def _is_module(obj: Any) -> bool:
    return isinstance(obj, nn.Module)


def find_trunk(model: nn.Module) -> nn.Module:
    """Locate the decoder stack across MLX-LM's model wrappers."""
    candidate: Any = model
    if _is_module(getattr(candidate, "language_model", None)):
        candidate = candidate.language_model
    if _is_module(getattr(candidate, "model", None)) and hasattr(candidate.model, "layers"):
        return candidate.model
    if hasattr(candidate, "layers"):
        return candidate
    for attr in ("transformer", "decoder"):
        inner = getattr(candidate, attr, None)
        if _is_module(inner) and hasattr(inner, "layers"):
            return inner
    raise BackendError(f"cannot locate decoder stack on {type(model).__name__}")


@dataclass
class BlockSite:
    """A legal push-through injection point."""

    layer: int
    attr: str
    parent: nn.Module
    module: nn.Module
    kind: str

    @property
    def path(self) -> str:
        return f"layers.{self.layer}.{self.attr}"


class PushThroughBlock(nn.Module):
    """Wraps a frozen block and adds low-rank deltas on its input path."""

    def __init__(self, base: nn.Module, entries: Sequence[tuple[mx.array, mx.array, float]]):
        super().__init__()
        self.base = base
        # Stored outside the module tree on purpose: cartridges are runtime
        # state, not parameters, and must never appear in save_weights().
        object.__setattr__(self, "_entries", tuple(entries))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        entries = object.__getattribute__(self, "_entries")
        if not entries:
            return y
        delta: mx.array | None = None
        for a, b, gamma in entries:
            term = (x @ a.T) @ b.T
            if gamma != 1.0:
                term = term * gamma
            delta = term if delta is None else delta + term
        if delta is None:  # pragma: no cover - entries is non-empty here
            return y
        return y + delta.astype(y.dtype)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        n = len(object.__getattribute__(self, "_entries"))
        return f"PushThroughBlock({type(self.base).__name__}, deltas={n})"


class MLXBackend:
    """Push-through backend for MLX / MLX-LM models."""

    name = "mlx"

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.trunk = find_trunk(model)
        self._sites: dict[int, BlockSite] = self._discover_sites()
        if not self._sites:
            raise BackendError("no feed-forward blocks found to inject into")
        self._hidden_size = self._infer_hidden_size()
        self._active: list[Cartridge] = []
        self._originals: dict[int, tuple[BlockSite, nn.Module]] = {}
        self._mount_param_ids: dict[str, int] | None = None

    # ------------------------------------------------------------- discovery
    def _discover_sites(self) -> dict[int, BlockSite]:
        layers = getattr(self.trunk, "layers", None)
        if layers is None:
            raise BackendError("decoder stack has no .layers attribute")
        sites: dict[int, BlockSite] = {}
        for idx, layer in enumerate(layers):
            for attr in _BLOCK_ATTRS:
                block = getattr(layer, attr, None)
                if not _is_module(block):
                    continue
                cls = type(block).__name__
                if attr == "mixer" and not any(tok in cls for tok in _MOE_TOKENS):
                    # Nemotron-H interleaves Mamba2 and attention mixers; only
                    # MoE mixers are feed-forward equivalents.
                    continue
                sites[idx] = BlockSite(layer=idx, attr=attr, parent=layer, module=block, kind=cls)
                break
        return sites

    def _infer_hidden_size(self) -> int:
        args = getattr(self.model, "args", None)
        hidden = getattr(args, "hidden_size", None)
        if isinstance(hidden, int) and hidden > 0:
            return hidden
        embed = getattr(self.trunk, "embed_tokens", None)
        weight = getattr(embed, "weight", None)
        if weight is not None:
            bits = getattr(embed, "bits", None)
            if isinstance(bits, int) and bits in (2, 4, 6, 8):
                return int(weight.shape[-1]) * 32 // bits
            return int(weight.shape[-1])
        raise BackendError("cannot infer hidden size; pass it explicitly")

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def injectable_layers(self) -> tuple[int, ...]:
        """Layer indices that expose an injectable feed-forward block."""
        return tuple(sorted(self._sites))

    @property
    def sites(self) -> tuple[BlockSite, ...]:
        return tuple(self._sites[i] for i in sorted(self._sites))

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "trunk": type(self.trunk).__name__,
            "hidden_size": self._hidden_size,
            "injectable_layers": list(sorted(self._sites)),
            "n_injectable": len(self._sites),
            "block_kinds": sorted({s.kind for s in self._sites.values()}),
        }

    # ----------------------------------------------------------------- mount
    def mount(self, cartridges: Sequence[Cartridge], *, scale: float = 1.0) -> MountReport:
        if not cartridges:
            raise BackendError("mount() requires at least one cartridge")
        if self._active:
            self.unmount()

        for cart in cartridges:
            if cart.hidden_size != self._hidden_size:
                raise GeometryMismatchError(
                    f"{cart.skill_id}: hidden_size={cart.hidden_size} but model is {self._hidden_size}"
                )
            missing = [layer for layer in cart.layer_indices if layer not in self._sites]
            if missing:
                raise GeometryMismatchError(
                    f"{cart.skill_id}: layers {missing} are not injectable; "
                    f"available: {sorted(self._sites)[:8]}..."
                )

        t0 = time.perf_counter()
        # Identity snapshot only: one tree walk (~1 ms). Hashing is deliberately
        # kept off the mount path and exposed separately as fingerprint().
        self._mount_param_ids = {name: id(arr) for name, arr in tree_flatten(self.model.parameters())}

        per_layer: dict[int, list[tuple[mx.array, mx.array, float]]] = {}
        for cart in cartridges:
            for delta in cart.layers:
                a = mx.array(np.ascontiguousarray(delta.a, dtype=np.float32))
                b = mx.array(np.ascontiguousarray(delta.b, dtype=np.float32))
                per_layer.setdefault(delta.layer, []).append((a, b, float(scale)))

        for layer_idx, entries in per_layer.items():
            site = self._sites[layer_idx]
            original = getattr(site.parent, site.attr)
            self._originals[layer_idx] = (site, original)
            setattr(site.parent, site.attr, PushThroughBlock(original, entries))

        # Materialize the factors once so the first token does not pay for it.
        # The base parameters are already resident; forcing a full re-eval here
        # would add tens of milliseconds of pure copy to the mount path.
        for entries in per_layer.values():
            for a, b, _ in entries:
                mx.eval(a, b)

        self._active = list(cartridges)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return MountReport(
            backend=self.name,
            skill_ids=tuple(c.skill_id for c in cartridges),
            layers=tuple(sorted(per_layer)),
            nbytes=sum(c.nbytes for c in cartridges),
            latency_ms=elapsed_ms,
            scale=float(scale),
            fused=len(cartridges) > 1,
            extra={"injection": "block", "hidden_size": self._hidden_size},
        )

    # --------------------------------------------------------------- unmount
    def unmount(self) -> UnmountReport:
        t0 = time.perf_counter()
        restored = True
        detail: dict[str, Any] = {}
        if self._active:
            for layer_idx, (site, original) in self._originals.items():
                current = getattr(site.parent, site.attr)
                if not isinstance(current, PushThroughBlock):
                    restored = False
                    detail.setdefault("unexpected_layer_state", []).append(layer_idx)
                    continue
                setattr(site.parent, site.attr, original)
            self._originals.clear()
            self._active = []

            if self._mount_param_ids is not None:
                current_ids = {n: id(a) for n, a in tree_flatten(self.model.parameters())}
                if current_ids != self._mount_param_ids:
                    restored = False
                    detail["parameter_identity_changed"] = True
            self._mount_param_ids = None

        return UnmountReport(
            backend=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            restored=restored,
            detail=detail,
        )

    # ------------------------------------------------------------ inspection
    def active_skills(self) -> tuple[str, ...]:
        return tuple(c.skill_id for c in self._active)

    def fingerprint(self, *, full: bool = False, sample: int = 24) -> str:
        """Content hash of the base parameters.

        ``full=True`` hashes every tensor (slow on 10 GB+ models). The default
        hashes the tensor inventory plus a bounded head/middle slice of a
        deterministic sample, which detects any mutation while staying fast:
        all slices are evaluated in a single ``mx.eval`` to avoid per-tensor
        device synchronization.
        """
        params = sorted(tree_flatten(self.model.parameters()), key=lambda kv: kv[0])
        names = [name for name, _ in params]
        chosen = names
        if not full and len(names) > sample:
            step = max(1, len(names) // sample)
            chosen = names[::step][:sample]
        index = dict(params)

        h = hashlib.sha256()
        h.update(f"n={len(names)}|bytes={sum(a.nbytes for _, a in params)}".encode())
        slices: list[tuple[str, mx.array]] = []
        for name in chosen:
            arr = index[name]
            pieces = [_flatten_slice(arr, None)] if full else _bounded_slices(arr, 2048)
            h.update(name.encode())
            h.update(str(arr.dtype).encode())
            h.update(str(tuple(arr.shape)).encode())
            slices.extend((name, piece) for piece in pieces)

        if slices:
            mx.eval([piece for _, piece in slices])
        for _, piece in slices:
            h.update(_array_bytes(piece))
        return h.hexdigest()[:32]

    def eval_all(self) -> None:
        mx.eval(self.model.parameters())


def _flatten_slice(arr: mx.array, limit: int | None) -> mx.array:
    flat = arr.reshape(-1)
    if flat.dtype == mx.bfloat16:
        # Cast before the batched eval: doing it afterwards would reintroduce a
        # lazy op (and therefore a device sync) inside the hashing loop.
        flat = flat.astype(mx.float32)
    return flat if limit is None else flat[:limit]


def _bounded_slices(arr: mx.array, limit: int) -> list[mx.array]:
    flat = arr.reshape(-1)
    if flat.dtype == mx.bfloat16:
        flat = flat.astype(mx.float32)
    if flat.size <= 2 * limit:
        return [flat]
    mid = flat.size // 2
    return [flat[:limit], flat[mid : mid + limit]]


def _array_bytes(arr: mx.array) -> bytes:
    """Bytes of an already-evaluated float16/float32/uint32 array."""
    return np.array(arr).tobytes()


def iter_blocks(backend: MLXBackend) -> Iterable[BlockSite]:
    """Convenience iterator over injection sites."""
    yield from backend.sites
