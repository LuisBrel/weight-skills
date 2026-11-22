"""PyTorch / Hugging Face backend (FP16 / BF16 / NF4 / FP8 compatible).

Mirrors the MLX backend: the base block is never modified, a thin wrapper adds
``(x @ A.T) @ B.T`` on the input path, and unmounting restores the original
module object.

This module imports torch lazily so it can be imported (and type-checked) in an
environment without torch, such as the Apple-Silicon-only development host used
for the reference benchmarks.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.cartridge import Cartridge
from ..errors import BackendError, GeometryMismatchError
from .base import MountReport, UnmountReport


@dataclass
class _Site:
    layer: int
    attr: str
    parent: Any
    module: Any


class _PushThrough:
    """Callable wrapper; only instantiated once torch is importable."""

    def __new__(cls, base: Any, entries: Sequence[tuple[Any, Any, float]]):  # noqa: D102
        import torch
        from torch import nn

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                self._factors = nn.ParameterList()
                for a, b, _ in entries:
                    _ = a, b  # kept in the list below, not as parameters
                self._entries = [
                    (
                        torch.as_tensor(a, dtype=torch.float32),
                        torch.as_tensor(b, dtype=torch.float32),
                        float(g),
                    )
                    for a, b, g in entries
                ]

            def forward(self, x):  # noqa: ANN001, ANN201
                y = self.base(x)
                delta = None
                for a, b, g in self._entries:
                    a_ = a.to(device=x.device, dtype=x.dtype)
                    b_ = b.to(device=x.device, dtype=x.dtype)
                    term = (x @ a_.T) @ b_.T
                    term = term * g if g != 1.0 else term
                    delta = term if delta is None else delta + term
                return y if delta is None else y + delta

        return _Wrapper()


def _find_layers(model: Any) -> Any:
    for path in ("model.layers", "transformer.h", "layers", "language_model.model.layers"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return node
    raise BackendError(f"cannot locate decoder layers on {type(model).__name__}")


def _infer_hidden(model: Any) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, attr, None)
        if isinstance(value, int):
            return value
    raise BackendError("cannot infer hidden size from model.config")


class TorchBackend:
    """Push-through backend for PyTorch / Hugging Face models."""

    name = "torch"

    def __init__(self, model: Any) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - host without torch
            raise BackendError("TorchBackend requires torch: pip install 'weight-skills[torch]'") from exc
        self.model = model
        self.layers = _find_layers(model)
        self._hidden = _infer_hidden(model)
        self._sites: dict[int, _Site] = self._discover()
        self._active: list[Cartridge] = []
        self._originals: dict[int, tuple[_Site, Any]] = {}
        self._ids: dict[str, int] | None = None

    def _discover(self) -> dict[int, _Site]:
        sites: dict[int, _Site] = {}
        for index, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "forward"):
                sites[index] = _Site(layer=index, attr="mlp", parent=layer, module=mlp)
        return sites

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "hidden_size": self._hidden,
            "n_injectable": len(self._sites),
            "injectable_layers": sorted(self._sites),
        }

    def mount(self, cartridges: Sequence[Cartridge], *, scale: float = 1.0) -> MountReport:
        import torch

        if not cartridges:
            raise BackendError("mount() requires at least one cartridge")
        if self._active:
            self.unmount()
        for cart in cartridges:
            if cart.hidden_size != self._hidden:
                raise GeometryMismatchError(
                    f"{cart.skill_id}: hidden_size={cart.hidden_size} but model is {self._hidden}"
                )
            missing = [layer for layer in cart.layer_indices if layer not in self._sites]
            if missing:
                raise GeometryMismatchError(f"{cart.skill_id}: layers {missing} are not injectable")

        t0 = time.perf_counter()
        self._ids = {name: id(p) for name, p in self.model.named_parameters()}
        per_layer: dict[int, list[tuple[Any, Any, float]]] = {}
        for cart in cartridges:
            for delta in cart.layers:
                per_layer.setdefault(delta.layer, []).append((delta.a, delta.b, float(scale)))

        for layer_index, entries in per_layer.items():
            site = self._sites[layer_index]
            original = getattr(site.parent, site.attr)
            self._originals[layer_index] = (site, original)
            setattr(site.parent, site.attr, _PushThrough(original, entries))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._active = list(cartridges)
        return MountReport(
            backend=self.name,
            skill_ids=tuple(c.skill_id for c in cartridges),
            layers=tuple(sorted(per_layer)),
            nbytes=sum(c.nbytes for c in cartridges),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            scale=float(scale),
            fused=len(cartridges) > 1,
            extra={"injection": "block", "hidden_size": self._hidden},
        )

    def unmount(self) -> UnmountReport:
        t0 = time.perf_counter()
        restored = True
        for site, original in self._originals.values():
            current = getattr(site.parent, site.attr)
            if type(current).__name__ != "_Wrapper":
                restored = False
            setattr(site.parent, site.attr, original)
        self._originals.clear()
        self._active = []
        if self._ids is not None:
            current = {name: id(p) for name, p in self.model.named_parameters()}
            if current != self._ids:
                restored = False
            self._ids = None
        return UnmountReport(
            backend=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            restored=restored,
        )

    def active_skills(self) -> tuple[str, ...]:
        return tuple(c.skill_id for c in self._active)

    def fingerprint(self, *, full: bool = False) -> str:
        import torch

        h = hashlib.sha256()
        params = sorted(self.model.named_parameters(), key=lambda kv: kv[0])
        chosen = params if full else params[:: max(1, len(params) // 24)][:24]
        for name, param in chosen:
            h.update(name.encode())
            h.update(str(tuple(param.shape)).encode())
            flat = param.detach().reshape(-1)
            sample = flat if full else torch.cat([flat[:2048], flat[flat.numel() // 2 : flat.numel() // 2 + 2048]])
            h.update(sample.to(torch.float32).cpu().numpy().tobytes())
        return h.hexdigest()[:32]
