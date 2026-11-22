"""Weight Skills: zero-token procedural skills for local language models.

A skill is a small, model-specific low-rank cartridge that is *pushed through*
the frozen base model at inference time::

    y = FFN_base(x) + gamma * ((x @ A.T) @ B.T)

The base weights are never written to, so mounting has no prefill cost, no
system-prompt tokens, and unmounting is exact (no floating-point drift).

Public API
----------
>>> from weight_skills import load, mount, unmount
>>> cart = load("cartridges/aws-cso-gemma4.safetensors")
>>> mount(model, cart)          # backend auto-detected from the model object
>>> unmount(model)

The package is deliberately dependency-light: ``numpy`` and ``safetensors`` are
the only hard requirements. MLX, PyTorch and vLLM are optional extras.
"""

from __future__ import annotations

from .core.arena import CartridgeArena
from .core.cartridge import Cartridge, LayerDelta
from .core.dispatch import DispatchDecision, SkillDispatcher
from .core.kv_cartridge import KVCartridge
from .core.subspace import (
    gram_schmidt,
    orthogonality_residual,
    orthogonalize_cartridges,
)
from .errors import (
    BackendError,
    CartridgeFormatError,
    GeometryMismatchError,
    RestorationError,
    WeightSkillsError,
)

__version__ = "0.1.0"

__all__ = [
    "BackendError",
    "Cartridge",
    "CartridgeArena",
    "CartridgeFormatError",
    "DispatchDecision",
    "GeometryMismatchError",
    "LayerDelta",
    "KVCartridge",
    "RestorationError",
    "SkillDispatcher",
    "WeightSkillsError",
    "__version__",
    "gram_schmidt",
    "load",
    "mount",
    "orthogonality_residual",
    "orthogonalize_cartridges",
    "unmount",
]


def load(path: str) -> Cartridge:
    """Load a cartridge from a ``.safetensors`` file (memory-mapped)."""
    return Cartridge.load(path)


def _resolve_backend(model: object):  # noqa: ANN202 - lazy import by design
    """Pick an execution backend for ``model`` without importing heavy deps eagerly."""
    module_root = type(model).__module__.split(".")[0]
    if module_root == "mlx":
        from .backends.mlx_backend import MLXBackend

        return MLXBackend(model)
    if module_root == "torch":
        from .backends.torch_backend import TorchBackend

        return TorchBackend(model)
    raise BackendError(
        f"No weight-skills backend for model type {type(model).__module__}.{type(model).__name__}"
    )


def mount(model: object, cartridge: Cartridge | str, *, scale: float = 1.0):  # noqa: ANN201
    """Attach one or more cartridges to ``model``.

    ``cartridge`` may be a :class:`Cartridge` or a path to a ``.safetensors``
    file. Multiple cartridges can be passed as a list; they are fused into a
    single push-through term per layer.
    """
    backend = _resolve_backend(model)
    carts = [cartridge] if isinstance(cartridge, (str, Cartridge)) else list(cartridge)
    resolved = [load(c) if isinstance(c, str) else c for c in carts]
    return backend.mount(resolved, scale=scale)


def unmount(model: object):
    """Remove every cartridge from ``model`` and verify bit-identical restoration."""
    return _resolve_backend(model).unmount()
