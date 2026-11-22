"""Backend protocol and shared report types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.cartridge import Cartridge


@dataclass(frozen=True)
class MountReport:
    """Result of a mount operation."""

    backend: str
    skill_ids: tuple[str, ...]
    layers: tuple[int, ...]
    nbytes: int
    latency_ms: float
    scale: float
    fused: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "skill_ids": list(self.skill_ids),
            "layers": list(self.layers),
            "nbytes": self.nbytes,
            "latency_ms": round(self.latency_ms, 4),
            "scale": self.scale,
            "fused": self.fused,
            **self.extra,
        }


@dataclass(frozen=True)
class UnmountReport:
    """Result of an unmount operation, including the restoration verdict."""

    backend: str
    latency_ms: float
    restored: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 4),
            "restored": self.restored,
            **self.detail,
        }


@runtime_checkable
class SkillBackend(Protocol):
    """Minimal contract every execution backend implements."""

    name: str

    def mount(self, cartridges: Sequence[Cartridge], *, scale: float = 1.0) -> MountReport: ...

    def unmount(self) -> UnmountReport: ...

    def active_skills(self) -> tuple[str, ...]: ...

    def fingerprint(self, *, full: bool = False) -> str: ...
