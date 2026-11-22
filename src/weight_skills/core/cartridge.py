"""Cartridge format: validation, memory-mapped loading, and serialization.

A cartridge is a set of per-layer low-rank factors ``(A, B)`` whose shapes match
one model geometry:

===============  ==========================  =================================
tensor           shape                       meaning
===============  ==========================  =================================
``layer_i.a``    ``(rank, hidden_size)``     input-side projection
``layer_i.b``    ``(hidden_size, rank)``     output-side projection
===============  ==========================  =================================

The push-through term contributed at layer ``i`` is ``(x @ A.T) @ B.T``.

Provenance (skill id, base model, rank, alpha, compiler settings) lives in the
safetensors string metadata under :data:`META_KEY` as JSON.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from ..errors import CartridgeFormatError

SCHEMA = "weight-skills/cartridge@1"
META_KEY = "weight_skills"
SUPPORTED_INJECTIONS = ("block",)


@dataclass(frozen=True)
class LayerDelta:
    """Low-rank factors for a single decoder block."""

    layer: int
    a: np.ndarray  # (rank, hidden_size), float32
    b: np.ndarray  # (hidden_size, rank), float32

    def validate(self, hidden_size: int, rank: int) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise CartridgeFormatError(f"layer {self.layer}: factors must be 2-D")
        if self.a.shape != (rank, hidden_size):
            raise CartridgeFormatError(
                f"layer {self.layer}: A has shape {self.a.shape}, expected {(rank, hidden_size)}"
            )
        if self.b.shape != (hidden_size, rank):
            raise CartridgeFormatError(
                f"layer {self.layer}: B has shape {self.b.shape}, expected {(hidden_size, rank)}"
            )
        if not np.isfinite(self.a).all() or not np.isfinite(self.b).all():
            raise CartridgeFormatError(f"layer {self.layer}: non-finite values in factors")

    @property
    def nbytes(self) -> int:
        return int(self.a.nbytes + self.b.nbytes)

    def to_float32(self) -> LayerDelta:
        return LayerDelta(
            layer=self.layer,
            a=np.ascontiguousarray(self.a, dtype=np.float32),
            b=np.ascontiguousarray(self.b, dtype=np.float32),
        )


@dataclass(frozen=True)
class Cartridge:
    """A validated, geometry-tagged set of push-through factors."""

    skill_id: str
    version: str
    hidden_size: int
    rank: int
    layers: tuple[LayerDelta, ...]
    base_model: str = ""
    injection: str = "block"
    alpha: float = 1.0
    compiler: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    notes: str = ""
    path: Path | None = None
    _digest: str = ""

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        if self.injection not in SUPPORTED_INJECTIONS:
            raise CartridgeFormatError(
                f"unsupported injection {self.injection!r}; expected one of {SUPPORTED_INJECTIONS}"
            )
        if self.rank < 1:
            raise CartridgeFormatError(f"rank must be >= 1, got {self.rank}")
        if self.hidden_size < 1:
            raise CartridgeFormatError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if not self.layers:
            raise CartridgeFormatError("cartridge contains no layers")
        seen: set[int] = set()
        for delta in self.layers:
            if delta.layer in seen:
                raise CartridgeFormatError(f"duplicate layer index {delta.layer}")
            seen.add(delta.layer)
            if delta.layer < 0:
                raise CartridgeFormatError(f"negative layer index {delta.layer}")
            delta.validate(self.hidden_size, self.rank)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(d.layer for d in self.layers))

    @property
    def nbytes(self) -> int:
        return int(sum(d.nbytes for d in self.layers))

    @property
    def digest(self) -> str:
        """Stable content hash over factors and geometry (sha256, first 16 hex)."""
        if self._digest:
            return self._digest
        h = hashlib.sha256()
        h.update(f"{self.skill_id}|{self.version}|{self.hidden_size}|{self.rank}".encode())
        for delta in sorted(self.layers, key=lambda d: d.layer):
            h.update(delta.a.tobytes())
            h.update(delta.b.tobytes())
        return h.hexdigest()[:16]

    def frobenius(self) -> dict[int, float]:
        """Per-layer ``||B A||_F`` — an interpretable proxy for edit magnitude."""
        return {
            d.layer: float(np.linalg.norm(d.b.astype(np.float64) @ d.a.astype(np.float64)))
            for d in self.layers
        }

    def with_scale(self, gamma: float) -> Cartridge:
        """Return a copy whose B factors are multiplied by ``gamma``."""
        if gamma == 1.0:
            return self
        scaled = tuple(
            LayerDelta(layer=d.layer, a=d.a, b=(d.b.astype(np.float32) * float(gamma))) for d in self.layers
        )
        return Cartridge(
            skill_id=self.skill_id,
            version=self.version,
            hidden_size=self.hidden_size,
            rank=self.rank,
            layers=scaled,
            base_model=self.base_model,
            injection=self.injection,
            alpha=self.alpha * float(gamma),
            compiler=dict(self.compiler),
            created_utc=self.created_utc,
            notes=self.notes,
            path=self.path,
        )

    def __iter__(self) -> Iterator[LayerDelta]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    # ------------------------------------------------------------------- I/O
    @classmethod
    def load(cls, path: str | Path) -> Cartridge:
        """Load and validate a cartridge file.

        The file is opened memory-mapped by safetensors and each factor is
        materialized as a contiguous float32 array. Cartridges are a few MB, so
        the copy is negligible and avoids relying on mmap lifetime semantics.
        """
        p = Path(path)
        if not p.is_file():
            raise CartridgeFormatError(f"cartridge not found: {p}")
        try:
            handle = safe_open(str(p), framework="numpy")
            raw_meta = handle.metadata() or {}
        except Exception as exc:  # noqa: BLE001 - normalized to our error type
            raise CartridgeFormatError(f"cannot open {p}: {exc}") from exc

        if META_KEY not in raw_meta:
            raise CartridgeFormatError(f"{p.name}: missing {META_KEY!r} metadata block")
        try:
            meta = json.loads(raw_meta[META_KEY])
        except json.JSONDecodeError as exc:
            raise CartridgeFormatError(f"{p.name}: metadata is not valid JSON: {exc}") from exc

        schema = meta.get("schema", "")
        if not str(schema).startswith("weight-skills/cartridge@"):
            raise CartridgeFormatError(f"{p.name}: unknown schema {schema!r}")

        groups: dict[int, dict[str, np.ndarray]] = {}
        for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
            if not key.startswith("layer_"):
                raise CartridgeFormatError(f"{p.name}: unexpected tensor {key!r}")
            try:
                layer_part, side = key[len("layer_") :].split(".", 1)
                idx = int(layer_part)
            except ValueError as exc:
                raise CartridgeFormatError(f"{p.name}: cannot parse tensor name {key!r}") from exc
            if side not in ("a", "b"):
                raise CartridgeFormatError(f"{p.name}: unexpected factor {key!r}")
            tensor = handle.get_tensor(key)
            groups.setdefault(idx, {})[side] = np.array(tensor, dtype=np.float32, copy=True)

        layers: list[LayerDelta] = []
        for idx in sorted(groups):
            pair = groups[idx]
            if set(pair) != {"a", "b"}:
                raise CartridgeFormatError(f"{p.name}: layer {idx} is missing one of a/b")
            layers.append(LayerDelta(layer=idx, a=pair["a"], b=pair["b"]))

        cart = cls(
            skill_id=str(meta.get("skill_id", p.stem)),
            version=str(meta.get("version", "0.0.0")),
            hidden_size=int(meta["hidden_size"]),
            rank=int(meta["rank"]),
            layers=tuple(layers),
            base_model=str(meta.get("base_model", "")),
            injection=str(meta.get("injection", "block")),
            alpha=float(meta.get("alpha", 1.0)),
            compiler=dict(meta.get("compiler", {})),
            created_utc=str(meta.get("created_utc", "")),
            notes=str(meta.get("notes", "")),
            path=p,
        )
        cart.validate()
        return cart

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "skill_id": self.skill_id,
            "version": self.version,
            "base_model": self.base_model,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "alpha": self.alpha,
            "layers": list(self.layer_indices),
            "injection": self.injection,
            "compiler": self.compiler,
            "created_utc": self.created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        """Serialize to ``.safetensors`` with embedded provenance metadata."""
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, np.ndarray] = {}
        for delta in sorted(self.layers, key=lambda d: d.layer):
            tensors[f"layer_{delta.layer}.a"] = np.ascontiguousarray(delta.a, dtype=np.float32)
            tensors[f"layer_{delta.layer}.b"] = np.ascontiguousarray(delta.b, dtype=np.float32)
        save_file(tensors, str(p), metadata={META_KEY: json.dumps(self.metadata())})
        return p

    def summary(self) -> str:
        layers = ",".join(str(i) for i in self.layer_indices)
        return (
            f"{self.skill_id}@{self.version} r={self.rank} d={self.hidden_size} "
            f"layers=[{layers}] {self.nbytes / 1e6:.2f} MB digest={self.digest}"
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        skill_id: str,
        layers: Sequence[tuple[int, np.ndarray, np.ndarray]],
        hidden_size: int,
        rank: int,
        version: str = "0.1.0",
        base_model: str = "",
        compiler: dict[str, Any] | None = None,
        alpha: float = 1.0,
        notes: str = "",
    ) -> Cartridge:
        deltas = tuple(
            LayerDelta(layer=int(idx), a=np.asarray(a, np.float32), b=np.asarray(b, np.float32))
            for idx, a, b in layers
        )
        cart = cls(
            skill_id=skill_id,
            version=version,
            hidden_size=int(hidden_size),
            rank=int(rank),
            layers=deltas,
            base_model=base_model,
            compiler=dict(compiler or {}),
            alpha=float(alpha),
            notes=notes,
        )
        cart.validate()
        return cart
