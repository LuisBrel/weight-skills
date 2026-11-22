"""vLLM backend: Punica SGMV multi-slot serving.

vLLM's LoRA path already implements exactly the algebra this package needs:
``y = W0 x + B A x`` per adapter slot, with the base weights untouched. A
weight-skill cartridge stores ``A (r, d_in)`` and ``B (d_out, r)``, which is the
Punica layout verbatim, so mounting is an adapter registration rather than a
weight edit.

The backend writes a minimal PEFT-style adapter directory per cartridge and
hands vLLM a ``LoRARequest``. It requires a running vLLM install; importing this
module does not import vllm.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.cartridge import Cartridge
from ..errors import BackendError, GeometryMismatchError
from .base import MountReport, UnmountReport

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


class VLLMBackend:
    """Registers cartridges as Punica LoRA slots on an ``vllm.LLM`` instance."""

    name = "vllm"

    def __init__(self, llm: Any, *, base_model: str, workdir: str = "/tmp/weight-skills-vllm") -> None:
        self.llm = llm
        self.base_model = base_model
        self.workdir = Path(workdir)
        self._active: list[tuple[Cartridge, Any]] = []
        self._next_id = 1
        self._hidden: int | None = None

    # ---------------------------------------------------------------- helpers
    def export_adapter(self, cartridge: Cartridge, *, target_modules: Sequence[str] | None = None) -> Path:
        """Write a PEFT-compatible adapter directory for one cartridge."""
        from safetensors.numpy import save_file

        out = self.workdir / f"{cartridge.skill_id}-{cartridge.version}"
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "peft_type": "LORA",
            "r": cartridge.rank,
            "lora_alpha": cartridge.rank,
            "lora_dropout": 0.0,
            "bias": "none",
            "base_model_name_or_path": self.base_model,
            "task_type": "CAUSAL_LM",
            "target_modules": list(
                target_modules or [f"layers.{layer}.mlp.down_proj" for layer in cartridge.layer_indices]
            ),
        }
        (out / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2))
        tensors: dict[str, Any] = {}
        for delta in cartridge.layers:
            tensors[f"base_model.model.model.layers.{delta.layer}.mlp.down_proj.lora_A.weight"] = delta.a
            tensors[f"base_model.model.model.layers.{delta.layer}.mlp.down_proj.lora_B.weight"] = delta.b
        save_file(tensors, str(out / ADAPTER_WEIGHTS))
        return out

    def _lora_request(self, adapter_dir: Path, adapter_id: int) -> Any:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:  # pragma: no cover - requires vllm
            raise BackendError("vllm is not installed") from exc
        return LoRARequest(lora_name=adapter_dir.name, lora_int_id=adapter_id, lora_path=str(adapter_dir))

    # ------------------------------------------------------------------ mount
    def mount(self, cartridges: Sequence[Cartridge], *, scale: float = 1.0) -> MountReport:
        if scale != 1.0:
            raise BackendError(
                "vLLM applies adapter scale itself; bake scale into B factors with Cartridge.with_scale()"
            )
        if not cartridges:
            raise BackendError("mount() requires at least one cartridge")
        if self._hidden is not None:
            for cart in cartridges:
                if cart.hidden_size != self._hidden:
                    raise GeometryMismatchError(
                        f"{cart.skill_id}: hidden_size={cart.hidden_size} but engine is {self._hidden}"
                    )
        t0 = time.perf_counter()
        for cart in cartridges:
            adapter_dir = self.export_adapter(cart)
            request = self._lora_request(adapter_dir, self._next_id)
            self._next_id += 1
            self._active.append((cart, request))
            self._hidden = cart.hidden_size
        return MountReport(
            backend=self.name,
            skill_ids=tuple(c.skill_id for c, _ in self._active),
            layers=tuple(sorted({layer for c, _ in self._active for layer in c.layer_indices})),
            nbytes=sum(c.nbytes for c, _ in self._active),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            scale=1.0,
            fused=len(self._active) > 1,
            extra={"slots": len(self._active), "punica": "sgmv"},
        )

    def unmount(self) -> UnmountReport:
        t0 = time.perf_counter()
        self._active.clear()
        return UnmountReport(
            backend=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            restored=True,
            detail={"note": "adapter slots removed; base weights were never modified"},
        )

    def active_skills(self) -> tuple[str, ...]:
        return tuple(c.skill_id for c, _ in self._active)

    @property
    def lora_requests(self) -> list[Any]:
        """Pass these to ``llm.generate(..., lora_request=...)``."""
        return [request for _, request in self._active]

    def fingerprint(self, *, full: bool = False) -> str:
        return "vllm:base-weights-immutable"
