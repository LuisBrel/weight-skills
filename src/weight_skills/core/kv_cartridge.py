"""KV-cache skill cartridges: behaviour that rides in the cache.

A weight cartridge edits the model. A KV cartridge does not: it prefills the
demonstrations once, keeps the resulting attention cache, and every request
appends only its user turn. The model then conditions on the demonstrations
exactly as if they were in the prompt, but they are never re-encoded and never
appear in the caller's prompt.

Measured on Gemma 4B / Qwen3.6-35B (see the research ledger, G.36-G.37):
persona 18/18, refusal 14/16 with 0/6 over-refusal on the 35B, arithmetic
10/24 -> 21/24 on mixed-difficulty problems.

Usage
-----
>>> cart = KVCartridge.from_messages("clu-persona", demos, model_id="gemma-4-e4b")
>>> cart.save("cartridges/kv/clu-persona.safetensors")
>>> cart = KVCartridge.load("cartridges/kv/clu-persona.safetensors")
>>> text = cart.generate(model, tokenizer, "What is a deadlock?")

The file stores the rendered demonstration token ids plus provenance; the cache
itself is rebuilt from those ids, so the artifact is model-portable in the same
sense a prompt is (same tokenizer family).
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from ..errors import CartridgeFormatError

SCHEMA = "weight-skills/kv-cartridge@1"
META_KEY = "weight_skills_kv"


@dataclass
class KVCartridge:
    """A prefilled demonstration prefix, stored as token ids + metadata."""

    skill_id: str
    demo_messages: list[dict[str, str]]
    demo_ids: list[int]
    model_id: str = ""
    version: str = "0.1.0"
    created_utc: str = ""
    notes: str = ""
    path: Path | None = None
    _cache: Any = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_messages(
        cls,
        skill_id: str,
        messages: Sequence[dict[str, str]],
        *,
        tokenizer: Any,
        model_id: str = "",
        version: str = "0.1.0",
        notes: str = "",
    ) -> KVCartridge:
        """Render the demonstration conversation with the model's chat template."""
        msgs = [dict(m) for m in messages]
        if not msgs:
            raise CartridgeFormatError("KV cartridge needs at least one demonstration message")
        if msgs[-1].get("role") != "assistant":
            raise CartridgeFormatError(
                "demonstrations must end with an assistant turn so the user query can continue"
            )
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        except Exception:  # noqa: BLE001 - template without the flag
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = list(tokenizer.encode(text))
        if not ids:
            raise CartridgeFormatError("tokenizer produced no tokens for the demonstrations")
        return cls(
            skill_id=skill_id,
            demo_messages=msgs,
            demo_ids=ids,
            model_id=model_id,
            version=version,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            notes=notes,
        )

    # ------------------------------------------------------------------- I/O
    @classmethod
    def load(cls, path: str | Path) -> KVCartridge:
        p = Path(path)
        if not p.is_file():
            raise CartridgeFormatError(f"KV cartridge not found: {p}")
        try:
            handle = safe_open(str(p), framework="numpy")
            raw = handle.metadata() or {}
            ids = np.asarray(handle.get_tensor("demo_ids"), dtype=np.int64).tolist()
        except Exception as exc:  # noqa: BLE001
            raise CartridgeFormatError(f"cannot open {p}: {exc}") from exc
        if META_KEY not in raw:
            raise CartridgeFormatError(f"{p.name}: missing {META_KEY!r} metadata")
        meta = json.loads(raw[META_KEY])
        schema = str(meta.get("schema", ""))
        if not schema.startswith("weight-skills/kv-cartridge@"):
            raise CartridgeFormatError(f"{p.name}: unknown schema {schema!r}")
        return cls(
            skill_id=str(meta["skill_id"]),
            demo_messages=list(meta.get("demo_messages", [])),
            demo_ids=ids,
            model_id=str(meta.get("model_id", "")),
            version=str(meta.get("version", "0.1.0")),
            created_utc=str(meta.get("created_utc", "")),
            notes=str(meta.get("notes", "")),
            path=p,
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": SCHEMA,
            "skill_id": self.skill_id,
            "model_id": self.model_id,
            "version": self.version,
            "demo_messages": self.demo_messages,
            "n_tokens": len(self.demo_ids),
            "created_utc": self.created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": self.notes,
        }
        save_file(
            {"demo_ids": np.asarray(self.demo_ids, dtype=np.int32)},
            str(p),
            metadata={META_KEY: json.dumps(meta)},
        )
        return p

    # ------------------------------------------------------------- inference
    @property
    def n_tokens(self) -> int:
        return len(self.demo_ids)

    def build_cache(self, model: Any) -> Any:
        """Prefill the demonstrations once and return the resulting cache."""
        import mlx.core as mx
        import mlx_lm  # noqa: F401 - ensures the runtime is the one the cache came from
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([self.demo_ids], dtype=mx.uint32), cache=cache))
        return cache

    def suffix_ids(self, tokenizer: Any, prompt: str) -> list[int]:
        """Token ids of the user turn, as a continuation of the demonstration prefix."""
        try:
            demo_text = tokenizer.apply_chat_template(
                self.demo_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            full_text = tokenizer.apply_chat_template(
                self.demo_messages + [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:  # noqa: BLE001 - templates without the thinking flag
            demo_text = tokenizer.apply_chat_template(
                self.demo_messages, tokenize=False, add_generation_prompt=False
            )
            full_text = tokenizer.apply_chat_template(
                self.demo_messages + [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        demo_ids = tokenizer.encode(demo_text)
        full_ids = tokenizer.encode(full_text)
        if full_ids[: len(demo_ids)] == demo_ids:
            return full_ids[len(demo_ids):]
        # Some templates (e.g. Qwen3.6) render demo turns with a thinking block but
        # the generation prompt without one. Fall back to concatenating a fresh
        # user turn onto the demonstration prefix; the cache still holds the demos.
        turn_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        turn_ids = list(tokenizer.encode(turn_text))
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None and turn_ids and turn_ids[0] == bos:
            turn_ids = turn_ids[1:]
        if not turn_ids:
            raise CartridgeFormatError("could not build a user-turn continuation for this template")
        return turn_ids

    def stream(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ):
        """Yield continuation text chunks conditioned on the cached demonstrations."""
        import copy

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if self._cache is None:
            self._cache = self.build_cache(model)
        suffix = self.suffix_ids(tokenizer, prompt)
        cache = copy.deepcopy(self._cache)
        sampler = make_sampler(temp=temperature)
        for response in stream_generate(
            model, tokenizer, suffix, max_tokens=max_tokens, prompt_cache=cache, sampler=sampler
        ):
            yield response.text

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """Generate a continuation conditioned on the cached demonstrations."""

        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        if self._cache is None:
            self._cache = self.build_cache(model)
        suffix = self.suffix_ids(tokenizer, prompt)
        cache = copy.deepcopy(self._cache)
        return mlx_lm.generate(
            model,
            tokenizer,
            prompt=suffix,
            prompt_cache=cache,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )

    def summary(self) -> str:
        return f"kv:{self.skill_id}@{self.version} n_tokens={self.n_tokens} model={self.model_id or '?'}"
