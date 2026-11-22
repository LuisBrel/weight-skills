"""Content-addressable system-prompt stripper.

Agent harnesses resend the same 4,000-token ``CLAUDE.md`` / ``AGENTS.md``
preamble on every turn. The stripper recognises that preamble by SimHash,
removes it from the outgoing prompt, and reports which weight skill should be
mounted instead. The model then runs with zero framework tokens.

The stripper never *guesses*: it only removes a system message when the
similarity to a registered reference exceeds the threshold, and it keeps the
original text available in the result for auditing.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..core.dispatch import SkillDispatcher, simhash64

CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) when no tokenizer is available."""
    return max(1, len(text) // 4) if text.strip() else 0


@dataclass
class StripResult:
    """Outcome of stripping one chat request."""

    messages: list[dict[str, Any]]
    removed_chars: int
    removed_tokens_estimate: int
    skill_id: str | None
    score: float
    latency_ms: float
    matched_reference: str | None = None
    reason: str = ""
    original_system: str = ""

    @property
    def stripped(self) -> bool:
        return self.removed_chars > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stripped": self.stripped,
            "removed_chars": self.removed_chars,
            "removed_tokens_estimate": self.removed_tokens_estimate,
            "skill_id": self.skill_id,
            "score": round(self.score, 4),
            "latency_ms": round(self.latency_ms, 4),
            "matched_reference": self.matched_reference,
            "reason": self.reason,
        }


class PromptStripper:
    """Detects and removes known harness preambles from chat requests."""

    def __init__(
        self,
        dispatcher: SkillDispatcher | None = None,
        *,
        min_chars: int = 1200,
        keep_system_remainder: bool = False,
    ) -> None:
        self.dispatcher = dispatcher or SkillDispatcher()
        self.min_chars = min_chars
        self.keep_system_remainder = keep_system_remainder
        self._references: dict[str, str] = {}
        self.stats = {"requests": 0, "stripped": 0, "chars_removed": 0, "tokens_saved_estimate": 0}

    # ------------------------------------------------------------- registry
    def register(self, skill_id: str, reference_text: str, *, cartridge_path: str | None = None) -> None:
        """Register a harness preamble that maps to ``skill_id``."""
        self.dispatcher.register(skill_id, reference_text, cartridge_path=cartridge_path)
        self._references[skill_id] = reference_text

    def register_file(self, skill_id: str, path: str, *, cartridge_path: str | None = None) -> None:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            self.register(skill_id, handle.read(), cartridge_path=cartridge_path)

    # --------------------------------------------------------------- strip
    def strip(self, messages: Iterable[dict[str, Any]]) -> StripResult:
        t0 = time.perf_counter()
        msgs = [dict(m) for m in messages]
        self.stats["requests"] += 1

        system_text = "\n\n".join(
            str(m.get("content", ""))
            for m in msgs
            if str(m.get("role", "")).lower() == "system"
        )
        if len(system_text) < self.min_chars:
            return StripResult(
                messages=msgs,
                removed_chars=0,
                removed_tokens_estimate=0,
                skill_id=None,
                score=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                reason="no substantial system prompt",
            )

        decision = self.dispatcher.route(system_text)
        if not decision.matched:
            return StripResult(
                messages=msgs,
                removed_chars=0,
                removed_tokens_estimate=0,
                skill_id=None,
                score=decision.score,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                reason=decision.reason,
                original_system=system_text,
            )

        reference = self._references.get(decision.skill_id or "", "")
        remainder = self._subtract(system_text, reference) if self.keep_system_remainder else ""
        kept = 0
        rebuilt: list[dict[str, Any]] = []
        for msg in msgs:
            if str(msg.get("role", "")).lower() == "system":
                if remainder and kept == 0:
                    rebuilt.append({**msg, "content": remainder})
                    kept += 1
                continue
            rebuilt.append(msg)

        removed = len(system_text) - len(remainder)
        tokens = estimate_tokens(system_text) - estimate_tokens(remainder)
        self.stats["stripped"] += 1
        self.stats["chars_removed"] += removed
        self.stats["tokens_saved_estimate"] += max(tokens, 0)
        return StripResult(
            messages=rebuilt,
            removed_chars=removed,
            removed_tokens_estimate=max(tokens, 0),
            skill_id=decision.skill_id,
            score=decision.score,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            matched_reference=decision.candidate,
            reason="fingerprint match",
            original_system=system_text,
        )

    @staticmethod
    def _subtract(full: str, reference: str) -> str:
        """Best-effort removal of the reference block, preserving extra rules."""
        if not reference:
            return ""
        normalized_ref = reference.strip()
        if normalized_ref and normalized_ref in full:
            return full.replace(normalized_ref, "").strip()
        # Fall back to dropping lines that are individually near-identical.
        ref_hashes = {simhash64(line, shingle=2) for line in reference.splitlines() if len(line) > 40}
        from ..core.dispatch import similarity

        kept = [
            line
            for line in full.splitlines()
            if not (len(line) > 40 and any(similarity(simhash64(line, shingle=2), h) > 0.9 for h in ref_hashes))
        ]
        return "\n".join(kept).strip()

    def summary(self) -> dict[str, Any]:
        return dict(self.stats)
