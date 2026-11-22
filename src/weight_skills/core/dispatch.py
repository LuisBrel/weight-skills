"""Content-addressable routing: rolling SimHash over system prompts.

The proxy sees a 4,000-token ``CLAUDE.md`` / ``AGENTS.md`` preamble on every
request. Hashing it once and comparing 64-bit fingerprints is O(tokens) with a
tiny constant, and it lets the proxy decide *which cartridge to mount* before
the request ever reaches the model.

This is a text-space probe. It is deliberately model-agnostic: no forward pass,
no GPU, no tokenizer dependency. An activation-space probe (hooking an early
decoder layer) can be supplied through ``probe_fn`` if a deployment wants it.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

_WORD = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")


def simhash64(text: str, *, shingle: int = 1) -> int:
    """64-bit SimHash of ``text``.

    ``shingle=3`` hashes word trigrams, which is far more robust to a paragraph
    being appended or removed than a unigram hash.
    """
    words = _tokens(text)
    if not words:
        return 0
    if shingle > 1 and len(words) >= shingle:
        features = [" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)]
    else:
        features = words
    counts = Counter(features)
    items = list(counts.items())
    # Vectorised accumulation: build the 64-bit matrix once and contract it with
    # the feature weights. The naive per-feature/per-bit loop costs ~5 ms on a
    # 4,000-token preamble; this is ~20x faster and numerically identical.
    hashes = np.fromiter((_hash64(f) for f, _ in items), dtype=np.uint64, count=len(items))
    weights = np.fromiter((w for _, w in items), dtype=np.float64, count=len(items))
    bit_positions = np.arange(64, dtype=np.uint64)
    bits = ((hashes[:, None] >> bit_positions[None, :]) & np.uint64(1)).astype(np.float64)
    signed = bits * 2.0 - 1.0  # +weight when the bit is set, -weight otherwise
    vector = weights @ signed
    out = 0
    for bit in np.nonzero(vector > 0)[0]:
        out |= 1 << int(bit)
    return out


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def similarity(a: int, b: int) -> float:
    """``1 - hamming/64`` in ``[0, 1]``."""
    return 1.0 - hamming64(a, b) / 64.0


@dataclass(frozen=True)
class DispatchDecision:
    """Outcome of a routing decision."""

    skill_id: str | None
    score: float
    latency_ms: float
    candidate: str | None = None
    reason: str = ""

    @property
    def matched(self) -> bool:
        return self.skill_id is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "score": round(self.score, 4),
            "latency_ms": round(self.latency_ms, 4),
            "candidate": self.candidate,
            "reason": self.reason,
        }


@dataclass
class _Entry:
    skill_id: str
    fingerprint: int
    label: str
    cartridge_path: str | None = None


class SkillDispatcher:
    """Registry of known prompt fingerprints -> skill ids."""

    def __init__(
        self,
        *,
        threshold: float = 0.86,
        probe_fn: Callable[[str], int] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self._entries: list[_Entry] = []
        self._probe_fn = probe_fn

    def register(
        self,
        skill_id: str,
        reference_text: str,
        *,
        cartridge_path: str | None = None,
        label: str = "",
    ) -> None:
        fingerprint = self._fingerprint(reference_text)
        self._entries = [e for e in self._entries if e.skill_id != skill_id]
        self._entries.append(
            _Entry(skill_id=skill_id, fingerprint=fingerprint, label=label or skill_id, cartridge_path=cartridge_path)
        )

    def register_cartridge(self, cartridge, reference_text: str) -> None:
        self.register(
            cartridge.skill_id,
            reference_text,
            cartridge_path=str(cartridge.path) if cartridge.path else None,
        )

    def _fingerprint(self, text: str) -> int:
        return self._probe_fn(text) if self._probe_fn is not None else simhash64(text, shingle=3)

    def route(self, text: str) -> DispatchDecision:
        t0 = time.perf_counter()
        if not text.strip() or not self._entries:
            return DispatchDecision(
                skill_id=None,
                score=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                reason="empty prompt or empty registry",
            )
        fp = self._fingerprint(text)
        best: _Entry | None = None
        best_score = -1.0
        for entry in self._entries:
            score = similarity(fp, entry.fingerprint)
            if score > best_score:
                best, best_score = entry, score
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if best is not None and best_score >= self.threshold:
            return DispatchDecision(
                skill_id=best.skill_id,
                score=best_score,
                latency_ms=elapsed_ms,
                candidate=best.label,
                reason="fingerprint match",
            )
        return DispatchDecision(
            skill_id=None,
            score=max(best_score, 0.0),
            latency_ms=elapsed_ms,
            candidate=best.label if best else None,
            reason=f"below threshold ({best_score:.3f} < {self.threshold:.3f})",
        )

    def identify(self, text: str, *, top_k: int = 3) -> list[tuple[str, float]]:
        """Rank all registered skills by similarity (diagnostics/tests)."""
        fp = self._fingerprint(text)
        scored = [(e.skill_id, similarity(fp, e.fingerprint)) for e in self._entries]
        return sorted(scored, key=lambda kv: kv[1], reverse=True)[:top_k]

    @property
    def skills(self) -> Iterable[str]:
        return (e.skill_id for e in self._entries)

    def __len__(self) -> int:
        return len(self._entries)
