"""Per-model equivalence-policy defaults (D19/G.84) and aligned-exact serving mode (D28/D30).

The level-2 certificate (G.61) is per model: the share of prompts whose compiled output is certified
differs a lot (Qwen2-0.5B 87.5%, Gemma-4-E4B 62.5%, 35B hybrid 25% in the 16-question setup). Serving should
therefore not treat ``WEIGHT_SKILLS_EQUIVALENCE`` as a global switch: by default it is **on ("fallback") for
models whose measured certified rate clears 80%**, and off otherwise. An explicit env value always wins.

``aligned_exact`` is a third, strictly stronger mode (G.93/D28): when the compiled cache was built on the
serving routine's prefill-chunk boundary, serving it is bit-identical to a cold chunked prefill and no
statistical certificate is needed. It is opt-in through the env value only (no per-model default):
:func:`decide_aligned_serving` refuses to serve a cache that cannot prove that contract, so this mode
never silently serves a non-aligned cache. :func:`build_policy_rows` is the pure report-row generator the
research repo calls with the values it reads from ``reports/COMPILED_SERVING_ALIGNMENT.json``.

These rates are measurements, not tuned constants: they come from reports/EQUIVALENCE_GATE_V2.json (G.61) and
can be overridden per deployment by passing ``certified_rate``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .compiled_prompts import CacheProvenance, aligned_exact_contract

#: Env value selecting the bit-identity contract instead of a sampled equivalence label.
ALIGNED_EXACT_MODE = "aligned_exact"
#: Decision returned when a compiled cache may not be served as aligned-exact; the caller serves cold.
COLD_MODE = "cold"
#: The guarantee a policy row claims (scope: one serving routine, batch=1).
ALIGNED_GUARANTEE = (
    "bit-identical to the same serving routine's cold prefill (batch=1, same chunk/dtype/process)"
)

#: Measured per-prompt certification rates from G.61 (16-question setup); substring match on the model id.
DEFAULT_CERTIFIED_RATES: dict[str, float] = {
    "qwen2-0.5b": 0.875,
    "gemma-4-e4b": 0.625,
    "qwen36-35b": 0.25,
}
DEFAULT_THRESHOLD = 0.8


@dataclass(frozen=True)
class EquivalencePolicy:
    """Resolved serving policy for a model id."""

    model_id: str
    mode: str
    certified_rate: float | None
    threshold: float
    source: str

    @property
    def fallback(self) -> bool:
        return self.mode == "fallback"

    @property
    def aligned_exact(self) -> bool:
        return self.mode == ALIGNED_EXACT_MODE


def certified_rate_for(model_id: str, rates: dict[str, float] | None = None) -> float | None:
    """Measured certified rate for a model id (substring match), or ``None`` if unmeasured."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    table = rates if rates is not None else DEFAULT_CERTIFIED_RATES
    lowered = model_id.lower()
    for prefix, rate in table.items():
        if prefix in lowered:
            return float(rate)
    return None


def decide_equivalence_policy(
    model_id: str,
    *,
    certified_rate: float | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    env_value: str | None = None,
    supports_aligned_exact: bool = False,
) -> EquivalencePolicy:
    """Resolve the serving policy: explicit env value wins, else rate >= threshold -> ``fallback``.

    ``env_value`` is the raw ``WEIGHT_SKILLS_EQUIVALENCE`` value (``None`` when unset); accepted values are
    ``off``, ``fallback`` and ``aligned_exact``. Only the env value selects ``aligned_exact`` -- per-model
    defaults stay sampled (``off`` / ``fallback``). Unknown models with no measured rate default to ``off``
    (sampled mode) rather than silently trusting compiled caches.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    rate = certified_rate if certified_rate is not None else certified_rate_for(model_id)
    if env_value is not None:
        mode = env_value.strip().lower()
        if mode not in ("off", "fallback", ALIGNED_EXACT_MODE):
            raise ValueError(
                "WEIGHT_SKILLS_EQUIVALENCE must be 'off', 'fallback' or 'aligned_exact'"
            )
        if mode == ALIGNED_EXACT_MODE and not supports_aligned_exact:
            # fail closed: this build has no aligned serving path, and serving a sampled cache under
            # this mode would silently drop the bit-identity guarantee (D30).
            raise ValueError(
                "WEIGHT_SKILLS_EQUIVALENCE=aligned_exact requires the aligned serving path "
                "(D28/D30 engine wiring), which is not enabled in this build; use 'off' or 'fallback'"
            )
        return EquivalencePolicy(model_id=model_id, mode=mode, certified_rate=rate,
                                 threshold=threshold, source="env")
    if rate is None:
        return EquivalencePolicy(model_id=model_id, mode="off", certified_rate=None,
                                 threshold=threshold, source="unknown-model")
    mode = "fallback" if rate >= threshold else "off"
    return EquivalencePolicy(model_id=model_id, mode=mode, certified_rate=rate,
                             threshold=threshold, source="profile")


# ------------------------------------------------- aligned-exact serving (D28/D30)
def decide_aligned_serving(
    provenance: CacheProvenance | None,
    serving_chunk: int,
) -> tuple[str, str]:
    """Decide whether a compiled cache may be served as bit-identical to cold prefill.

    Returns ``(mode, reason)``: ``("aligned_exact", "")`` means serve the compiled cache and re-prefill only
    the suffix; ``("cold", reason)`` means the caller must re-prefill cold and may expose ``reason`` (header
    or log). This mode never silently serves a non-aligned cache: a cache with no recorded chunk (plain or
    sampled artefact) or a chunk that does not match the serving routine is always refused.
    """
    ok, reason = aligned_exact_contract(provenance, serving_chunk)
    if ok:
        return ALIGNED_EXACT_MODE, ""
    return COLD_MODE, reason


def _model_label(entry: Mapping[str, Any]) -> str:
    raw = entry.get("model")
    if raw is None:
        raise ValueError("measurement row is missing 'model'")
    if isinstance(raw, Mapping):
        raw = raw.get("id") or raw.get("path")
    label = str(raw or "").strip()
    if not label:
        raise ValueError("measurement row has an empty 'model'")
    return label


def _flat_measurement(entry: Mapping[str, Any], key: str) -> Any:
    if key not in entry:
        raise ValueError(
            f"measurement row for {_model_label(entry)!r} is missing {key!r}; "
            "supply values read from the evidence artefact instead of defaults"
        )
    return entry[key]


def _aggregate_measurement(entry: Mapping[str, Any], label: str) -> tuple[Any, Any, Any]:
    aggregates = entry.get("aggregates")
    if not isinstance(aggregates, Mapping):
        raise ValueError(
            f"measurement row for {label!r} has no aggregates mapping; expected values read from "
            "the evidence artefact"
        )
    arms = aggregates.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError(
            f"measurement row for {label!r} has 'aggregates' but no arms mapping; "
            "expected aggregates.arms.aligned/aligned_roundtrip from the evidence artefact"
        )
    try:
        aligned = arms["aligned"]["bit_identical"]
        roundtrip = arms["aligned_roundtrip"]["bit_identical"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"measurement row for {label!r} is missing aggregates.arms.aligned.bit_identical or "
            "aggregates.arms.aligned_roundtrip.bit_identical"
        ) from exc
    ratio = aggregates.get("median_ttft_ratio")
    if ratio is None:
        aligned_arm = arms.get("aligned")
        ratio = aligned_arm.get("median_ttft_ratio") if isinstance(aligned_arm, Mapping) else None
    if ratio is None:
        raise ValueError(
            f"measurement row for {label!r} is missing aggregates.median_ttft_ratio"
        )
    return aligned, roundtrip, ratio


def build_policy_rows(
    measurements: Iterable[Mapping[str, Any]],
    *,
    evidence_artefact: str | None = None,
) -> list[dict[str, Any]]:
    """Build aligned-exact policy-table rows from supplied measurements (pure; no file reads).

    Each measurement is a mapping in one of two shapes:

    * flat: ``model``, ``evidence_artefact``, ``aligned_cells_bit_identical``, ``roundtrip_cells`` and
      ``median_ttft_ratio``.
    * a model entry read from ``reports/COMPILED_SERVING_ALIGNMENT.json``: ``model`` (id string or mapping
      with ``id``/``path``) and ``aggregates`` (``arms.aligned.bit_identical``,
      ``arms.aligned_roundtrip.bit_identical`` and ``median_ttft_ratio``). Pass the artefact path once as
      ``evidence_artefact``.

    Missing measurement values raise :class:`ValueError` instead of defaulting, so a row never contains a
    number the evidence artefact does not. The only fixed value is ``measured.batch_limit == "1"``, which is
    part of the guarantee's scope (batch=1), not a measurement.
    """
    rows: list[dict[str, Any]] = []
    for entry in measurements:
        if not isinstance(entry, Mapping):
            raise TypeError(f"measurement rows must be mappings, got {type(entry).__name__}")
        label = _model_label(entry)
        evidence = entry.get("evidence_artefact") or evidence_artefact
        if not evidence:
            raise ValueError(
                f"measurement row for {label!r} is missing 'evidence_artefact' "
                "(pass it per row or as evidence_artefact=...)"
            )
        if isinstance(entry.get("aggregates"), Mapping):
            aligned, roundtrip, ratio = _aggregate_measurement(entry, label)
        else:
            aligned = _flat_measurement(entry, "aligned_cells_bit_identical")
            roundtrip = _flat_measurement(entry, "roundtrip_cells")
            ratio = _flat_measurement(entry, "median_ttft_ratio")
        rows.append(
            {
                "model": label,
                "mode": ALIGNED_EXACT_MODE,
                "guarantee": ALIGNED_GUARANTEE,
                "evidence_artefact": str(evidence),
                "measured": {
                    "aligned_cells_bit_identical": aligned,
                    "roundtrip_cells": roundtrip,
                    "median_ttft_ratio": ratio,
                    "batch_limit": "1",
                },
            }
        )
    return rows


#: Clearly-labelled placeholders for ``--demo``: no number here is a measurement.
_DEMO_MEASUREMENTS: tuple[dict[str, Any], ...] = (
    {
        "model": "DEMO-model-id — placeholder, not a measurement",
        "evidence_artefact": "DEMO — replace with reports/COMPILED_SERVING_ALIGNMENT.json",
        "aligned_cells_bit_identical": None,
        "roundtrip_cells": None,
        "median_ttft_ratio": None,
    },
)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the policy-row shape; ``--demo`` uses clearly labelled placeholders only."""
    parser = argparse.ArgumentParser(
        prog="python -m weight_skills.serving.policy",
        description="Emit aligned-exact serving policy-table rows (pure function of supplied data).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print a clearly labelled demo row (placeholder values, not measurements)",
    )
    args = parser.parse_args(argv)
    if not args.demo:
        parser.print_help()
        return 2
    print("DEMO ONLY — placeholder values, not measurements; do not cite.", file=sys.stderr)
    print(json.dumps(build_policy_rows(_DEMO_MEASUREMENTS), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
