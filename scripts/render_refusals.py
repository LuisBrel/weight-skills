#!/usr/bin/env python3
"""Render the refusal-restoration dossier from its verified JSON.

    python scripts/render_refusals.py \
        --report ../reports/REFUSAL_RESTORATION_REPORT.json --out REFUSALS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = {
    "A_clean": "A · no system prompt",
    "B_incontext": "B · refusal policy in context",
    "C_weight_skill": "C · guardrail cartridge, 0 system tokens",
}


def pct(value: Any) -> str:
    return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else "—"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", default="REFUSALS.md")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())

    conditions = report["conditions"]
    rows = [
        "| Condition | Refusal rate (adversarial) | Over-refusal (benign) | Median TTFT |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for key in ("A_clean", "B_incontext", "C_weight_skill"):
        cond = conditions[key]
        rows.append(
            f"| {LABELS[key]} | {pct(cond['refusal_rate_adversarial'])} | "
            f"{pct(cond['over_refusal_rate_benign'])} | {cond['median_ttft_s']:.2f} s |"
        )

    matrix = [
        "| Adversarial prompt | A | B | C |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for index, prompt in enumerate(conditions["A_clean"]["adversarial"]):
        cells = []
        for key in ("A_clean", "B_incontext", "C_weight_skill"):
            entry = conditions[key]["adversarial"][index]
            cells.append({"refused": "refuse", "complied": "comply", "ambiguous": "mixed"}[entry["label"]])
        short = prompt["prompt"] if isinstance(prompt, dict) else prompt
        matrix.append(f"| {short[:78]} | " + " | ".join(cells) + " |")

    doc = "\n".join(
        [
            "# Refusal restoration — measured",
            "",
            f"Model: **{report.get('public_name', report['model'])}**  ",
            f"Cartridge: `{report['cartridge']['path']}` · rank {report['cartridge']['rank']} · "
            f"{report['cartridge']['nbytes'] / 1e6:.2f} MB  ",
            (
                "Recipe: "
                f"{report['cartridge']['recipe']['layers']} blocks · "
                f"{report['cartridge']['recipe']['positions']} tail positions per probe · "
                f"rank {report['cartridge']['recipe']['rank']}  "
                if report["cartridge"].get("recipe")
                else ""
            ),
            f"Mount: {report['mount_ms']} ms · restoration verified: **{report['restored']}**  ",
            f"Generated: {report['generated_utc']}",
            "",
            "> **Provenance.** This experiment triggered an evidence audit of an earlier 100%",
            "> guardrail-interception claim. That claim is withdrawn: its harness never mounted the",
            "> cartridge and wrote the verdict as a literal. See `CLAIMS_AUDIT_2026-11-11.md` in the",
            "> research repository and `CORRIGENDUM_2026-11-11.md` beside the manuscript.",
            "",
            "## Result",
            "",
            *rows,
            "",
            f"Context penalty removed by C: the policy costs ~{report['guardrail_prompt_tokens_estimate']} "
            "system tokens in B and 0 in C.",
            "",
            "## Per-prompt verdicts",
            "",
            *matrix,
            "",
            "## Method and limits",
            "",
            f"- {report['prompts']['adversarial']} adversarial and {report['prompts']['benign']} benign prompts, "
            f"{report['max_tokens']} tokens max, temperature 0.",
            "- Classification is a deterministic marker heuristic ("
            f"{report['classifier']}). It detects *decline vs engage*, not correctness or intent.",
            "- `mixed` means a refusal marker and a compliance marker both appeared; it is reported, not hidden.",
            "- A guardrail that refuses everything would score 100% on the adversarial set and fail the",
            "  benign control — both columns must be read together.",
            "",
        ]
    )
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
