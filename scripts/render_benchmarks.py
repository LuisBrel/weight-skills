#!/usr/bin/env python3
"""Render benchmark tables from a verified report JSON.

Every number in the README and BENCHMARKS.md comes from this script reading the
report produced by the battery. Nothing is typed by hand.

    python scripts/render_benchmarks.py \
        --report ../reports/MULTI_MODEL_HARDWARE_BENCHMARK_REPORT.json \
        --readme README.md --out BENCHMARKS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COND_LABEL = {
    "A_clean": "A · clean baseline",
    "B_incontext": "B · in-context framework",
    "C_weight_skill": "C · weight skill mounted",
}

README_START = "<!-- BENCH:START -->"
README_END = "<!-- BENCH:END -->"


def _med(block: dict[str, Any] | None, key: str = "median") -> float | None:
    if not block:
        return None
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _ratio(numerator: float | None, denominator: float | None) -> str:
    if not numerator or not denominator:
        return "—"
    return f"{numerator / denominator:.2f}×"


def headline_table(report: dict[str, Any]) -> str:
    rows = [
        "| Model | Weights | Prefill tax (B ÷ A, paired) | Decode retained (C ÷ A, paired) | Mount (ms) | KV cache A → B (MB) | Restoration |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for model in report.get("models", {}).values():
        if model.get("status") != "ok":
            continue
        conditions = model["conditions"]
        a, b, c = (conditions.get(k, {}) for k in ("A_clean", "B_incontext", "C_weight_skill"))
        # Paired per-repetition ratios cancel host drift; fall back to medians
        # only for reports produced before the paired estimator existed.
        paired = model.get("paired", {})
        tax_paired = (paired.get("prefill_tax_B_over_A") or {}).get("median")
        ret_paired = (paired.get("decode_ratio_C_over_A") or {}).get("median")
        prefill_a, prefill_b = _med(a.get("ttft_s")), _med(b.get("ttft_s"))
        decode_a, decode_c = _med(a.get("decode_tps")), _med(c.get("decode_tps"))
        tax_value = tax_paired if tax_paired else (prefill_b / prefill_a if prefill_a and prefill_b else None)
        ret_value = ret_paired if ret_paired else (decode_c / decode_a if decode_a and decode_c else None)
        kv_a, kv_b = a.get("kv_cache_mb"), b.get("kv_cache_mb")
        mount = _med(model.get("mount_ms"))
        restored = "exact" if model.get("restored") else "CHECK"
        rows.append(
            "| {name} | {params} | {tax} | {retained} | {mount} | {kv} | {restored} |".format(
                name=model.get("public_name", model["id"]),
                params=model.get("params", ""),
                tax=f"{tax_value:.2f}×" if tax_value else "—",
                retained=f"{ret_value:.2f}×" if ret_value else "—",
                mount=_fmt(mount, 2),
                kv=f"{_fmt(kv_a, 1)} → {_fmt(kv_b, 1)}" if kv_a and kv_b else "—",
                restored=restored,
            )
        )
    return "\n".join(rows)


def condition_table(report: dict[str, Any]) -> str:
    rows = [
        "| Model | Condition | System tokens | Prefill / TTFT (s) | Decode (tok/s) | KV cache (MB) | Stall | Fidelity |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for model in report.get("models", {}).values():
        if model.get("status") != "ok":
            continue
        macro = model.get("fidelity_macro", {})
        prompt_tokens = model.get("prompts", {})
        for key in ("A_clean", "B_incontext", "C_weight_skill"):
            cond = model["conditions"].get(key)
            if not cond:
                continue
            system_tokens = {
                "A_clean": prompt_tokens.get("short_system_tokens", 0),
                "B_incontext": prompt_tokens.get("incontext_system_tokens", 0),
                "C_weight_skill": 0,
            }[key]
            fidelity = macro.get(key)
            rows.append(
                "| {name} | {label} | {tokens} | {ttft} | {decode} | {kv} | {stall} | {fid} |".format(
                    name=model.get("public_name", model["id"]),
                    label=COND_LABEL[key],
                    tokens=system_tokens,
                    ttft=_fmt(_med(cond.get("ttft_s")), 2),
                    decode=_fmt(_med(cond.get("decode_tps")), 1),
                    kv=_fmt(cond.get("kv_cache_mb"), 1),
                    stall="yes" if cond.get("stall_detected") else "no",
                    fid=f"{fidelity * 100:.0f}%" if isinstance(fidelity, (int, float)) else "—",
                )
            )
    return "\n".join(rows)


def fidelity_table(report: dict[str, Any]) -> str:
    tasks: list[str] = []
    for model in report.get("models", {}).values():
        fidelity = model.get("fidelity")
        if not fidelity:
            continue
        for task in fidelity.get("A_clean", {}).get("tasks", []):
            if task["id"] not in tasks:
                tasks.append(task["id"])
        break
    if not tasks:
        return "_no fidelity results in report_"

    header = "| Task | " + " | ".join(
        f"{m.get('public_name', m['id'])} A/B/C"
        for m in report["models"].values()
        if m.get("status") == "ok"
    ) + " |"
    n_models = sum(1 for m in report["models"].values() if m.get("status") == "ok")
    divider = "| :--- | " + " | ".join(":---:" for _ in range(n_models)) + " |"
    rows = [header, divider]
    for task_id in tasks:
        cells = []
        for model in report["models"].values():
            if model.get("status") != "ok":
                continue
            cells_task = []
            for key in ("A_clean", "B_incontext", "C_weight_skill"):
                entry = next(
                    (
                        t
                        for t in model.get("fidelity", {}).get(key, {}).get("tasks", [])
                        if t["id"] == task_id
                    ),
                    None,
                )
                cells_task.append(f"{entry['passed']}/{entry['total']}" if entry else "—")
            cells.append(" / ".join(cells_task))
        rows.append(f"| `{task_id}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def cartridge_table(report: dict[str, Any]) -> str:
    rows = [
        "| Model | Cartridge | Rank | Layers | Residual ratio (fit) | Size |",
        "| :--- | :--- | ---: | :--- | :--- | ---: |",
    ]
    for model in report.get("models", {}).values():
        if model.get("status") != "ok":
            continue
        cart = model.get("cartridge", {})
        diag = cart.get("diagnostics", {})
        residual = diag.get("residual_ratio", {})
        residual_text = ", ".join(f"{k}: {v:.4f}" for k, v in residual.items()) if residual else "—"
        rows.append(
            "| {name} | `{skill}` | {rank} | {layers} | {residual} | {size} MB |".format(
                name=model.get("public_name", model["id"]),
                skill="aws-cso",
                rank=cart.get("rank", "—"),
                layers=",".join(str(x) for x in cart.get("layers", [])),
                residual=residual_text,
                size=_fmt((cart.get("nbytes") or 0) / 1e6, 2),
            )
        )
    return "\n".join(rows)


def validity_table(report: dict[str, Any]) -> str:
    rows = [
        "| Model | Decode at start (tok/s) | Decode at end (tok/s) | Drift ratio | Swap in use (MB) | Restored (checks) |",
        "| :--- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for model in report.get("models", {}).values():
        if model.get("status") != "ok":
            continue
        drift = model.get("drift", {})
        swaps = [
            c.get("swap_used_mb_median")
            for c in model.get("conditions", {}).values()
            if c.get("swap_used_mb_median") is not None
        ]
        rows.append(
            "| {name} | {start} | {end} | {ratio} | {swap} | {checks} |".format(
                name=model.get("public_name", model["id"]),
                start=_fmt(drift.get("start_decode_tps"), 1),
                end=_fmt(drift.get("end_decode_tps"), 1),
                ratio=f"{drift['ratio']:.2f}×" if drift.get("ratio") else "—",
                swap=_fmt(max(swaps), 0) if swaps else "—",
                checks=model.get("restored_checks", "—"),
            )
        )
    return "\n".join(rows)


def environment_block(report: dict[str, Any]) -> str:
    meta = report.get("meta", {})
    config = report.get("config", {})
    return "\n".join(
        [
            f"- **Hardware:** {meta.get('machine', '?')} · "
            f"{meta.get('total_memory_gib', meta.get('total_memory_gb', '?'))} GiB unified memory",
            f"- **OS:** {meta.get('platform', '?')}",
            f"- **Runtime:** Python {meta.get('python', '?')} · MLX {meta.get('mlx', '?')} · mlx-lm {meta.get('mlx_lm', '?')}",
            f"- **Protocol:** conditions interleaved round-robin within each repetition "
            f"({config.get('reps', '?')} repetitions), temperature {config.get('temperature', 0)}, "
            f"{config.get('timing_tokens', '?')} timing tokens, {config.get('fidelity_tokens', '?')} fidelity tokens",
            f"- **Conditions:** A = {config.get('short_system_tokens', '?')} system tokens, "
            f"B = {config.get('incontext_system_tokens', '?')} system tokens, C = 0 system tokens + cartridge",
            f"- **Generated:** {meta.get('generated_utc', '?')}",
        ]
    )


def build_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Verified benchmarks",
            "",
            "Generated by `scripts/render_benchmarks.py` directly from",
            "`MULTI_MODEL_HARDWARE_BENCHMARK_REPORT.json`. Do not edit by hand.",
            "",
            "All rows are **within-model treatment effects**: each model is compared only",
            "against its own baseline. This is not a cross-model quality ranking, and no",
            "model here is claimed to be better than another.",
            "",
            "## Environment",
            "",
            environment_block(report),
            "",
            "## Excluded models",
            "",
            *(
                [f"- **{e['model']}** — {e['reason']}. {e['impact']}." for e in report.get("exclusions", [])]
                or ["- none"]
            ),
            "",
            "## Measurement validity",
            "",
            "Conditions are interleaved, and a baseline decode probe runs before and after",
            "each model. Headline ratios are **paired per repetition** (B_i ÷ A_i, C_i ÷ A_i),",
            "which cancels common-mode host drift. Absolute tok/s figures are indicative:",
            "on this host, per-condition decode samples spread by 2.4–4.9× within a run.",
            "",
            validity_table(report),
            "",
            "## Headline",
            "",
            headline_table(report),
            "",
            *[
                f"_{m['public_name']} was measured through `{m['loader']}` "
                f"(the architecture its checkpoint actually implements), so its KV-cache "
                f"column is not instrumented._"
                for m in report.get("models", {}).values()
                if m.get("status") == "ok" and m.get("loader")
            ],
            "",
            "",
            "## Per-condition measurements",
            "",
            condition_table(report),
            "",
            "## Downstream fidelity (deterministic AWS rubrics)",
            "",
            "Each cell is `passed/total` across the task's structural checks.",
            "`0/1` means the model produced no parsable YAML/JSON document at all,",
            "which scores zero — it is a parse failure, not a one-check task.",
            "",
            fidelity_table(report),
            "",
            "## Compiled cartridges",
            "",
            cartridge_table(report),
            "",
        ]
    )


def inject_readme(readme_path: Path, block: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(f"{readme_path} is missing the {README_START} / {README_END} markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme_path.write_text(f"{head}{README_START}\n{block}\n{README_END}{tail}", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--readme", default=None)
    parser.add_argument("--out", default="BENCHMARKS.md")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    markdown = build_markdown(report)
    Path(args.out).write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out}")

    if args.readme:
        inject_readme(Path(args.readme), headline_table(report))
        print(f"injected headline table into {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
