#!/usr/bin/env python3
"""03 — Dual-brain pairing: cloud plans, local workstation executes.

Pattern B: a frontier model receives the user query and emits a compact Intent
DAG (~120 tokens). The local model — with domain cartridges mounted — expands
each abstract step into a deterministic tool call and executes it.

This example uses a stub cloud planner so it runs offline; swap ``StubPlanner``
for your Claude/Gemini client and keep everything else identical.
"""

from __future__ import annotations

import json

from weight_skills.hybrid.dual_brain import DualBrainExecutor, run_pattern_b

INTENT_DAG = {
    "goal": "audit the acme-reports bucket",
    "steps": [
        {"id": "describe", "tool": "describe_bucket", "args": {"name": "acme-reports"}},
        {"id": "pab", "tool": "check_public_access_block", "args": {"name": "acme-reports"}, "depends_on": ["describe"]},
        {"id": "report", "tool": "summarize", "args": {"checks": ["describe", "pab"]}, "depends_on": ["pab"]},
    ],
}


class StubPlanner:
    """Stands in for Claude 3.7 / Gemini 3 — returns the Intent DAG as JSON."""

    def plan(self, query: str) -> str:
        print(f"[cloud] planning for: {query}")
        return "```json\n" + json.dumps(INTENT_DAG) + "\n```"


def describe_bucket(name: str):
    return {"name": name, "region": "eu-west-1", "exists": True}


def check_public_access_block(name: str):
    return {
        "name": name,
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }


def summarize(checks):
    return {"checks": checks, "verdict": "bucket is private"}


def main() -> int:
    executor = DualBrainExecutor(
        {
            "describe_bucket": describe_bucket,
            "check_public_access_block": check_public_access_block,
            "summarize": summarize,
        }
    )
    result = run_pattern_b(
        planner=StubPlanner(),
        executor=executor,
        query="Is the acme-reports bucket locked down?",
    )
    print(json.dumps(result, indent=2))
    print(
        f"\ncloud {result['cloud_seconds']:.3f}s | local {result['local_seconds']:.3f}s | ok={result['ok']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
