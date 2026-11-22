"""Hybrid local/cloud pairing — Pattern B (dual-brain agent split).

A frontier model plans; the local model executes. The cloud model receives the
user query and emits a compact **Intent DAG** (a JSON graph of abstract steps).
The workstation mounts the relevant weight skills and runs the DAG's tools
deterministically at native decode speed.

The DAG is treated as untrusted input: it is validated (unique ids, known tools,
acyclic, step budget) before any tool runs.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_STEPS = 24
MAX_SECONDS = 120.0


class PlanError(ValueError):
    """Raised when an Intent DAG is malformed or unsafe."""


@dataclass(frozen=True)
class IntentStep:
    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "tool": self.tool, "args": dict(self.args), "depends_on": list(self.depends_on)}


@dataclass
class IntentDAG:
    goal: str
    steps: list[IntentStep]

    def validate(self, allowed_tools: Sequence[str] | None = None) -> list[IntentStep]:
        if not self.steps:
            raise PlanError("intent DAG has no steps")
        if len(self.steps) > MAX_STEPS:
            raise PlanError(f"intent DAG exceeds {MAX_STEPS} steps")
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise PlanError("duplicate step ids")
        known = set(ids)
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in known:
                    raise PlanError(f"step {step.id!r} depends on unknown step {dep!r}")
                if dep == step.id:
                    raise PlanError(f"step {step.id!r} depends on itself")
            if allowed_tools is not None and step.tool not in allowed_tools:
                raise PlanError(f"tool {step.tool!r} is not allow-listed")
        return self.topological_order()

    def topological_order(self) -> list[IntentStep]:
        graph = {s.id: set(s.depends_on) for s in self.steps}
        by_id = {s.id: s for s in self.steps}
        ordered: list[IntentStep] = []
        while graph:
            ready = sorted(node for node, deps in graph.items() if not deps)
            if not ready:
                raise PlanError("intent DAG contains a cycle")
            for node in ready:
                ordered.append(by_id[node])
                graph.pop(node)
            for deps in graph.values():
                deps.difference_update(ready)
        return ordered


def parse_intent_dag(text: str) -> IntentDAG:
    """Extract the first JSON object containing ``steps`` from model output."""
    candidates: list[str] = []
    fence = re.findall(r"```(?:json)?\s*\n(.*?)```", text, flags=re.DOTALL)
    candidates.extend(fence)
    start = text.find("{")
    if start >= 0:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
            steps: list[IntentStep] = []
            for raw in payload["steps"]:
                if not isinstance(raw, dict) or "tool" not in raw:
                    raise PlanError("each step needs a 'tool'")
                steps.append(
                    IntentStep(
                        id=str(raw.get("id") or f"step{len(steps) + 1}"),
                        tool=str(raw["tool"]),
                        args=dict(raw.get("args") or {}),
                        depends_on=tuple(str(d) for d in (raw.get("depends_on") or raw.get("depends") or [])),
                    )
                )
            return IntentDAG(goal=str(payload.get("goal", "")), steps=steps)
    raise PlanError("no intent DAG found in model output")


class CloudPlanner(Protocol):
    def plan(self, query: str) -> str: ...


@dataclass
class ExecutionResult:
    step_id: str
    tool: str
    ok: bool
    output: Any
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "ok": self.ok,
            "seconds": round(self.seconds, 4),
            "output": self.output,
        }


@dataclass
class DualBrainExecutor:
    """Executes an Intent DAG locally against an allow-listed tool registry."""

    tools: dict[str, Callable[..., Any]]
    max_seconds: float = MAX_SECONDS

    def execute(self, dag: IntentDAG) -> list[ExecutionResult]:
        ordered = dag.validate(allowed_tools=tuple(self.tools))
        started = time.perf_counter()
        results: list[ExecutionResult] = []
        for step in ordered:
            if time.perf_counter() - started > self.max_seconds:
                results.append(
                    ExecutionResult(step.id, step.tool, False, {"error": "budget exceeded"}, 0.0)
                )
                break
            tool = self.tools[step.tool]
            t0 = time.perf_counter()
            try:
                output = tool(**step.args)
                ok = True
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                output = {"error": f"{type(exc).__name__}: {exc}"}
                ok = False
            results.append(ExecutionResult(step.id, step.tool, ok, output, time.perf_counter() - t0))
        return results


def run_pattern_b(
    *,
    planner: CloudPlanner,
    executor: DualBrainExecutor,
    query: str,
) -> dict[str, Any]:
    """Full dual-brain turn: cloud plans, workstation executes."""
    t0 = time.perf_counter()
    plan_text = planner.plan(query)
    dag = parse_intent_dag(plan_text)
    plan_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    results = executor.execute(dag)
    return {
        "query": query,
        "goal": dag.goal,
        "steps": [s.as_dict() for s in dag.steps],
        "results": [r.as_dict() for r in results],
        "ok": all(r.ok for r in results),
        "cloud_seconds": round(plan_seconds, 4),
        "local_seconds": round(time.perf_counter() - t1, 4),
    }
