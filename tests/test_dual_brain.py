from __future__ import annotations

import pytest

from weight_skills.hybrid.dual_brain import (
    DualBrainExecutor,
    IntentDAG,
    IntentStep,
    PlanError,
    parse_intent_dag,
    run_pattern_b,
)

PLAN = """
```json
{
  "goal": "verify bucket posture",
  "steps": [
    {"id": "s1", "tool": "describe_bucket", "args": {"name": "acme"}},
    {"id": "s2", "tool": "check_pab", "args": {"name": "acme"}, "depends_on": ["s1"]}
  ]
}
```
"""


def test_parse_fenced_plan():
    dag = parse_intent_dag(PLAN)
    assert dag.goal == "verify bucket posture"
    assert [s.tool for s in dag.steps] == ["describe_bucket", "check_pab"]


def test_parse_bare_json():
    dag = parse_intent_dag('{"steps": [{"tool": "noop"}]}')
    assert dag.steps[0].id == "step1"


def test_parse_rejects_garbage():
    with pytest.raises(PlanError):
        parse_intent_dag("no json here")


def test_cycle_detection():
    dag = IntentDAG(
        goal="",
        steps=[
            IntentStep("a", "t1", {}, ("b",)),
            IntentStep("b", "t2", {}, ("a",)),
        ],
    )
    with pytest.raises(PlanError):
        dag.validate()


def test_unknown_dependency():
    dag = IntentDAG(goal="", steps=[IntentStep("a", "t1", {}, ("ghost",))])
    with pytest.raises(PlanError):
        dag.validate()


def test_tool_allowlist_enforced():
    dag = IntentDAG(goal="", steps=[IntentStep("a", "rm_rf", {})])
    with pytest.raises(PlanError):
        dag.validate(allowed_tools=("safe_tool",))


def test_topological_execution_order():
    dag = IntentDAG(
        goal="",
        steps=[
            IntentStep("last", "t", {}, ("first",)),
            IntentStep("first", "t", {}),
        ],
    )
    order = [s.id for s in dag.validate(allowed_tools=("t",))]
    assert order == ["first", "last"]


def test_executor_runs_tools_and_captures_errors():
    calls: list[str] = []

    def good(**kwargs):
        calls.append("good")
        return {"ok": True, **kwargs}

    def bad(**_kwargs):
        raise RuntimeError("boom")

    dag = IntentDAG(
        goal="g",
        steps=[IntentStep("a", "good", {"x": 1}), IntentStep("b", "bad", {"depends": "a"}, ("a",))],
    )
    results = DualBrainExecutor({"good": good, "bad": bad}).execute(dag)
    assert calls == ["good"]
    assert results[0].ok is True
    assert results[1].ok is False
    assert "boom" in results[1].output["error"]


def test_step_budget_enforced():
    dag = IntentDAG(goal="", steps=[IntentStep(f"s{i}", "t", {}) for i in range(40)])
    with pytest.raises(PlanError):
        dag.validate()


def test_run_pattern_b_end_to_end():
    class Planner:
        def plan(self, query: str) -> str:
            return PLAN

    def describe_bucket(name):
        return {"name": name, "exists": True}

    def check_pab(name):
        return {"name": name, "blocked": True}

    out = run_pattern_b(
        planner=Planner(),
        executor=DualBrainExecutor({"describe_bucket": describe_bucket, "check_pab": check_pab}),
        query="is acme locked down?",
    )
    assert out["ok"] is True
    assert len(out["results"]) == 2
    assert out["steps"][1]["depends_on"] == ["s1"]
