from __future__ import annotations

from weight_skills.core.dispatch import SkillDispatcher, hamming64, simhash64, similarity
from weight_skills.proxy.prompt_stripper import PromptStripper

HARNESS_PROMPT = "\n".join(
    [
        "# Workspace instructions",
        "Every agent must read the following operating rules before acting.",
        "Rule 1: never run destructive commands without checking the resolved path.",
        "Rule 2: batch shell work and avoid unbounded log reads.",
        "Rule 3: prefer targeted edits over full-file rewrites.",
        "Rule 4: every benchmark number must be read from a verified JSON report.",
        "Rule 5: do not disclose internal training recipes in public artefacts.",
        "Rule 6: keep public naming clean and free of internal nicknames.",
        "Rule 7: record hypotheses and outcomes in the research ledger.",
        "Rule 8: use the brand palette for any user interface work.",
        "Rule 9: never poll long-running jobs on a short timer.",
        "Rule 10: report residual failures instead of re-probing indefinitely.",
    ]
    * 6
)

OTHER_PROMPT = "\n".join(
    [
        "# Cooking instructions",
        "Preheat the oven to two hundred degrees and line a tray with parchment.",
        "Whisk the eggs with sugar until the mixture ribbons off the whisk.",
        "Fold the flour in three additions to keep the batter aerated.",
        "Bake for twenty five minutes and cool on a wire rack before icing.",
    ]
    * 10
)


def test_simhash_identical_is_one():
    assert similarity(simhash64(HARNESS_PROMPT), simhash64(HARNESS_PROMPT)) == 1.0


def test_simhash_different_is_far():
    score = similarity(simhash64(HARNESS_PROMPT), simhash64(OTHER_PROMPT))
    assert score < 0.75


def test_simhash_robust_to_appended_paragraph():
    mutated = HARNESS_PROMPT + "\nAdditional note: this paragraph was appended later."
    assert similarity(simhash64(HARNESS_PROMPT), simhash64(mutated)) > 0.86


def test_hamming_distance_is_symmetric():
    a, b = simhash64(HARNESS_PROMPT), simhash64(OTHER_PROMPT)
    assert hamming64(a, b) == hamming64(b, a)


def test_dispatcher_requires_threshold():
    dispatcher = SkillDispatcher(threshold=0.9)
    dispatcher.register("harness", HARNESS_PROMPT)
    assert dispatcher.route(HARNESS_PROMPT).matched
    assert not dispatcher.route(OTHER_PROMPT).matched


def test_dispatcher_returns_ranking():
    dispatcher = SkillDispatcher()
    dispatcher.register("harness", HARNESS_PROMPT)
    dispatcher.register("cooking", OTHER_PROMPT)
    ranking = dispatcher.identify(HARNESS_PROMPT, top_k=2)
    assert ranking[0][0] == "harness"
    assert ranking[0][1] > ranking[1][1]


def test_stripper_removes_known_preamble():
    stripper = PromptStripper()
    stripper.register("harness", HARNESS_PROMPT)
    messages = [
        {"role": "system", "content": HARNESS_PROMPT},
        {"role": "user", "content": "Fix the failing test."},
    ]
    result = stripper.strip(messages)
    assert result.stripped
    assert result.skill_id == "harness"
    assert result.removed_chars >= len(HARNESS_PROMPT) - 10
    assert [m["role"] for m in result.messages] == ["user"]
    assert result.latency_ms < 50


def test_stripper_leaves_short_prompt_alone():
    stripper = PromptStripper()
    stripper.register("harness", HARNESS_PROMPT)
    result = stripper.strip([{"role": "system", "content": "Be terse."}])
    assert not result.stripped
    assert result.messages[0]["content"] == "Be terse."


def test_stripper_leaves_unknown_prompt_alone():
    stripper = PromptStripper()
    stripper.register("harness", HARNESS_PROMPT)
    result = stripper.strip([{"role": "system", "content": OTHER_PROMPT}])
    assert not result.stripped
    assert result.skill_id is None


def test_stripper_stats_accumulate():
    stripper = PromptStripper()
    stripper.register("harness", HARNESS_PROMPT)
    for _ in range(3):
        stripper.strip([{"role": "system", "content": HARNESS_PROMPT}, {"role": "user", "content": "hi"}])
    stats = stripper.summary()
    assert stats["requests"] == 3
    assert stats["stripped"] == 3
    assert stats["tokens_saved_estimate"] > 500
