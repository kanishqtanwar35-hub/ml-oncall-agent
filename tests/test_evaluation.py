"""The agent's own scorecard, as assertions.

Every number the README quotes is pinned here, so it cannot go stale while the
weights change underneath it.
"""

import pytest

from oncall.evaluate import ablate, degradation, recovery, run, tool_selection
from oncall.incidents import SCENARIOS


@pytest.fixture(scope="module")
def report():
    return run()


def test_the_agent_gets_every_root_cause_right(report):
    assert report.task_success == 1.0, report.table()


def test_the_agent_recommends_the_right_action_every_time(report):
    assert report.action_accuracy == 1.0, report.table()


def test_every_scenario_is_scored(report):
    assert len(report.results) == len(SCENARIOS)


def test_the_run_is_short(report):
    """Steps per task is a cost metric. An agent that takes forty steps to read
    three files is burning tokens on ceremony."""
    assert report.mean_steps <= 12


def test_calibration_is_honestly_unmeasurable_here(report):
    """At 100% task success there are no wrong answers to compare confidence
    against, so the calibration number does not exist. Reporting a figure here
    would be inventing one - the honest output is 'n/a, and that is a limit of
    the scenario set'."""
    calibration = report.calibration()
    assert calibration["when_right"] > 0.0
    assert calibration["when_wrong"] != calibration["when_wrong"]      # NaN


def test_every_artifact_is_consulted(report):
    assert tool_selection(report)["consulted_per_incident"] == 3.0


def test_removing_an_artifact_never_helps():
    """Ablation's floor condition. If dropping evidence ever IMPROVED the
    score, the weights would be fitting the scenario set rather than reasoning
    about it."""
    scores = ablate()
    baseline = scores["all three"]["task_success"]
    for label, result in scores.items():
        if label == "all three":
            continue
        assert result["task_success"] <= baseline, label


def test_the_drift_report_and_the_evals_both_carry_signal():
    scores = ablate()
    assert scores["without drift"]["task_success"] < 1.0
    assert scores["without evals"]["task_success"] < 1.0


def test_the_change_log_currently_changes_no_verdicts():
    """An honest negative result, asserted so it cannot be quietly forgotten.

    Removing the change log costs nothing on this scenario set: `bad_deploy` is
    already separable from the drift and eval evidence alone. The change log
    earns its place by NAMING THE COMMIT - which is what a human needs to act -
    not by changing the diagnosis.

    If a future scenario makes it load-bearing, this test fails and the claim
    in the README has to be updated. That is the point of asserting it.
    """
    scores = ablate()
    assert scores["without changes"]["task_success"] == \
        scores["all three"]["task_success"]


def test_a_missing_artifact_is_always_noticed():
    """An agent that quietly diagnoses from partial evidence is worse than one
    that refuses, because nobody knows to distrust it."""
    for row in degradation():
        assert row["noticed"], row


def test_the_critic_recovers_from_skipped_evidence():
    result = recovery()
    assert result["recovered"]
    assert "changes" in result["consulted"]
    assert any("never read" in c for c in result["critiques"])
