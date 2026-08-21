"""The evidence readers, the specialists, and the deterministic verdict."""

import json

import pytest

from oncall.agents.graph import (
    MAX_REVISIONS,
    build_graph,
    investigate,
    investigate_without_graph,
    recommended_action,
)
from oncall.agents.specialists import (
    SPECIALISTS,
    ChangeSpecialist,
    Citation,
    DriftSpecialist,
    EvalSpecialist,
    Finding,
    diagnose,
)
from oncall.evidence.models import (
    Change,
    ChangeLog,
    DriftReport,
    EvalRun,
    Evidence,
    MetricChange,
)
from oncall.incidents import ROOT_CAUSES, SCENARIOS, build, evidence_for, write


# ---------------------------------------------------------------------------
# Evidence readers
# ---------------------------------------------------------------------------

def test_a_missing_artifact_is_a_finding_not_a_crash(tmp_path):
    """The one thing worse than a missing artifact at 3am is a tool that
    crashes on it."""
    report = DriftReport.load(tmp_path / "nope.json")
    assert report.present is False
    assert report.severity == "none"


def test_malformed_json_is_also_not_a_crash(tmp_path):
    path = tmp_path / "drift.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert DriftReport.load(path).present is False


def test_the_reader_matches_the_real_drift_schema(tmp_path):
    """Field names verified against an artifact `model-drift-monitor` actually
    wrote, not copied from documentation. The two drift apart, and the failure
    mode is an agent silently reading None."""
    payload = build("data_shift")["drift"]
    (tmp_path / "drift.json").write_text(json.dumps(payload), encoding="utf-8")

    report = DriftReport.load(tmp_path / "drift.json")
    assert report.present
    assert report.severity == "high"
    assert report.drifted_count == 2
    assert "debt_ratio" in report.features
    assert report.features["debt_ratio"].psi == pytest.approx(4.13)
    assert report.prediction_drift is not None
    assert report.performance["accuracy"] == pytest.approx(0.661)


def test_schema_problems_are_kept_apart_from_distribution_drift():
    """A column that has DISAPPEARED is a pipeline failure, not a population
    change, and it points at a different team."""
    report = DriftReport.from_dict(build("pipeline_break")["drift"])
    problems = report.schema_problems()
    assert any("absent" in p for p in problems)
    assert any("never trained on" in p for p in problems)


def test_a_healthy_report_has_no_schema_problems():
    assert DriftReport.from_dict(build("healthy")["drift"]).schema_problems() == []


def test_evidence_reports_what_it_is_missing():
    evidence = Evidence(drift=DriftReport(present=True), evals=EvalRun(),
                        changes=ChangeLog())
    assert evidence.available() == ["drift"]
    assert evidence.missing() == ["evals", "changes"]


def test_write_and_load_round_trip(tmp_path):
    write("bad_deploy", tmp_path)
    evidence = Evidence.load(tmp_path)
    assert evidence.available() == ["drift", "evals", "changes"]
    assert evidence.changes.suspects


# ---------------------------------------------------------------------------
# Metric and change semantics
# ---------------------------------------------------------------------------

def test_a_move_smaller_than_the_noise_is_not_a_regression():
    """Detection and significance are two questions. The first version used a
    hardcoded 0.02 tolerance, which swallowed a 0.006 move before the noise
    check could look at it - so a flaky eval was diagnosed 'healthy' instead of
    'noise'. That is the same folklore-threshold mistake model-drift-monitor
    argues against."""
    evals = EvalRun(present=True, metrics={"accuracy": 0.7488},
                    baseline={"accuracy": 0.755}, noise_std=0.006)
    regressions = evals.regressions()
    assert regressions, "the move must be DETECTED"
    assert evals.within_noise(regressions[0]), "and then judged insignificant"


def test_an_unknown_noise_level_is_not_treated_as_significant():
    """'Unknown' must not silently become 'significant'. Assuming otherwise is
    how an on-call rota gets woken for a rerun."""
    evals = EvalRun(present=True, metrics={"accuracy": 0.70},
                    baseline={"accuracy": 0.755}, noise_std=None)
    assert not evals.within_noise(evals.regressions()[0])


def test_slice_regressions_find_concentrated_failures():
    evals = EvalRun(present=True,
                    metrics={"accuracy": 0.68}, baseline={"accuracy": 0.755},
                    by_tag={"short": {"accuracy": 0.44}},
                    baseline_by_tag={"short": {"accuracy": 0.74}})
    assert evals.slice_regressions()[0].name == "short"


def test_a_readme_edit_is_not_a_suspect():
    """A README edit in the same window is a coincidence. Treating it like a
    pipeline change is how a blameless postmortem becomes a witch hunt."""
    assert not Change("abc", "docs: typo", files=["README.md"]).is_suspect


def test_a_feature_pipeline_change_is_a_suspect():
    assert Change("abc", "perf", files=["src/pipeline/features.py"]).is_suspect


def test_a_review_finding_makes_any_change_a_suspect():
    change = Change("abc", "chore", files=["notes.txt"],
                    review_findings=["this looks wrong"])
    assert change.is_suspect


# ---------------------------------------------------------------------------
# Findings must be citable
# ---------------------------------------------------------------------------

def test_a_finding_cannot_name_an_unknown_root_cause():
    """The verdict space is an allowlist so a diagnosis can be scored. A
    free-text root cause is one no runbook has an entry for."""
    with pytest.raises(ValueError, match="not a known root cause"):
        Finding("x", Citation("drift", "severity", "high"),
                supports={"gremlins": 1.0})


def test_every_finding_carries_a_citation():
    """Not decoration. At 3am the first thing a sceptical engineer does is
    check one claim, and a report that cannot be checked gets ignored on its
    second wrong call."""
    for scenario in SCENARIOS:
        evidence = evidence_for(scenario.name)
        for specialist in SPECIALISTS:
            for finding in specialist.analyse(evidence):
                assert finding.citation.artifact in {"drift", "evals", "changes"}
                assert finding.citation.field_path
                assert finding.specialist == specialist.name


def test_no_drift_is_reported_as_positive_evidence():
    """The most valuable finding in the file, and the one an LLM summarising
    the JSON would skip because nothing happened."""
    findings = DriftSpecialist().analyse(evidence_for("concept_drift"))
    assert any("did not cause this" in f.claim for f in findings)
    assert any("concept_drift" in f.supports for f in findings)


def test_no_deploy_is_reported_as_positive_evidence():
    findings = ChangeSpecialist().analyse(evidence_for("concept_drift"))
    assert any("none touch model behaviour" in f.claim for f in findings)
    assert any(f.rules_out.get("bad_deploy", 0) > 0 for f in findings)


def test_auc_below_random_is_called_out_specifically():
    """Under pure covariate shift a correct model keeps its RANKING. An AUC
    below 0.5 means the ranking itself inverted, which is a changed
    relationship, not a changed population."""
    findings = EvalSpecialist().analyse(evidence_for("concept_drift"))
    assert any("WORSE THAN RANDOM" in f.claim for f in findings)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def test_diagnose_is_a_pure_function():
    findings = []
    for specialist in SPECIALISTS:
        findings += specialist.analyse(evidence_for("data_shift"))
    assert diagnose(findings).root_cause == diagnose(findings).root_cause


def test_no_findings_means_no_confident_verdict():
    result = diagnose([])
    assert result.confidence == 0.0


def test_every_scenario_is_diagnosed_correctly():
    """Ground truth is known by construction, so this is scored rather than
    admired. 'The agent produced a plausible report' is not a result."""
    wrong = []
    for scenario in SCENARIOS:
        outcome = investigate(evidence_for(scenario.name))
        if outcome.diagnosis.root_cause != scenario.root_cause:
            wrong.append(f"{scenario.name}: expected {scenario.root_cause}, "
                         f"got {outcome.diagnosis.root_cause}")
    assert not wrong, "; ".join(wrong)


def test_every_scenario_gets_the_right_action():
    """Being right about the cause and wrong about the action is still a bad
    night."""
    for scenario in SCENARIOS:
        outcome = investigate(evidence_for(scenario.name))
        assert recommended_action(outcome.diagnosis) == scenario.expected_action


def test_the_recommended_action_covers_every_root_cause():
    from oncall.agents.specialists import RECOMMENDED_ACTION
    assert set(RECOMMENDED_ACTION) == set(ROOT_CAUSES)


def test_concept_drift_is_diagnosed_from_the_ABSENCE_of_evidence():
    """The case the whole repository is built around. No drift, no deploy, and
    the model's ranking has collapsed - so the answer comes from combining two
    negatives with one positive, which is exactly what a summary of any single
    artifact would miss."""
    outcome = investigate(evidence_for("concept_drift"))
    assert outcome.diagnosis.root_cause == "concept_drift"
    assert outcome.diagnosis.confidence > 0.5


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def test_the_graph_and_the_plain_loop_agree():
    """Proves the graph adds ORCHESTRATION, not reasoning. If these ever
    diverge, logic has leaked into the wiring where it cannot be unit-tested."""
    for scenario in SCENARIOS:
        evidence = evidence_for(scenario.name)
        assert investigate(evidence).diagnosis.root_cause == \
            investigate_without_graph(evidence).diagnosis.root_cause


def test_each_specialist_is_consulted_exactly_once():
    outcome = investigate(evidence_for("bad_deploy"))
    assert sorted(outcome.consulted) == ["changes", "drift", "evals"]
    assert len(outcome.consulted) == len(set(outcome.consulted))


def test_a_specialist_with_no_artifact_is_not_consulted():
    evidence = Evidence(drift=DriftReport.from_dict(build("healthy")["drift"]),
                        evals=EvalRun(), changes=ChangeLog())
    outcome = investigate(evidence)
    assert outcome.consulted == ["drift"]


def test_the_trace_records_the_routing():
    outcome = investigate(evidence_for("healthy"))
    assert any("supervisor ->" in step for step in outcome.trace)
    assert any("diagnose ->" in step for step in outcome.trace)
    assert any("critic:" in step for step in outcome.trace)


def test_the_run_terminates_on_every_scenario():
    """An agent that can loop is an agent that can loop forever, and a runaway
    on-call agent generates pages instead of answering them."""
    for scenario in SCENARIOS:
        outcome = investigate(evidence_for(scenario.name))
        assert outcome.steps < 25


# ---------------------------------------------------------------------------
# The critic
# ---------------------------------------------------------------------------

def test_the_critic_catches_evidence_that_was_never_read():
    """A critic that never fires is decoration. The fault is injected in the
    ROUTING - pre-seeding `consulted` would hide the gap from the very check
    being tested."""
    compiled = build_graph(evidence_for("bad_deploy"),
                           skip_first_pass=("changes",))
    final = compiled.invoke({"question": "t", "consulted": [], "findings": [],
                             "critiques": [], "trace": [], "revisions": 0},
                            config={"recursion_limit": 40})
    assert final["revisions"] >= 1
    assert any("never read" in c for c in final["critiques"])
    assert "changes" in final["consulted"], "and it went back and read it"


def test_the_critic_accepts_a_complete_investigation():
    outcome = investigate(evidence_for("data_shift"))
    assert outcome.revisions == 0


def test_the_revision_budget_is_bounded():
    """MAX_REVISIONS caps the loop; recursion_limit catches anything that
    escapes it."""
    compiled = build_graph(evidence_for("bad_deploy"), max_revisions=0,
                           skip_first_pass=("changes",))
    final = compiled.invoke({"question": "t", "consulted": [], "findings": [],
                             "critiques": [], "trace": [], "revisions": 0},
                            config={"recursion_limit": 40})
    assert final["revisions"] == 0
    assert any("never read" in c for c in final["critiques"]), \
        "it still complains, it just does not loop"


def test_max_revisions_is_small():
    assert 1 <= MAX_REVISIONS <= 3


def test_an_ambiguous_verdict_is_flagged_rather_than_committed_to():
    """Two causes within a hair of each other is a coin flip. Committing sends
    someone down the wrong path with false confidence."""
    findings = [
        Finding("a", Citation("drift", "severity", "high"),
                supports={"data_shift": 2.0}),
        Finding("b", Citation("evals", "metrics.accuracy", 0.5),
                supports={"bad_deploy": 2.0}),
    ]
    assert diagnose(findings).ambiguous


def test_a_clear_verdict_is_not_flagged_ambiguous():
    assert not investigate(evidence_for("concept_drift")).diagnosis.ambiguous
