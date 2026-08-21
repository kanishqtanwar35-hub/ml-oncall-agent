"""Incidents with a known root cause, so the agent can be scored rather than admired.

**Why this file exists.** "The agent produced a plausible incident report" is
not a result. Plausible is what language models are *for*; it is the failure
mode, not the evidence of success.

To measure an agent you need incidents whose true cause you already know, and
real incidents do not come with a label — that is precisely why diagnosing them
is hard. So the scenarios below construct the three artifacts *from* a declared
root cause, and the agent's verdict is scored against it.

This is the same argument as `model-drift-monitor`'s synthetic data, applied to
diagnosis instead of detection. The numbers characterise the *method*, not any
real production system.

**The six scenarios, and why each is here**

| scenario | drift | evals | changes | the trap |
|---|---|---|---|---|
| `healthy` | none | flat | none | does it stay quiet? |
| `data_shift` | high | regressed | none | the easy case |
| `bad_deploy` | none | regressed | risky | drift is silent; blame the code |
| `concept_drift` | **none** | **regressed** | none | nothing to point at |
| `pipeline_break` | schema | regressed | none | not a population change |
| `flaky_eval` | none | "regressed" | none | inside the noise band |

`concept_drift` is the one that matters. The inputs are unchanged, so the drift
monitor is correctly silent, and an agent that reasons "no drift and no deploy,
therefore nothing is wrong" gets it exactly backwards. That is the same blind
spot `model-drift-monitor` documents about itself, one layer up.

`flaky_eval` is the counterweight: an agent that always finds a cause is as
useless as one that never does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

#: The verdicts an agent is allowed to reach. An allowlist, like everything
#: else in this portfolio: a free-text root cause cannot be scored, and a
#: diagnosis outside this list is one no runbook has an entry for.
ROOT_CAUSES = (
    "healthy",          # nothing is wrong
    "data_shift",       # the population being scored changed
    "bad_deploy",       # a code change altered model behaviour
    "concept_drift",    # inputs unchanged, the relationship changed
    "pipeline_break",   # upstream schema or ingestion failure
    "noise",            # the move is inside run-to-run variance
)


@dataclass(frozen=True)
class Scenario:
    name: str
    root_cause: str
    description: str
    #: What a competent human would do next. Scored separately from the
    #: diagnosis, because being right about the cause and wrong about the
    #: action is still a bad night.
    expected_action: str


SCENARIOS: List[Scenario] = [
    Scenario("healthy", "healthy",
             "Everything is within tolerance.",
             "no action"),
    Scenario("data_shift", "data_shift",
             "The scored population moved: debt_ratio and income shifted, "
             "accuracy fell with it. No deploy in the window.",
             "retrain on recent data"),
    Scenario("bad_deploy", "bad_deploy",
             "Inputs unchanged, evals regressed, and a change to the feature "
             "pipeline landed in the window with a review finding against it.",
             "roll back"),
    Scenario("concept_drift", "concept_drift",
             "Inputs are statistically identical and nothing was deployed, yet "
             "the model's ranking collapsed. The relationship changed.",
             "retrain with fresh labels"),
    Scenario("pipeline_break", "pipeline_break",
             "A column is missing and a categorical contains values the model "
             "was never trained on. Upstream, not the model.",
             "fix the upstream pipeline"),
    Scenario("flaky_eval", "noise",
             "The headline metric moved less than the harness's own run-to-run "
             "standard deviation. Nothing happened.",
             "rerun the evaluation"),
]

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


# ---------------------------------------------------------------------------
# Artifact construction
# ---------------------------------------------------------------------------

def _feature(name, severity="none", psi=0.004, js=0.02, p=0.4,
             importance=0.1, significant=False, out_of_range=0.0,
             new_categories=(), notes=()):
    return {
        "name": name, "kind": "numeric", "severity": severity,
        "psi": psi, "jensen_shannon": js, "p_value": p,
        "wasserstein": 0.0, "ks_statistic": 0.0,
        "significant": significant, "importance": importance,
        "out_of_range_fraction": out_of_range,
        "new_categories": list(new_categories),
        "missing_fraction_reference": 0.0, "missing_fraction_current": 0.0,
        "notes": list(notes),
    }


def _drift(severity="none", features=None, metadata=None,
           prediction_severity="none", performance=None) -> Dict[str, Any]:
    features = features or {}
    return {
        "severity": severity,
        "rows_reference": 8000,
        "rows_current": 2000,
        "alpha": 0.05,
        "drifted_count": sum(1 for f in features.values()
                             if f["severity"] != "none"),
        "feature_count": max(len(features), 8),
        "metadata": metadata or {},
        "prediction_drift": {"name": "prediction", "kind": "numeric",
                             "severity": prediction_severity,
                             "psi": 0.01, "jensen_shannon": 0.03},
        "performance": performance or {"accuracy": 0.755, "roc_auc": 0.729,
                                       "f1": 0.376, "brier": 0.172,
                                       "positive_rate": 0.126, "n": 2000},
        "features": features,
    }


def _evals(metrics, baseline, by_tag=None, baseline_by_tag=None,
           noise_std=0.006) -> Dict[str, Any]:
    return {
        "run_id": "nightly",
        "cases": 220,
        "metrics": metrics,
        "baseline": baseline,
        "by_tag": by_tag or {},
        "baseline_by_tag": baseline_by_tag or {},
        "noise_std": noise_std,
    }


def _changes(entries, window_hours=24.0) -> Dict[str, Any]:
    return {"window_hours": window_hours, "changes": entries}


_NEUTRAL_FEATURES = {
    "income": _feature("income", importance=0.04),
    "credit_score": _feature("credit_score", importance=0.53),
    "debt_ratio": _feature("debt_ratio", importance=0.25),
    "age": _feature("age", importance=0.07),
}

_QUIET_CHANGES = [
    {"sha": "a1b2c3d4e5f6", "message": "docs: fix a typo in the README",
     "author": "kanishq", "when": "2026-08-20T18:04:00Z",
     "files": ["README.md"], "review_findings": []},
]


def build(scenario: str) -> Dict[str, Dict[str, Any]]:
    """The three artifacts for one scenario."""
    if scenario not in SCENARIOS_BY_NAME:
        raise ValueError(f"unknown scenario {scenario!r}; "
                         f"expected one of {sorted(SCENARIOS_BY_NAME)}")

    if scenario == "healthy":
        return {
            "drift": _drift(features=dict(_NEUTRAL_FEATURES)),
            "evals": _evals({"accuracy": 0.757, "roc_auc": 0.731},
                            {"accuracy": 0.755, "roc_auc": 0.729}),
            "changes": _changes(_QUIET_CHANGES),
        }

    if scenario == "data_shift":
        features = dict(_NEUTRAL_FEATURES)
        features["debt_ratio"] = _feature(
            "debt_ratio", severity="high", psi=4.13, js=0.51, p=2.2e-202,
            importance=0.25, significant=True, out_of_range=0.022,
            notes=["2.2% of values are outside the training range "
                   "[0.0048, 1.213]"])
        features["income"] = _feature("income", severity="medium", psi=0.14,
                                      js=0.16, p=1.2e-24, importance=0.04,
                                      significant=True)
        return {
            "drift": _drift(severity="high", features=features,
                            prediction_severity="high",
                            performance={"accuracy": 0.661, "roc_auc": 0.724,
                                         "f1": 0.31, "brier": 0.21,
                                         "positive_rate": 0.21, "n": 2000}),
            "evals": _evals({"accuracy": 0.661, "roc_auc": 0.724},
                            {"accuracy": 0.755, "roc_auc": 0.729}),
            "changes": _changes(_QUIET_CHANGES),
        }

    if scenario == "bad_deploy":
        return {
            "drift": _drift(features=dict(_NEUTRAL_FEATURES)),
            "evals": _evals(
                {"accuracy": 0.681, "roc_auc": 0.652},
                {"accuracy": 0.755, "roc_auc": 0.729},
                by_tag={"short": {"accuracy": 0.44}, "long": {"accuracy": 0.75}},
                baseline_by_tag={"short": {"accuracy": 0.74},
                                 "long": {"accuracy": 0.76}}),
            "changes": _changes([
                *_QUIET_CHANGES,
                {"sha": "9f8e7d6c5b4a",
                 "message": "perf: vectorise the feature transform",
                 "author": "kanishq", "when": "2026-08-20T21:12:00Z",
                 "files": ["src/pipeline/features.py",
                           "src/pipeline/preprocess.py"],
                 "review_findings": [
                     "features.py:88 the fillna default changed from the "
                     "training median to 0, which shifts every imputed row"]},
            ]),
        }

    if scenario == "concept_drift":
        # The hard one. Inputs statistically identical, no deploy, and the
        # model's ranking has inverted. Nothing to point at.
        return {
            "drift": _drift(features=dict(_NEUTRAL_FEATURES)),
            "evals": _evals({"accuracy": 0.615, "roc_auc": 0.326},
                            {"accuracy": 0.755, "roc_auc": 0.729}),
            "changes": _changes(_QUIET_CHANGES),
        }

    if scenario == "pipeline_break":
        features = dict(_NEUTRAL_FEATURES)
        features["channel"] = _feature(
            "channel", severity="high", psi=0.73, js=0.34, p=0.0,
            importance=0.011, significant=True,
            new_categories=["partner_api"],
            notes=["1 category value(s) the model was never trained on, "
                   "22.10% of the batch"])
        return {
            "drift": _drift(severity="high", features=features,
                            metadata={"missing_columns": ["n_recent_queries"]},
                            performance={"accuracy": 0.71, "roc_auc": 0.68,
                                         "f1": 0.29, "brier": 0.22,
                                         "positive_rate": 0.18, "n": 2000}),
            "evals": _evals({"accuracy": 0.710, "roc_auc": 0.680},
                            {"accuracy": 0.755, "roc_auc": 0.729}),
            "changes": _changes(_QUIET_CHANGES),
        }

    # flaky_eval
    return {
        "drift": _drift(features=dict(_NEUTRAL_FEATURES)),
        "evals": _evals({"accuracy": 0.7488, "roc_auc": 0.7261},
                        {"accuracy": 0.755, "roc_auc": 0.729},
                        noise_std=0.006),
        "changes": _changes(_QUIET_CHANGES),
    }


def write(scenario: str, directory) -> Path:
    """Write the three artifacts to a directory, as the real tools would."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in build(scenario).items():
        (directory / f"{name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return directory


def evidence_for(scenario: str):
    """An `Evidence` bundle for a scenario, without touching the filesystem."""
    from oncall.evidence.models import ChangeLog, DriftReport, EvalRun, Evidence

    payload = build(scenario)
    drift = DriftReport.from_dict(payload["drift"], source=f"{scenario}:drift")

    evals_raw = payload["evals"]
    evals = EvalRun(
        present=True, run_id=evals_raw["run_id"], cases=evals_raw["cases"],
        metrics=evals_raw["metrics"], baseline=evals_raw["baseline"],
        by_tag=evals_raw["by_tag"], baseline_by_tag=evals_raw["baseline_by_tag"],
        noise_std=evals_raw["noise_std"], source=f"{scenario}:evals")

    from oncall.evidence.models import Change
    changes_raw = payload["changes"]
    changes = ChangeLog(
        present=True, window_hours=changes_raw["window_hours"],
        changes=[Change(sha=c["sha"], message=c["message"], author=c["author"],
                        when=c["when"], files=c["files"],
                        review_findings=c["review_findings"])
                 for c in changes_raw["changes"]],
        source=f"{scenario}:changes")

    return Evidence(drift=drift, evals=evals, changes=changes)
