"""The three specialists, and the finding they are allowed to produce.

**The design decision the whole repository turns on: the diagnosis is
deterministic, and the language model only narrates it.**

The obvious build is to paste three JSON files into a prompt and ask a model
what went wrong. It produces something fluent every time, including when the
evidence supports nothing — and fluent-when-there-is-nothing-there is the
failure mode, not an edge case. It is also untestable: you cannot assert on
prose, and you cannot tell a correct answer from a lucky one.

So the reasoning is ordinary Python:

  * Each specialist reads ONE artifact and emits `Finding`s.
  * Every finding carries a **citation** — the file, the field, and the value.
    A claim with no citation cannot be made, because `Finding` requires one.
  * Each finding names which root causes it **supports** and which it **rules
    out**, with a weight.
  * `diagnose()` sums the weights. That is the verdict, and it is a pure
    function of the evidence, so it can be unit-tested against incidents whose
    true cause is known.

The model's job is to turn that into a paragraph a tired person can read. If
the model is unavailable, the paragraph is templated and the *diagnosis is
identical* — which is the test that it was never doing the reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from oncall.evidence.models import Evidence
from oncall.incidents import ROOT_CAUSES


@dataclass(frozen=True)
class Citation:
    """Where a claim came from. Required, not optional."""

    artifact: str            # "drift" | "evals" | "changes"
    field_path: str          # e.g. "features.debt_ratio.psi"
    value: object

    def __str__(self) -> str:
        return f"{self.artifact}:{self.field_path}={self.value}"


@dataclass(frozen=True)
class Finding:
    """One observation, with the evidence for it."""

    claim: str
    citation: Citation
    #: Root causes this finding makes MORE likely, and by how much.
    supports: Dict[str, float] = field(default_factory=dict)
    #: Root causes this finding makes less likely. Negative evidence is where
    #: most of the diagnostic power is: "no deploy in the window" does not
    #: prove anything on its own and eliminates a great deal.
    rules_out: Dict[str, float] = field(default_factory=dict)
    specialist: str = ""

    def __post_init__(self) -> None:
        for name in list(self.supports) + list(self.rules_out):
            if name not in ROOT_CAUSES:
                raise ValueError(
                    f"'{name}' is not a known root cause; the verdict space is "
                    f"an allowlist so that a diagnosis can be scored"
                )

    def describe(self) -> str:
        return f"{self.claim}  [{self.citation}]"


class Specialist:
    """One artifact, one reader. Named so the supervisor can route by name."""

    name = "specialist"
    reads = ""

    def applicable(self, evidence: Evidence) -> bool:
        raise NotImplementedError

    def analyse(self, evidence: Evidence) -> List[Finding]:
        raise NotImplementedError


# ---------------------------------------------------------------------------

class DriftSpecialist(Specialist):
    """Reads the drift report from `model-drift-monitor`."""

    name = "drift"
    reads = "drift.json"

    def applicable(self, evidence: Evidence) -> bool:
        return evidence.drift.present

    def analyse(self, evidence: Evidence) -> List[Finding]:
        drift = evidence.drift
        findings: List[Finding] = []

        # Schema problems first. A missing column is a pipeline failure, not a
        # population change, and it points at a different team entirely.
        for problem in drift.schema_problems():
            findings.append(Finding(
                claim=problem,
                citation=Citation("drift", "metadata/new_categories", problem),
                supports={"pipeline_break": 3.0},
                rules_out={"concept_drift": 1.0, "healthy": 2.0},
                specialist=self.name))

        if drift.rank >= 2 and not drift.schema_problems():
            worst = drift.worst(2)
            named = ", ".join(f"{f.name} (PSI {f.psi:.2f})" for f in worst
                              if f.psi is not None)
            findings.append(Finding(
                claim=f"input drift is '{drift.severity}' on "
                      f"{drift.drifted_count} of {drift.feature_count} "
                      f"features: {named}",
                citation=Citation("drift", "severity", drift.severity),
                supports={"data_shift": 3.0},
                rules_out={"healthy": 2.0, "concept_drift": 2.0, "noise": 1.5},
                specialist=self.name))

            for feature in worst:
                if feature.importance and feature.importance >= 0.2 \
                        and feature.rank >= 2:
                    findings.append(Finding(
                        claim=f"'{feature.name}' carries "
                              f"{feature.importance:.0%} of the model's "
                              f"importance and drifted '{feature.severity}'",
                        citation=Citation(
                            "drift", f"features.{feature.name}.importance",
                            feature.importance),
                        supports={"data_shift": 1.5},
                        specialist=self.name))

                if feature.out_of_range_fraction > 0.01:
                    findings.append(Finding(
                        claim=f"{feature.out_of_range_fraction:.1%} of "
                              f"'{feature.name}' values are outside the "
                              f"training range - the model has never seen "
                              f"inputs like these",
                        citation=Citation(
                            "drift", f"features.{feature.name}."
                                     f"out_of_range_fraction",
                            feature.out_of_range_fraction),
                        supports={"data_shift": 1.5},
                        specialist=self.name))

        elif drift.rank <= 1:
            # The most valuable finding in the file, and the one an LLM
            # summarising the JSON would skip because nothing happened.
            findings.append(Finding(
                claim=f"input drift is '{drift.severity}': the scored "
                      f"population is statistically the same as training, so "
                      f"the inputs did not cause this",
                citation=Citation("drift", "severity", drift.severity),
                supports={"concept_drift": 2.0, "bad_deploy": 1.5,
                          "noise": 1.0, "healthy": 1.0},
                rules_out={"data_shift": 4.0},
                specialist=self.name))

        if drift.prediction_drift is not None:
            prediction = drift.prediction_drift
            if prediction.rank >= 2:
                findings.append(Finding(
                    claim=f"the model's OUTPUT distribution also moved "
                          f"('{prediction.severity}'), so its behaviour "
                          f"changed and not merely its inputs",
                    citation=Citation("drift", "prediction_drift.severity",
                                      prediction.severity),
                    supports={"data_shift": 1.0, "bad_deploy": 1.0},
                    rules_out={"noise": 1.0},
                    specialist=self.name))

        return findings


class EvalSpecialist(Specialist):
    """Reads the offline evaluation run."""

    name = "evals"
    reads = "evals.json"

    def applicable(self, evidence: Evidence) -> bool:
        return evidence.evals.present

    def analyse(self, evidence: Evidence) -> List[Finding]:
        evals = evidence.evals
        findings: List[Finding] = []

        regressions = evals.regressions()
        if not regressions:
            findings.append(Finding(
                claim="no metric regressed beyond tolerance",
                citation=Citation("evals", "metrics", evals.metrics),
                supports={"healthy": 3.0},
                rules_out={"data_shift": 2.0, "bad_deploy": 2.0,
                           "concept_drift": 2.0, "pipeline_break": 2.0},
                specialist=self.name))
            return findings

        # Noise before cause. "Is this even real" is the first question, and
        # skipping it is how an on-call rota gets woken for a rerun.
        inside_noise = [c for c in regressions if evals.within_noise(c)]
        if len(inside_noise) == len(regressions):
            worst = regressions[0]
            findings.append(Finding(
                claim=f"{worst.name} moved {worst.delta:+.4f}, which is inside "
                      f"the harness's own run-to-run spread "
                      f"(sigma {evals.noise_std}); this is not a regression",
                citation=Citation("evals", "noise_std", evals.noise_std),
                supports={"noise": 4.0},
                rules_out={"data_shift": 2.0, "bad_deploy": 2.0,
                           "concept_drift": 2.0, "pipeline_break": 2.0},
                specialist=self.name))
            return findings

        for change in regressions[:2]:
            findings.append(Finding(
                claim=f"{change.name} fell {change.delta:+.3f} "
                      f"({change.relative:+.1%}) against baseline",
                citation=Citation("evals", f"metrics.{change.name}",
                                  change.current),
                supports={"data_shift": 1.0, "bad_deploy": 1.0,
                          "concept_drift": 1.0, "pipeline_break": 1.0},
                rules_out={"healthy": 3.0, "noise": 1.0},
                specialist=self.name))

        # AUC collapsing while accuracy holds up is the signature of the
        # relationship inverting rather than the population moving. Under pure
        # covariate shift a correct model keeps its RANKING and loses its
        # threshold; here it is the ranking itself that is gone.
        auc = next((c for c in regressions if "auc" in c.name.lower()), None)
        if auc is not None and auc.current < 0.5:
            findings.append(Finding(
                claim=f"{auc.name} is {auc.current:.3f} - WORSE THAN RANDOM. "
                      f"The model's ranking has inverted, which is what a "
                      f"changed relationship looks like, not a changed "
                      f"population",
                citation=Citation("evals", f"metrics.{auc.name}", auc.current),
                supports={"concept_drift": 5.0},
                rules_out={"noise": 3.0, "healthy": 3.0},
                specialist=self.name))

        for slice_change in evals.slice_regressions()[:2]:
            findings.append(Finding(
                claim=f"the '{slice_change.name}' slice fell "
                      f"{slice_change.delta:+.3f} while the aggregate moved "
                      f"far less - a concentrated failure, not a broad one",
                citation=Citation("evals", f"by_tag.{slice_change.name}",
                                  slice_change.current),
                supports={"bad_deploy": 2.0, "pipeline_break": 1.0},
                rules_out={"noise": 1.0},
                specialist=self.name))

        return findings


class ChangeSpecialist(Specialist):
    """Reads the deploy window and the code-review findings."""

    name = "changes"
    reads = "changes.json"

    def applicable(self, evidence: Evidence) -> bool:
        return evidence.changes.present

    def analyse(self, evidence: Evidence) -> List[Finding]:
        log = evidence.changes
        findings: List[Finding] = []

        if not log.changes:
            findings.append(Finding(
                claim=f"nothing was deployed in the last "
                      f"{log.window_hours:.0f} hours",
                citation=Citation("changes", "changes", []),
                supports={"data_shift": 1.0, "concept_drift": 2.0},
                rules_out={"bad_deploy": 4.0},
                specialist=self.name))
            return findings

        suspects = log.suspects
        if not suspects:
            # Negative evidence, stated explicitly. "Three commits landed and
            # none of them touch anything that can change model behaviour" is a
            # real result and the thing that makes a blameless postmortem
            # possible.
            files = sorted({f for c in log.changes for f in c.files})
            findings.append(Finding(
                claim=f"{len(log.changes)} change(s) landed in the window but "
                      f"none touch model behaviour "
                      f"({', '.join(files[:3])})",
                citation=Citation("changes", "changes[].files", files[:3]),
                supports={"data_shift": 1.0, "concept_drift": 2.0},
                rules_out={"bad_deploy": 3.0},
                specialist=self.name))
            return findings

        for change in suspects[:2]:
            findings.append(Finding(
                claim=f"{change.sha} '{change.message}' touched "
                      f"{', '.join(change.risky_files[:2])}",
                citation=Citation("changes", f"changes.{change.sha}.files",
                                  change.risky_files[:2]),
                supports={"bad_deploy": 3.0},
                rules_out={"concept_drift": 1.0, "healthy": 1.0},
                specialist=self.name))

            for note in change.review_findings[:1]:
                findings.append(Finding(
                    claim=f"the review bot flagged {change.sha}: {note}",
                    citation=Citation("changes",
                                      f"changes.{change.sha}.review_findings",
                                      note),
                    supports={"bad_deploy": 3.0},
                    rules_out={"noise": 2.0},
                    specialist=self.name))

        return findings


SPECIALISTS: Sequence[Specialist] = (
    DriftSpecialist(), EvalSpecialist(), ChangeSpecialist(),
)

SPECIALISTS_BY_NAME = {s.name: s for s in SPECIALISTS}


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    root_cause: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    runner_up: Optional[str] = None
    margin: float = 0.0

    @property
    def ambiguous(self) -> bool:
        """Two causes close enough that picking either is a coin flip.

        The right response is to say so. An on-call report that commits to one
        of two equally-supported causes sends someone down the wrong path with
        false confidence, which is worse than sending them down two.
        """
        return self.margin < 1.5

    def supporting(self) -> List[Finding]:
        return [f for f in self.findings if self.root_cause in f.supports]


def diagnose(findings: Sequence[Finding]) -> Diagnosis:
    """Sum the evidence. A pure function, so it can be tested against truth."""
    scores = {cause: 0.0 for cause in ROOT_CAUSES}
    for finding in findings:
        for cause, weight in finding.supports.items():
            scores[cause] += weight
        for cause, weight in finding.rules_out.items():
            scores[cause] -= weight

    if not findings:
        return Diagnosis("healthy", 0.0, scores, [], None, 0.0)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ordered[0]
    runner_up, runner_score = ordered[1]

    total = sum(max(0.0, v) for v in scores.values()) or 1.0
    confidence = max(0.0, best_score) / total

    return Diagnosis(root_cause=best, confidence=confidence, scores=scores,
                     findings=list(findings), runner_up=runner_up,
                     margin=best_score - runner_score)


RECOMMENDED_ACTION = {
    "healthy": "no action",
    "data_shift": "retrain on recent data",
    "bad_deploy": "roll back",
    "concept_drift": "retrain with fresh labels",
    "pipeline_break": "fix the upstream pipeline",
    "noise": "rerun the evaluation",
}
