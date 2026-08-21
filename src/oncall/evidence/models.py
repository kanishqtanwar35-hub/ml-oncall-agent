"""The evidence an on-call engineer actually has at 3am.

**The real problem.** An ML system degrades silently. When someone finally
notices, the evidence is scattered across three places that do not talk to each
other:

  * a **drift report** — did the input distribution move?
  * an **eval run** — did the offline scores regress, and on which slices?
  * a **change log** — did anyone deploy anything?

Correlating them is a job, and it is a job people do badly at 3am because it
requires holding three JSON files in your head at once. That is what this
repository automates.

Everything here is a typed reader over artifacts that **already exist** in the
rest of the portfolio — `model-drift-monitor` writes the drift reports,
`llm-eval-pipeline` writes the eval results, `ai-code-review-bot` writes the
findings. Nothing is invented for the demo.

**Why the readers are strict.** Every field is `Optional` and every accessor
returns a default rather than raising, because the one thing worse than a
missing artifact at 3am is a tool that crashes on it. An agent that cannot read
the eval file should say "I could not read the eval file", not stack-trace into
a supervisor that has no idea what happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Severity ladder shared with `model-drift-monitor`, so the two agree on what
#: "high" means without either importing the other.
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _load(path) -> Dict[str, Any]:
    """Read JSON, returning {} rather than raising.

    A missing or malformed artifact is a *finding*, not a crash. The specialist
    that reads it reports "absent" and the supervisor routes around it.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureDrift:
    name: str
    severity: str = "none"
    psi: Optional[float] = None
    jensen_shannon: Optional[float] = None
    p_value: Optional[float] = None
    importance: Optional[float] = None
    significant: bool = False
    out_of_range_fraction: float = 0.0
    new_categories: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)


@dataclass(frozen=True)
class DriftReport:
    """A `model-drift-monitor` report_*.json, typed.

    The field names below are the ones that file actually contains — verified
    against a real artifact rather than copied from documentation, because the
    two drift apart and the failure mode is an agent silently reading `None`.
    """

    present: bool = False
    severity: str = "none"
    drifted_count: int = 0
    feature_count: int = 0
    rows_current: int = 0
    rows_reference: int = 0
    alpha: float = 0.05
    features: Dict[str, FeatureDrift] = field(default_factory=dict)
    prediction_drift: Optional[FeatureDrift] = None
    performance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def load(cls, path) -> "DriftReport":
        raw = _load(path)
        if not raw:
            return cls(present=False, source=str(path))
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], source: str = "") -> "DriftReport":
        features = {}
        for name, entry in (raw.get("features") or {}).items():
            features[name] = FeatureDrift(
                name=entry.get("name", name),
                severity=entry.get("severity", "none"),
                psi=entry.get("psi"),
                jensen_shannon=entry.get("jensen_shannon"),
                p_value=entry.get("p_value"),
                importance=entry.get("importance"),
                significant=bool(entry.get("significant", False)),
                out_of_range_fraction=entry.get("out_of_range_fraction") or 0.0,
                new_categories=list(entry.get("new_categories") or []),
                notes=list(entry.get("notes") or []),
            )

        prediction = raw.get("prediction_drift")
        return cls(
            present=True,
            severity=raw.get("severity", "none"),
            drifted_count=int(raw.get("drifted_count", 0)),
            feature_count=int(raw.get("feature_count", 0)),
            rows_current=int(raw.get("rows_current", 0)),
            rows_reference=int(raw.get("rows_reference", 0)),
            alpha=float(raw.get("alpha", 0.05)),
            features=features,
            prediction_drift=FeatureDrift(
                name="prediction",
                severity=prediction.get("severity", "none"),
                psi=prediction.get("psi"),
                jensen_shannon=prediction.get("jensen_shannon"),
            ) if isinstance(prediction, dict) else None,
            performance=dict(raw.get("performance") or {}),
            metadata=dict(raw.get("metadata") or {}),
            source=source,
        )

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)

    def worst(self, limit: int = 3) -> List[FeatureDrift]:
        return sorted(self.features.values(),
                      key=lambda f: (-f.rank, -(f.jensen_shannon or 0.0)))[:limit]

    def schema_problems(self) -> List[str]:
        """Missing columns and unseen categories.

        Kept separate from distribution drift on purpose: a column that has
        *disappeared* is a pipeline failure, not a population change, and it
        points at a completely different team.
        """
        problems = []
        for name in self.metadata.get("missing_columns") or []:
            problems.append(f"column '{name}' is absent from the scored batch")
        for feature in self.features.values():
            if feature.new_categories:
                problems.append(
                    f"'{feature.name}' contains category values the model was "
                    f"never trained on: {', '.join(feature.new_categories[:3])}"
                )
        return problems


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricChange:
    name: str
    current: float
    baseline: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def relative(self) -> float:
        return self.delta / self.baseline if self.baseline else 0.0


@dataclass(frozen=True)
class EvalRun:
    """One offline evaluation, with a baseline to compare against."""

    present: bool = False
    run_id: str = ""
    cases: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline: Dict[str, float] = field(default_factory=dict)
    #: Per-slice scores. The most useful field in the file and the one most
    #: often missing: an aggregate that moved 2% while one slice moved 30% is a
    #: different incident from one where everything moved a little.
    by_tag: Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline_by_tag: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Run-to-run standard deviation of the headline metric, where the harness
    #: reports it. Without it there is no way to tell a regression from noise,
    #: and "is this just noise" is the first question worth asking.
    noise_std: Optional[float] = None
    source: str = ""

    @classmethod
    def load(cls, path) -> "EvalRun":
        raw = _load(path)
        if not raw:
            return cls(present=False, source=str(path))
        return cls(
            present=True,
            run_id=str(raw.get("run_id", "")),
            cases=int(raw.get("cases", 0)),
            metrics=dict(raw.get("metrics") or {}),
            baseline=dict(raw.get("baseline") or {}),
            by_tag=dict(raw.get("by_tag") or {}),
            baseline_by_tag=dict(raw.get("baseline_by_tag") or {}),
            noise_std=raw.get("noise_std"),
            source=str(path),
        )

    def changes(self) -> List[MetricChange]:
        out = []
        for name, value in self.metrics.items():
            if name in self.baseline:
                out.append(MetricChange(name, float(value),
                                        float(self.baseline[name])))
        return sorted(out, key=lambda c: c.delta)

    #: Anything a dashboard would render as a downward tick. Deliberately small,
    #: and NOT the significance test.
    #:
    #: The first version used 0.02 here, which is a folklore threshold of
    #: exactly the kind `model-drift-monitor` argues against: it swallowed a
    #: 0.006 move before the noise check could look at it, so a flaky eval
    #: was diagnosed "healthy" instead of "noise". Detection and significance
    #: are two questions and this constant only answers the first - `within_noise`
    #: answers the second, from the harness's own measured spread.
    DETECTION_FLOOR = 0.005

    def regressions(self, threshold: Optional[float] = None) -> List[MetricChange]:
        """Every metric that moved down at all. Significance comes later."""
        limit = self.DETECTION_FLOOR if threshold is None else threshold
        return [c for c in self.changes() if c.delta < -limit]

    def slice_regressions(self, metric: str = "accuracy",
                          threshold: float = 0.05) -> List[MetricChange]:
        """Slices that moved more than the aggregate would suggest.

        This is where the actionable detail usually is. An aggregate holds
        steady while one tag collapses, and the aggregate is what the dashboard
        shows.
        """
        out = []
        for tag, scores in self.by_tag.items():
            base = (self.baseline_by_tag.get(tag) or {}).get(metric)
            current = scores.get(metric)
            if base is None or current is None:
                continue
            change = MetricChange(tag, float(current), float(base))
            if change.delta < -threshold:
                out.append(change)
        return sorted(out, key=lambda c: c.delta)

    def within_noise(self, change: MetricChange, sigmas: float = 2.0) -> bool:
        """Whether a move is smaller than the harness's own run-to-run spread.

        Returns False when the harness reports no noise estimate, because
        "unknown" must not be treated as "significant". Assuming otherwise is
        how an on-call rota gets woken for a rerun.
        """
        if self.noise_std is None:
            return False
        return abs(change.delta) <= sigmas * self.noise_std


# ---------------------------------------------------------------------------
# Changes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Change:
    sha: str
    message: str
    author: str = ""
    when: str = ""
    files: List[str] = field(default_factory=list)
    #: Findings raised against this change by `ai-code-review-bot`.
    review_findings: List[str] = field(default_factory=list)

    #: Paths whose modification can plausibly change model behaviour. A README
    #: edit in the same window is a coincidence; a change to the feature
    #: pipeline is a suspect, and treating them alike is how a blameless
    #: postmortem becomes a witch hunt.
    RISKY = ("feature", "preprocess", "transform", "pipeline", "model",
             "train", "encode", "schema", "config", "prompt", "requirements")

    @property
    def risky_files(self) -> List[str]:
        lowered = [(f, f.lower()) for f in self.files]
        return [f for f, low in lowered if any(k in low for k in self.RISKY)]

    @property
    def is_suspect(self) -> bool:
        return bool(self.risky_files) or bool(self.review_findings)


@dataclass(frozen=True)
class ChangeLog:
    present: bool = False
    changes: List[Change] = field(default_factory=list)
    window_hours: float = 24.0
    source: str = ""

    @classmethod
    def load(cls, path) -> "ChangeLog":
        raw = _load(path)
        if not raw:
            return cls(present=False, source=str(path))
        return cls(
            present=True,
            window_hours=float(raw.get("window_hours", 24.0)),
            changes=[
                Change(
                    sha=str(entry.get("sha", ""))[:12],
                    message=str(entry.get("message", "")),
                    author=str(entry.get("author", "")),
                    when=str(entry.get("when", "")),
                    files=list(entry.get("files") or []),
                    review_findings=list(entry.get("review_findings") or []),
                )
                for entry in (raw.get("changes") or [])
            ],
            source=str(path),
        )

    @property
    def suspects(self) -> List[Change]:
        return [c for c in self.changes if c.is_suspect]


# ---------------------------------------------------------------------------
# The bundle an agent is handed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    """Everything available about one incident."""

    drift: DriftReport = field(default_factory=DriftReport)
    evals: EvalRun = field(default_factory=EvalRun)
    changes: ChangeLog = field(default_factory=ChangeLog)
    question: str = "why did the last run regress?"

    @classmethod
    def load(cls, directory, question: str = "why did the last run regress?"
             ) -> "Evidence":
        directory = Path(directory)
        return cls(
            drift=DriftReport.load(directory / "drift.json"),
            evals=EvalRun.load(directory / "evals.json"),
            changes=ChangeLog.load(directory / "changes.json"),
            question=question,
        )

    def available(self) -> List[str]:
        return [name for name, present in
                (("drift", self.drift.present), ("evals", self.evals.present),
                 ("changes", self.changes.present)) if present]

    def missing(self) -> List[str]:
        return [name for name, present in
                (("drift", self.drift.present), ("evals", self.evals.present),
                 ("changes", self.changes.present)) if not present]
