"""How good is the agent? Measured, not asserted.

"The agent produced a plausible incident report" is not a result — plausible is
what language models do, including when the evidence supports nothing. So this
file scores the agent the way you would score a classifier, against incidents
whose true cause is known by construction.

Four things are measured, and the last two are the ones people skip:

  **Task success** — did it name the right root cause?
  **Action accuracy** — did it recommend the right next step? Being right about
    the cause and wrong about the action is still a bad night.
  **Tool selection** — did it read the artifacts it should have, and *only*
    those? An agent that calls every tool every time has no routing policy; it
    has a for-loop, and it will not scale past three tools.
  **Ablation** — how much does each artifact actually contribute? Remove one
    and re-measure. This is the only way to find out that a source you spent a
    week integrating changes no verdicts.

There is also a **degradation** check: artifacts go missing in production —
a job fails, a bucket is empty — and the interesting question is not whether
the agent still answers but whether it *notices* and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from oncall.agents.graph import investigate, recommended_action
from oncall.agents.specialists import SPECIALISTS
from oncall.evidence.models import ChangeLog, DriftReport, EvalRun, Evidence
from oncall.incidents import SCENARIOS, Scenario, evidence_for


@dataclass
class Result:
    scenario: str
    truth: str
    verdict: str
    confidence: float
    margin: float
    action: str
    expected_action: str
    consulted: List[str]
    steps: int
    revisions: int
    critiques: List[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.verdict == self.truth

    @property
    def action_correct(self) -> bool:
        return self.action == self.expected_action


@dataclass
class Report:
    results: List[Result] = field(default_factory=list)

    @property
    def task_success(self) -> float:
        return self._rate(lambda r: r.correct)

    @property
    def action_accuracy(self) -> float:
        return self._rate(lambda r: r.action_correct)

    @property
    def mean_steps(self) -> float:
        return (sum(r.steps for r in self.results) / len(self.results)
                if self.results else 0.0)

    def _rate(self, predicate) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if predicate(r)) / len(self.results)

    def calibration(self) -> Dict[str, float]:
        """Mean confidence when right versus when wrong.

        The number that says whether the confidence means anything. A system
        that is equally confident about its mistakes has a confidence field
        that is decoration, and decoration is worse than nothing because people
        act on it.
        """
        right = [r.confidence for r in self.results if r.correct]
        wrong = [r.confidence for r in self.results if not r.correct]
        return {
            "when_right": sum(right) / len(right) if right else float("nan"),
            "when_wrong": sum(wrong) / len(wrong) if wrong else float("nan"),
        }

    def table(self) -> str:
        lines = [f"{'scenario':<16} {'truth':<15} {'verdict':<15} "
                 f"{'conf':>5} {'margin':>7} {'steps':>6}  action"]
        lines.append("-" * 96)
        for r in self.results:
            mark = " " if r.correct else "X"
            action_mark = "" if r.action_correct else "  <- wrong action"
            lines.append(
                f"{mark}{r.scenario:<15} {r.truth:<15} {r.verdict:<15} "
                f"{r.confidence:>5.2f} {r.margin:>7.1f} {r.steps:>6}  "
                f"{r.action}{action_mark}")
        return "\n".join(lines)


def run(scenarios: Sequence[Scenario] = SCENARIOS) -> Report:
    report = Report()
    for scenario in scenarios:
        outcome = investigate(evidence_for(scenario.name))
        report.results.append(Result(
            scenario=scenario.name,
            truth=scenario.root_cause,
            verdict=outcome.diagnosis.root_cause,
            confidence=outcome.diagnosis.confidence,
            margin=outcome.diagnosis.margin,
            action=recommended_action(outcome.diagnosis),
            expected_action=scenario.expected_action,
            consulted=outcome.consulted,
            steps=outcome.steps,
            revisions=outcome.revisions,
            critiques=outcome.critiques,
        ))
    return report


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def tool_selection(report: Report) -> Dict[str, float]:
    """Did the agent read what it should have, and nothing else?

    With three artifacts and all three always present, the honest answer is
    that this is trivially 1.0 and the metric is not yet earning its keep. It
    is here because it is the metric that *starts* mattering the moment a
    fourth tool is added, and adding it retroactively is how you discover the
    agent has been calling everything for months.
    """
    expected, actual = 0, 0
    for result in report.results:
        expected += 3
        actual += len(set(result.consulted))
    return {
        "consulted_per_incident": actual / len(report.results)
        if report.results else 0.0,
        "coverage": actual / expected if expected else 0.0,
    }


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def _without(evidence: Evidence, artifact: str) -> Evidence:
    return Evidence(
        drift=DriftReport() if artifact == "drift" else evidence.drift,
        evals=EvalRun() if artifact == "evals" else evidence.evals,
        changes=ChangeLog() if artifact == "changes" else evidence.changes,
        question=evidence.question,
    )


def ablate() -> Dict[str, Dict[str, float]]:
    """Remove one artifact at a time and re-measure.

    The result worth predicting before you run it: removing the drift report
    barely moves task success, because the *eval* run carries most of the
    signal. That is a genuinely useful thing to know about a pipeline somebody
    is paying to maintain.
    """
    out: Dict[str, Dict[str, float]] = {}

    baseline = run()
    out["all three"] = {"task_success": baseline.task_success,
                        "action_accuracy": baseline.action_accuracy}

    for artifact in ("drift", "evals", "changes"):
        report = Report()
        for scenario in SCENARIOS:
            evidence = _without(evidence_for(scenario.name), artifact)
            outcome = investigate(evidence)
            report.results.append(Result(
                scenario=scenario.name, truth=scenario.root_cause,
                verdict=outcome.diagnosis.root_cause,
                confidence=outcome.diagnosis.confidence,
                margin=outcome.diagnosis.margin,
                action=recommended_action(outcome.diagnosis),
                expected_action=scenario.expected_action,
                consulted=outcome.consulted, steps=outcome.steps,
                revisions=outcome.revisions, critiques=outcome.critiques))
        out[f"without {artifact}"] = {
            "task_success": report.task_success,
            "action_accuracy": report.action_accuracy,
        }
    return out


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def degradation() -> List[Dict[str, object]]:
    """With an artifact missing, does the agent NOTICE?

    Artifacts go missing in production: a job fails, a bucket is empty, a path
    changes. The interesting question is not whether the agent still produces
    an answer - it will - but whether it flags that it answered on partial
    evidence. An agent that quietly diagnoses from two of three files is worse
    than one that refuses, because nobody knows to distrust it.
    """
    rows = []
    for artifact in ("drift", "evals", "changes"):
        evidence = _without(evidence_for("bad_deploy"), artifact)
        outcome = investigate(evidence)
        rows.append({
            "missing": artifact,
            "verdict": outcome.diagnosis.root_cause,
            "consulted": outcome.consulted,
            "noticed": any(artifact in c for c in outcome.critiques)
            or artifact not in outcome.consulted,
            "critiques": outcome.critiques,
        })
    return rows


def recovery() -> Dict[str, object]:
    """Does the critic actually send work back when evidence was skipped?

    Constructed by starting the graph with a specialist already marked as
    consulted, so the supervisor skips it and the critic must catch the gap.
    A critic that never fires is decoration; this proves it fires.
    """
    from oncall.agents.graph import build_graph

    evidence = evidence_for("bad_deploy")
    # Inject the fault in the ROUTING: the supervisor declines to read the
    # change log on its first pass. Pre-seeding `consulted` instead would hide
    # the gap from the very check being tested.
    compiled = build_graph(evidence, skip_first_pass=("changes",))

    final = compiled.invoke(
        {"question": "test", "consulted": [], "findings": [],
         "critiques": [], "trace": [], "revisions": 0},
        config={"recursion_limit": 40},
    )
    critiques = final.get("critiques", [])
    return {
        "revisions": final.get("revisions", 0),
        "critiques": critiques,
        "recovered": final.get("revisions", 0) > 0,
        "consulted": final.get("consulted", []),
    }
