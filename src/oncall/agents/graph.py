"""Supervisor, specialists, critic — as a LangGraph, and runnable with no LLM.

    question
        │
        ▼
   SUPERVISOR ──► drift ──┐
        ▲   │             │
        │   ├──► evals ───┤   one specialist per artifact, each consulted once
        │   │             │
        │   └──► changes ─┤
        │                 ▼
        │             DIAGNOSE          sum the weighted findings
        │                 │
        │                 ▼
        └──────────────  CRITIC         "is this conclusion supported?"
           (bounded)      │
                          ▼
                        REPORT

**Why a graph rather than three function calls.** Two properties fall out of it
that a straight-line pipeline does not have:

  * **The critic can send work back.** If the diagnosis rests on one artifact
    while another was never read, the critic returns control to the supervisor
    rather than publishing a conclusion drawn from half the evidence.
  * **The cycle is bounded.** `MAX_REVISIONS` caps it, and `recursion_limit`
    catches anything that escapes. An agent that can loop is an agent that can
    loop forever, and a runaway on-call agent generates pages instead of
    answering them.

**It runs with no LLM and no API key.** The supervisor routes by a plain Python
rule and the specialists are deterministic analysers, so the entire graph — the
routing, the delegation, the critic's loop-back — is unit-testable offline. The
model, when configured, writes only the closing paragraph.

That ordering matters. If the model did the routing, "did the agent consult the
right specialist" would be a question you could only answer by sampling.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    # LangGraph and pydantic both prefer typing_extensions - on Python 3.11 a
    # `typing.TypedDict` used as a pydantic field raises outright. So prefer it
    # when it is installed.
    from typing_extensions import Annotated, TypedDict
except ImportError:
    # ...but do not REQUIRE it. `investigate_without_graph` and the whole
    # deterministic verdict need no third-party package at all, and importing
    # this module must not be what breaks that.
    #
    # CI caught this, which is the entire reason the `core-only` job exists:
    # it installs nothing but pytest, so a module-scope dependency here fails
    # it while the full matrix stays green. The claim in the README was simply
    # untrue until this ran.
    from typing import Annotated, TypedDict        # type: ignore[assignment]

from oncall.agents.specialists import (
    RECOMMENDED_ACTION,
    SPECIALISTS,
    SPECIALISTS_BY_NAME,
    Diagnosis,
    Finding,
    diagnose,
)
from oncall.evidence.models import Evidence

#: How many times the critic may send the supervisor back. Two is enough to
#: consult a skipped specialist and re-diagnose; more than that and the graph
#: is thrashing, not thinking.
MAX_REVISIONS = 2

#: LangGraph's own backstop, in case a routing bug escapes MAX_REVISIONS. It
#: raises rather than hanging, which is the behaviour you want from something
#: that runs unattended at 3am.
RECURSION_LIMIT = 40


class OncallState(TypedDict, total=False):
    """The graph's shared state.

    `consulted` and `findings` both carry `operator.add` reducers because more
    than one node writes to them. A key written by several nodes without a
    reducer is silently OVERWRITTEN by whichever ran last — the same bug the
    `agentic-rag` repo documents about its latency map, and it is silent both
    times.
    """

    question: str
    consulted: Annotated[List[str], operator.add]
    findings: Annotated[List[Finding], operator.add]
    critiques: Annotated[List[str], operator.add]
    trace: Annotated[List[str], operator.add]
    diagnosis: Optional[Diagnosis]
    revisions: int
    next: str


@dataclass
class Run:
    """What one invocation produced, including how it got there."""

    diagnosis: Diagnosis
    consulted: List[str] = field(default_factory=list)
    critiques: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    revisions: int = 0

    @property
    def steps(self) -> int:
        return len(self.trace)


def _applicable(evidence: Evidence) -> List[str]:
    return [s.name for s in SPECIALISTS if s.applicable(evidence)]


def build_graph(evidence: Evidence, max_revisions: int = MAX_REVISIONS,
                skip_first_pass: Sequence[str] = ()):
    """Compile the graph for one incident.

    `evidence` is closed over rather than carried in state. It is immutable
    input, and putting it in state would make every node's signature imply it
    could be modified.

    `skip_first_pass` is a FAULT INJECTION hook, and it exists because a critic
    that never fires is decoration. It makes the supervisor decline to consult
    the named specialists on its first pass, so the critic has a real gap to
    find and a real correction to make. On the second pass - after a critique -
    the skip is lifted and the specialist is read.

    An earlier version of the recovery test tried to fake this by pre-seeding
    `consulted`, which does not work: the critic compares `available` against
    `consulted`, so marking something consulted hides the gap from the very
    check being tested. The injection has to happen in the ROUTING, not in the
    bookkeeping.
    """
    from langgraph.graph import END, START, StateGraph

    available = _applicable(evidence)

    def supervisor(state: OncallState) -> Dict[str, Any]:
        """Pick the next specialist, or move to diagnosis when done."""
        consulted = state.get("consulted", [])
        remaining = [name for name in available if name not in consulted]

        # The injected fault is lifted once the critic has objected, which is
        # what makes the loop a RECOVERY rather than a retry of the same plan.
        if state.get("revisions", 0) == 0:
            remaining = [n for n in remaining if n not in skip_first_pass]

        if remaining:
            chosen = remaining[0]
            return {"next": chosen,
                    "trace": [f"supervisor -> {chosen}"]}

        return {"next": "diagnose", "trace": ["supervisor -> diagnose"]}

    def make_specialist(name: str):
        specialist = SPECIALISTS_BY_NAME[name]

        def node(state: OncallState) -> Dict[str, Any]:
            findings = specialist.analyse(evidence)
            return {
                "consulted": [name],
                "findings": findings,
                "trace": [f"{name}: {len(findings)} finding(s)"],
            }
        return node

    def diagnose_node(state: OncallState) -> Dict[str, Any]:
        result = diagnose(state.get("findings", []))
        return {"diagnosis": result,
                "trace": [f"diagnose -> {result.root_cause} "
                          f"(margin {result.margin:.1f})"]}

    def critic(state: OncallState) -> Dict[str, Any]:
        """Check the conclusion against the evidence, not against plausibility.

        Three questions, each of which has sent a real diagnosis back:

        1. Was every available artifact actually read? A conclusion drawn from
           two of three files is a guess wearing a citation.
        2. Does anything support the verdict? A root cause reached only by
           elimination is worth saying out loud rather than asserting.
        3. Is the margin large enough to commit to? Two causes within a hair of
           each other is a coin flip, and an on-call report that picks one
           sends someone down the wrong path with false confidence.
        """
        result: Optional[Diagnosis] = state.get("diagnosis")
        consulted = state.get("consulted", [])
        revisions = state.get("revisions", 0)
        critiques: List[str] = []

        unread = [name for name in available if name not in consulted]
        if unread:
            critiques.append(
                f"{', '.join(unread)} was available and never read; a "
                f"conclusion from part of the evidence is a guess")

        if result is not None and not result.supporting():
            critiques.append(
                f"nothing positively supports '{result.root_cause}' - it was "
                f"reached only by eliminating the others, which is worth "
                f"stating rather than asserting")

        if result is not None and result.ambiguous:
            critiques.append(
                f"'{result.root_cause}' beats '{result.runner_up}' by only "
                f"{result.margin:.1f}; that is close enough to be a coin flip "
                f"and both should be investigated")

        # Only send back for the fixable complaint. Ambiguity is a property of
        # the evidence: re-running the same specialists cannot resolve it, and
        # looping on it would burn the revision budget achieving nothing.
        should_retry = bool(unread) and revisions < max_revisions

        return {
            "critiques": critiques,
            "revisions": revisions + (1 if should_retry else 0),
            "next": "supervisor" if should_retry else "done",
            "trace": [f"critic: {len(critiques)} concern(s), "
                      f"{'sending back' if should_retry else 'accepting'}"],
        }

    graph = StateGraph(OncallState)
    graph.add_node("supervisor", supervisor)
    for name in available:
        graph.add_node(name, make_specialist(name))
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("critic", critic)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", "diagnose"),
        {**{name: name for name in available}, "diagnose": "diagnose"},
    )
    for name in available:
        graph.add_edge(name, "supervisor")
    graph.add_edge("diagnose", "critic")
    graph.add_conditional_edges(
        "critic",
        lambda state: state.get("next", "done"),
        {"supervisor": "supervisor", "done": END},
    )
    return graph.compile()


def investigate(evidence: Evidence, max_revisions: int = MAX_REVISIONS) -> Run:
    """Run the graph over one incident."""
    compiled = build_graph(evidence, max_revisions=max_revisions)
    final = compiled.invoke(
        {"question": evidence.question, "consulted": [], "findings": [],
         "critiques": [], "trace": [], "revisions": 0},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    result = final.get("diagnosis") or diagnose([])
    return Run(
        diagnosis=result,
        consulted=final.get("consulted", []),
        critiques=final.get("critiques", []),
        trace=final.get("trace", []),
        revisions=final.get("revisions", 0),
    )


def investigate_without_graph(evidence: Evidence) -> Run:
    """The same investigation as a plain loop.

    Exists so the graph can be proved to add orchestration and not *reasoning*:
    `test_the_graph_and_the_plain_loop_agree` asserts the two reach an identical
    diagnosis on every scenario. If they ever diverge, logic has leaked into the
    wiring, where it cannot be unit-tested.
    """
    findings: List[Finding] = []
    consulted: List[str] = []
    for specialist in SPECIALISTS:
        if specialist.applicable(evidence):
            findings.extend(specialist.analyse(evidence))
            consulted.append(specialist.name)
    return Run(diagnosis=diagnose(findings), consulted=consulted,
               trace=["plain loop"], revisions=0)


def recommended_action(diagnosis: Diagnosis) -> str:
    return RECOMMENDED_ACTION.get(diagnosis.root_cause, "investigate")
