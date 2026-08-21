"""The incident report, and the one job the language model is given.

**The model writes the paragraph. It does not reach the conclusion.**

That split is the whole design. `render()` produces a complete, correct report
with no model at all — verdict, evidence, citations, recommended action. A
model, when configured, rewrites only the opening paragraph into something a
tired person reads more easily.

`test_the_narration_does_not_change_the_diagnosis` asserts the verdict is
byte-identical with and without it. If that ever fails, the model has started
doing the reasoning, and the reasoning stops being testable the moment it does.

**Every claim cites its source.** Not decoration: at 3am the first thing a
sceptical engineer does is check one claim, and a report that cannot be checked
gets ignored on its second wrong call and never trusted again.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import List, Optional

from oncall.agents.graph import Run, recommended_action
from oncall.evidence.models import Evidence

SEVERITY_EMOJI = {
    "healthy": "OK", "noise": "OK",
    "data_shift": "ACT", "concept_drift": "ACT",
    "bad_deploy": "ACT", "pipeline_break": "ACT",
}


def render(run: Run, evidence: Evidence, narration: Optional[str] = None) -> str:
    """A complete incident report. No model required."""
    diagnosis = run.diagnosis
    lines: List[str] = []

    lines.append(f"# Incident: {diagnosis.root_cause}")
    lines.append("")
    lines.append(f"**{SEVERITY_EMOJI.get(diagnosis.root_cause, 'ACT')}** - "
                 f"recommended action: **{recommended_action(diagnosis)}**")
    lines.append("")
    lines.append(narration or _template_narration(run, evidence))
    lines.append("")

    if diagnosis.ambiguous:
        lines.append(
            f"> **This is close.** '{diagnosis.root_cause}' beats "
            f"'{diagnosis.runner_up}' by only {diagnosis.margin:.1f} points of "
            f"evidence. Investigate both rather than committing to one.")
        lines.append("")

    lines.append("## Evidence")
    lines.append("")
    lines.append("| finding | source |")
    lines.append("|---|---|")
    for finding in diagnosis.findings:
        supports = " ".join(sorted(finding.supports)) or "-"
        lines.append(f"| {finding.claim} | `{finding.citation}` |")
    lines.append("")

    supporting = diagnosis.supporting()
    if supporting:
        lines.append(f"**{len(supporting)} of {len(diagnosis.findings)} "
                     f"findings positively support '{diagnosis.root_cause}'.**")
    else:
        lines.append(f"**Nothing positively supports "
                     f"'{diagnosis.root_cause}'** - it was reached by "
                     f"eliminating the alternatives, which is weaker evidence "
                     f"and is worth saying rather than hiding.")
    lines.append("")

    lines.append("## How it was reached")
    lines.append("")
    lines.append(f"Consulted: {', '.join(run.consulted) or 'nothing'}. "
                 f"{run.steps} steps, {run.revisions} revision(s).")
    lines.append("")
    lines.append("```")
    lines.extend(run.trace)
    lines.append("```")
    lines.append("")

    if run.critiques:
        lines.append("### The critic's concerns")
        lines.append("")
        for critique in run.critiques:
            lines.append(f"- {critique}")
        lines.append("")

    lines.append("### Evidence weights")
    lines.append("")
    ordered = sorted(diagnosis.scores.items(), key=lambda kv: -kv[1])
    lines.append("| root cause | score |")
    lines.append("|---|---|")
    for cause, score in ordered:
        marker = " **<- chosen**" if cause == diagnosis.root_cause else ""
        lines.append(f"| {cause} | {score:+.1f}{marker} |")
    lines.append("")

    missing = evidence.missing()
    if missing:
        lines.append(f"> **Partial evidence.** {', '.join(missing)} was not "
                     f"available, so this conclusion rests on "
                     f"{', '.join(evidence.available()) or 'nothing'}.")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_The diagnosis above is computed deterministically from weighted "
        "evidence, not generated. A language model, when configured, rewrites "
        "only the opening paragraph; a test asserts the verdict is identical "
        "with and without it._")
    return "\n".join(lines)


def _template_narration(run: Run, evidence: Evidence) -> str:
    """The no-model paragraph. Correct, if a little flat."""
    diagnosis = run.diagnosis
    top = diagnosis.supporting()[:2]
    because = "; ".join(f.claim for f in top) or "the evidence eliminated the alternatives"
    return (f"The most likely cause is **{diagnosis.root_cause}** "
            f"(confidence {diagnosis.confidence:.0%}, margin "
            f"{diagnosis.margin:.1f} over '{diagnosis.runner_up}'), because "
            f"{because}.")


# ---------------------------------------------------------------------------
# Optional narration
# ---------------------------------------------------------------------------

ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:generateContent")
DEFAULT_MODEL = "gemini-flash-latest"

PROMPT = """You are writing the opening paragraph of an ML incident report for
an on-call engineer at 3am.

The diagnosis has ALREADY been made from weighted evidence. Do not second-guess
it, do not add causes, and do not soften it. Rewrite it as two or three plain
sentences that a tired person can act on.

Diagnosis: {cause} (confidence {confidence:.0%})
Recommended action: {action}

Evidence:
{evidence}

Write only the paragraph. No heading, no bullet list, no preamble."""


def read_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Key from the argument, then the environment. BOM-stripped.

    A key saved from a Windows editor carries a UTF-8 BOM, which survives into
    the HTTP header and produces `'latin-1' codec can't encode character
    '\\ufeff'` - an error that names the encoding and not the cause.
    """
    key = explicit or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    return key.strip().lstrip("﻿").strip() or None


def narrate(run: Run, api_key: Optional[str] = None,
            model: str = DEFAULT_MODEL, timeout_s: float = 8.0
            ) -> Optional[str]:
    """Ask a model for the opening paragraph. Returns None on any failure.

    None rather than an exception, because a narration failure must not fail an
    incident report. The template paragraph is already correct; the model is a
    readability improvement, not a dependency.
    """
    import json

    key = read_api_key(api_key)
    if not key:
        return None

    diagnosis = run.diagnosis
    evidence_lines = "\n".join(f"- {f.claim} [{f.citation}]"
                               for f in diagnosis.findings[:8])
    prompt = PROMPT.format(cause=diagnosis.root_cause,
                           confidence=diagnosis.confidence,
                           action=recommended_action(diagnosis),
                           evidence=evidence_lines)

    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
    }).encode("utf-8")

    request = urllib.request.Request(
        ENDPOINT.format(model=model), data=body,
        headers={
            "Content-Type": "application/json",
            # A header, never `?key=` in the URL. URLs reach exception
            # messages, proxy logs and CI output; headers do not.
            "x-goog-api-key": key,
        },
        method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError,
            IndexError, TypeError, ValueError, OSError):
        # The original exception is deliberately discarded: urllib exceptions
        # can carry the Request object, and the Request carries the API key.
        return None
