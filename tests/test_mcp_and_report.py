"""The MCP server and the incident report.

The MCP tests use `Client(server)` **in process** — mcp 2.0 accepts a server
object directly, so the whole protocol round trip is exercised with no
subprocess, no port, and no flakiness from either. That affordance is the only
reason a protocol server is cheap to test at all.
"""

import asyncio
import json

import pytest

from oncall.agents.graph import investigate
from oncall.incidents import SCENARIOS, evidence_for
from oncall.mcp_server import (
    SERVER_NAME,
    TOOLS,
    build_server,
    compare_incidents,
    get_drift_report,
    investigate_incident,
    list_incidents,
    list_specialists,
)
from oncall.report import _template_narration, narrate, read_api_key, render

mcp = pytest.importorskip("mcp", reason="the mcp package is optional")


def call(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# The tool functions, without MCP
# ---------------------------------------------------------------------------

def test_the_tools_work_as_plain_functions():
    """They are ordinary functions first and MCP tools second, so most of the
    surface is testable without the protocol at all."""
    assert {i["name"] for i in list_incidents()} == {s.name for s in SCENARIOS}
    assert get_drift_report("data_shift")["severity"] == "high"
    assert investigate_incident("bad_deploy")["root_cause"] == "bad_deploy"
    assert len(list_specialists()) == 3


def test_an_unknown_incident_is_refused_with_the_known_list():
    with pytest.raises(ValueError, match="Known:"):
        get_drift_report("does_not_exist")


def test_compare_returns_both():
    result = compare_incidents("bad_deploy", "concept_drift")
    assert result["bad_deploy"]["root_cause"] == "bad_deploy"
    assert result["concept_drift"]["root_cause"] == "concept_drift"


def test_every_finding_in_a_tool_response_has_a_citation():
    for scenario in SCENARIOS:
        payload = investigate_incident(scenario.name)
        assert payload["findings"]
        for finding in payload["findings"]:
            assert finding["citation"]
            assert ":" in finding["citation"] and "=" in finding["citation"]


# ---------------------------------------------------------------------------
# Over the protocol
# ---------------------------------------------------------------------------

def test_the_server_advertises_every_tool():
    async def run():
        from mcp.client import Client
        async with Client(build_server()) as client:
            listed = await client.list_tools()
            return {t.name for t in listed.tools}
    assert call(run()) == set(TOOLS)


def test_tool_schemas_are_generated_from_type_hints():
    async def run():
        from mcp.client import Client
        async with Client(build_server()) as client:
            listed = await client.list_tools()
            return {t.name: t.input_schema for t in listed.tools}

    schemas = call(run())
    # snake_case on the model, camelCase on the wire. `tool.inputSchema` does
    # not exist in mcp 2.0 and reading it returns nothing useful.
    assert schemas["investigate_incident"]["required"] == ["incident"]
    assert schemas["compare_incidents"]["required"] == ["first", "second"]


def test_a_tool_call_returns_structured_content():
    """mcp 2.0 derives `structuredContent` from the RETURN annotation, and a
    bare `dict` produces none at all - the text is still there, so it looks
    fine in a chat client and silently breaks anything reading the structured
    field. Every tool here returns a parameterised generic for that reason."""
    async def run():
        from mcp.client import Client
        async with Client(build_server()) as client:
            return await client.call_tool("investigate_incident",
                                          {"incident": "concept_drift"})

    result = call(run())
    assert result.is_error is False
    assert result.structured_content is not None, \
        "a bare `-> dict` annotation would make this None"
    assert result.structured_content["result"]["root_cause"] == "concept_drift"


def test_a_bad_argument_is_an_error_not_a_crash():
    async def run():
        from mcp.client import Client
        async with Client(build_server()) as client:
            return await client.call_tool("investigate_incident",
                                          {"incident": "nonsense"})

    result = call(run())
    assert result.is_error is True
    assert "unknown incident" in result.content[0].text


def test_the_server_exposes_a_readable_resource():
    async def run():
        from mcp.client import Client
        async with Client(build_server()) as client:
            listed = await client.list_resources()
            contents = await client.read_resource("oncall://incidents")
            return [str(r.uri) for r in listed.resources], contents

    uris, contents = call(run())
    assert "oncall://incidents" in uris
    assert len(json.loads(contents.contents[0].text)) == len(SCENARIOS)


def test_a_templated_resource_is_advertised():
    """A URI with {params} registers as a template rather than a static
    resource, which is how a client offers 'diagnosis for <which incident>'."""
    async def run():
        from mcp.client import Client
        async with Client(build_server()) as client:
            listed = await client.list_resource_templates()
            return [str(t.uri_template) for t in listed.resource_templates]
    assert "oncall://incident/{name}/diagnosis" in call(run())


def test_the_server_identifies_itself():
    async def run():
        from mcp.client import Client
        async with Client(build_server()) as client:
            return client.server_info.name, client.protocol_version
    name, version = call(run())
    assert name == SERVER_NAME
    assert version


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_is_complete_without_a_model():
    outcome = investigate(evidence_for("data_shift"))
    text = render(outcome, evidence_for("data_shift"))
    assert "# Incident: data_shift" in text
    assert "recommended action" in text
    assert "## Evidence" in text
    assert "retrain on recent data" in text


def test_every_claim_in_the_report_names_its_source():
    outcome = investigate(evidence_for("bad_deploy"))
    text = render(outcome, evidence_for("bad_deploy"))
    for finding in outcome.diagnosis.findings:
        assert str(finding.citation) in text


def test_the_narration_does_not_change_the_diagnosis():
    """The test that keeps the model out of the reasoning. `render` takes the
    paragraph as an argument, so the verdict, the evidence table and the
    recommended action are byte-identical whatever it says."""
    outcome = investigate(evidence_for("concept_drift"))
    evidence = evidence_for("concept_drift")

    plain = render(outcome, evidence)
    narrated = render(outcome, evidence, narration="Something else entirely.")

    def without_narration(text):
        return [line for line in text.splitlines()
                if "most likely cause is" not in line
                and "Something else entirely" not in line]

    assert without_narration(plain) == without_narration(narrated)


def test_narration_returns_none_without_a_key(monkeypatch):
    """A narration failure must not fail an incident report. The templated
    paragraph is already correct."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert narrate(investigate(evidence_for("healthy"))) is None


def test_the_api_key_reader_strips_a_bom(monkeypatch):
    """A key saved from a Windows editor carries a UTF-8 BOM, which survives
    into the HTTP header and produces an error that names the encoding and not
    the cause."""
    monkeypatch.setenv("GEMINI_API_KEY", "﻿  secret-value  ")
    assert read_api_key() == "secret-value"


def test_the_template_narration_names_the_cause_and_the_evidence():
    outcome = investigate(evidence_for("pipeline_break"))
    text = _template_narration(outcome, evidence_for("pipeline_break"))
    assert "pipeline_break" in text


def test_a_report_on_partial_evidence_says_so():
    """An agent that quietly diagnoses from two of three files is worse than
    one that refuses, because nobody knows to distrust it."""
    from oncall.evidence.models import ChangeLog, Evidence
    full = evidence_for("bad_deploy")
    partial = Evidence(drift=full.drift, evals=full.evals, changes=ChangeLog())

    text = render(investigate(partial), partial)
    assert "Partial evidence" in text
    assert "changes" in text


def test_an_unsupported_verdict_is_labelled_as_such():
    from oncall.agents.specialists import Citation, Finding, diagnose
    from oncall.agents.graph import Run
    result = diagnose([Finding("only negative evidence",
                               Citation("drift", "severity", "none"),
                               rules_out={"data_shift": 5.0})])
    text = render(Run(diagnosis=result), evidence_for("healthy"))
    assert "Nothing positively supports" in text
