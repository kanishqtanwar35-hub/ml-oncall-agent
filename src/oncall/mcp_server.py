"""An MCP server, so the on-call tools work from any MCP client.

    python -m oncall.mcp_server                 # stdio, for Claude Desktop et al
    python -m oncall.mcp_server --http --port 8931

**Why MCP rather than a REST API.** The consumer here is a person's assistant,
not another service. MCP is the protocol that lets a chat client discover these
tools, read their schemas and call them without anybody writing an integration —
which is exactly the shape of "let me ask about last night's incident from
wherever I already am".

**Written against mcp 2.0.0, which is a breaking rewrite.** Worth stating,
because most material online is 1.x and does not work:

  * `FastMCP` **is gone.** `from mcp.server.fastmcp import FastMCP` raises
    `ModuleNotFoundError`. The replacement is `MCPServer` from
    `mcp.server.mcpserver`.
  * The low-level `@server.list_tools()` / `@server.call_tool()` decorators are
    gone too; the low-level `Server` now takes constructor callbacks.
  * `@server.tool()` **must be called** — bare `@server.tool` raises a
    `TypeError` that says so.
  * Tool schemas are generated from **type hints** via pydantic. On Python 3.11
    a parameter annotated with `typing.TypedDict` fails at registration;
    pydantic requires `typing_extensions.TypedDict`. Everything below uses
    plain scalars and `dict` to stay clear of it.

**The tools return evidence, not prose.** Each one hands back the artifact or
the structured diagnosis, so the client's model reasons over *data with
citations* rather than over a paragraph somebody already summarised. A summary
is where the detail goes to die.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from oncall.agents.graph import investigate, recommended_action
from oncall.agents.specialists import SPECIALISTS_BY_NAME
from oncall.incidents import SCENARIOS, SCENARIOS_BY_NAME, build, evidence_for

SERVER_NAME = "ml-oncall"
SERVER_VERSION = "1.0.0"

INSTRUCTIONS = """Tools for diagnosing an ML incident.

Start with `list_incidents`, then `investigate` for a full diagnosis with
citations. The `get_*` tools return the raw artifacts if you want to check the
reasoning yourself - every claim in a diagnosis names the file and field it
came from, so you can.

The diagnosis is computed deterministically from weighted evidence, not
generated. Two independent implementations (a LangGraph and a plain loop) are
asserted to agree on every scenario.
"""


def _known(incident: str) -> None:
    if incident not in SCENARIOS_BY_NAME:
        raise ValueError(
            f"unknown incident '{incident}'. Known: "
            f"{', '.join(sorted(SCENARIOS_BY_NAME))}"
        )


# ---------------------------------------------------------------------------
# The tool implementations. Plain functions, so they are testable without MCP.
# ---------------------------------------------------------------------------

def list_incidents() -> List[Dict[str, str]]:
    """Every incident available to investigate, with its description."""
    return [{"name": s.name, "description": s.description} for s in SCENARIOS]


def get_drift_report(incident: str) -> Dict[str, Any]:
    """The raw drift report for an incident, as `model-drift-monitor` writes it."""
    _known(incident)
    return build(incident)["drift"]


def get_eval_run(incident: str) -> Dict[str, Any]:
    """The raw offline evaluation results, with baseline and per-slice scores."""
    _known(incident)
    return build(incident)["evals"]


def get_changes(incident: str) -> Dict[str, Any]:
    """Code changes in the incident window, with any code-review findings."""
    _known(incident)
    return build(incident)["changes"]


def investigate_incident(incident: str) -> Dict[str, Any]:
    """Diagnose an incident. Returns the verdict, the evidence and the trace.

    Every finding carries a citation naming the artifact and field it came
    from, so the conclusion can be checked rather than trusted.
    """
    _known(incident)
    run = investigate(evidence_for(incident))
    diagnosis = run.diagnosis

    return {
        "incident": incident,
        "root_cause": diagnosis.root_cause,
        "confidence": round(diagnosis.confidence, 3),
        "ambiguous": diagnosis.ambiguous,
        "runner_up": diagnosis.runner_up,
        "margin": round(diagnosis.margin, 2),
        "recommended_action": recommended_action(diagnosis),
        "specialists_consulted": run.consulted,
        "revisions": run.revisions,
        "critiques": run.critiques,
        "findings": [
            {
                "claim": f.claim,
                "citation": str(f.citation),
                "specialist": f.specialist,
                "supports": f.supports,
            }
            for f in diagnosis.findings
        ],
        "scores": {k: round(v, 2) for k, v in diagnosis.scores.items()},
        "trace": run.trace,
    }


def compare_incidents(first: str, second: str) -> Dict[str, Any]:
    """Diagnose two incidents side by side.

    Useful for the case that trips people up: two incidents with nearly
    identical eval regressions and completely different causes.
    """
    return {name: investigate_incident(name) for name in (first, second)}


def list_specialists() -> List[Dict[str, str]]:
    """The specialists and the artifact each one reads."""
    return [{"name": s.name, "reads": s.reads}
            for s in SPECIALISTS_BY_NAME.values()]


TOOLS = {
    "list_incidents": list_incidents,
    "get_drift_report": get_drift_report,
    "get_eval_run": get_eval_run,
    "get_changes": get_changes,
    "investigate_incident": investigate_incident,
    "compare_incidents": compare_incidents,
    "list_specialists": list_specialists,
}


# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------

def build_server():
    """Construct the MCP server. Imported lazily so the tools stay usable
    without the `mcp` package installed - the CLI and the tests do not need it."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name=SERVER_NAME, version=SERVER_VERSION,
                       instructions=INSTRUCTIONS)

    # NOTE: @server.tool() with parentheses. Bare @server.tool raises a
    # TypeError in mcp 2.0 that explains exactly this.
    # Return annotations are PARAMETERISED generics on purpose. mcp 2.0 derives
    # `structuredContent` from the return type, and a bare `dict` or `list`
    # produces NOTHING structured - the text content is still there, so it looks
    # fine in a chat client and silently breaks any client that reads the
    # structured field. Verified: `-> dict` gives structured_content=None,
    # `-> Dict[str, Any]` gives {'result': {...}}.
    @server.tool(name="list_incidents", description=list_incidents.__doc__)
    def list_incidents_tool() -> List[Dict[str, str]]:
        return list_incidents()

    @server.tool(name="get_drift_report", description=get_drift_report.__doc__)
    def get_drift_report_tool(incident: str) -> Dict[str, Any]:
        return get_drift_report(incident)

    @server.tool(name="get_eval_run", description=get_eval_run.__doc__)
    def get_eval_run_tool(incident: str) -> Dict[str, Any]:
        return get_eval_run(incident)

    @server.tool(name="get_changes", description=get_changes.__doc__)
    def get_changes_tool(incident: str) -> Dict[str, Any]:
        return get_changes(incident)

    @server.tool(name="investigate_incident",
                 description=investigate_incident.__doc__)
    def investigate_tool(incident: str) -> Dict[str, Any]:
        return investigate_incident(incident)

    @server.tool(name="compare_incidents", description=compare_incidents.__doc__)
    def compare_tool(first: str, second: str) -> Dict[str, Any]:
        return compare_incidents(first, second)

    @server.tool(name="list_specialists", description=list_specialists.__doc__)
    def list_specialists_tool() -> List[Dict[str, str]]:
        return list_specialists()

    # A resource rather than a tool: it is a document to read, not an action to
    # take, and clients present the two differently.
    @server.resource("oncall://incidents",
                     description="The incident catalogue as JSON.",
                     mime_type="application/json")
    def incidents_resource() -> str:
        return json.dumps(list_incidents(), indent=2)

    @server.resource("oncall://incident/{name}/diagnosis",
                     description="A full diagnosis for one incident.",
                     mime_type="application/json")
    def diagnosis_resource(name: str) -> str:
        return json.dumps(investigate_incident(name), indent=2)

    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="oncall.mcp_server")
    parser.add_argument("--http", action="store_true",
                        help="serve streamable-HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8931)
    args = parser.parse_args(argv)

    server = build_server()
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
