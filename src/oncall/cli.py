"""Command line entry point.

    python -m oncall.cli incidents                  list what can be investigated
    python -m oncall.cli investigate concept_drift  the full report
    python -m oncall.cli evaluate                   score the agent against truth
    python -m oncall.cli ablate                     what each artifact contributes
    python -m oncall.cli serve                      run the MCP server
    python -m oncall.cli write ./artifacts --scenario bad_deploy

Everything runs with no API key. `investigate --narrate` adds a model-written
opening paragraph if `GEMINI_API_KEY` is set, and changes nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys

from oncall.agents.graph import investigate, recommended_action
from oncall.evaluate import ablate, degradation, recovery, run as run_eval, tool_selection
from oncall.incidents import SCENARIOS, SCENARIOS_BY_NAME, evidence_for, write
from oncall.report import narrate, render


def _stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def cmd_incidents(args) -> int:
    for scenario in SCENARIOS:
        print(f"{scenario.name:16} truth={scenario.root_cause:15} "
              f"{scenario.description}")
    return 0


def cmd_investigate(args) -> int:
    if args.incident not in SCENARIOS_BY_NAME:
        print(f"unknown incident '{args.incident}'. Known: "
              f"{', '.join(sorted(SCENARIOS_BY_NAME))}", file=sys.stderr)
        return 2

    evidence = evidence_for(args.incident)
    outcome = investigate(evidence)

    narration = None
    if args.narrate:
        narration = narrate(outcome)
        if narration is None:
            print("(no GEMINI_API_KEY, or the call failed - using the "
                  "templated paragraph. The diagnosis is unaffected.)\n",
                  file=sys.stderr)

    print(render(outcome, evidence, narration=narration))

    truth = SCENARIOS_BY_NAME[args.incident].root_cause
    if outcome.diagnosis.root_cause != truth:
        print(f"\n(ground truth for this scenario was '{truth}')")
        return 1
    return 0


def cmd_evaluate(args) -> int:
    report = run_eval()
    print(report.table())
    print()
    print(f"task success    : {report.task_success:.0%}")
    print(f"action accuracy : {report.action_accuracy:.0%}")
    print(f"mean steps      : {report.mean_steps:.1f}")

    calibration = report.calibration()
    print(f"confidence when right: {calibration['when_right']:.2f}")
    if calibration["when_wrong"] != calibration["when_wrong"]:      # NaN
        print("confidence when wrong: n/a - the agent got everything right on "
              "this set, so calibration is UNMEASURABLE here. That is a "
              "limitation of the scenario set, not a result.")
    else:
        print(f"confidence when wrong: {calibration['when_wrong']:.2f}")

    selection = tool_selection(report)
    print(f"tools consulted per incident: "
          f"{selection['consulted_per_incident']:.1f} of 3")

    print()
    print("recovery: does the critic send work back when evidence was skipped?")
    result = recovery()
    print(f"  recovered={result['recovered']} after {result['revisions']} "
          f"revision(s); consulted {result['consulted']}")
    for critique in result["critiques"]:
        print(f"  critic: {critique}")
    return 0 if report.task_success == 1.0 else 1


def cmd_ablate(args) -> int:
    print("Each artifact removed in turn. This is the only way to find out")
    print("that a source you spent a week integrating changes no verdicts.")
    print()
    print(f"{'evidence':<18} {'task success':>13} {'action accuracy':>17}")
    print("-" * 52)
    for label, scores in ablate().items():
        print(f"{label:<18} {scores['task_success']:>12.0%} "
              f"{scores['action_accuracy']:>16.0%}")

    print()
    print("With an artifact missing, does the agent NOTICE?")
    for row in degradation():
        print(f"  missing {row['missing']:<9} -> verdict "
              f"{row['verdict']:<15} flagged={row['noticed']}")
    return 0


def cmd_serve(args) -> int:
    from oncall.mcp_server import main as serve_main
    argv = ["--http", "--host", args.host, "--port", str(args.port)] \
        if args.http else []
    return serve_main(argv)


def cmd_write(args) -> int:
    path = write(args.scenario, args.directory)
    print(f"wrote drift.json, evals.json and changes.json to {path}")
    print("These are the shapes model-drift-monitor, llm-eval-pipeline and")
    print("ai-code-review-bot actually produce, so `Evidence.load(dir)` reads")
    print("a real pipeline's output unchanged.")
    return 0


def cmd_json(args) -> int:
    from oncall.mcp_server import investigate_incident
    print(json.dumps(investigate_incident(args.incident), indent=2))
    return 0


def main(argv=None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(
        prog="oncall", description="diagnose an ML incident from its evidence")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("incidents", help="list the incidents").set_defaults(
        func=cmd_incidents)

    investigate_parser = sub.add_parser("investigate", help="full report")
    investigate_parser.add_argument("incident")
    investigate_parser.add_argument(
        "--narrate", action="store_true",
        help="let a model write the opening paragraph (needs GEMINI_API_KEY)")
    investigate_parser.set_defaults(func=cmd_investigate)

    sub.add_parser("evaluate", help="score the agent").set_defaults(
        func=cmd_evaluate)
    sub.add_parser("ablate", help="what each artifact contributes").set_defaults(
        func=cmd_ablate)

    json_parser = sub.add_parser("json", help="the diagnosis as JSON")
    json_parser.add_argument("incident")
    json_parser.set_defaults(func=cmd_json)

    serve = sub.add_parser("serve", help="run the MCP server")
    serve.add_argument("--http", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8931)
    serve.set_defaults(func=cmd_serve)

    write_parser = sub.add_parser("write", help="write artifacts to a directory")
    write_parser.add_argument("directory")
    write_parser.add_argument("--scenario", default="bad_deploy")
    write_parser.set_defaults(func=cmd_write)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
