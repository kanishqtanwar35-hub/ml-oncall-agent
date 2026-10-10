https://m8ven.ai/badge/mcp/kanishqtanwar35-hub-ml-oncall-agent-1aup5n?v=c0d514391b09a058726d8a4ae193dfd3
https://m8ven.ai/badge/mcp/kanishqtanwar35-hub/ml-oncall-agent
# ML On-Call Agent

**"Why did last night's run regress?"** — a multi-agent system that answers it
by correlating a drift report, an eval run and a deploy log, and **cites every
claim it makes**.

Exposed over **MCP**, so the tools work from any MCP client.

**Status:** 65 tests. The diagnosis is computed deterministically from weighted
evidence, so every number below is an assertion, not a demo.

---

## The real problem

An ML system degrades silently. When somebody finally notices, the evidence is
scattered across three places that do not talk to each other:

- a **drift report** — did the input distribution move?
- an **eval run** — did the offline scores regress, and on which slices?
- a **deploy log** — did anyone ship anything?

Correlating them is a job, and it is a job people do badly at 3am because it
means holding three JSON files in your head at once.

Those three artifacts are not invented for this repo. They are what
[`model-drift-monitor`](https://github.com/kanishqtanwar35-hub/model-drift-monitor),
[`llm-eval-pipeline`](https://github.com/kanishqtanwar35-hub/llm-eval-pipeline)
and [`ai-code-review-bot`](https://github.com/kanishqtanwar35-hub/ai-code-review-bot)
already produce. The reader was written against a **real artifact**, not against
documentation.

---

## The result

```bash
python -m oncall.cli evaluate
```

```
 scenario         truth           verdict          conf  margin  steps  action
 healthy          healthy         healthy          0.57     2.0      9  no action
 data_shift       data_shift      data_shift       0.71     8.0      9  retrain on recent data
 bad_deploy       bad_deploy      bad_deploy       0.66     8.5      9  roll back
 concept_drift    concept_drift   concept_drift    0.81     9.0      9  retrain with fresh labels
 pipeline_break   pipeline_break  pipeline_break   0.62     5.0      9  fix the upstream pipeline
 flaky_eval       noise           noise            0.62     3.0      9  rerun the evaluation

task success    : 100%
action accuracy : 100%
mean steps      : 9.0
```

**Ground truth is known by construction**, so the agent is *scored* rather than
admired. "The agent produced a plausible incident report" is not a result —
plausible is what language models do, including when the evidence supports
nothing.

### The scenario that is the point

`concept_drift`: the inputs are statistically identical, nothing was deployed,
and the model's ranking has **inverted** (AUC 0.729 → 0.326, worse than random).

There is nothing to point at. The answer comes from combining **two negatives
and one positive** — no drift, no deploy, ranking collapsed — which is exactly
what a summary of any single artifact would miss. It is also the blind spot
`model-drift-monitor` documents about *itself*, diagnosed one layer up.

---

## The design decision everything rests on

**The diagnosis is deterministic. The language model only narrates.**

The obvious build is to paste three JSON files into a prompt and ask what went
wrong. It produces something fluent every time — including when the evidence
supports nothing — and it is untestable, because you cannot assert on prose or
tell a correct answer from a lucky one.

So the reasoning is ordinary Python:

- each specialist reads **one** artifact and emits `Finding`s
- every finding carries a **citation** — file, field, value. `Finding` *requires*
  one, so a claim without a source cannot be constructed
- each finding names which root causes it **supports** and which it **rules out**
- `diagnose()` sums the weights — a pure function, unit-tested against truth

```
| finding                                                    | source                     |
|------------------------------------------------------------|----------------------------|
| input drift is 'none': the scored population is             | `drift:severity=none`      |
| statistically the same as training                          |                            |
| roc_auc is 0.326 - WORSE THAN RANDOM. The model's ranking   | `evals:metrics.roc_auc     |
| has inverted, which is a changed relationship               |  =0.326`                   |
| 1 change(s) landed but none touch model behaviour           | `changes:changes[].files`  |
```

`test_the_narration_does_not_change_the_diagnosis` asserts the verdict is
identical with and without a model. If it ever fails, the model has started
doing the reasoning — and the reasoning stops being testable the moment it does.

**Negative evidence is where most of the diagnostic power is.** "No drift" and
"nothing was deployed" are findings, with weights. An LLM summarising the drift
JSON would skip them, because nothing happened.

---

## The graph

```
   SUPERVISOR ──► drift ──┐
        ▲   ├──► evals ───┤   one specialist per artifact, each consulted once
        │   └──► changes ─┤
        │                 ▼
        │             DIAGNOSE          sum the weighted findings
        │                 ▼
        └──────────────  CRITIC         "is this conclusion supported?"
           (bounded)      ▼
                        REPORT
```

Two properties a straight-line pipeline does not have:

**The critic can send work back.** If the verdict rests on two artifacts while a
third was never read, control returns to the supervisor rather than publishing a
conclusion drawn from half the evidence.

**The cycle is bounded.** `MAX_REVISIONS = 2` caps it and `recursion_limit`
catches anything that escapes. An agent that can loop is an agent that can loop
forever, and a runaway on-call agent generates pages instead of answering them.

It runs **with no LLM and no API key** — the supervisor routes by a plain Python
rule — so the routing, the delegation and the critic's loop-back are all
unit-tested offline. `test_the_graph_and_the_plain_loop_agree` asserts a
LangGraph run and a plain `for` loop reach identical verdicts on every scenario,
which proves the graph adds *orchestration*, not reasoning.

---

## What each artifact is actually worth

```bash
python -m oncall.cli ablate
```

| evidence | task success | action accuracy |
|---|---|---|
| all three | **100%** | **100%** |
| without drift | 67% | 67% |
| without evals | 67% | 67% |
| **without changes** | **100%** | **100%** |

**An honest negative result, and it is asserted so it cannot be quietly
forgotten.** Removing the change log costs *nothing* on this scenario set —
`bad_deploy` is already separable from the drift and eval evidence alone. The
change log earns its place by **naming the commit**, which is what a human needs
in order to act, not by changing the diagnosis.

`test_the_change_log_currently_changes_no_verdicts` pins that. If a future
scenario makes it load-bearing, the test fails and this table has to change.
That is what the assertion is for.

Ablation is the only way to discover that a source you spent a week integrating
changes no verdicts.

### And when an artifact goes missing

Jobs fail, buckets are empty, paths change. The interesting question is not
whether the agent still answers — it will — but whether it **notices**:

```
  missing drift     -> verdict bad_deploy   flagged=True
  missing evals     -> verdict bad_deploy   flagged=True
  missing changes   -> verdict bad_deploy   flagged=True
```

An agent that quietly diagnoses from two of three files is worse than one that
refuses, because nobody knows to distrust it.

---

## The MCP server

```bash
python -m oncall.mcp_server                 # stdio, for Claude Desktop et al
python -m oncall.mcp_server --http --port 8931
```

Seven tools — `list_incidents`, `get_drift_report`, `get_eval_run`,
`get_changes`, `investigate_incident`, `compare_incidents`, `list_specialists`
— plus a resource and a templated resource.

**The tools return evidence, not prose.** Each hands back the artifact or the
structured diagnosis, so the client's model reasons over *data with citations*
rather than over a paragraph somebody already summarised. A summary is where the
detail goes to die.

### Written against `mcp 2.0.0`, which is a breaking rewrite

Worth stating, because almost everything published is 1.x and does not work.
Each of these was verified by running it:

| 1.x | 2.0.0 |
|---|---|
| `from mcp.server.fastmcp import FastMCP` | **gone** — `ModuleNotFoundError`. Use `MCPServer` from `mcp.server.mcpserver` |
| `@server.list_tools()` / `@server.call_tool()` | **gone** — the low-level `Server` takes constructor callbacks |
| `stdio_client` + `ClientSession` by hand | `Client(server_or_url_or_transport)` |
| `tool.inputSchema` | `tool.input_schema` — snake_case on the model, camelCase on the wire |

Two more that cost real time:

- **`@server.tool()` must be called.** Bare `@server.tool` raises a `TypeError`
  that says so.
- **A bare `-> dict` produces no `structuredContent` at all.** The text content
  is still there, so it looks fine in a chat client and silently breaks any
  client reading the structured field. Every tool here returns a parameterised
  generic (`Dict[str, Any]`) for that reason, and there is a test.

### Testing a protocol server without a subprocess

`Client(server)` accepts a server object directly, so the entire protocol round
trip runs **in process** — no subprocess, no port, no flakiness from either.
That affordance is the only reason protocol tests are cheap here.

---

## Quickstart

```bash
git clone https://github.com/kanishqtanwar35-hub/ml-oncall-agent
cd ml-oncall-agent
pip install -r requirements.txt
export PYTHONPATH=src

python -m oncall.cli incidents                  # what can be investigated
python -m oncall.cli investigate concept_drift  # the full report
python -m oncall.cli evaluate                   # score it against truth
python -m oncall.cli ablate                     # what each artifact is worth
pytest -q                                       # 65 tests
```

Point it at real artifacts:

```bash
python -m oncall.cli write ./artifacts --scenario bad_deploy
# then: Evidence.load("./artifacts") reads a real pipeline's output unchanged
```

Everything runs with **no API key**. `investigate --narrate` adds a
model-written opening paragraph if `GEMINI_API_KEY` is set, and changes nothing
else.

---

## Bugs worth reading about

**A hardcoded tolerance swallowed the flaky-eval case.** `regressions()` used a
0.02 threshold, so a 0.006 move never registered and the noise check never ran —
the agent said `healthy` where the truth was `noise`. That is exactly the
folklore-threshold mistake `model-drift-monitor` argues against, reproduced one
repo over. Detection and significance are two questions: the floor is now 0.005
("anything a dashboard would show") and the harness's own measured `noise_std`
does the discriminating.

**The recovery test could not fail.** It faked a skipped specialist by
pre-seeding `consulted` — but the critic compares *available* against
*consulted*, so marking something consulted hid the gap from the very check
being tested. It reported `recovered: False` and looked like a broken critic.
The fault has to be injected in the **routing**, not the bookkeeping;
`build_graph(skip_first_pass=...)` does that, and the critic now demonstrably
catches the gap and sends the supervisor back.

---

## Limitations, stated plainly

- **The scenarios are synthetic.** Deliberately — real incidents do not come
  with a labelled root cause, which is precisely why diagnosing them is hard.
  The numbers characterise the *method*, not any production system.
- **Six scenarios is a small set.** 100% on six is not 100% in general, and the
  next scenario added is more likely to break it than confirm it.
- **Calibration is unmeasurable here.** With no wrong answers there is nothing
  to compare confidence against. The CLI prints `n/a` and says why rather than
  inventing a figure — `test_calibration_is_honestly_unmeasurable_here` pins
  that.
- **The weights are hand-set.** They encode my judgement about what evidence is
  worth, and a larger labelled incident set would let them be *fitted* instead —
  with a held-out split, because fitting them on six scenarios would be
  memorisation.
- **Tool-selection accuracy is trivially 1.0** with three always-present
  artifacts. The metric is here because it starts mattering at the fourth tool,
  and adding it retroactively is how you discover the agent has been calling
  everything for months.
- **No live integrations.** It reads artifacts from disk. Wiring it to a real
  warehouse, CI system and git host is deployment work, not reasoning work.
- **The critic checks completeness and support, not correctness.** It cannot
  tell you the weights are wrong, only that the evidence was thin.

## Roadmap

1. A larger labelled incident set, and fit the weights on a held-out split.
2. More specialists — serving latency, cost ledger, feature-store freshness —
   which is where tool-selection accuracy starts to mean something.
3. Wire the MCP server to real artifact sources instead of a scenario builder.
4. Multi-incident correlation: three services regressing at once is one
   incident, not three.

## Licence

MIT.
