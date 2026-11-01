# Frontier calibration skill

[`frontier-calibration`](frontier-calibration/SKILL.md) guides an agent through
comparing Frontier with a chosen inference engine, finding the causes of
prediction errors, and correcting the simulator. It covers time to first token
(TTFT), time per output token (TPOT), request latency, and throughput. The reference can be vLLM, your
own fork, SGLang, TensorRT-LLM, or another engine whose behavior you want to
model in Frontier.

The goal is to predict the behavior users normally see when they run the
selected engine with the same configuration. Calibration starts with enough independent clean runs
to establish a useful reference, including normal variation between runs.
Detailed traces help explain errors; users should be able to assess prediction
accuracy through ordinary repeated runs. Modeling changes need evidence that
they address recurring behavior and improve predictions on fresh runs.

## How calibration is organized

For a new engine or build, the [engine integration procedure](frontier-calibration/references/engine-integration.md)
first checks collection tools, metric conversion, and the Frontier behavior
needed by the case. Existing vLLM collectors are available for compatible
builds; other engines may need a reader, probe, or simulator capability added.
Missing optional probes limit those diagnostics while supported E2E comparison
can proceed.

The skill has three main entry points. An agent can start with a new case or
continue from evidence you already have.

| Entry | Responsibility |
| --- | --- |
| [`parity-run`](frontier-calibration/entries/parity-run.md) | Match the model, workload, and runtime settings; collect and validate reference-engine runs; run Frontier on the same case. |
| [`error-analysis`](frontier-calibration/entries/error-analysis.md) | Compare E2E metrics, assess whether the clean reference is sufficient, and trace errors to configuration semantics, workflow, operators, communication, or CPU overhead. |
| [`error-correction`](frontier-calibration/entries/error-correction.md) | Apply agreed corrections, run regressions, and collect fresh evidence before comparing metrics again. |

Specialized entries handle scheduling and KV-cache behavior, operator/kernel
coverage, CPU overhead, and communication such as collectives and KV transfers.
The [call graph](frontier-calibration/references/call-graph.md) defines their
order and when to use each branch. Workflow and operator errors are resolved
before CPU overhead corrections, so host timing does not absorb a logic error.

An optional [workflow check with measured layer timings](frontier-calibration/references/layer-timing-calibration.md)
is useful when the target GPU allocation and timing support are available.
It measures the whole model at the target scale, accounts for compute,
communication, and CPU work, then substitutes those timings while Frontier
still controls topology, scheduling, KV-cache maintenance, and preemption.
This reduces timing-estimation error while checking workflow behavior. Once
that check passes, calibration progressively replaces layer timings with
operator/kernel, communication, and CPU estimates. Final accuracy is checked
with Frontier's normal predictors against clean ground truth.

`SKILL.md` selects the entry; `entries/` contains the steps; `references/`
defines shared metric, evidence, and correction rules. These entries belong to
one skill and are loaded as needed. The repository's E2E gate computes the
per-metric 10% verdict. The skill also requires enough clean-run evidence to
support an accuracy claim.

## Installation and local configuration

The same skill directory can be used by Claude Code, Codex CLI, pi, and
OpenCode. Copy or link the entire `frontier-calibration/` directory into your
agent's skills directory so its supporting files remain available.

| Agent | User skills directory |
| --- | --- |
| Claude Code | `$HOME/.claude/skills/` |
| Codex CLI | `${CODEX_HOME:-$HOME/.codex}/skills/` |
| pi | `$HOME/.pi/agent/skills/` |
| OpenCode | `${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills/` |

For a new installation, run this from the Frontier repository root. Set
`skills_dir` to the directory for your agent; the example selects Claude Code.
A symlink keeps the installed skill in sync with this checkout.

```bash
skills_dir="$HOME/.claude/skills"
# Codex: skills_dir="${CODEX_HOME:-$HOME/.codex}/skills"
# pi: skills_dir="$HOME/.pi/agent/skills"
# OpenCode: skills_dir="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills"

mkdir -p "$skills_dir"
ln -s "$PWD/skills/frontier-calibration" "$skills_dir/frontier-calibration"
```

Start a new agent session after installation. If you use a custom agent
configuration directory, use its skills directory in place of the defaults
above. If the agent already discovers this skill through another directory,
keep that installation to avoid duplicate names.

Keep reusable local settings in `site.local.md` beside the skill's `SKILL.md`,
or provide them with the case. Record the reference engine and its source
checkout or installed build, calibration artifacts root, scratch root
(`FRONTIER_TMP_ROOT`), Python command, GPU platform and device generation,
and the local rules for submitting jobs. Git ignores this file so each machine
can keep its own settings. Case-specific engine choices take precedence over
these local defaults. Running the tools requires access to the Frontier
checkout and the appropriate Python environments. Collecting new ground truth
also requires the GPU resources specified by the case.

## Usage

Open the agent in the Frontier checkout and give it a case directory or the
model, workload, and GPU layout for a new case. For an existing case, a request
can be as simple as:

```text
Use frontier-calibration to calibrate Frontier against my SGLang build.
The engine configuration, workload, and clean runs are in <case-directory>.
Verify the engine integration and that the clean runs provide a sufficient
reference. Compare TTFT, TPOT, request latency, and throughput. Investigate any
metric outside the 10% gate and propose a correction.
```

Replace `<case-directory>` with the path to your case. To invoke the skill
explicitly, use the syntax for your agent and append your request:

| Agent | Invocation |
| --- | --- |
| Claude Code | `/frontier-calibration <request>` |
| Codex CLI | `$frontier-calibration <request>` |
| pi | `/skill:frontier-calibration <request>` |
| OpenCode | Ask the agent to load `frontier-calibration` with its `skill` tool, then carry out the request. |

You can name an internal entry in the request, such as "Start at
`error-analysis` and use the existing runs." The names under `entries/` are
steps within the skill, not separate slash commands.

To check the skill's structure after editing it, run from the repository root:

```bash
python3 skills/frontier-calibration/tests/pressure_scenarios.py skills/frontier-calibration
```

A successful check prints `v2_pressure=PASS`. It validates the entry catalog,
references, repository paths, and portable wording; it does not run a
calibration or verify prediction accuracy.
