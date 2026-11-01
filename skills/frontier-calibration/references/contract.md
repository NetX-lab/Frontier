# Frontier Calibration v2 Contract

This is the shared contract for every entry. `registry.json` is the entry
catalog, `call-graph.md` owns execution order, and this file owns comparison
scope, evidence, decision, and approval rules. Gate arithmetic lives in the
repository harness, `tests/comparison/e2e_gate/` (`README.md`, `gate.py`);
derived readings reuse that arithmetic without changing the native harness
reading. Engine-specific collection and conversion must first
pass `engine-integration.md`; tool availability is not implied by this contract.

## Calibration Objective

Frontier should predict the representative behavior and performance of the
selected reference engine and build under the same user-visible configuration
and workload. An ordinary user may run that configuration repeatedly and compare a mean or a tail statistic;
they should not need internal scheduler, kernel, or routing traces to explain
away a prediction error. Calibration uses those traces to find causes and
improve the model, while its accuracy claim rests on ordinary clean runs.

### Establish a sufficient clean reference

Declare the comparison statistic and its aggregation level before collection:
a percentile of per-request latency differs from a percentile of run-level
means. Record the independent clean-run count, per-run values, variability,
and uncertainty of the claimed statistic. Warmups and many requests within one
run do not substitute for independent run repetitions. Choose repeat counts
from observed variability and the precision needed to distinguish a modeling
effect or a gate verdict. Ten runs is an example, not a universal sufficient
sample size; a tail estimate may require more evidence than a mean.

Collect additional independent clean runs within the approved resources until
the conclusion is stable at that precision. If uncertainty could change the
conclusion, report `INSUFFICIENT_EVIDENCE` and the collection needed; a numeric
gate pass does not resolve it. Record this assessment in the existing E2E
summary, preserving the harness reading. Budget exhaustion leaves an explicit
limit, never an assumed stable reference. The existing gate keeps its defined
means, pooling, and 10% rule; a requested p90 comparison is reported separately
unless a gate change is explicitly decided. A mean prediction alone does not
establish tail accuracy.

Keep all admissible clean runs of the declared population, including ordinary
slow runs and recurrent mode changes. Exclude a run only for an evidenced
validity failure under the collection rules, with its reason recorded
independently of whether exclusion improves agreement. A probe's host-speed
admission check does not select which otherwise valid clean runs count.
Mode-conditioned analysis and declared abstractions retain their diagnostic
value, but report their scope alongside the ordinary clean-run comparison;
a restricted gate pass supports only that restricted claim.

### Correct general behavior

Base a repair on a causal behavior that recurs across independent runs or
representative workloads. Its normal inputs must be available from the user's
configuration/workload and supported profiling inputs, without per-run
fine-grained ground-truth traces. Validate the proposed effect on fresh runs
or cases that were not used to choose the correction. A run-specific offset,
a temporary assumed behavior, selected mode replay, or a changed metric window
introduced to pass the gate is not a modeling correction.

When missing modeling causes a material gap, keep that gap explicit. Gather
repeated clean-run evidence, source/event evidence of the mechanism, its
frequency and contribution to the user-visible metric, and the affected
configurations. Propose the smallest causal model with expected benefit,
implementation cost, and validation on fresh evidence for the user's decision
under `repair-approval.md`. Normal variability is not automatically an external
fault or a reason to exclude behavior. If evidence or value remains unclear,
record an unresolved limitation and the next evidence needed; neither a
speculative patch nor a narrower accuracy claim fixes the missing model.

### Warm steady state

Comparison and calibration target the selected engine's warm steady state.
One-time warmup costs belong to service start-up: kernel or JIT compilation,
autotuning, graph capture, lazily opened communication connections (KV
connector or collective handshakes), and other first-use setup. Frontier does
not model them, and no correction prices them.

A clean reference is warm only when its warmup has run every one-time path the
formal window uses: each prompt/context shape, each compile and graph path, and
each lazily connected communication pair. Load-based routing can leave pairs
unopened after a fixed warmup count, so readiness evidence shows the coverage
itself from the engine's own records (handshake or connection log lines,
lifecycle records), matched against the path and connection identities the
topology requires. Preflight records the policy, expected coverage and checking
method; the collector verifies actual coverage and warmup drain at the formal
boundary, and the post-run check audits the formal window.

A formal request that still pays a one-time cost makes its run incompletely
warmed, a `groundtruth_defect`. Identify such a request by mechanism: the
engine source shows the one-time path, and that run's event records bind each
event to every request that waited for it. Apply one rule to every case and
repeat, fixed before reading the comparison. Clean-run validity and warm
readiness are separate: a complete valid run remains in the reference population
when it is incompletely warmed. Keep every valid repeat, including one with no
remaining warm rows; that repeat leaves the comparison insufficient rather than
being discarded.

For each repeat, the warm reading removes exactly the event-bound requests
from both its reference rows and the matching Frontier rows, recomputes each
side's retained-request window, and pools with the harness arithmetic. It
decides the analysis status when admission, event coverage and reference
sufficiency are established; the unchanged all-request `run_gate.py` reading
stays on record beside it. Unknown readiness, unbound events or insufficient
remaining evidence yield `INSUFFICIENT_EVIDENCE`, not a favorable subset or a
substitute repeat. `../entries/e2e-metrics-gap.md` owns the reproducible calculation
and the harness's shared-ID limitation. Latency alone never identifies a
request; retained requests keep their observed queueing and timing. The next
collection warms the uncovered paths rather than carrying the exclusion
forward.

## Case Layout

A calibration task keeps one artifacts bundle. Record how each collector reads
and writes that bundle; repository harnesses use `--artifacts-root`. The bundle
is task-local and stays out of the repository; the repository pins its files by SHA-256 in
`tests/comparison/e2e_gate/artifacts.json` and `cases/scenario_matrix.json`.

```text
<artifacts root>/
  <case_id>/
    manifest.yaml                      case binding (see Case Binding)
    case_init.md                       purpose and the decisions that created the case
    inputs/
      engine_config.*                  native engine configuration, per mode when needed
      frontier_<generation>.json       Frontier settings, one file per table generation
      workload.json, trace/, cells/<cell>/trace/
      cpu_overhead*/                   published CPU-overhead tables with provenance.json
      groundtruth_overlay.patch, fork_changed_files.txt  when source changes apply
    runs/groundtruth_<mode>/<run_tag>/ run_manifest.json, launcher receipt, run/, worker evidence
    analysis/                          run checks, gate readings, analysis tables and status files
    reports/                           review-ready summaries
    repairs/                           change markers, review receipts, result receipts
  scenario_matrix/                     cell matrix and coverage check
  e2e_profiling/tables/<table set>/<model>/
```

Frontier outputs are large and go under the scratch root
(`$FRONTIER_TMP_ROOT/<task>/<case_id>/<label>/`); every receipt records that path.
An entry's output path such as `analysis/e2e_metrics_status.json` is relative to
the case directory.

## Case Binding

`manifest.yaml` binds a case before any run. A case-local writer may regenerate
it; each regeneration that changes a run input increments `run_generation` and
keeps the earlier values in a history field. Required content:

- identity: `case_id`, `run_generation`, `created_at_utc`, `purpose`,
  `requesting_user`, `reviewer_identity`;
- `selected_checkout`: Frontier worktree, branch, commit, dirty paths (empty
  for a gate run), the commit under test when it differs, and the A/B reference
  revision when one is used;
- the ground-truth engine/build identity (next section) and `engine_binding`
  from `engine-integration.md`;
- model/tokenizer and dtype, architecture, native parallel layout and its
  verified Frontier mapping, scheduler limits in the engine's own terms,
  chunked prefill, prefix caching, graph mode and capture sizes, attention
  backend, KV budget and its source, KV transfer, routing status and method,
  weight mode, and measurement families;
- the Frontier settings file of the current table generation;
- cells: trace path, `formal_request_ids` and `warmup_request_ids` (from the
  trace's `request_ids.json`), load level, ground-truth run tags;
- the decision IDs that set any of the above.

An unresolved required field is written `BLOCKED` or `PENDING`; the affected
runner stops instead of substituting a default. Record `not_applicable` with a
reason for concepts the engine does not use. Native defaults may be used only
when their effective values and source are verified and bound to the case.

## Ground-Truth Engine and Build

The user selects the reference engine and build. No repository URL, branch,
upstream version, or private fork is mandatory. Record `groundtruth_engine`,
`groundtruth_version`, the runtime environment, and one reproducible build
identity in the manifest and every run manifest:

- For source-based runs: `groundtruth_checkout_path`, `groundtruth_commit`,
  `groundtruth_tree_dirty`, and any applied instrumentation or overlay with
  its diff artifact and hash. A detached checkout at the recorded commit is
  valid. Record `groundtruth_branch`, `groundtruth_ref`,
  `groundtruth_remote_url`, and `groundtruth_remote_tip` when present; a local
  fork does not need a remote or a particular branch.
- For installed or compiled runtimes: package/build version and an immutable
  executable, package, or container digest. Pin compiled model artifacts,
  build options, and plugins when they affect execution. Mark source-checkout
  fields not applicable when source is unavailable; do not invent a commit.

Use a clean pinned source commit or an immutable recorded source/build snapshot.
Record `groundtruth_diff_artifact`, `groundtruth_dirty_patch_sha256`, and
`groundtruth_overlay_patch_applied` when an overlay exists. All modes of one
case use the same serving build; probes change only the declared observations.
A build, source, or instrumentation change starts a new run generation.
Unrecorded changes or a run that differs from its pinned identity fail admission.

Inspect a fork's serving changes against its declared base when needed to
understand behavior. An intentional fork difference is part of the selected
target, not a `groundtruth_defect` merely because upstream behaves differently.
A release claim must name the engine/build actually measured. Publishing a
fork or its artifacts remains an external action requiring authorization.

## Run Modes

Modes name evidence purposes. `clean` runs preserve the selected serving path
without timing-distorting diagnostic probes; normal production telemetry stays
as configured. They provide the E2E reference. Separate diagnostic runs collect
workflow, layer, operator/kernel, CPU, communication, or routing observations
as the case needs. Their native flags and combinations come from the verified
`engine_binding`, not from another engine's mode names. Diagnostic request
metrics, if emitted, remain diagnostic.

Plan and verify readiness under Warm steady state. Pair timing observations
with clean runs on
the same node group and topology; record each run separately and assess repeat
counts under Calibration Objective. Isolate conflicting ports and resources.
The existing vLLM mode names, switches, and published warmup recipe live in
`tool-boundaries.md`, Existing vLLM Collection Profile.

### Weights and routing

Use the selected workload's model weights, tokenizer, generation policy, and
native routing behavior. Record weight identity and any loading substitution.
Dummy weights or forced routing are valid only for an explicitly selected
synthetic case whose measured path and limits are established. Its result
supports that synthetic target, not ordinary native-model performance. Neither
substitution is a default way to make an unsupported case pass.

## Status and Evidence

Every analysis writes `analysis_state=COMPLETE|INCOMPLETE`,
`status=PASS|FAIL|INSUFFICIENT_EVIDENCE|PENDING_HUMAN_REVIEW|BLOCKED`,
`correction_state=pending|complete|not_applicable`, and a non-empty
`next_action`. `pending` means a scoped correction awaits the user's decision;
`complete` means the approved change, its rerun, and a fresh analysis are done;
`not_applicable` means a complete analysis found nothing to change in its
domain. The CPU lane treats `complete` and `not_applicable` alike as a finished
workflow or operator attempt.

## E2E Gate

The gate compares clean Frontier rows with clean reference-engine rows for the
formal requests of one cell. Each metric is decided on its own and passes when
`abs(frontier - groundtruth) / abs(groundtruth) <= 0.10`; a cell passes only
when every gated metric passes.

| Metric | Value |
| --- | --- |
| TTFT | mean, request arrival to prefill completion (`request_arrival_to_prefill_completion`) |
| TPOT | mean over requests with `request_num_decode_tokens > 1` |
| request E2E | mean |
| request throughput | formal request count over the formal window |
| token throughput | `sum(prefill_tokens + decode_tokens)` over the formal window |

The formal window is `last_formal_completion_s - first_formal_arrival_s` on each
side's own clock. Latency rows record `metric_statistic=mean`; median and p95
are diagnostic. A reference value of zero, or a missing, zero, or non-finite
denominator, is `INSUFFICIENT_EVIDENCE`.

Admission comes before any number: each side holds every formal request exactly
once, Frontier rows carry the reference token counts, and
`request_e2e_time_ms = 1000 * (completion_s - arrival_s)` within 1e-3 ms with
`0 <= ttft_ms <= request_e2e_time_ms`. A failed admission makes every metric
`INSUFFICIENT_EVIDENCE`.

Gate kinds (definitions in the harness README): select by validated metric
semantics and evidence in `engine_binding`, not architecture labels alone.
The specialized vLLM studies below do not automatically apply to other engines.

- `single` (ordinary pooled comparison): one Frontier run against every
  admissible clean run of the cell, numerators and denominators summed over the runs.
- `pd_s33` (the existing vLLM PDD study): `single` with each reference run's
  TTFT reduced by that run's minimum endpoint offset (proxy hop plus prefill
  API-server time), the declared bound of the S33 abstraction.
- `dp_pp_pairs` (the existing vLLM DP x PP study): each request is labeled
  with the DP-lane mode of its window (`in_phase`, `anti_phase`, `split`,
  `double_split`); every reference run pairs with every Frontier ensemble member; a mode pool is gated once 20
  distinct requests reach it; no gated pool means `INSUFFICIENT_EVIDENCE`.

Coverage: a cell whose formal batches leave the profiled ranges or grid spacing
is `outside_profile`; its numbers are reported and are not in-domain evidence.

Reporting: every metric shows both sides' absolute values with units (ms,
req/s, tok/s), with the relative error beside them. Instrumented, aggregate,
system, plot, and trace files never close the gate.

### MoE routing gate

The selected engine's routing is the reference. Record both sides' router
semantics, scoring function, `top_k`, weight renormalization, expert count, EP
participant scope, and representative expert-load distribution. Choose the
comparison criterion before observing the gap. Source inspection and diagnostic
samples establish the mapping; ordinary accuracy claims still use clean runs.

A deterministic forced-routing case can compare per-request, logical-batch,
layer, and phase counts exactly:

```text
count_abs_error = sum_i(abs(frontier_count_i - groundtruth_count_i))
```

The existing vLLM round-robin profile requires the same semantics and scope
with `count_abs_error = 0`. A native or stochastic router instead needs a
Frontier model and evidence that represent its distribution and dependencies;
matching engine names or replacing it with a balanced split proves nothing.
Record `MATCH` only when the declared criterion is met, `MISMATCH` for a
supported comparison that fails, and `UNSET` for missing evidence. Unsupported
routing remains an explicit semantic limitation. These statuses feed the
existing routing gate without changing its arithmetic.

## Operator Comparability Predicate

Operator gaps are comparable only when the evidence contains the first formal
scheduler batch, at least three distinct aligned logical batches, and every
active phase, with the same request, batch, phase, lane, rank, and
measurement-family identity on both sides. Missing coverage sets
`status=INSUFFICIENT_EVIDENCE` and routes to the owning supplement. The
`lane_registry` in `registry.json` is the only map from a residual class to a
repair lane.

## Error-Source Classes

Every analysis assigns each attributed error to one class. The class decides
the owner.

| Class | Meaning | Owner |
| --- | --- | --- |
| `simulation_logic` | Frontier's scheduling, execution, or pricing logic disagrees with the selected engine | a Frontier repair entry |
| `measurement_input` | a profiling table, CPU-overhead table, kernel gap, collective fit, or coverage gap | a supplement entry and a new table generation |
| `groundtruth_defect` | reference collection fails to represent the selected build or its warm steady state, or a harness/probe corrupts its measurements | a reference fix and new run generation; change build/tool identity only when that build/tool changes |
| `reference_instability` | between-run variation whose cause and representative effect need evidence | repeated clean collection and causal attribution under Calibration Objective; exclusion only for evidenced invalid runs |
| `declared_abstraction` | a boundary term Frontier deliberately leaves out, with a recorded bound | the gate kind that removes the bound |

A warmup-only collection defect belongs to `ground_truth_tool`: revise its
collection policy and start a new run generation, retaining the serving build
identity. Record a new collector/tool identity only if its implementation
changes; an actual serving-build change records a new build identity. This is
an owner within `groundtruth_defect`, not an additional error-source class.

Value rule: the highest-value corrections fix `simulation_logic`. Measurement
engineering (re-collecting or tuning tables, probe rules, pooling rules) runs
only when the user approves it for the task. A Frontier change never models a
`groundtruth_defect`. A change that adds configuration surface needs a measured
fidelity effect and a benefit that justifies its implementation and maintenance
cost for the case. An intentional engine or fork behavior belongs to the
modeling target even when it differs from another engine.

## Decisions and Approval

A decision changes topology, scheduling, metric meaning, gate rules, the
ground-truth build identity, a table generation, or Frontier numbers. For each
one:

1. Gather the evidence and prepare options, each with its cost and expected
   effect, the recommended option first.
2. Put one focused question to the requesting user (the `grill-me` workflow
   when the user runs it).
3. Record a decision ID, the question, the options, the user's verbatim answer,
   and the UTC date in the task's `requirements.md`; cite the ID in the
   manifest, the ticket, and the commit message.

An unresolved decision is `BLOCKED`. When a result departs from what the chosen
option predicted, stop and ask again before building on it.

A code change follows `repair-approval.md`: one ticket and one scoped commit per
correction, after a complete analysis and the user's decision. A standing
authorization from the user (for example, GPU jobs already planned in the
current tickets) is recorded once in `requirements.md` and cited in each
receipt it covers; it never covers a new ticket, a gate-rule change, or larger
resources.

## Supplement Integrity

A supplement publishes a new table set; it never edits a published one.
`tests/comparison/calibration/merge_profiling_csvs.py` writes into a fresh
output directory, retains every base row, records repeated identities instead of
appending them, rejects conflicting identities and rows without measurements,
checks every source hash, and writes `merge_receipt.json` and `COMPLETE` last.
A table set's name carries its purpose, device, scope, and date plus a letter
(for example, `e2e-h200-t42-dense-20261007a`). The owning
analysis regenerates its status after a supplement; a new table generation
gets its own Frontier settings file, fresh Frontier runs, and republished
readings.

## Command Receipts

`exec capture: UNKNOWN`. No platform hook records every shell call, so each
entry writes a receipt for each command it launches: the command, working
directory, declared environment, UTC start and end, exit code, Frontier commit
and dirty state, ground-truth build identity, and expected output paths. The
receipt covers that entry's own calls only.

## Compatibility

Older receipts may carry `mutation_marker`; it means the same as
`change_marker`, the identifier that ties an approved code-change intent to its
result receipt. New records use `change_marker` and plain "code change"
wording.
