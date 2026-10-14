# Frontier Calibration v2 Contract

This reference is the shared contract for the active v2 skills. The machine
readable routing source is `registry.json`; this file defines the smallest
evidence and approval envelope that every entry follows.

## Case Binding

Every case has one `manifest.yaml` and one immutable `case_init` record. Before
any run, the manifest records:

- `case_id`, `run_generation`, `formal_request_ids`, and warmup request ids;
- the selected Frontier checkout and the selected ground-truth checkout tuple:
  path, ref, commit, tree status, remote URL/tip, diff artifact, and preserved
  patch/applied state;
- model, dtype, request trace, architecture, scheduler, parallel domains,
  backend, graph/eager mode, KV budget, and user-approved settings;
- the requesting user identity and any named human delegate allowed to review a
  code change;
- each active mode, its source path, producer profile, request-id namespace and
  encoding, artifact path, artifact timestamp, and fresh-file assertion;
- `groundtruth_weight_mode` and `real_weight_download`, which declare the
  groundtruth model-loading policy;
- `case_init.auto_recycle`, frozen once at initialization.

The minimum normative provenance envelope is therefore:
`case_id`, `run_generation`, `manifest_path`, `mode`, `selected_checkout`,
`source_path`, `producer_profile`, `request_id_namespace`,
`request_id_encoding`, `formal_request_ids`, `warmup_request_ids`,
`artifact_created_at`, `fresh_file`, and `status`. A row without this envelope
is evidence-incomplete even when its numeric fields look valid.

### Decisions and requested user

When a missing setting or material design choice changes topology, scheduling,
metric meaning, or repair scope, invoke `$grill-me` before running the case.
Record the resulting `decision_id`, question, evidence-backed options, answer,
`requesting_user`, `reviewer_identity`, and UTC decision time in the immutable
manifest. An unresolved decision is `BLOCKED`; the runner never fills the value
from an unstated default.

The human review receipt repeats `requesting_user` and `reviewer_identity`.
`reviewer_identity` is the requesting user or a named delegate listed in
`case_init` before the review. The repair process cannot issue its own human
review receipt.

### Selected ground-truth checkout

Every case selects a clean checkout whose branch is exactly
`feature/frontier-comparison-instrumentation` and whose ref is
`refs/heads/feature/frontier-comparison-instrumentation`. The expected remote is
`https://github.com/fwyc0573/vLLM-BS.git`. The manifest and each mode run
manifest record `groundtruth_checkout_path`, `groundtruth_branch`,
`groundtruth_ref`, `groundtruth_commit`, `groundtruth_tree_dirty`,
`groundtruth_remote_url`, `groundtruth_remote_tip`, `groundtruth_diff_artifact`,
`groundtruth_dirty_patch_sha256`, and `groundtruth_overlay_patch_applied`.
Require `groundtruth_tree_dirty=false` for a parity run and verify the recorded
branch, ref, commit, remote tip, and patch state before launch. A detached,
dirty, wrong-branch, or unproven checkout returns `FAIL` or
`INSUFFICIENT_EVIDENCE`; the original dirty checkout is never silently used.

## Run Modes

`groundtruth-run-check` validates mode predicates before `groundtruth-run`.
Clean ground truth uses at least three warmups, enables E2E runtime metrics, and
disables operator and CPU probes. Instrumented ground truth enables operator
diagnostics and disables E2E runtime metrics. CPU mode enables only the CPU probe
and disables operator instrumentation and E2E runtime metrics. Frontier uses the
clean-style E2E predicate. Each mode owns a disjoint run directory.

### vLLM groundtruth weight policy

Clean vLLM groundtruth uses `groundtruth_weight_mode=dummy` and
`real_weight_download=false` by default. The groundtruth comparison target is
runtime timing and throughput; it does not compare generated text, accuracy,
quality, or model intelligence. The run manifest records both fields and the
effective branch-specific loading command before launch. A missing, contradictory,
or unproven declaration returns `FAIL` or `INSUFFICIENT_EVIDENCE`.

Dummy loading preserves the model architecture, dtype, request shapes, scheduler,
parallel domains, backend, graph/eager mode, KV budget, and other settings that
determine the measured execution path. A dummy run whose loading mode changes the
measured execution path is diagnostic-only and cannot close the E2E gate. Formal
request rows from a valid dummy-weight run remain eligible for the runtime gate;
only setup rows explicitly marked as dummy-shape or dummy-request rows are
excluded with warmup rows.

## Status and Evidence

Use `analysis_state=COMPLETE|INCOMPLETE` for analysis completeness and
`status=PASS|FAIL|INSUFFICIENT_EVIDENCE|PENDING_HUMAN_REVIEW|BLOCKED` for the
result. Use `correction_state=pending|complete|not_applicable` for domains that
can feed the correction chain. `pending` means a scoped correction remains to be
reviewed; `complete` means an approved code change, isolated rerun, and fresh
analysis are complete; `not_applicable` means a complete `PASS` analysis found
no scoped code change to apply. `not_applicable` is a terminal no-change result,
not a repair authorization. The CPU lane treats `complete` and
`not_applicable` as completed workflow/operator correction attempts. Every
status has a non-empty `next_action`.

## E2E Gate

Compare only clean Frontier and clean vLLM normalized request rows. Join each
formal request exactly once and report TTFT, TPOT, request E2E, request
throughput, and token throughput independently. Latency uses the arithmetic mean
over eligible formal rows and records `metric_statistic=mean`; median and p95
remain diagnostic only. Use the shared TTFT semantic
`request_arrival_to_prefill_completion`.

Use the formal window
`last_formal_completion_s - first_formal_arrival_s` as the throughput duration.
Request throughput uses the formal request count as its numerator. Token
throughput uses `sum(prefill_tokens + decode_tokens)`, where the canonical row
fields are `request_num_prefill_tokens` and `request_num_decode_tokens`. TPOT is
eligible only when `request_num_decode_tokens > 1`; one-token rows enter
`excluded_count` and never create a zero denominator. Every metric records
`sample_count`, `eligible_count`, `excluded_count`, `numerator`, `denominator`,
`missing_request_ids`, and `duplicate_request_ids`.

Require every denominator to be positive and finite. A zero, negative,
non-finite, or missing denominator is `FAIL` or `INSUFFICIENT_EVIDENCE`, never a
passing zero error. Calculate
`abs(frontier - groundtruth) / abs(groundtruth)` and require each applicable
metric to be at most `0.10`. Instrumented, aggregate-only, system, plot, and
trace files never close this gate.

### MoE routing-distortion gate

MoE cases require a routing-alignment record before request-level parity is
accepted. The record treats ground-truth routing as authoritative and binds
each comparison to formal request ID, logical batch, layer, phase, and EP
scope. It records, for both Frontier and ground truth, the router implementation
path, scoring function, `top_k`, weight renormalization, expert count, EP
participant count, routing bias or grouped-top-k settings, and profile context.

Instrumented evidence includes per-expert token counts whenever the runtime
exposes them. The analysis reports total routed tokens, expert utilization,
load-imbalance CV, minimum and maximum load ratios, load entropy, Gini
coefficient, and the per-expert count-vector difference. The default distortion
is:

```text
count_abs_error = sum_i(abs(frontier_count_i - groundtruth_count_i))
count_relative_distortion = count_abs_error / max(sum_i(groundtruth_count_i), 1)
```

The analysis may use another explicitly justified vector metric, but it records
the formula, denominator, aggregation window, and every excluded or missing
record. A routing implementation or participant-scope mismatch is `MISMATCH`;
missing routing or profile evidence is `UNSET`. Either status blocks the E2E
numeric gate. A nonzero distortion can pass only when the case declares an
approved simulator abstraction and records its bound and rationale; otherwise
the case remains outside the gate even when TTFT relative error is at most
`0.10`.

## Operator Comparability Predicate

Operator gaps are comparable only when the evidence contains the first formal
scheduler batch, at least three distinct aligned logical batches, and every
active phase for the selected case. The same request, batch, phase, lane, rank,
and measurement-family identity is used on both sides. Missing first-batch,
three-batch, phase, or profiling coverage sets `status=INSUFFICIENT_EVIDENCE`
and routes to the owning supplement; it never authorizes a code change. The
machine-readable residual `lane_registry` in `registry.json` is the sole map
from a classifier label to a repair lane.

## Analysis Before Code Change

The required sequence is:

```text
analysis_state=COMPLETE and status=PASS
-> human review decision=PASS
   or case_init.auto_recycle=true with a complete auto receipt
-> record a plain code-change intent and change marker
-> apply one scoped Frontier code change
-> isolated rerun
-> fresh analysis and fresh approval before another change
```

Analysis, dispatch conversion, and data supplementation never authorize a code
change. Ground-truth core scheduling and inference semantics remain the
reference; repairs target Frontier unless an explicitly declared experiment
knob is used.

## Supplement Integrity

Supplement output is a complete merged dataset under the case `supplements/`
directory. Retain every base row, append accepted rows deterministically, keep
`dropped_rows=[]`, and record each added or deduplicated row with identity,
source path, source row number, fingerprint, full payload, and coverage basis.
Reject same-context contradictory values. Publish the completion marker only
after all payloads and their digests validate. The owning analysis regenerates
its status after the supplement; the supplement never edits an analysis status.

## Command Receipts

`exec capture: UNKNOWN`. The platform provides no proven hook that records every
shell call made outside the invoking skill. Each skill records an explicit
caller-written command receipt containing command, working directory,
environment declarations, start/end UTC, exit code, and declared artifact
paths. The receipt covers only commands issued by that skill and makes no claim
about unrelated external calls.

## Compatibility

The existing v1 artifact contract remains the compatibility reference during
staging. Existing machine fields such as `mutation_marker` may be read when a
legacy receipt requires them; new v2 prose calls the same value a `change_marker`
and uses plain “code change” language.
