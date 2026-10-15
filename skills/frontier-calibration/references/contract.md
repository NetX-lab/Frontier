# Frontier Calibration v2 Contract

This is the shared contract for every entry. `registry.json` is the entry
catalog, `call-graph.md` owns execution order, and this file owns the evidence,
decision, and approval rules. Gate arithmetic lives in the repository harness,
`tests/comparison/e2e_gate/` (`README.md`, `gate.py`); this contract summarizes
it and never overrides it.

## Case Layout

A calibration task keeps one artifacts bundle, the directory every harness
receives as `--artifacts-root`. The bundle is task-local and stays out of the
repository; the repository pins its files by SHA-256 in
`tests/comparison/e2e_gate/artifacts.json` and `cases/scenario_matrix.json`.

```text
<artifacts root>/
  <case_id>/
    manifest.yaml                      case binding (see Case Binding)
    case_init.md                       purpose and the decisions that created the case
    inputs/
      engine*.json                     vLLM engine files, one per mode when they differ
      frontier_<generation>.json       Frontier settings, one file per table generation
      workload.json, trace/, cells/<cell>/trace/
      cpu_overhead*/                   published CPU-overhead tables with provenance.json
      groundtruth_overlay.patch, fork_changed_files.txt
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
- the ground-truth checkout tuple (next section);
- model and dtype, architecture, the vLLM parallel layout and its Frontier
  mapping (repository `AGENTS.md`, "vLLM Parallel Semantics and Frontier
  Mapping"), scheduler limits (`max_num_seqs`, `max_num_batched_tokens`,
  `max_model_len`, block size), chunked prefill, prefix caching, graph mode and
  capture sizes, attention backend, KV budget and its source, KV transfer,
  routing status and method, weight mode, measurement families;
- the Frontier settings file of the current table generation;
- cells: trace path, `formal_request_ids` and `warmup_request_ids` (from the
  trace's `request_ids.json`), load level, ground-truth run tags;
- the decision IDs that set any of the above.

A field the user has not decided is written `BLOCKED` or `PENDING`. Runners
stop on such a field; they never substitute a default. The same holds for a
null value in a settings file.

## Ground-Truth Checkout

The ground truth is vLLM-BS, `https://github.com/fwyc0573/vLLM-BS.git`, branch
`feature/frontier-comparison-instrumentation`, ref
`refs/heads/feature/frontier-comparison-instrumentation`, a Python-only
instrumentation fork of vLLM v0.10.2. The manifest and each run manifest record
`groundtruth_checkout_path`, `groundtruth_branch`, `groundtruth_ref`,
`groundtruth_commit`, `groundtruth_tree_dirty`, `groundtruth_remote_url`,
`groundtruth_remote_tip`, `groundtruth_diff_artifact`,
`groundtruth_dirty_patch_sha256`, and `groundtruth_overlay_patch_applied`.

- The tree is clean at the recorded commit (`groundtruth_tree_dirty=false`).
- The commit may be local and ahead of `groundtruth_remote_tip`. Record the
  ahead count, save `groundtruth_instrumentation_<sha>.diff` and the changed
  file list in the bundle, and record the diff SHA-256. Pushing the fork is an
  external action that needs the user's approval.
- Probes are off by default, so a clean run on a probe commit measures the same
  serving path. All modes of one case use one commit; a new commit starts a new
  run generation.
- Before claiming vLLM behavior, check the fork diff against upstream v0.10.2:
  the fork changes some scheduling details.

A detached, dirty, wrong-branch, or unrecorded checkout is `FAIL`.

## Run Modes

`tests/comparison/dp_placement_pp/vllm_replay.py` (`MODES`, `server_env`) is
the source of truth for each mode's switches. Modes are mutually exclusive in
one server run.

| Mode | Records | Use |
| --- | --- | --- |
| `clean` | request metrics only | the only rows that decide the E2E gate |
| `schedule_timing` | scheduler step log | step membership, queueing, step periods |
| `op_timing` | CUDA-event operator scopes, batch and PP-boundary logs | operator gap; launch gaps inside scopes included |
| `kernel_timing` | the same with device sync per scope | kernel-only scope times; its step times are distorted and never read |
| `instrumented` | MoE routing records | routing audit; not operator timing |
| `cpu` | CPU probe (engine and per DP/PP stage logs) | host-time attribution and CPU-overhead tables |
| `device_timeline` | CPU probe and torch profiler traces | device busy and idle per forward, collective waits |
| `kv_save_timing`, `kv_save_device_ids` | CPU probe and KV-save probe | PD KV-send diagnosis |

Collection rules:

- At least three warmups, and warmups cover every prompt-length class of the
  formal trace. The formal window starts after the warmups have drained.
- Eager MoE runs follow an unmeasured warm pass over the cell's prompt lengths
  in the same job, so compile caches are warm before measurement.
- The clean run and the `cpu` / `schedule_timing` runs that feed tables or
  analysis run in one job on one node. The KV block count comes from that cell's
  own clean run.
- MoE references run with vLLM-BS round-robin routing (engine key
  `moe_uniform_routing`), which matches Frontier's `balanced` split.
- Jobs with static ports run one at a time.

### Weight policy

Clean ground truth uses `groundtruth_weight_mode=dummy` and
`real_weight_download=false`. The gate compares runtime timing and throughput,
never generated text or accuracy. Dummy loading keeps the architecture, dtype,
request shapes, scheduler, parallel domains, backend, graph mode, and KV budget.
A loading mode that changes the measured path makes the run diagnostic-only.

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

The gate compares clean Frontier rows with clean vLLM rows for the formal
requests of one cell. Each metric is decided on its own and passes when
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
are diagnostic. A vLLM value of zero, or a missing, zero, or non-finite
denominator, is `INSUFFICIENT_EVIDENCE`.

Admission comes before any number: each side holds every formal request exactly
once, Frontier rows carry the vLLM token counts, and
`request_e2e_time_ms = 1000 * (completion_s - arrival_s)` within 1e-3 ms with
`0 <= ttft_ms <= request_e2e_time_ms`. A failed admission makes every metric
`INSUFFICIENT_EVIDENCE`.

Gate kinds (definitions in the harness README):

- `single` (co-location): one Frontier run against every admissible clean run
  of the cell, numerators and denominators summed over the runs.
- `pd_s33` (PDD): `single` with each vLLM run's TTFT reduced by that run's
  minimum endpoint offset (proxy hop plus prefill API-server time), the
  declared bound of the S33 abstraction.
- `dp_pp_pairs` (DP x PP): each request is labeled with the DP-lane mode of its
  window (`in_phase`, `anti_phase`, `split`, `double_split`); every reference
  run pairs with every Frontier ensemble member; a mode pool is gated once 20
  distinct requests reach it; no gated pool means `INSUFFICIENT_EVIDENCE`.

Coverage: a cell whose formal batches leave the profiled ranges or grid spacing
is `outside_profile`; its numbers are reported and are not in-domain evidence.

Reporting: every metric shows both sides' absolute values with units (ms,
req/s, tok/s), with the relative error beside them. Instrumented, aggregate,
system, plot, and trace files never close the gate.

### MoE routing gate

Ground-truth routing is the reference. A MoE case records, for both sides, the
router path, scoring function, `top_k`, weight renormalization, expert count, EP
participant scope, and the load distribution. The current practice runs the
reference with round-robin routing and audits an `instrumented` run against
Frontier's `balanced` split per formal request, logical batch, layer, and
phase:

```text
count_abs_error = sum_i(abs(frontier_count_i - groundtruth_count_i))
```

`MATCH` needs the same router semantics and participant scope and
`count_abs_error = 0`. A routing path or scope difference is `MISMATCH` (cell
`FAIL`); missing records are `UNSET` (cell `INSUFFICIENT_EVIDENCE`). A native
router reference stays outside the gate until Frontier models its load
distribution and the user accepts the bound.

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
| `simulation_logic` | Frontier's scheduling, execution, or pricing logic disagrees with vLLM | a Frontier repair entry |
| `measurement_input` | a profiling table, CPU-overhead table, kernel gap, collective fit, or coverage gap | a supplement entry and a new table generation |
| `groundtruth_defect` | the fork, harness, or a probe departs from upstream vLLM or distorts timing | a ground-truth fix: new commit, new run generation |
| `reference_instability` | the reference varies by node, host state, DP x PP mode, API stalls, or compile caches | collection and admission rules, reruns, or a declared exclusion |
| `declared_abstraction` | a boundary term Frontier deliberately leaves out, with a recorded bound | the gate kind that removes the bound |

Value rule: the highest-value corrections fix `simulation_logic`. Measurement
engineering (re-collecting or tuning tables, probe rules, pooling rules) runs
only when the user approves it for the task. A Frontier change never models a
`groundtruth_defect`. A change that adds configuration surface needs a measured
fidelity effect; the user has declined knobs worth less than 0.5% of a gate
metric.

## Decisions and Approval

A decision changes topology, scheduling, metric meaning, gate rules, the
ground-truth commit, a table generation, or Frontier numbers. For each one:

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
and dirty state, ground-truth commit, and expected output paths. The receipt
covers that entry's own calls only.

## Compatibility

Older receipts may carry `mutation_marker`; it means the same as
`change_marker`, the identifier that ties an approved code-change intent to its
result receipt. New records use `change_marker` and plain "code change"
wording.
