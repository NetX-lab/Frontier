# D30 remaining adjacent regression scope

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Resolve both reviewer P2 findings in approved scope; record passing affected matrices and parent-only limits. |
| 2026-10-22 | YC approved D30-SCOPE-7-750 at b8be287; implementation proceeds independently of D29. |
| 2026-10-22 | Record reviewer consumer/fixture findings and complete seven-file proposal; no source/test edits authorized. |
| 2026-10-22 | Separate five new fixture failures from the baseline READ-byte mismatch. |

## Historical failures and cause

Candidate extra suite: `tests/unit/test_step5_measured_pd.py`,
`test_step5_structural_pd.py`, `test_step5_pd_placement.py`:68PASS/6FAIL.
Same files with only the scheduler restored in memory from `fa5bae2`:
73PASS/1FAIL. No disk source or parent file was restored/overwritten.

The original run identified five newly failing parameter instances that schedule
only the selected lane after each output. It did not prove a live-driver
deadlock. The independent reviewer subsequently identified one more decode
instance hidden behind a baseline byte assertion; the complete list is below.

| File/function | Parameter instance(s) | Decisive direct-lane call at candidate base |
| --- | --- | --- |
| `tests/unit/test_step5_measured_pd.py::test_step_nixl_cost_shape_uses_computed_frontier_without_changing_outputs` | `offline`, `online` (2) | `54-63`: only target scheduling/output callbacks |
| `tests/unit/test_step5_measured_pd.py::test_capacity_retry_rejects_changed_coverage_and_releases_reserved_owners` | `changed=False` (1) | `363-374`: nine selected-target iterations |
| `tests/unit/test_step5_measured_pd.py::test_measured_nixl_uses_covered_read_cost_and_prompt_tail_in_existing_owners` | no parameters (1) | `186-194`: eight decode iterations, hidden behind `159` |
| `tests/unit/test_step5_pd_placement.py::test_p_client_finishes_at_response_while_source_kv_is_retained` | no parameters (1) | `156-161`: two P-lane iterations |
| `tests/unit/test_step5_pd_placement.py::test_selected_d_lane_survives_capacity_retry_read_ready_and_decode` | no parameters (1) | `197-201`: nine selected-D iterations |

Raw fixture arithmetic: `2 + 1 + 1 + 1 + 1 = 6` instances in five functions
across two test modules. The `changed=True` capacity-error branch is not one of
these six, but its original assertions remain intact. Parent commit
`d581625ed8bbc238836a6376cb7690f0b490466e` changes the stale READ expectation
at measured-PD `159` from103051776 to102203904 and preserves1020 allocated
blocks. That exposes the later decode loop; do not undo or reprice this fix.

The Step finish boundary must not be bypassed to make these tests pass.
Replace the six direct-lane loops with a narrowly scoped CPU event queue that
dispatches the real all-lane arrival/schedule, dummy completion and peer-release
events. Preserve request ownership, transfer bytes/costs, READ/readiness,
context/token/output counts, P response/source-KV retention and retry assertions.
Pause at real transfer or output event boundaries where the original test needs
an intermediate observation. Do not invoke `_end_iteration`, override barrier
state, fabricate a peer result or suppress returned events.

## Reviewer receipt and live-driver evidence

Reviewer `ses_e9e108a90ffe6P2irzkxFevd06`, relayed by parent on2026-10-22:
two P2 findings are the verdict metadata rejection and the complete six-instance
fixture matrix. No specific live-driver deadlock was found. Child read-only
inspection confirms the existing production wake/release chain:

- `frontier/scheduler/utils/forward_start_agreement.py:117-130` wakes an idle
  missing lane with `ReplicaScheduleEvent`; queued lanes get stage events.
- `frontier/scheduler/utils/sync_entry.py:8-23` wakes missing idle engines at
  per-layer synchronization through the same replica event.
- `frontier/events/batch_stage_end_event.py:163-178` reschedules after the actual
  dummy completion callback; it does not inject a finish-consensus result.
- `frontier/events/replica_schedule_event.py:93-102` dispatches popped outputs
  and emits `ReplicaScheduleEvent` for each lane released by final consensus.

These source paths explain why bypassing the DES queue in a fixture is not
evidence that the real driver deadlocks. The records-only follow-up ran no
simulation; the approved continuation now drives the CPU DES lifecycle below.
Live native correctness remains unverified.

## Historical missing observer consumer boundary

Emitter: `tests/integration/step5_pd/r2_observation.py:1188-1210` produces
`native_dp_finish_check` and `native_dp_wave_counter_zero` as host scalar rows
without request, engine or TP identities. Consumer:
`tests/integration/step5_pd/r2_verdict.py:130-131,267-298` does not list these
events as metadata and rejects them at277-278 as Unknown event. Merely adding
them to requestless or worker/canonical sets would impose unrelated identities
or incorrectly admit them as request/transfer evidence.

Minimal proposed consumer change: classify both as diagnostic metadata, handle
them before normalization/request/engine/TP validation, retain each unmodified
raw row with its file/line location in instrumentation diagnostic details, and
exclude them from the request-bearing `records` returned by `load()`. Keep the
existing `require(records, ...)` at320 and native-transfer requirements at1399
unchanged: metadata alone must not satisfy evidence completeness. Do not create
a D30/native PASS criterion or alter R2 status/qualification or source pins.
Existing observer error events still take the error path before metadata.

Natural regression owner: the already changed
`tests/unit/test_step5_finish_sync.py:129-185` emits both events through the
normal `r2.instrument`/writer path. Extend it to feed those actual JSONL rows
through `r2_verdict.load()`/`judge()`, reusing the unchanged verdict test
`observations`/`write_stage` fixtures (`test_step5_native_pd_r2_verdict.py:34-143`).
Check P and D rows without request/engine/TP fields, exact raw retention, no
request-record pollution, unchanged valid-request verdict, and metadata-only
evidence remaining UNVERIFIED/no transfer or D30/native qualification. Keep
unknown-event and observer-error rejection. This reuse adds no eighth file.
The existing metadata parametrization in verdict tests at174-178 would also
exercise the two new metadata names when that unchanged suite is run later.

## Approved numerical expansion proposal

The previous six-file/<450-line proposal was incomplete and is superseded.
Complete D30 source/test file list and explicit budget:

| File | Current changed lines | Proposed additional budget | Result budget |
| --- | --- | --- | --- |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` |24 |0 |24 |
| `tests/integration/step5_pd/r2_observation.py` |39 |0 |39 |
| `tests/unit/test_step5_finish_sync.py` |261 |70: emitter-to-loader/judge regression |331 |
| `tests/unit/test_step5_native_pd_r2_observation.py` |3 |0 |3 |
| `tests/integration/step5_pd/r2_verdict.py` |0 |30: metadata classification/raw retention |30 |
| `tests/unit/test_step5_pd_placement.py` |0 |180: <=130 event/predictor/metric fixture support, <=50 two-loop assertions/callbacks |180 |
| `tests/unit/test_step5_measured_pd.py` |0 |120: reuse driver, stage-fixture input and three functions/four instances |120 |

Raw accounting: `4 current files + 3 additional files = 7 files`;
`319 insertions + 8 deletions = 327 current changed lines`;
`70 + 30 + 180 + 120 = 400 additional budgeted changed lines`;
`327 + 400 = 727`, with23 lines of headroom in a proposed **7 files / <=750
changed lines** D30 cap. This counts all existing candidate files, not just
follow-up edits; insertions and deletions both count. It is a design ceiling,
not measured future diff size. Parent must count the D29/D30 union of paths and
actual merged hunks; the parent-owned d581625 READ hunk remains owned/countable
there, not a replacement D30 fix.

The driver needs more than the earlier70-100-line estimate: placement `_scheduler`
currently passes `predictor=object()` (`28-51`), and measured `_FixtureCosts`
(`74-102`) implements only NIXL cost APIs, not stage/layer execution. Budget
therefore includes explicit test-only stage predictor inputs and metric callbacks
needed by the real event handlers, deterministic CPU timing and pause/resume
support. Reuse existing event/predictor types where compatible, but do not
assume current doubles can already drive the full lifecycle. None of these
fixture inputs is profiling evidence or a production timing change.

Current module sizes are1811,1258,261,1364,1473,205,420 lines respectively;
even gross budget additions stay below2000 lines per changed module. No extra
driver module, verdict-test file, production scheduler, CLI flag or new ticket
is proposed. If the complete event fixture cannot fit the stated budgets,
stop and return measured accounting rather than hiding files or bypassing events.

Historical status at0a0b9fe: numerical expansion was pending approval. No production/test edit, broader
test run, profiling, child agent or platform operation occurred in this
follow-up. Candidate `cf1a53f` remains unmerged and not merge-ready; the parent
alone combines this proposal with D29 for one approval request. Main D30
authority/spec is parent `9ed57630f2f564498f9706c7a5af0243928d1370`; no root or
parent files were overwritten and no D29 native design was investigated.

Current authority: b8be287 and `repairs/human_review_receipt_D30-SCOPE-7-750_2026-10-22.json`
record YC's direct `批准，继续`. Seven files/750 added+deleted lines against
b8be287, including cf1a53f, are approved. The D29 approval dependency is retired.
This task now implements the recorded proposal; native acceptance stays open.

Evidence: `/data/ycfeng/tmp/d30_finish_pd_adjacent01.{log,xml}` and
`/data/ycfeng/tmp/d30_finish_pd_baseline01.{log,xml}`.

## Approved implementation resolution

Both reviewer P2 issues are resolved at the source/CPU acceptance level:

- `r2_verdict.py:130-132,268-271` classifies both counters as diagnostic metadata
  before normalization and identity validation. It retains each untouched raw
  row and JSONL file/line in `instrumentation.details['diagnostic_metadata']`,
  never request/transfer records. Real emitter-to-load/judge P/D regression
  RED2FAIL to GREEN2PASS; valid request verdicts stay unchanged, metadata alone
  stays UNVERIFIED, unknown/observer errors retain rejection.
- The five functions/six instances use `_PDExecution` in existing placement
  tests, dispatched by the production event priority. Real stage arrivals,
  schedules, dummy ends and final-consensus peer events are queued and processed.
  Retry wake/output events are retained, not recreated; existing assertions for
  request/source/target ownership, READ102203904 bytes/1020 blocks, readiness,
  context/token output and retry faults remain. No direct `_end_iteration`,
  barrier write or fabricated peer return exists in either repaired module.
- `_StageCosts` supplies existing typed per-layer execution with registered
  attention identity and structured dispatch/combine times. These fixed CPU
  fixture inputs are not profiling or fitted latency evidence. Existing actual
  EP workloads are materialized by their unchanged owners; no workload,
  distribution, routing source or measured READ price is changed.

Final affected CPU PD matrix74PASS297.53s includes all six formerly failing
instances and unchanged capacity `changed=True`. Boundary/observer/legacy matrix
32PASS83.14s; R2 observer/stage/census/verdict388PASS10.06s; verdict rerun260PASS8.40s.
The final exact-peer-ID assertion rerun7PASS203.78s confirms all six repaired
instances plus unchanged fault branch. Complete source/test scope is
7files/528insertions+72deletions=600lines against b8be287, leaving150 lines
under the approved cap. Numerical expansion is no longer a blocker; D30 does
not wait for D29. Both documented source/CPU P2 findings are resolved.

Remaining acceptance is parent-owned: independent review, common integration,
strict77, fresh stage/package/pins and live all-DP counter completeness, first
idle boundary, active-peer continuation and wave reset. Metadata records do not
qualify any of these. No GPU/platform/CLI/performance action was performed.
