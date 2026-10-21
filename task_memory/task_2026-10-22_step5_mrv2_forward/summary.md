# D23-MRV2-FORWARD Completion Archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive the scoped CPU implementation and parent integration contract. |
| 2026-10-22 | Archive P2 correction,489 affected passes and explicit request-capacity handoff. |
| 2026-10-22 | Record P2 source commit and clean-branch completion receipt. |
| 2026-10-22 | Primary verifies fixed-capacity binding with242 passing affected checks. |

## Outcome

Commit `3cad8dbe398304d798a323389f9e3d18500a42de` binds measured Step useful
and dummy lanes before any first predictor call. Base is immutable 5c2319cd;
branch `feat/step5-mrv2-forward`, worktree
`/data/ycfeng/Frontier/.worktrees/step5-mrv2-forward`. Inputs worktree stayed clean.
No GPU/native qualification or accuracy claim is made.

Final fixed-capacity source `ef476fe` has independent Standards/Spec
**No findings** and is integrated into `feat/step5-preview` at `32fb934`.
All16 changed core/test files match the reviewed bytes. Parent's clean legacy
matrix and fresh full CLI remain in progress; see the main task integration report.

P2 review correction (follow-up to record HEAD `d8e3e941`) removes illegal
uniform FULL captures above the existing V1 request capacity. With cap3 and
capture sizes[1,2,4,8], decode3+dummy1 now binds PIECEWISE4 under
`full_and_piecewise`, or NONE with local3/1 widths under `full_decode_only`.
Legal wider PIECEWISE prefill captures remain supported. Two real-engine
controls and four exact RED failures became6 PASS; the affected suites now
pass221 stage +268 measured =489 checks. No producer/full CLI/strict77 rerun.
Correction/tests/records commit:
`c9e9388f117c625c6d546d1d083f8aaf3f824f51`; post-commit branch is clean.
The following docs-only commit records that completed receipt.

## Deliverables

The fixed-capacity follow-up uses existing `_config.batch_size_cap` directly
at the agreement boundary. Native graph capacity is fixed at initialization;
current supported Step requests cannot use the thinking-dependent admission
override. Therefore no reachable current-workload arrival-order defect is
claimed. Primary inspected that constraint and verified242 stage/engine/public
predictor/Step structural-PD checks in33.92s after the interrupted owner handoff.
All pass without errors or skips; exact command/log/XML are in the test report.

- `frontier/scheduler/utils/forward_start_agreement.py:14,23,83`: final descriptor,
  min-mode/capture binding and existing-event hold/resume. Timing is external.
- `frontier/scheduler/replica_stage_scheduler/stage_execution_context.py:78-115,433-435`:
  shared readiness, single wake and group reset; existing owners remain canonical.
- `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py:51-72`:
  held batch reference and exactly-once continuation, no new queue.
- `frontier/events/replica_stage_schedule_event.py:81-125`: agreement before real
  P/D and dummy prediction. Existing local metadata is not readiness.
- `frontier/entities/batch.py:806-844`: bound mode/physical width applies to
  PREFILL too; original arrays and request ownership remain local.
- `frontier/config/graph_policy.py`, `config.py` and
  `vllm_v1_iteration_policy.py`: one policy table, FULL_AND_PIECEWISE and shared
  capture resolver; existing policy values retain behavior.
- `frontier/execution_time_predictor/base_execution_time_predictor.py:42`:
  measured-Step-only capability, dummy-cost/other families unaffected.
- `frontier/execution_time_predictor/step5_cost.py:23,62,126` and seven-line
  sklearn delegate: P now requires honest CUDA_EVENT attention as well as
  KERNEL_ONLY; dummy uses the shared MRv2 split/provenance; missing descriptor
  fails, no native relabel of isolated diagnostic rows.
- `frontier/scheduler/utils/ep_wave_inputs.py:74-98`: explicit native NONE keeps
  local widths; predictor-only aggregate carries summed physical rows.
- `tests/unit/test_step5_mrv2_forward.py`: 30 focused regressions, including
  actual event first call with shared fitted/bundled root/count lookup.
- Updated measured manager/public/union fixtures supply the newly required P
  CUDA_EVENT MRv2 dummy source; no validation is weakened.

## Validation

30 focused, 176 affected stage/width/engine, 268 measured, 467 producer and 55
DP-runtime/EP/placement tests pass. Original event source is RED in all 12
first-call cases. Broader selector failures match base exactly (7 cases/errors).
Commands, versions, detailed logs and XML are in
`test_report_2026-10-22_forward_agreement.md` and `/data/ycfeng/tmp/d23_forward_*`.

## Exact fresh CLI input requirements for parent

1. Set `decode_cuda_graph_mode=full_and_piecewise` and the actual native capture
   size list. Measured agreement intentionally rejects graph policy with a
   missing list. Keep existing PP1/no-prefix/no-MTP/topology restrictions.
2. Supply both `atten_kernel_only_input_file` and `atten_input_file` for **P and
   D**. Under graph policies, P can execute FULL dummy and PIECEWISE useful
   work; it cannot use a missing event path or borrow KERNEL_ONLY timings.
   Union requires compatible device/model/TP/block/cache provenance and includes
   both declared P/D sources in cache identity.
3. Every reached request-owned row retains native request-owned scope and local
   `query_lens/context_lens`, with final agreed `graph_mode`, `valid_tokens`,
   `padded_tokens`, real request count and actual physical table width. For FULL
   one-token prompt extensions, physical phase is decode while DES prefill
   ownership/credit remains unchanged. In asymmetric graph groups, pad **real
   and dummy** lanes to the common capture width; NONE keeps local width.
   Current simple `Step5ProfileInput` helper does not represent additional real
   graph padding; parent-owned integration fixture must record that truthful
   native physical extent instead of duplicating request query rows.
4. Reached dummy rows use `native_workload=mrv2_dp_dummy` and scope
   `mrv2_dp_dummy_attention_layer_without_tp_allreduce`, `has_attention_metadata=1`,
   full shared `MRV2_DUMMY_ROW_IDENTITY` and `dummy_descriptor`. Preserve
   `dummy_owned_requests=0`, PAD_SLOT_ID/cache semantics and the actual query
   split: NONE local 1/1; FULL width requests of one token with raw
   `max_query_len=null, uniform_token_count=1`; PIECEWISE raw `num_reqs=null`
   creates one actual local request spanning the agreed width and null query
   classification fields. `[1]` remains DES bookkeeping only.
5. Collect independent KERNEL_ONLY GPU-active rows/count plus eager gap coverage
   and CUDA_EVENT graph rows/count at actual reached dimensions; keep typed
   scopes, root/count pairing, SWA/DSA backends and real provenance. Existing
   all-layer attention/input-norm, FFN/merge/boundary, CPU/communication and NIXL
   requirements remain. CPU fixture numbers are not native input evidence.
6. Rebuild caches through the existing manager/publication path from fresh
   sources. Same-shape request-owned/dummy keys cannot substitute; isolated
   no-metadata diagnostic coverage is not native dummy coverage. Re-run the
   parent full actual-main CLI and strict77 clean matrix after consolidation.

Actual-main runner `tests/integration/run_step5_measured_main.py` and its owned
integration fixtures were not edited. G3 diagnostics/publication and legacy
selector fixture repairs remain parent-owned.

Task/session: `D23-MRV2-FORWARD` / `ses_ea08efc2cffe6Z1Rc7srkVdebh`.

## P2 deliverables and remaining acceptance

- `frontier/config/graph_policy.py:11-21`: resolve capture candidates through the
  existing policy fields; FULL token/request buckets must fit request capacity,
  then PIECEWISE candidates remain available. No new registry or policy flag.
- `frontier/scheduler/utils/forward_start_agreement.py:21,39-50,107-110`:
  required existing request capacity, local candidate selection before DP min,
  shared-width re-dispatch through the same resolver. Event/lifecycle and raw
  descriptor fields are unchanged.
- `tests/unit/test_step5_mrv2_forward.py:163-225,281-349`:40 arrival-order/role
  cases and6 actual V1 wakeup/arrival/first-call cases replace the former1 case.
  All earlier focused coverage remains; total75 PASS. Exact failure messages,
  commands, log/XML paths and counts are in the test report's P2 section.
- `tests/unit/test_step5_measured_public_predictor.py:126,148`: direct agreement
  calls now supply their existing scheduler config's `batch_size_cap`.
- Parent fresh fixture/direct agreement callers must add required keyword
  `request_capacity` from the existing role scheduler/config; do not infer it
  from captures. No parent fresh CLI fixture was read or edited here.
- Scope remains CPU forward correction only. Source re-review and merge are
  complete at32fb934; full CLI and strict77 remain unfinished acceptance.
