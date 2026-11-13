# D23-ACTUAL-MAIN Handoff

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record implemented fixture slice; full subprocess acceptance pending. |
| 2026-10-23 | Exact compute coverage repaired; actual04 pending on retained output. |
| 2026-10-22 | Archive paired actual04 core ledger blocker and verified integration slice. |
| 2026-10-22 | Parent accepts root/owner repair26db654 and launches paired actual05 with existing inputs. |
| 2026-10-22 | Final root/owner clean legacy comparison77/77 identical. |
| 2026-10-22 | Actual05 both exceed2400s after partial decode; CPU bottleneck diagnosis is active. |
| 2026-10-22 | Both CPU reuse corrections accepted; merged224 PASS and frozen-checkout actual06/candidate77 launched. |
| 2026-10-22 | CPU reuse strict comparison passes77/77; actual06 terminal acceptance remains pending. |
| 2026-10-22 | Actual06 times out after357/362 tokens; record measured speedup and resume whole-iteration profiling. |
| 2026-10-22 | Confirm P-side repeated prompt work and separate SWA payload arithmetic; semantic corrections await decision. |

## Status

BLOCKED BY RUNTIME: actual06 offline/online both exceed2400s on immutable7e8bea3.
They reach357/362 emitted tokens, versus actual05's42/41. Initialization falls
from about1000s to338/340s; median decode iteration falls to5.165/5.069s. The
87/85 common prior batch events preserve exact simulated times and token fields.
Terminal metrics and full6144/1024 acceptance are still absent. Whole-iteration
CPU profiling is active. Both reuse corrections remain source-accepted, with224
combined checks and77/77 exact legacy comparison passing. See
`actual06_timeout_receipt.json` and the latest test-report section.

Native-aligned acceptance also has a confirmed semantic blocker: current P6144
plus D16 repeats16 prompt-token computations, whereas native is P6128 plus D16.
Parent verified frozen source pins, native forward arrays and both current
partial-run P records. A separate847,872-byte/rank payload difference closes
numerically to one extra SWA page per69 layers. P lifecycle/new fixture and
native page-selection correction await their decision/proof; see
`prefill_census/report.md`. Neither issue is explained away by runtime speedup.

Historical attempts: shared fit generation02 is complete and retained. Actual01 rejected
mutually exclusive runner graph flags (fixed); actual02 rejected an uninitialized
local collective-sim dependency (pinned local initialization and build complete).
Actual03 offline/online both reachP layer3 EP wave and reject old compute fixture
coverage at6145 above6144; exact compute-only refresh is complete. Actual04
offline/online exit1 at1032.6621s/1032.4320s without timeout. Both reach P-stage
completion, where production metrics rejects `step5_layer_e2e` with KeyError.
No D scheduler records or terminal request/READ/final metrics exist. Fixture tests and
initialized profiles do not establish full-request acceptance or native accuracy.

## Deliverables

- `tests/integration/step5_measured_compute_fixture.py`: labelled synthetic CPU
  rows through existing typed producer/workload/fit/cache/manager/bundle paths.
  Exact92-layer coverage;1023 decode contexts6144..7166. Truthful MRv2 NONE1,
  PIECEWISE16/null-requests, FULL1/raw-null/uniform1 dummy inputs.
  Local1/16/6144 and aggregate2/32/6145 linear widths; routed rows retain exact
  balanced22-expert EP loads through the existing materializer. Compute-only
  refresh preserves attention CSVs/cache and backs up old compute CSVs.
- `tests/integration/run_step5_measured_main.py`: real `python -B -m frontier.main`,
  no simulator patching; native7168/8192/128,TP8/DP2/EP16/PP1 and graph policy.
  Preserves first-tail16/cached6128,3.5ms synthetic NIXL READ, event/source release
  and final positive finite metrics validation. Bounded2400s subprocess per mode,
  based on observed1068s original manager fit, not simulated time changes.
- `tests/integration/test_step5_measured_main.py`:8 CPU regressions, including
  honest physical17->24 padding without duplicated query/context/request rows,
  plus exact2/32/6145 aggregate and22-expert local EP materializer features;
  existing typed shared cache/Step lookup matches duration and count7 for both
  precision groups and retains missing7168 rejection.

## Source and limits

New worktree base d8e3e941, supplied c9e9388f correction merged unchanged.
No independent core edits, legacy matrix, native/platform operations or old runs.
Synthetic costs are not measurements/admitted profiles. Formal128-request/native
timing/accuracy acceptance remains outside this slice.

## Genuine producer API gap

`Step5ProfileInput.workload()` fixes physical_query_rows to valid token sum;
it cannot express additional **real** graph padding observed in a native batch.
The CPU fixture explicitly uses existing `Step5AttentionWorkload` for the intended
physical extent while preserving the real query/context arrays and typed row.
Native producer API/metadata binding remains a source-backed follow-up; no fake
additional request or query is introduced here.

## Retained evidence

- `/data/ycfeng/tmp/d23_main_fixture02/`: original94392/184 attention rows/cache,
  refreshed compute CSVs and preserved `*.before_mrv2_compute` sources.
- `/data/ycfeng/tmp/d23_main_actual03_{offline,online}/`: exact layer3 runtime
  rejection at6145; selected `actual03_failure_receipt.json` records log hashes.
- `/data/ycfeng/tmp/d23_main_actual04_{offline,online}/`: current bounded actual
  subprocess failures; `actual04_input_receipt.json` locks exact source/input hashes.
- `compute_refresh_receipt.json`: unchanged attention SHA256, new/old compute
  SHA256, manifest and physical source widths. No profile admission claim.
- `actual04_failure_receipt.json`: execution, scheduler census and precise absence
  of terminal artifacts; commands/log/config SHA256.
- `actual04_root_source_receipt.json`: cached P6144 typed source root.6244ms/count23,
  explicitly not a captured event cost.
- `core_blocker.md`: executable minimal CPU reproduction and source change request.
- `scoped_test_receipt.json`: eight scoped PASS,0 errors/failures/skips,5.843s;
  XML/log SHA256 retained.

## Parent-owned remaining acceptance

The root/owner projection repair is complete; see `root_metrics_review.md` for
the initial P1 and final independent acceptance. Fixed graph creation capacity
is integrated at32fb934. The generic mixed-precision warning was checked by
source and factor-injection controls in `quantization_warning.md`; it does not
rescale this typed Step path.

Build a short red-capable whole-iteration feedback loop to account for the
remaining5.1s decode cadence, then select a measured correction. Full
READ/readiness/source release,1024 outputs and final metrics remain unverified.
The new candidate77 matches the retained clean26db654 baseline exactly; see
`cpu_reuse_fidelity.json`. The execution bound remains2400s. Parent source/proof checks
are in `test_report_2026-10-22_cpu_reuse.md`; full acceptance is outstanding.
Actual05 phase evidence and hashes are in `actual05_timeout_receipt.json`.
The clean77 comparison is complete:
77/77 identical, including predictor cache names and clean provenance; see
`root_metrics_fidelity.json`. Native/profile/accuracy qualification and the real
additional-padding producer gap remain open.

## Commit and final scope

Implementation commit:`e54f9457aed7e31336cb6cb553aec016a737ae2a`.
Only the three authorized integration files changed since suppliedc9e9388f;
`committed_source_receipt.json` verifies committed runner/fixture SHA256 matches
the actual04 source exactly. Eight scoped checks passed before commit. The
archive commit records the demonstrated core blocker; it does not turn failed
actual CLI acceptance into completion. No duplicate fit, extra actual attempt,
production core edit, native operation or77matrix was performed afteractual04.
