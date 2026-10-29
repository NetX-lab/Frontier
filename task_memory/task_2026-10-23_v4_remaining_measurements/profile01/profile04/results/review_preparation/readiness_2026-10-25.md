## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Prepared R6/R7 evidence paths, remaining checks and limits from retained local records only. |
| 2026-10-25 | Updated the failed fresh group pilot, conditional-rank block and segment-only allocation; preserved preflight0/8 as history. |

# R6/R7 evidence readiness

Snapshot: the 2026-10-25 Hong Kong continuation supersedes historical sections; retained receipts are dated 2026-10-24 UTC.
Current frontier: fresh group pilot FAILED, child124/driver1/60.32408195s, actual30GiB, no final JSON/useful-step metrics; affected1/8, remaining7 NOT_RUN/BLOCKED, R5 still24/32.
The new `p0_cpu_loop/group_metrics_tail02/primary/allocation.json` permits only one tiny control and one segment diagnostic; affected and complete-control ceilings are zero.
Task: `task_memory/task_2026-10-23_v4_remaining_measurements/`; results below are relative to its `profile01/profile04/results/` directory.
Inspection used local Markdown, compact JSON, source text and file existence only. Large audit JSON, raw traces and kernel indices were not loaded; no tests, drivers, ranks, Git, network, platform or GPU operations were executed.

## Original criteria and current boundary

The task `plan.md`, “Current profile04 verification”, defines R6 as independent phase review plus primary reproduction, then R7 as limited evidence acceptance/archive.
The current continuation requires complete actual 32-rank outputs -> R5 acceptance -> R6 full review/primary synthesis -> R7; R6 evidence preparation can proceed now.
`requirements.md`, D-V4-PROFILE04-VERIFY, requires 432 unique successful canonical client rows/three repetitions, four actual hosts/32 H800 UUIDs, 32 published formal metadata records/raw traces, at least three useful kernel-bearing steps per role/lane, no unintended loss, direct raw DP/stream evidence and post-stop evidence.
Optional text statistics tables must remain absent; missing evidence retains FAIL/UNKNOWN/INSUFFICIENT_EVIDENCE. Captured ownership, GPU dependency/cost additivity and reference admission are separate.

## Existing paths and accepted work to reuse

| Evidence | Exact existing path(s) | Retained result / inspection limit |
| --- | --- | --- |
| Instructions/current route | Workspace `AGENTS.md`; task `requirements.md`, `plan.md`, `progress.md`, `summary.md`; `profile01/profile04/requirements.md` and `plan.md` | Current continuation and original R6/R7/runtime criteria inspected; older pending entries are historical. |
| R1/R2 execution and input closure | `terminal_platform.json`, `platform_group_binding.json`, `download_receipt.json`, `download_input_check.json`, `host_package_check.json` | Job exp-1024-172225-512616, four Succeeded/Completed/exit0/restart0 hosts; 202 files/6,024,578,382 bytes; 45 sealed inputs. Receipt observations, not fresh platform/raw hashing. |
| R3/R4 accepted decision | `run_metadata_acceptance.json`, `review_2026-10-24_run_metadata.md`, `root_cpu_replay_01.json`, `root_cpu_replay_02_metadata_p2.json` | 432 successes; primary 193 checks/99 bindings; corrected 32-rank replay and P2 closure already accepted. Reuse these receipts, not the superseded pre-fix metadata report. |
| R4 detailed evidence references | `metadata_audit.json`, `metadata_p2/semantic_equality.json` | Acceptance binds corrected metadata SHA256 `8ba69802bdbef62f1dd8cd6e5c979cbf4a60f1a1eb317c84f060b5399712a587`; 32 closed windows, zero drops/errors, 80 direct decode joins/1320 sampling rows. Large audit JSON not loaded here. |
| R5 retained delivery | `test_report_2026-10-24_trace.md`, `trace_command.md`, `trace_checkpoint.md`; referenced `trace_audit.json` | Report/delivery bind SHA256 `78376357cec6b374f2343e73bac6e1da1bb58c6dcc296eef34054d917783d5e0`; 24 complete/8 original timeouts. Large aggregate not loaded. |
| Accepted metrics correction | `p0_cpu_loop/metrics_correction02/primary/source_acceptance.json`, `complete_controls.json`; sibling `review_spec.md`, `review_standards.md` | Owner/primary each 9 tests/42 checks; two exact complete controls, 142,567/570,964 index rows and 361/240 metric groups; both reviews No findings. Unit/control allowances spent. |
| Historical source/affected preflight | `p0_cpu_loop/metrics_correction02/primary/affected_allocation.json`, `allocation.json`, `resume_source_binding_2026-10-25.json` | Source commit `8b4d6993d8daf738352c117a091cb4f801c0a429`; eight pinned input triplets, 60s/32,212,254,720 bytes/max2; source/prelaunch0/8 is an earlier snapshot, superseded by current affected1/8. |
| Current fresh pilot failure | `p0_cpu_loop/metrics_correction02/primary/pilot_failure.json`, `p0_cpu_loop/metrics_correction02/test_report_2026-10-25_group_metrics_failed_pilot.md` | Child124/driver1/60.32408195s; actual30GiB; clean index1,215,919 rows only, no final JSON/useful metrics; affected1/8, remaining7 NOT_RUN/BLOCKED. Source/exact-control acceptance remains unchanged. |
| Current segment-only allocation | `p0_cpu_loop/group_metrics_tail02/primary/allocation.json` | D-V4-P0-GROUP-METRICS-SEGMENTS: one tiny/one diagnostic, 60s/actual30GiB/max2, selected concurrency1/no retry; zero affected/complete controls. Further correction/controls/affected work needs a separate measured allocation. |
| Separate scope acceptance | Task `profile01/scope_lifetime/results/primary_acceptance.json` | Ordinary sealed 1024-element pointwise P1-P6 only; GPU1/1 spent; control topology/output identity only. Native serving ownership/dependency/timing/cost/reference excluded. |

Trace scratch: `/data/ycfeng/tmp/glm53_dsv4/profile04_trace_of9lmcmi/`.
Inspected there: `freeze.json`, `native_shape_control.json`, `delivery_verification.json`, `launch_coordinator_binding.json`, `coordinator.py`, all 32 `*.status.json` and `*.scope.json`.
Observed: `prefill_dp1_rank{0..7}`, `decode_dp0_rank{0..7}`, `decode_dp1_rank{0..7}` each have exit0, final `.json` and `.kernels.jsonl.gz`; all eight `prefill_dp0_rank{0..7}` have exit124/no final JSON; six partial indices exist. Every recorded memory limit is 32,212,254,720 bytes.
The retained trace report states 7,758,672 kernels/102,248 exact annotations and no loss for the completed 24; useful counts per rank are P1=4, D0=61, D1=61. P0 remains UNKNOWN here.
Inspected frozen source: `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py` and workspace `tests/analysis/v4_trace_audit_group_metrics.py`; the frozen reader is referenced by `freeze.json` and affected source pins.
Local input root: `/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04/`; its inventory is `/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/inventory.json`. These paths were obtained from receipts, without reading raw inputs.

## R6 preparation now; synthesis after complete R5

Observed in the new failure receipt: kernel_pass_done55.532s, metrics batchstart55.992s/end absent; zero recorded gen2callbacks after batchstart. Global order, group sort and union durations remain UNKNOWN; this censored interval does not identify the slow segment or establish sufficient overall speed.

1. Reuse R3/R4 acceptance, corrected replay/P2 closure, frozen-source pins and accepted metrics reviews/controls. Prepare the review checklist and evidence map now; do not rerun accepted work or import spent drivers.
2. Preserve the failed fresh prefix `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_affected_20261025_01/prefill_dp0_rank0` and hold all seven conditional ranks. Primary next reviews the segment source/AST and exact tiny evidence, then uses only the separately allocated one diagnostic; do not retry this pilot or borrow the seven blocked allowances.
3. Future R5 closure still requires each P0 rank to have complete final JSON/clean gzip EOF/index, exact input/source/host/header/filename/rank/generation joins, zero missing/duplicate annotations or unexpected drops/errors, and at least three useful steps. The segment diagnostic, even with complete output, is not an affected-rank acceptance run.
4. Useful means first-repeat formal context, canonical request membership, positive local tokens, retained annotations and correlated kernels. Do not inherit the old partial run's 39 useful steps or treat a complete index/exit0 as sufficient.
5. Bind actual count/sum/union/hierarchy/leaf/per-stream fields, raw IDs and index row counts. Accepted exact controls establish algorithm equivalence; absent original P0 final JSON supplies no complete metrics baseline for an equality claim.
6. After measured diagnosis and separately authorized work supply eight complete P0 outputs, build a fresh 32-rank evidence manifest from those outputs plus the preserved 24, keyed by role/lane/rank. Bind commands, outputs, indices, cgroups, inventory/source hashes; preserve original `trace_audit.json`, timeouts and both failed pilots.
7. Complete the remaining independent full-trace phase review and primary artifact reproduction/synthesis against that manifest. Existing R3/R4 reviews/replays and source exact controls support their own slices; none alone closes R6 full-trace review. This report supplies no additional execution allowance.

## Limits that remain through R7

- R4 has 16 raw DP groups keyed by generation/role/ordered members, only 80 direct decode joins and asymmetric partitions; whole-window DP alignment remains false. No counter offsets, nearest-time joins or cross-run remapping.
- The completed-24 trace report records 7,334,656 captured kernels with unknown per-op ownership and 7,152,256 kernels without a direct collector-stream match. Outer graph/scope correlation and nonadditive hierarchy sums cannot establish serving cost.
- P1 collector record/wait pairs prove preceding same-object/generation/source array order only. Decode has zero formal collector sync rows; empty pairs do not establish complete device dependencies. Cross-host clocks remain unaligned.
- Preserve 13 child SIGKILL escalations, UNKNOWN NET/IB/Socket and worker-binary rehash, preferred-versus-required affinity discrepancy, original profile03 failure and every P0 timeout/partial. Requested parent exit0 is not all-natural child exit.
- Scope P2 thread equality is observation; P3 freshness uses exact names/monotonic windows; P4 default keep_graph=False has no post-capture source dereference. Six unmatched-created causes remain UNKNOWN and host driver release UNRECORDED; control.verdict=UNVERIFIED stays unchanged.
- After complete R5 and R6, primary may record R7's limited diagnostic decision and archive exact evidence paths/hashes, review coverage, observed metrics and open gaps in new owned records/main task completion records. Complete parsing does not make every ownership/dependency predicate PASS.
- Native06 R4 remains the reference; full05/09/19 OPEN, GLM PAUSED and CPU-overhead-last unchanged. No native serving ownership, timing, cost, clean E2E or reference upgrade follows.

## Bounded check-progress self-check

PASS: this sole readiness file preserves inspected evidence and updates the requested frontier. Fresh pilot FAILED; segment diagnosis is the current allocated work; seven ranks/R6 final/R7 remain BLOCKED. D1-D7 CLEAR; S1-S6 show no bounded-sidecar defect; Branch A. Historical source0/8 is not live status.
Next route: segment source/AST + one tiny -> one segment diagnostic -> measured findings/separate correction, control and affected allocations -> complete32 evidence/R5 -> independent R6 review/synthesis -> limited R7 archive. This sidecar adds no execution, delegation or authority; primary owns integration and closure.
