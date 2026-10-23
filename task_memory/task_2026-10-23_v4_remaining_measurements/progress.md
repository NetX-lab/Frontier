# Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded approval and began CPU audit/design; previous native reference evidence verified intact. |
| 2026-10-23 | E/F/G1 accepted at merged732406aa; H launch increment authorized and implementation resumed. |
| 2026-10-23 | Closed five H findings, integrated64f3c950, verified merged suites/input/matrix binding, and finalized the G2 handoff. |
| 2026-10-23 | YC resumed work locally from77668066; G2 window/API design and real replay integration are active. |

## Current execution — G2 resumed

- completed: H closure archived at77668066; main tracked worktree clean on restart, unrelated `.pi/` and `.cortexkit/` retained.
- in-progress G2a: root inspects the real replay path while W128-G2-WINDOW-DESIGN independently inspects pinned native profile API/iteration/window behavior. The H implementation owner retains ownership of driver/collector edits after root records the design and allocation.
- pending G2b/c/d: implement and verify the selected window/retention mechanism, independent reviews plus merged checks, then a concrete GPU request. No new GPU approval or runtime qualification is inferred.
- scheduling correction: the handoff below is the completed77668066 snapshot. YC explicitly chose to continue in this session; its technical boundaries and unfinished G2 work still apply.

## Current handoff — 2026-10-23

- completed A/B/C/D/E/F/G1/H: H source `f8621a8258ffa445abfb51f5ea73258ad271494e` is integrated on `feat/glm53-dsv4-vllm028` at `64f3c9509b09279f321d237e3a95cbe904c6dd22`. Both independent re-reviews closed all five original issues with no new findings. Scope: five non-core files,880/987changed lines. Root acceptance: `collector_primary/primary_acceptance.json`, status `ACCEPTED_CPU_COLLECTOR_COMPONENT`.
- completed final-source and merged verification: collector28, inventory28, V2inventory45, R4reader135, nativeworker16, DPharness75, P/Dharness68, manifest14 PASS, each suite separately with zero failures/errors/skips. Merged automatic input94 has full receipt equality with the accepted732406aa result. `collector_primary/test_report_2026-10-23_final.md` records commands and exact evidence.
- completed H default matrix gate: `collector_primary/merged_matrix_binding.json` explicitly binds64f3c950 to the accepted4a1b53e6 capture,77/77identical; simulator/data/examples/matrix bytes are unchanged. This is not a fresh capture or native-profile qualification.
- completed source/CPU workstream closure: implementer and both reviewers delivered and stopped; owner code/records are consolidated on the main task branch. No new GPU run was launched. The original failed checks and initial review findings remain historical evidence below and in `collector_primary/`.
- pending G2: real replay driver, collection-window/retention limits, CPU driver verification and a concrete GPU proposal. No G2 driver implementation or new GPU approval exists. Profile output remains UNVERIFIED, with stream mapping, actual PDL, compile/breakable survival, captured-segment ownership and GPU CUPTI restart unresolved.

### Numbered next actions

1. Read `handoff_prompt.md`, `requirements.md`, `driver_design.md`, accepted receipts and actual Git state; preserve `.pi/`, `.cortexkit/`, original evidence and existing worktrees.
2. Assign the real `pd_replay.py` driver/collector coordination to one owner. Add both-head start-before-formal-dispatch and stop-after-drain through the existing invocation path. Set explicit `serving_profile_limits` and profiler windows. The collector records from startup and does not reset on `/start_profile`; prove selected-window records survive the capacity limits.
3. Verify unselected compatibility, original returns/exceptions, both-head ordering and auxiliary-failure isolation with CPU controls; independently review and freeze that source.
4. Present the pinned source/workload/windows, GPU count/duration, artifact volume and runtime gates for YC's explicit run approval. Then follow workflow/GPU/operator-gap before CPU-overhead work. Full05/09/19 and four-metric calibration remain unfinished; GLM stays paused.

## Execution history

Intermediate H pending/in-progress statements below describe their recorded time and are superseded by the current handoff above.

- completed: YC approved the recommended KERNEL_ONLY CPU implementation/measurement-design route. New GPU runs require a concrete later proposal.
- completed: Root rehashed R4 freeze dependencies/evidence/originals and original R3 prelaunch evidence; all intact. Prior admission records archived at972f0cb1 (24record files; no source changes).
- completed: Private-input design recorded before edits. Scope:2core files/180lines and3non-core files/350lines; a candidate exporter cannot issue its own PASS admission.
- historical E implementation: `/data/ycfeng/Frontier/.worktrees/v4-attention-input` (session ses_e992b3667ffeN5frEhBm9plm59). Its first182-test run predates later edits; final-source and merged reruns below supersede that intermediate evidence.
- completed E owner delivery: source `4a1b53e6`, records `04a72b08`, clean worktree; 2 core / 94 and 3 non-core / 285 changed lines. Owner reported 194 tests, dual94 exact, legacy102/72 and one fixed-source77 capture. Root acceptance below supersedes the delivery-time pending gates.
- completed F source checks: both independent reviewers returned no findings (`input_primary/review_2026-10-23_input_source.md`). Root fixed-source unit rerun: 194 PASS, zero failures/errors/skips. Root strict comparison against the bound576d278 capture: 77/77 identical, zero cache-name differences or missing evidence (`/data/ycfeng/tmp/glm53_dsv4/root_attention_input_4a1b53e6_compare/`).
- observed root preparation check: fresh sidecar bytes differ from candidate02, triggering the root script's over-strict byte assertion after all three CLI commands succeeded. Investigation found exact parsed JSON equality and identical original evidence/CSV bytes; only dictionary key order differs, from `build_isolated_prefill_layouts`' metadata-column set. No source or original artifact change is needed. Representation acceptance uses exact layout values and each candidate's own digest; the failed first root check is retained.
- completed F: root independently verified seven non-timing records reconstruct all twelve table/layer layouts, rehashed529package files and451matrix artifacts, and issued candidate02's external isolated-only admission. Actual admitted automatic lookup reproduces all94 explicit medians/layouts. Evidence: `input_primary/{representation_verification.json,input_admission.json,test_report_2026-10-23_primary.md}`. Legacy temporary-path strings and comparator-list schema required two further root-check corrections, documented in the report; no product changes.
- completed G1: integrated as `732406aa235ff63c97e7b9bbd6d5836fe1881de7`. Merged unit module194 PASS (no errors/failures/skips); merged admitted automatic94 gives full receipt equality. `input_primary/merged_matrix_binding.json` binds the existing4a1b53e6 capture to identical merged simulator/data/example/matrix bytes; only the standalone audit script differs outside task records. Final `input_primary/primary_acceptance.json`: ACCEPTED_ISOLATED_INPUT_COMPONENT. No fresh capture at merge.
- in-progress H scope correction: `9366b6e4` delivered three reproduced launch blockers and no collector implementation. Root allocated its proposed two paths / 12 changed lines (five non-core files / 787 total), returning implementation to the same owner. Prior unchanged-suite passes are baseline checks, not collector evidence.
- completed B: Kernel/dependency audit `audit/test_report_2026-10-23_kernel_coverage.md`, `audit/kernel_coverage_receipt_r2.json`, `audit/native_source_evidence.json`; tool `tests/analysis/v4_kernel_coverage.py` (227 lines). 1815 kernels/132 annotations/366 owned kernels reproduced; 1449 kernels unowned, no outer attention boundary, 85 stream waits without event/stream handles, 93 same-stream overlaps (PDL reachable). Retained trace cannot support a serving total; minimum capture additions listed in the report.
- completed D: `native_profile_seam.md` (two rounds) and `native_profile_checks/`. Full native TP8/DP2/EP16 already exists; no shared TP parameter needed. Actual rank0: P NONE at1request/6144tokens; D FULL+PIECEWISE at descriptors 1/2/4/8/16 with 1..11 scheduled requests per lane (root read all 32 rank files, dropped_rows 0; the earlier "16-token only" statement was wrong and is corrected in seam §3); captured128 is capacity only. Breakable capture is auto-enabled for V4 (server.log line16), so eager attention segments re-run Python at replay. Profiler start/stop restarts work at API level (CPU control); CUPTI re-init under graphs and capture-event survival stay runtime gates.
- completed: Checkpoint corrections applied to plan.md (G1/G2/H) and proposal.md (outer-excluded CUDA_EVENT sums; J3 premise).
- in-progress H: native V2 profiling extension in `/data/ycfeng/Frontier/.worktrees/v4-native-serve-profile` (branch feat/v4-native-serve-profile from993f05f5, session ses_e9916feb6ffeDDJENvata6ugYg). Allocation: new `native_serve_profile.py`<=450, `native_pd_worker.py`<=25, one test file<=300, `dp_placement_pp/vllm_replay.py`<=8 and `calibration/pd_replay.py`<=4 changed lines. Driver is not built; root traced its real invocation and recorded the required lifecycle in `driver_design.md`. CPU only.
- completed root: audit records, tool and corrected plan/proposal committed as993f05f5 (receipts excluded by size).
- completed root: native06 decode descriptors corrected from all 32 rank files (ea1b021c) and bound to pinned DP padding source (f1f503bd). H acceptance now also requires per-step lane-local and DP-synced counts; this was added after H started, so root checks it at H review and returns it to the H owner if missing.
- completed root: `baseline_binding.json` regenerated (SHA256 88f0496e…cbf): pre-change972f0cb1 binds to the576d278 fixed77 capture; d278659..972f0cb1 changes24task-record paths only. E's new4a1b53e6 capture was subsequently compared strictly and passed77/77.
- completed root: audit tool rerun reproduces `kernel_coverage_receipt_r2.json` (SHA256 caafcedf…c9e1) except the output path and start/end timestamps; rerun at `/data/ycfeng/tmp/glm53_dsv4/root_kernel_coverage_rerun_20261023.json`. Both receipts (~3.9 MB) stay local and untracked, bound by hash and regenerable from the committed tool; `kernel_coverage_receipt.json` (face6d04…d12a) is the superseded first audit pass.
- pending: H collector source/CPU completion, independent review and root integration; G2 actual driver integration and a concrete supplemental measurement proposal. E/F/G1 are complete.
- completed H initial delivery: d94b821a/728lines, source/CPU controls reported; root reviews found five unique issues (oneP1/fourP2), so this is not accepted. `collector_primary/review_2026-10-23_initial.md` retains both reports. Root's attempted old-source rerun failed before pytest on a receipt-field assumption; no old-source root PASS claim.
- in-progress H repairs: same owner, five files/987lines authorized for lifecycle, metadata/PDL boundaries, final-record admission and bounded logging. Session-close request requires completing these repairs, final verification/review and branch integration, then committed handoff. G2 driver/GPU preparation is deferred to the next session.
- limits: No additive serving total, physicalTP8/graph costs or full prediction yet. Uncompressed0.9699/1.5241ms difference remains unclassified; no CPU attribution/scaling.
