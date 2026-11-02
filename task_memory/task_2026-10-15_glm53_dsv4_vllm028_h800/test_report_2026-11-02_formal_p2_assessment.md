# Test report: formal P2 replays and the Q4 ticket 19 assessment, lead review

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | GPU-hours: the 19 UNKNOWN jobs filled from read-only platform container clocks; 35 jobs total 158.27 container GPU-hours. |
| 2026-11-02 | Created: formal P2-32/P2-64 replays with the adopted H200 NIXL pair, lead review of Codex Q4, independent checks, and the per-case verdicts (P2-32 PASS; P2-64 BLOCKED_ON_DECISION on the pre-launch promotion receipt). |

Authority: ticket 19 (`frontier_reference_run`, `comparison_and_calibration`, `promotion_32_to_64`, `final_acceptance`, Acceptance); requirements D-P2-INPUT-QUALIFICATION, D-P2-Q1-CHOICES, D-P2-Q2-NIXL-TIMING, D-P2-PAYLOAD-C, D-WARM-TARGET, D-WARM-TWO-FIX, D-WARM-EP-DEFER, decision 46 and D-NO-INTERRUPT.

Assignment: [Q4](h3o_operator_costs/formal_p2_20261102/codex_assignment_Q4_formal_p2_assessment.md). Codex request `75ccdbe8-d7c5-4b92-a3aa-97a0dc65ad21` (default model and effort, outcome `implemented`), report [implementation.md](h3o_operator_costs/codex-implement-75ccdbe8-d7c5-4b92-a3aa-97a0dc65ad21/implementation.md). Evidence: [assessment/report.md](h3o_operator_costs/formal_p2_20261102/assessment/report.md) and the machine outputs beside it (`input_status.json`, `binding.json`, `formal_vs_two_fix.json`, `gate_readings.csv`, `milestones.json`, `promotion.json`, `gpu_hours.csv`, `verdict.json`).

## Formal replays

Runner `h3o_operator_costs/formal_p2_20261102/run_formal_replay.py` (commit `abd1e141`). It runs the B29 formal inputs from the committed source and overrides only the two analytical KV transfer fields with the adopted H200 pair from `input_qualification_20261101/nixl_timing_q2/proposed_input.json` (sha256 `74374415...f673`): 390.39349928254796 Gbit/s and 0.0236521842818856 ms. Fresh cache, 64 GiB address-space limit, scratch `/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/`.

| Case | Command record | Started (UTC) | Finished (UTC) | Elapsed | Simulator | Gate (all-request) | Readings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2-32 | `formal_p2_20261102/p2_32_command.json` | 03:09:44 | 05:59:27 | 10,183 s | exit 0 | PASS | warm and empty, exit 0 |
| P2-64 | `formal_p2_20261102/p2_64_command.json` | 03:09:44 | 06:37:22 | 12,459 s | exit 0 | FAIL (repeat01 TTFT −12.254941%, preserved) | warm and empty, exit 0 |

## Codex result (reviewed)

1. **Inputs (27 ledger rows):** 22 QUALIFIED, 4 OPEN, 1 FAIL. Row 26, formal hardware/operator coverage as a whole, is PASS under its unchanged criterion: its three recorded missing items (attention, H200 timing, payload) are resolved. OPEN rows: 18 (H800 pair, retired for H200 and no longer consumed), 23 and 24 (independent memory peaks and autonomous planner agreement; the ledger states they are not prerequisites of the measured-capacity isolation), 27 (promotion receipt, needed only for P2-64). Row 19 (rejected NIXL02 fit) is FAIL and not consumed.
2. **Binding:** seven profiling tables, trace and population, resolved config, timing input, driver and dependencies, source `abd1e141`, and 879 generated cache files per case bind by path and SHA-256. Five disclosed finding groups per case (config roots and the two timing fields; the timing successor adopted after the frozen ledger; the new source commit; the new runner; auxiliary helpers bound through qualified owners). No unexpected consumed-cost change, omission or fallback.
3. **Formal versus two-fix:** the comparison method first passed its control (two-fix versus B29). Formal versus two-fix: zero forward-row differences (23,040 / 39,680 rows), identical request metrics, scheduler KV start/completion, collectives (6,310,644 / 9,002,176 calls), gate rows and readings. Only provenance and the two timing fields differ.
4. **Readings:** 24/24 warm checks PASS at 10%. The P2-64 repeat01 all-request TTFT FAIL (12.254941%) stays beside its warm PASS.
5. **Milestones:** 16 checkboxes mapped. The GPU-hours checkbox is INSUFFICIENT_EVIDENCE (19 jobs lack full allocation clocks).
6. **Promotion:** the P2-32 to P2-64 30% predicate is true (every finite error at most 5.607176%). Receipt drafted only (`draft_only=true`, `issued_receipt=false`). The existing P2-64 native reference `exp-1025-155441-668927` was submitted 2026-10-25T15:54:41Z, before any P2-32 result existed, under the D-H200-TEMP-MAINLINE authorization; no pre-launch receipt exists in the retained records.
7. **GPU-hours:** 35 distinct H200 jobs. 141.41 container GPU-hours witnessed (135.00 for the 32 later P2 jobs, 6.41 for three earlier shared DSV4 prerequisites), plus 19 jobs with incomplete clocks. The actual total is unknown.
8. **Verdicts:** P2-32 PASS; P2-64 BLOCKED_ON_DECISION.

## Lead checks

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), cwd the worktree, `PYTHONDONTWRITEBYTECODE=1`, each script under `timeout`. Scripts and logs: `h3o_operator_costs/formal_p2_20261102/lead_review/`.

| Check | Command / evidence | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Scope | `git status --porcelain`; `find -newer <assignment>` outside `assessment/` and the Codex report directory | no edits outside the write scope | none; other new files are the runner's command records and readings, and the unrelated `g0d_launch/launcher.log` | PASS |
| Inventory integrity | `check_artifacts.py`: rehash every row of `changed_files.csv`, list unlisted files | 66 rows, all hashes match, no unlisted file | 66 rows, 0 mismatches, 0 unlisted, 0 missing | PASS |
| Metric files | `cmp` of the six metric files, formal versus two-fix run directories | only `config.json` differs | `config.json` DIFF; `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `op_precision_metadata.csv`, `request_metrics.csv`, `system_metrics.json` SAME, both cases | PASS |
| Config difference | key-level diff of `config.json` | the two timing fields and three output/cache roots | exactly 5 keys per case: bandwidth 126.2 → 390.39349928254796, latency 0.0647 → 0.0236521842818856, `cc_backend_config/cache_dir`, `metrics_config/cache_dir`, `metrics_config/output_dir` | PASS |
| Readings | `check_readings.py`: formal versus two-fix reading JSON with `path` keys removed | identical | identical for all four reading files | PASS |
| Per-repeat gate | `check_readings.py`: per-repeat warm and all-request errors from the formal warm reading, matched against `gate_readings.csv` | 24 warm PASS; values match the CSV exactly | 24 warm PASS (maximum 5.675933%); one all-request FAIL (P2-64 repeat01 TTFT 12.254941%); every CSV value matches | PASS |
| 30% predicate | same script, every finite error of every P2-32 repeat | at most 30% | maximum 5.607176% (warm and all-request are equal for P2-32: no exclusions) | PASS |
| Chronology | `test_report_2026-10-25_h200_package_preparation.md:74`; `promotion.json#/chronology` | P2-64 reference launched before any receipt could exist | submitted 15:54:41Z, 31 minutes after the P2-32 reference (`exp-1025-152327-225209`) was submitted; no receipt | confirms BLOCKED |
| GPU-hour arithmetic | recompute container intervals × 8 / 3600 from the two reference platform receipts | equals `gpu_hours.csv` | 24.322222 (P2-64, 8 containers) and 11.311111 (P2-32, 4 containers), equal to the receipts and the CSV | PASS |
| Ledger criteria | `ledger.json` rows 18, 23, 24, 26, 27 | dispositions follow the recorded criterion and limitation text | row 26 "missing evidence" lists exactly attention, H200 timing and payload; rows 23/24 limitations exclude them from measured-capacity isolation; row 27 requires a pre-launch receipt | PASS |
| Quantization warnings | `sklearn_execution_time_predictor.py:796-814`, `deepseek_v4.py:606`, `metrics_store.py:2307` | the approximation-factor warnings in the replay logs do not scale predictions | the factor is logged only; the precision metadata feeds trace metadata and `op_precision_metadata.csv` | no effect |

## Decision

- **P2-32: PASS** (ticket 19 formal result for this case). Qualified inputs, all four warm errors at most 10% in every admitted repeat (maximum 5.607176%), unchanged all-request readings retained (also PASS), own-hardware coverage, correct resource class.
- **P2-64: BLOCKED_ON_DECISION.** The numbers pass (warm maximum 5.675933%), but Acceptance requires a valid pre-launch 30% promotion receipt for 64-GPU formal runs, and the existing reference was launched before any receipt could exist. Options for YC:
  - **A. Exception for the existing reference.** YC explicitly waives the launch order for `exp-1025-155441-668927`. Cost: 0 GPU-hours, records only. P2-64 then becomes PASS, with the exception recorded.
  - **B. New formal P2-64 native campaign.** The lead issues the P2-32 receipt (records only), prepares a pair-warmup step, and submits through `/ds4-exec`. Resources: 64 H200 in `step_main` (8 nodes × 8), not counted against the 32 non-preemptible limit (wholly preemptible, as the nixl6144 run). Cost: about 24.3 container GPU-hours from the original campaign; the 3,600 s timeout caps execution at 64 GPU-hours; queue time unknown.
  - Lead finding for B: D-WARM-TARGET requires future native runs to warm every lazily connected (P lane, D lane) pair before the formal window and prove it from lifecycle/server records. The retained package uses 16 fixed warmup requests, which its own manifest says do not prove coverage. B therefore needs a package change (a warmup that opens all 16 P x D lane pairs) and a check-only verification before submission. The new reference would not need the paired exclusion.
  - Lead recommendation: B. The Acceptance text says not to waive an active DSV4 criterion, and the pair-warmed reference also tests the warm target directly. Choose A if 64 H200 for about 25 minutes of container time (the original containers ran 1,304–1,473 s), plus the queue wait, is not worth it now.
- **GPU-hours checkbox:** INSUFFICIENT_EVIDENCE, not blocking the Acceptance predicate. Authoritative allocation records for the 19 incomplete-clock jobs can be requested read-only through `/ds4-exec`; no GPU run is needed.
- **GPU-hours, follow-up (same day):** the 19 jobs were read through `/ds4-exec` (Pi `gpuh-clocks-1102a`, read-only, personal client). 16 have `terminal_receipt.py` receipts; 3 have purged job objects (HTTP 404) and keep their clocks on the replica objects. `h3o_operator_costs/formal_p2_20261102/gpu_hours_clocks/merge_gpu_hours.py` recomputes each from container start/finish × GPU limit and checks creator, group, replica × GPU count and the receipt value (19/19 PASS). Total for the 35 jobs: **158.271667 container GPU-hours** (151.864444 P2 H200 campaign, 6.407222 earlier shared DSV4 prerequisites); `gpu_hours_complete.{csv,json}` beside the script. Basis: container start-to-finish, including initialization, compilation, warm-up and failures; queue time and pre-container pod time (image pull, 153–165 s where visible) are excluded. The checkbox stays open until the formal P2-64 campaign is added.

## Limits

- Attention: exact retained coordinates and the independent-pass supplement; no unseen-geometry claim.
- Payload: source-derived, metric-invariant; native descriptor arrays not observed.
- NIXL timing: qualified at 6144 tokens, eight pairs, one request in flight, `engine_step`; the measured 12.554511 ms maximum is a finite sample.
- Host cost zero, ingress 10.3 ms, handoff 11.4 ms and K19 stay the approved inputs.
- EP dispatch count stays deferred (D-WARM-EP-DEFER, at most 0.0468 ms per step).
- Capacity is measured-pool isolation (109,934 / 112,914 blocks); planner agreement and phase peaks stay OPEN.
- Native graph proof is serving host/replay returns, not device timing.
- Only the two H200 P2 cases are assessed; H800 cases and paused GLM work are not complete.
