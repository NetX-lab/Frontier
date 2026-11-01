**1. Assignment and outcome**

Completed the H200 NIXL READ timing proposal for DSV4-Flash P2-32 and P2-64. Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. All new files are under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/`.

The complete report, including every measured case/pass, tails, candidate residuals, concurrency evidence and exact verification commands, is [report.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/report.md).

Outcome is **implemented as a proposal and verification handoff**. Formal adoption and acceptance remain with YC/Claude. Frontier source, profiling tables, formal timing inputs, thresholds, references, folds, task records and retained evidence were not changed. No Git command, commit, dependency installation, GPU, platform or cloud action was performed. The pre-existing `.cortexkit/`, `.pi/` and `remaining_measurements/` were not edited.

The proposal follows D-P2-INPUT-QUALIFICATION and D-P2-Q1-CHOICES timing route B. It preserves D-WARM-TARGET, the committed packed-byte calculation and the original NIXL02 FAIL. The collective exact-point rule was not treated as an adopted NIXL rule.

**2. Findings and proposed input**

Integrity was independently rebuilt: 23 downloaded file hashes and sizes match; 30 cases × three passes give 90 unique records; all 12,150 reader trials and 80,352 notifications are retained; all 90 producer CSV rows reconstruct exactly. Every measured case/pass appears in the report and machine-readable tables.

At T6144 with one request in flight, the pooled critical-path median is 0.964047 ms for one pair and 0.971057 ms for eight pairs. The largest same-case pass-median spread is 0.882168%; the median spread is 0.090399%. Equal-byte T6143/T6144 matched-pass medians differ by at most 0.836141%. Completion grows approximately with total bytes per reader.

Tails remain material. The largest reader-batch trial is 27.497038 ms at T6145, eight pairs, sixteen requests in flight, pass 1. At the actual P2 coordinate, T6144/eight pairs/one request in flight, pass maxima are 3.958579, 12.554511 and 0.980675 ms. No slow trial or pass was removed.

The producer records completion of **all requests in a reader's batch**, including preparation and posting. It does not record individual request completion times at concurrency greater than one. Full-batch maxima were never divided by the number of requests to claim per-request bounds.

Both scales reach **one logical transfer per decode pod at a time**, including the entire wait for engine-step reporting. TP8 maps that request to eight physical READs: one per pair, with eight pairs reading together. Full transfer windows never overlap across DP lanes. P2-32 uses two DP lanes; P2-64 uses four, but never overlaps READs from those lanes. The measured grid covers both replays at `t6144_pairs8_inflight1`. This conclusion comes from all 128 windows in each of the four retained B29/two-fix traces, plus the authoritative TP/DP mapping; it is not a native in-flight observer result.

Recommend one YC choice: **approve `AFFINE_MEDIAN_P2_SINGLE_READ`**, specified in [proposed_input.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/proposed_input.json).

The proposed existing configuration fields are:

- `analytical_kv_cache_transfer_config_network_bandwidth_gbps = 390.39349928254796`
- `analytical_kv_cache_transfer_config_network_latency_ms = 0.0236521842818856`

OLS uses all 90 full-batch case/pass medians with equal row weights. The fitted intercept is positive without clamping. At one 46,103,040-byte READ, the price is **0.9684023419505275 ms**.

The proposed rule requires every two-pass training fit to have positive finite intercept and slope, and every held-out case/pass median to have at most 5% relative error. All 90 checks pass; maximum error is **0.919518%**. Every observed trial at the actual P2 coordinate also passes the empirical event-window check. This is a proposed median-input qualification rule, not an adopted trial-p90 or universal-tail rule.

Candidate results are:

| Candidate | Definition and proposed validation | Actual result | Interface and outside-domain behavior |
| --- | --- | --- | --- |
| Empirical bound | Every positive observed completion must precede every relevant next polling start | Actual P2 coordinate passes all three passes at both scales. Full-grid comparison fails 11/90 case-passes at P2-32 and 4/90 at P2-64 | No interface change for a case-metric independence exception. Finite samples do not establish a future serving bound; outside-domain qualification fails. This alone does not qualify the H800 price |
| Exact concurrency point | Pooled full-batch median keyed by bytes/request, pairs and requests in flight; leave one pass out; each held-out median error ≤5% | 90/90 median checks pass; maximum 0.808380%. The separate trial median5%/p90 10% rule fails 7/90 case-passes | Twenty byte-keyed coordinates; equal-byte prompts share a point. General use requires concurrency/cohort context and table admission. Reject unmeasured coordinates; no interpolation |
| Positive affine, recommended for P2 n=1 | All 90 median targets; physical full and held-out fits; each held-out median error ≤5%; empirical P2 tail/window check | All fits physical, 90/90 median checks pass. The separate trial median5%/p90 10% rule fails 8/90 case-passes | Existing configuration suffices for retained P2 n=1. The bytes-only runtime does not enforce domain restrictions; an external qualification census must reject other workloads. General concurrent-batch use requires interface work |

All candidates cost zero new GPU-hours for this proposal. General table/cohort implementation would require separate source approval. The recommended candidate needs no fresh CPU replay because no completion can move.

**3. Changes**

Every changed file is new under `nixl_timing_q2/`:

- `analyze.py`: raw reconstruction, repeatability, equal-byte comparisons, candidate fitting and held-out validation, interval concurrency sweep, and event-window proof.
- `checks.py`: twelve focused checks, including half-open interval handling, physical-fit rejection, independent closed-form OLS verification, complete populations and unchanged retained-input hashes.
- `verify.py`: executes the retained lead checker, existing NIXL lifecycle regression and production configuration price check.
- `render_report.py`: renders the complete decision report from machine-readable evidence.
- `proposed_input.json` and `results.json`: proposed input/rules/domain, all case/pass outcomes, four replay proofs, environment and evidence hashes.
- `case_pass_results.csv`, `repeatability.csv`, `equal_byte_passes.csv`, `reader_summary.csv`: complete measurement and candidate tables, including all 405 reader/pass summaries.
- `transfer_windows.csv` and `possible_post_windows.csv`: all 512 transfer windows and 2,034 possible posting-window checks.
- `analysis.log`, `checks.log`, `lead_retained_check.log`, `lifecycle_regression.log`, `production_price.log`, `verification.json`: actual command results.
- `report.md`: full analysis and adoption recommendation.
- `scratch/reader_trials.csv` and initial logs: all 12,150 reader trials with explicit millisecond column names and initial successful check logs. Scratch is not intended for commit.

No production behavior changed. The proposed configuration and qualification rule remain unapplied.

**4. Verification**

Executed with `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), repository cwd, `PYTHONPATH=$PWD:$PWD/tests/analysis`, bytecode disabled, OpenBLAS/OMP/MKL threads each set to one, and temporary files in the authorized scratch directory. Heavy inspections were streamed and bounded by `timeout 120s`.

Exact executable arguments, expected values and observed output are preserved in [verification.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/verification.json) and report section 6.

- `timeout 120s <python> <nixl_timing_q2>/analyze.py`: **PASS**, exit 0. Complete raw matrix and CSV equality, physical fits, concurrency census and all event assertions.
- `timeout 120s <python> <nixl_timing_q2>/checks.py`: **PASS**, twelve checks.
- Retained `lead_check_nixl6144.py`: **PASS**, 23 matching hashes, 90 unique records, 12,150 trials, 80,352 notifications, zero producer CSV mismatches.
- `<python> -m pytest tests/unit/test_nixl_read_completion.py -q -p no:cacheprovider --basetemp <authorized scratch>/pytest`: **PASS**, nine tests in 0.79 seconds.
- Production `AnalyticalKVCacheTransferConfig`/`AnalyticalKVCacheTransferPredictor` calculation: **PASS**, 0.9684023419505275 ms, exactly equal to the proposed formula.
- Report rendering and final artifact inspection: **PASS**, four Python ASTs, valid JSON, 33 valid local links, 90 unique case/pass rows and 12,150 correctly labelled reader trials.
- Fresh P2 CPU replays: **NOT_RUN**, discharged by the event proof.
- New GPU/platform/native-payload checks: **NOT_RUN**, outside assignment; retained receipts were inspected.

For every possible posting step inside each transfer envelope, both old/current prices and the proposed positive price finish before the first strictly later decode step. Minimum gaps are 18.724965326230 ms and 24.380075962759 ms. All retained DECODE forwards have positive duration, excluding an idle zero-cost wire-end wake. The same later step therefore finds each READ, and the same output reports it. All **2,034 checks have zero crossings**. Downstream event/metric invariance follows from the deterministic event sequence; no new E2E replay result is claimed.

**5. Remaining decision and review handoff**

YC/Claude must choose whether to adopt the displayed H200 pair under the proposed median rule and exact retained-P2 domain. If the required rule is instead per-trial median5%/p90 10% on the full grid, both deterministic median candidates must remain rejected. All failures are recorded individually in the report and tables.

The original NIXL02 negative-intercept fit remains FAIL. Standalone finite measurements do not establish a universal serving-tail guarantee or reproduce serving compute/collective contention. Native request descriptor bytes remain a separate OPEN edge; this proposal depends on the committed source-derived 46,103,040-byte payload. No blocker remains for the assigned proposal work.

Review the fit targets, positive intercept, held-out pass errors, complete tail failures, byte-keyed exact points, TP/DP mapping, ledger envelope sweep, zero-duration guard and outside-domain policy. Rerun `analyze.py`, `checks.py`, `verify.py` and `render_report.py` using report section 6. Claude retains adoption, acceptance, any optional formal binding replay, records and commits.
