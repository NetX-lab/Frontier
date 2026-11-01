# A6 CSA FULL decode variation: cause and measurement correction options

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Added section 6: what the variation does to the forward time and metrics a same-configuration vLLM comparison observes, with a revised recommendation. Diagnostic only; no input, threshold or verdict changed. |
| 2026-11-01 | Created: cause of the CSA FULL decode variation from the retained 16-repeat probe, phase-level repeatability, phase-level held-out on the recorded A6 folds (all 15 keys within the A6 limits), and three correction options. No formal input, threshold or acceptance verdict changed. |

## Purpose and scope

Decision D-A6-CSA-FULL-CAUSE: find why CSA FULL decode `dsv4_index_compressor_fused_compress_store` varies between measurements, using CPU analysis of the retained 16-repeat probe only, and propose a measurement correction. The scope stays FAIL while this runs. All results here are diagnostic. The published table (`01a9d51b…`), the A6 thresholds (median ≤ 5%, p90 ≤ 10%, KFold 5, seed 1234) and every A6 verdict are unchanged.

## Environment and commands

- Workspace `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6. CPU only; no platform operation. Temporary files under `/data/ycfeng/tmp/glm53_dsv4/`.
- Scripts, committed in `384d1552`:
  - `tests/analysis/v4_attention_stream_overlap.py` (SHA `3b85a9c2…`): kernel intervals, row-level spread across repeats, bootstrap of the median and mean.
  - `tests/analysis/v4_attention_phase_heldout.py`: replays the recorded A6 folds with the A6 production trainer and compares the composed attention phase.
- Runner: `h3o_operator_costs/a6_csa_full_cause_20261101/run_analysis.py`. It records the exact command, source commit, dirty source paths, script SHA, start/finish time, exit code and output SHA in `<name>_command.json`, and the full output in `<name>.log`. `phase_heldout.log` (11,979,942 bytes, SHA `67296548…`) stays in the local task directory and is not committed, like earlier A6 run logs.
- Inputs:
  - The A6 repeatability probe `exp-1101-015820-503227` (`h3o_operator_costs/a6_repeatability_20261101/package_plan.json` and `case_mapping.json`; raw `dsv4_raw_e0.jsonl` and `dsv4_raw_e1.jsonl` under `/data/ycfeng/tmp/glm53_dsv4/a6_repeatability_20261101_download/A6_REPEATABILITY/bridge/FULL_AND_PIECEWISE/raw/`). Eight workloads, each measured 16 times with one decode step per measurement: four CSA FULL decode workloads (`g1-e0-0262`, `g1-e0-0263`, `g1hs-e0-0100`, `g1hs-e0-0101`) and four CSA PIECEWISE workloads (`g1-e1-0729`, `g1-e1-0730` first_tail; `g1hs-e0-0253`, `g1hs-e0-0254` decode).
  - The formal A6 result `h3o_operator_costs/a6_codex_20261101/result.json` (table, config binding and fold assignment).

## Results

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Stream overlap (`stream_overlap.json`) | Exit 0; zero dirty source; every scope's kernel-duration sum equals its recorded call time within 0.001 µs | Exit 0 in 55 s; source `384d1552`, dirty 0; assertion held for every scope instance | PASS |
| 2 | Phase-level held-out (`phase_heldout.json`) | Exit 0; zero dirty source; table SHA and config binding equal the formal A6 run; per-scope errors on the evaluated rows reproduce the formal A6 values | Exit 0 in 2,698 s; source `384d1552`, dirty 0; script SHA `ae515b8d…`; output SHA `36285b1d…`; both bindings checked by the script; per-scope values below | PASS (diagnostic) |

### 1. Cause: start order of concurrent kernels in FULL graph replay

In a CSA layer, `compose_attention_phase` places the query scopes on the main stream, the compressor on auxiliary stream 1 and the index query on auxiliary stream 0. In FULL mode the index compressor forks to auxiliary stream 2; in PIECEWISE mode it runs after the index query on auxiliary stream 0. The kernels on different streams run at the same time between one fork and one join.

In FULL graph replay, the relative start time of the two fused compress-store kernels changes from replay to replay, and even from layer to layer within one replay. The offset (compressor-store start minus index-store start) falls into two clusters:

- **Together**: |offset| < 0.4 µs.
- **Staggered**: offset above 3 µs, up to 5.5 µs.

A few instances are negative (as low as −5.95 µs); the analysis separates the two classes at |offset| = 1 µs, so these count as staggered. In PIECEWISE replay the index compressor waits for the index query, so the offset is always −5.5 to −11.6 µs.

The start order decides which kernels slow each other down. Median kernel durations in µs, together / staggered:

| Workload, layer group | Instances (together share) | Index compress-store | q_projection | Compressor compress-store | Index-store CV |
| --- | --- | --- | --- | --- | ---: |
| g1-e0-0262, first CSA layer | 16 (0.50) | 11.06 / 7.55 | 7.41 / 10.29 | 18.37 / 17.92 | 0.200 |
| g1-e0-0262, reuse layers | 64 (0.42) | 11.33 / 7.87 | 7.30 / 10.37 | 19.04 / 18.27 | 0.184 |
| g1-e0-0263, first CSA layer | 16 (0.38) | 11.20 / 7.81 | 7.25 / 10.34 | 17.62 / 18.43 | 0.191 |
| g1-e0-0263, reuse layers | 64 (0.52) | 11.30 / 7.97 | 7.26 / 10.40 | 19.10 / 18.78 | 0.179 |
| g1hs-e0-0100, first CSA layer | 16 (0.12) | 13.42 / 9.52 | 9.55 / 11.42 | 19.14 / 19.44 | 0.138 |
| g1hs-e0-0100, reuse layers | 64 (0.28) | 12.06 / 9.76 | 9.98 / 11.42 | 18.10 / 19.52 | 0.109 |
| g1hs-e0-0101, first CSA layer | 16 (0.31) | 12.45 / 9.22 | 10.11 / 11.42 | 17.86 / 19.94 | 0.126 |
| g1hs-e0-0101, reuse layers | 64 (0.34) | 12.05 / 9.42 | 10.00 / 11.46 | 18.14 / 19.52 | 0.131 |
| PIECEWISE workloads, all groups | 16 or 64 (0.00) | 5.31–5.62 (staggered only) | 7.23–10.10 | — | 0.024–0.161 |

- When the two stores start together, the index store is 2.3–3.9 µs (24–47%) slower and q_projection is 1.3–3.1 µs faster. When they are staggered, the index store overlaps q_projection more and slows it down instead.
- q_projection's duration follows the time it overlaps one of the two stores. Its strongest correlation is with the index store in six of eight FULL groups (0.97–0.99 for g1-e0, 0.77–0.87 for g1hs) and with the compressor store in the other two (0.99 and 0.96).
- The compressor store changes little between the classes (CV 0.07–0.10) and ends last before the join in 16/16, 63/64, 16/16, 64/64, 16/16, 64/64, 16/16 and 64/64 instances, so it sets the end of the concurrent section.
- PIECEWISE workloads never start the stores together. Their index-store CV is 0.024–0.078, except one slow outlier in g1hs-e0-0253's first CSA layer (CV 0.161, maximum 9.12 µs).

The start order is set by the GPU at replay time; the probe records it but not why it changes. No hardware counter, clock or scheduler evidence exists, so the mechanism behind the start order is not established.

### 2. Why single-run row medians are noisy

The published table has one decode step per CSA FULL decode workload: the first-CSA-layer row holds one sample (layer 2) and the reuse row holds four samples (layers 4, 6, 8 and 10). Each index-store sample comes from one of two modes 24–47% apart, so a row median of one or four samples lands near either mode.

Bootstrap of n pooled kernel samples per layer group (10,000 draws, seed 1234), p90 relative deviation from the pooled value:

| Workloads | Scope | Statistic | n = 1 | n = 4 | n = 16 | n = 64 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| g1-e0 (together share 0.38–0.52) | index compress-store | median | 0.21–0.47 | 0.18–0.36 | 0.17–0.31 | 0.03–0.29 |
| | | mean | 0.21–0.31 | 0.15–0.16 | 0.07–0.08 | 0.04 |
| | q_projection | median | 0.27–0.38 | 0.23–0.35 | 0.14–0.32 | 0.00–0.32 |
| | | mean | 0.21–0.23 | 0.13–0.17 | 0.06–0.08 | 0.03–0.04 |
| g1hs-e0 FULL (together share 0.12–0.34) | index compress-store | median | 0.24–0.31 | 0.07–0.16 | 0.03–0.05 | 0.01–0.03 |
| | | mean | 0.20–0.24 | 0.08–0.10 | 0.04–0.05 | 0.02–0.03 |

- The n = 1 and n = 4 deviations match the noise floor already measured for this key (first layer p90 35.05%, reuse layers 21.10%, in `test_report_2026-11-01_a6_noise_floor.md`).
- When the two modes are close to equally likely (g1-e0), the median does not settle even at 64 samples: it jumps between the modes. The mean settles to about 8% at 16 samples and 4% at 64.
- When one mode dominates (g1hs-e0), the median settles to 5% by 16 samples.

This is consistent with median-of-three targets leaving CSA FULL decode at 21–25% p90: the median of three row medians, each from one or four samples of a near 50/50 mixture, can still land near either mode.

### 3. The attention phase the simulator consumes is repeatable

The simulator does not consume individual scope durations; it composes them into the attention-phase critical path with `compose_attention_phase`. Across the 16 repeats, using each repeat's row medians and gaps:

| Quantity across 16 repeats (CSA FULL decode rows) | CV | First-eight vs last-eight median difference |
| --- | ---: | ---: |
| Composed attention phase | 0.006–0.020 | ≤ 0.005 |
| Measured layer window | 0.005–0.020 | — |
| Index compress-store row median | 0.064–0.200 | up to 0.205 |
| q_projection row median | 0.042–0.173 | — |

Composed phase mean / layer window mean is 0.9959–1.0000 across all 16 CSA rows (FULL and PIECEWISE), so the composition reproduces the measured layer. The index-store and q_projection slowdowns trade against each other within one replay, so the composed phase stays within about 2% while each concurrent scope varies by up to 20%.

### 4. Phase-level held-out on the recorded A6 folds

`v4_attention_phase_heldout.py` retrains each recorded A6 fold with the A6 production trainer, predicts every scope's kernel time and gap for each held-out row (exact lookup or the fitted estimator, as A6 does), and composes the predicted attention phase with `compose_attention_phase`. The reference is the same composition of the row's measured medians and gaps; the measured layer window is a second reference. A row is skipped when any of its scopes is outside held-out coverage.

Population: 6,923 of 7,105 rows. The 182 skipped rows are 112 decode and first_tail rows with at least one scope outside held-out coverage (26 in CSA FULL decode), all 65 dummy rows and all 5 prefill rows; A6 evaluates no dummy or prefill row either. Per-scope kernel-time errors on the evaluated rows reproduce the formal A6 values, so the same failing predictions are being composed:

| Scope | Formal A6 (pooled over keys) median / p90 | Evaluated rows median / p90 |
| --- | ---: | ---: |
| q_projection | 1.70 / 10.23% | 1.69 / 10.20% |
| index_fused_query_prepare | 4.65 / 12.27% | 4.65 / 12.33% |
| compressor_fused_compress_store | 3.24 / 13.99% | 3.22 / 13.80% |
| index_compressor_fused_compress_store | 2.93 / 17.51% | 2.91 / 17.51% |

Composed attention phase, median / p90 absolute relative error (A6 limits: 5% / 10%):

| Model key (variant / mode / phase / layer group) | Rows | Phase vs composed measured | Phase vs layer window | Same rows, per scope: index compress-store / q_projection / compressor compress-store |
| --- | ---: | ---: | ---: | --- |
| All evaluated rows | 6,923 | 0.69 / 2.07% | 0.69 / 2.07% | 2.91 / 17.51%, 1.69 / 10.20%, 3.22 / 13.80% |
| CSA / FULL / decode / reuse layers | 467 | 0.95 / 2.81% | p90 2.47% | 9.22 / 24.12%, 7.89 / 14.77%, 4.73 / 13.88% |
| CSA / FULL / decode / first CSA layer | 467 | 1.17 / 3.53% | p90 3.53% | 13.76 / 28.17%, 11.91 / 27.39%, 6.51 / 30.41% |
| CSA / PIECEWISE / decode / reuse, first | 472, 472 | 0.86 / 2.56%, 1.01 / 3.16% | p90 2.47%, 3.16% | — |
| CSA / PIECEWISE / first_tail / reuse, first | 440, 440 | 0.97 / 3.17%, 1.13 / 3.70% | p90 3.09%, 3.70% | — |
| HCA, six keys | 441–473 each | 0.60–0.69 / 1.50–1.72% | p90 1.50–1.68% | — |
| Uncompressed, three keys | 441–478 each | 0.42–0.47 / 0.95–1.06% | p90 0.95–1.06% | — |

- All 15 evaluable keys and the whole population are within the A6 limits at phase level. No row exceeds 10% against either reference; the largest phase error is 8.48% (`g1hs-e0-0096-full`, HCA FULL decode reuse layers) and 78 rows exceed 5%.
- In CSA FULL decode, the same predictions that miss index compress-store by up to 28% p90 give the phase within 2.8–3.5% p90. The signed phase error there has median −0.27% and −0.28% and mean −0.07% and +0.08%, so the prediction has no systematic bias.
- The subset whose scopes all hit exact lookups (101 rows per CSA FULL decode key) gives 0.88 / 2.28% and 1.19 / 2.74%, so interpolation does not drive the result.
- The reference phase is itself one run. Its repeat-to-repeat CV is at most 2% (section 3), which leaves most of the 10% limit to the predictor.

### 5. Measurement correction options

The variation is a real property of FULL graph replay, not an instrument error, and it cancels in the composition the simulator uses. Each option below changes measurement, predictor code or acceptance, so each needs YC's decision. The scope stays FAIL until then.

| Option | What changes | Gain | Cost and risk | Evidence |
| --- | --- | --- | --- | --- |
| 1. Phase-level gate for concurrent scopes (recommended) | Scopes that run on concurrent streams between a fork and its join (query, compressor, index-query and index-compressor scopes, and the outer input projections) are judged by the composed attention phase per model key (sections 3 and 4), with unchanged folds, seed and limits (median ≤ 5%, p90 ≤ 10%). Serial scopes keep today's per-scope gate, which they already pass. Per-scope errors of concurrent scopes stay reported as diagnostics. | Tests the quantity the simulator times: the attention layer time is the composed elapsed (`frontier/entities/execution_time.py:1138`). That quantity repeats within 2%, so the gate tests the predictor rather than the GPU's start order. A per-key phase verdict is also stricter than today's gate, which pools each scope over all keys. | No GPU time; `v4_attention_phase_heldout.py` becomes part of the A6 evaluator; no simulator change. Risk: a concurrent scope's own error is not gated where it does not move the phase; per-scope durations still appear in the Chrome trace (`frontier/metrics/metrics_store.py:1186-1193`). | All 15 keys pass on the current single-run table (section 4). |
| 2. More samples per row, mean target for concurrent scopes | Measure several decode steps per CSA FULL decode workload and train the concurrent scopes on the mean instead of the median. | Per-scope targets settle: the mean of 16 samples stays within 8% p90 for a 50/50 mixture (section 2). | New GPU measurement (cost not estimated here) and a change of the predictor target in `dsv4_predictor.py`. A pooled per-scope gate may still fail on other keys. | Bootstrap only; not measured. |
| 3. Serialize the auxiliary streams while measuring | Remove the overlap during profiling. | Stable per-scope numbers. | Measures an execution the serving engine never runs; the composed phase would then be wrong. Not recommended. | — |

The four scopes that fail the formal per-scope gate are all concurrent scopes. The 21 failing per-key breakdowns of the formal run also include six PIECEWISE keys of the serial `dsv4_output_inv_rope_quant` (p90 10.25–11.92%); these do not decide the formal verdict, which is per scope, and option 1 leaves them as they are.

Interaction with D-A6-THREE-RUN: on the 989 workloads that already have three runs, median-of-three targets left CSA FULL decode at 21–25% p90 for q_projection and index compress-store (noise-floor report), consistent with section 2. Under option 1 the attention gate passes on the current single-run table, so the approved supplement passes (0.93 GPUh, blocked by I-009) would no longer be needed to qualify attention; they would only check robustness.

### 6. Impact on what a same-configuration vLLM comparison observes

YC's follow-up asks for the correction from the user's side: a user compares Frontier with a same-configuration vLLM run and does not collect kernel-level ground truth. The user therefore sees the four request metrics, which follow the forward time. This section propagates the variation to that level.

Command (CPU only, source `47cead2a` plus this script; output SHA `c4c30396…`, script SHA `3428c578…`):

```
env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp \
  timeout 600s /data/ycfeng/envs/frontier-py310/bin/python h3o_operator_costs/a6_csa_full_cause_20261101/user_impact.py \
  --phase-heldout h3o_operator_costs/a6_csa_full_cause_20261101/phase_heldout.json \
  --model-config data/config/models/deepseek-ai__DeepSeek-V4-Flash.json --layers 43 \
  --replay /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_32_warm_k19b29/forwards.jsonl.gz /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_k19b29/forwards.jsonl.gz \
  --package-plan h3o_operator_costs/a6_repeatability_20261101/package_plan.json --case-mapping h3o_operator_costs/a6_repeatability_20261101/case_mapping.json \
  --raw <the two probe raw files of section "Inputs"> --output h3o_operator_costs/a6_csa_full_cause_20261101/user_impact.json
```

**Metrics on the table that fails A6.** The B29 replays use the same attention table (`01a9d51b…`, in their receipts). Absolute relative error against the pooled native references, `codex_resume_20261031/*/reading.json`:

| Case | TTFT | TPOT | E2E | THPT |
| --- | ---: | ---: | ---: | ---: |
| P2-32 | 5.26% | 0.04% | 0.05% | 0.09% |
| P2-64 | 7.68% (repeat01 12.25%, FAIL) | 2.22% | 2.37% | 0.20% |

P2-64 native TTFT itself ranges 718.27–779.89 ms across its three repeats.

**Share of a real decode forward** (B29 per-forward records, serving, real decode forwards):

| Quantity | P2-32 | P2-64 |
| --- | ---: | ---: |
| Forwards; FULL share | 14,678; 98.3% | 23,708; 97.9% |
| Mean composed forward | 19.881 ms | 25.298 ms |
| Attention scope sum (upper limit: overlap counted twice) | 4.681 ms (23.5%) | 4.697 ms (18.6%) |
| The four varying scopes, sum | 0.827 ms (4.16%) | 0.964 ms (3.81%) |
| Those scopes each wrong by their worse-key p90 (27.4%, 13.1%, 30.4%, 28.2%), same sign, all on the critical path | 0.228 ms (1.15%) | 0.269 ms (1.06%) |

**Start order versus elapsed time** (16-repeat probe, four FULL workloads):

- Layer elapsed, together against staggered: −0.30% to −1.85%.
- Whole-forward attention (43 layers weighted by layer counts: CSA 1 + 20, HCA 1 + 19, uncompressed 2), every CSA layer together against every CSA layer staggered: −0.020 to −0.040 ms, which is −0.57% to −1.06% of attention and at most about 0.20% of a 19.88 ms forward.
- Whole-forward attention across 16 repeats, all eight workloads: CV 0.29–1.17%; mean over median −0.09% to +0.16%. The median-based table is therefore not a systematic error against the mean-based official metrics.

**Forward-level held-out.** Rolling the recorded held-out rows over the 43 layers, predicted attention against the measured layer windows (unchanged folds; 1,373 workloads, 24 skipped because a key is outside held-out coverage):

| Forward class | Workloads | Median / p90 absolute | Maximum | Signed mean |
| --- | ---: | ---: | ---: | ---: |
| FULL decode | 464 | 0.63 / 1.67% | 7.55% | −0.28% |
| PIECEWISE decode | 469 | 0.58 / 1.65% | 3.74% | −0.20% |
| PIECEWISE first_tail | 440 | 0.65 / 1.89% | 4.07% | −0.35% |

Inference, not a replay: with the upper-limit attention share, a 1.67% attention error moves the forward by about 0.39% (P2-32) and 0.31% (P2-64), and the 7.55% maximum by about 1.8% and 1.4%.

**Revised recommendation (replaces option 1 as stated in section 5).**

1. No simulator change and no new measurement for this variation. Modeling the start order would add machinery without moving any observable metric; options 2 and 3 settle per-scope numbers no user sees.
2. The attention verdict of A6 moves to whole-forward attention per workload against the measured layer windows, per forward class, with unchanged folds, seed and limits. Per-scope and per-key phase errors stay reported as diagnostics for every scope, with no separate concurrent/serial rule. A6 stays a required check for workloads the P2 cases do not exercise; the four metrics against vLLM stay the final verdict.
3. Evidence still needed, using data already paid for: replay P2-32 and P2-64 with attention tables built from each of the three independent passes (CPU only). The spread of the four metrics is how much measurement noise reaches the user, measured instead of inferred.

Trade-offs of a forward-level verdict: errors of opposite sign between layer types cancel. That cancellation is real for this model because the layer pattern is fixed; a different layer pattern needs its own run. TPOT agreement alone cannot replace the check, because EP collectives take 48–66% of the forward and could compensate.

Outside this decision, same viewpoint: the only failing official reading is P2-64 repeat01 TTFT. A6 evaluates no prefill or dummy row, and the attention table's prefill rows cover one length (6144 padded rows). What Frontier does for another prompt length was not run here.

## Conclusions

Observations:

1. In FULL graph replay, the two fused compress-store kernels of a CSA layer either start together or about 4–5 µs apart, varying between replays and between layers of one replay. Together, the index store is 24–47% slower and q_projection faster; staggered, the reverse. PIECEWISE replay always staggers them and stays stable.
2. A CSA FULL decode row of the published table holds one sample (first CSA layer) or four (reuse layers) of this two-mode mixture. That explains the measured noise floor of 35% and 21% p90 for those rows.
3. The composed attention phase repeats within a CV of 2% across 16 repeats and matches the measured layer window within 0.5%.
4. On the recorded A6 folds, the composed predicted phase is within 0.69 / 2.07% (median / p90) overall and within 3.70% p90 in every key, with no row above 10%. The same predictions miss index compress-store by up to 28% p90 in CSA FULL decode.

Inferences, not verified:

- Contention between the concurrent kernels is the most direct reading of the together/staggered durations and the overlap correlations. What sets the start order (graph launch timing, SM occupancy or clocks) is not measured.
- Option 1 would let attention qualify without new GPU time. Whether YC accepts a phase-level unit is a decision, not a finding.

## Limits

- One retained probe: four CSA FULL decode workloads at 16 and 24 padded rows, one H200, one source revision. Other batch sizes and context lengths are not covered.
- No hardware counter, clock or scheduler evidence: the start-order classes are observed, not explained.
- The bootstrap treats the pooled samples as independent. Samples from layers of the same replay can share a mode; this would make the real median spread larger, not smaller.
- The phase-level held-out evaluates a different quantity (the composed phase per row) than A6 (each scope per row). Its rows include rows where some scope is interpolated; the all-scopes-exact subsets are reported separately and give the same result.
- The phase reference is one measured run per row; only the 16-repeat probe (eight workloads) measures its repeatability.
- The probe's prompt contents and GPU cache/clock state were not fixed, as recorded in the probe report.
- Section 6: all attention times come from graph replay on one H200, not from a serving run. No native per-forward attention comparison exists, so the together/staggered mix of a real serving run is unmeasured; the sensitivity bound assumes these two classes are the ones that occur. The P2 cases are one workload per scale. H800 is not covered.
