# A6 measurement noise floor, three-run target tables and touched collective points

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Created: measurement noise floor, three diagnostic A6 reruns, per-key breakdown, touched collective points and extra-pass costs. No formal input, threshold or GPU run changed. |

## Purpose and scope

Next action 1 after the F comparisons: explain the A6 attention failures and the collective-curve qualification problem using retained data only. The 16-repeat full re-measurement draft is not executed. Measured CC, K19, ingress 10.3 ms and handoff 11.4 ms stay unchanged. Every result here is diagnostic; the published table (`01a9d51b…`) and the A6 thresholds (median ≤ 5%, p90 ≤ 10%, KFold 5, seed 1234) are unchanged.

## Environment and commands

- Workspace `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6. Temporary files under `/data/ycfeng/tmp/glm53_dsv4/`. CPU only; no platform operation.
- Scripts: `tests/analysis/v4_attention_noise_floor.py` (SHA `bb0b5f65…`) and `tests/analysis/v4_attention_multipass_table.py` (SHA `93e56b58…`), committed in `c4a0244b`; the A6 evaluator `tests/analysis/v4_attention_heldout.py` (SHA `3a13ad22…`) is unchanged.
- Inputs: G1h-02 reference rows `/data/ycfeng/tmp/b13r_accept/after/rows.json` (`ba1ad088…`), G2h-01 `/data/ycfeng/tmp/g2h_accept/rows_01/rows.json` (`ad8cd954…`), G2h-04 `/data/ycfeng/tmp/g2h_accept/rows_04/rows.json` (`9006bbab…`). All three measure the same 4,945 rows (989 workloads × 5 layer rows). The 432 supplement-only workloads (`g1hs-*`) have one run.
- Exact commands, exit codes, wall times and output hashes: `h3o_operator_costs/a6_noise_20261101/commands.json`. The A6 reruns used source `c4a0244b` with zero dirty source paths.
- Incident: the first table-builder run was piped through `grep`, which lost its exit status after one table was written. The partial scratch directory was removed and the unchanged command rerun with redirected output (exit 0).

## Results

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Noise floor (`noise_floor.json`) | Three runs share every identity; report per-scope error of the reference against the mean of the two other runs | 4,945 shared identities; per-scope table below | PASS (diagnostic) |
| 2 | Three-run tables (`multipass_tables_manifest.json`, `check_multipass_tables.log`) | Published table rewrites byte-identically; only shared-row kernel-time medians change | Rewrite byte-identical; 4,945 shared rows, 67,244 medians replaced (41,266 cells numerically differ); no other column differs | PASS |
| 3 | A6 reruns on the three tables | Exit 0; legally-zero PASS; zero dirty source | Exit 0 (2,690 s, 2,685 s, 2,801 s); legally-zero PASS; dirty 0 | PASS |
| 4 | Per-key breakdown (`per_key_breakdown.json`) | Locate the remaining failures | CSA FULL decode keys hold the remaining large errors | Diagnostic |
| 5 | Touched collective points (`touched_collective_points.json`) | Census price equals pooled median; report per-pass spread of the points P2 uses | Assertion holds for every point; spread below | PASS (diagnostic) |
| 6 | Extra-pass costs (`supplement_pass_costs.json`) | Same timing model as the repeatability probe | Costs below | Estimate only |

### 1. Noise floor

Error of the G1h-02 reference value against the mean of G2h-01 and G2h-04, divided by the reference (as A6 does). No model is fitted, so this is the error a perfect predictor would show against a single-run target.

| Scope | Formal A6 median / p90 | Noise median / p90 | Rows above 10% | G2h-01 vs G2h-04 median / p90 |
| --- | ---: | ---: | ---: | ---: |
| q_projection | 1.70% / 10.23% FAIL | 1.41% / 11.38% | 11.1% | 1.61% / 12.44% |
| index_fused_query_prepare | 4.65% / 12.27% FAIL | 4.68% / 13.89% | 19.6% | 5.24% / 15.46% |
| compressor_fused_compress_store | 3.24% / 13.99% FAIL | 2.85% / 13.07% | 14.6% | 3.31% / 14.83% |
| index_compressor_fused_compress_store | 2.93% / 17.51% FAIL | 2.90% / 17.11% | 18.9% | 3.17% / 19.01% |
| fused_qkv_rmsnorm (passing) | 3.57% / 8.72% | 3.68% / 10.63% | 11.5% | 4.00% / 12.43% |
| compressor_save_partial_states (passing) | 4.00% / 9.48% | 3.78% / 10.46% | 11.5% | 4.38% / 12.02% |

The four failing p90 values sit at the noise floor of their own targets. Two passing scopes also have a noise p90 above 10%; they pass A6 because the training mean averages several noisy rows.

Noise by layer group (median / p90, groups with p90 above 8%; full list in `noise_floor.json`):

- q_projection: CSA FULL decode first layer 13.55% / 36.71%, reuse layers 6.68% / 17.92%; CSA PIECEWISE first layer 2.4–2.7% / 19–20%.
- index_compressor_fused_compress_store: CSA FULL decode first layer 11.96% / 35.05%, reuse layers 7.62% / 21.10%; PIECEWISE decode first layer 2.60% / 9.34%.
- compressor_fused_compress_store: CSA FULL decode first layer 6.74% / 24.83%; PIECEWISE decode first layer 6.31% / 21.90%; PIECEWISE first_tail first layer 5.38% / 33.80%; HCA FULL decode first layer 4.59% / 11.64%.
- index_fused_query_prepare: every CSA group, first layer 16.6–18.3% p90, reuse layers 8.3–10.0%.
- HCA and uncompressed layers are low for the other scopes.

Noise concentrates in CSA layers, most of all the first CSA layer (the one that plans the tile schedule) and FULL graph mode. The cause is not established.

### 2–3. A6 with three-run targets

`original_shared` keeps the 989 shared workloads with published values. `median3_shared` replaces each of their kernel-time medians by the median of the three runs. `median3_all` keeps all 1,421 workloads with only the 989 replaced. Folds and thresholds are unchanged. Median / p90:

| Scope | formal (1,421, one run) | original_shared (989, one run) | median3_shared (989, three runs) | median3_all (1,421, mixed) |
| --- | --- | --- | --- | --- |
| q_projection | 1.70 / 10.23 FAIL | 1.42 / 8.83 PASS | 0.98 / 6.34 PASS | 1.38 / 8.90 PASS |
| index_fused_query_prepare | 4.65 / 12.27 FAIL | 4.59 / 12.21 FAIL | 3.08 / 8.44 PASS | 3.38 / 10.48 FAIL |
| compressor_fused_compress_store | 3.24 / 13.99 FAIL | 2.97 / 11.52 FAIL | 2.03 / 7.30 PASS | 2.54 / 10.33 FAIL |
| index_compressor_fused_compress_store | 2.93 / 17.51 FAIL | 3.16 / 17.48 FAIL | 2.50 / 13.86 FAIL | 2.53 / 15.25 FAIL |

All other evaluable scopes pass in all four runs, and the three-run target lowers p90 in every scope. Comparing `original_shared` with `median3_shared` isolates the target change on the same population: three of four failing scopes pass. `median3_all` shows that three runs on only 989 workloads are not enough for the full population.

Median-of-three to original ratio for the replaced values (p10 / p50 / p90): q_projection 0.966 / 1.000 / 1.025; index_fused_query_prepare 0.901 / 0.988 / 1.073; compressor_fused_compress_store 0.934 / 0.991 / 1.055; index_compressor_fused_compress_store 0.913 / 0.995 / 1.056. The median does not shift systematically.

### 4. Remaining failures by model key

Median / p90 per model key, columns formal | original_shared | median3_shared | median3_all:

| Key | Values |
| --- | --- |
| index_compressor_fused_compress_store / CSA / FULL / decode | 11.15 / 26.27 \| 11.08 / 25.53 \| 8.87 / 21.64 \| 9.31 / 22.76 |
| index_compressor_fused_compress_store / CSA / PIECEWISE / decode | 2.00 / 5.99 \| 2.05 / 6.11 \| 1.65 / 4.43 \| 1.74 / 4.70 |
| q_projection / CSA / FULL / decode | 9.42 / 25.61 \| 9.23 / 26.21 \| 7.52 / 25.00 \| 8.05 / 24.80 |
| q_projection / CSA / PIECEWISE / decode | 4.75 / 13.12 \| 2.39 / 9.00 \| 1.34 / 4.35 \| 3.79 / 10.99 |
| compressor_fused_compress_store / CSA / FULL / decode | 5.48 / 20.96 \| 4.41 / 15.56 \| 2.87 / 7.81 \| 3.94 / 15.28 |
| compressor_fused_compress_store / CSA / PIECEWISE / first_tail | 5.06 / 23.38 \| 3.79 / 16.32 \| 2.13 / 8.11 \| 3.23 / 20.14 |

- Three runs fix most keys but barely change CSA FULL decode for q_projection (p90 26.21% → 25.00%) and index_compressor (25.53% → 21.64%). q_projection passes A6 only because the gate is per scope and its other keys are accurate.
- The 16-repeat probe showed the FULL index-compression control with 19.96% CV and a 20.50% difference between its first-eight and last-eight medians. More repeats per workload are therefore unlikely to bring this key under 10% p90.
- Size (`csa_full_decode_share.json`): in CSA FULL decode rows, q_projection holds 5.56% (first layer) and 6.60% (reuse layers) of the summed attention kernel-time medians; index_compressor holds 4.53% and 5.37%. The attention layer is only part of a decode step, so these are upper bounds on the share of step time.

### 5. Collective points P2 uses

The A8 census shows P2-32 and P2-64 price every collective call at an exact measured point, with zero interpolation. The leave-one-out A6 rule tests interpolation between points, which P2 does not use. Per-pass spread of the exact points P2 uses (uniform rows only):

| Replay | Points | Predicted time | Share of time on points with a pass > 10% from the pooled median | Time-weighted max pass deviation |
| --- | ---: | ---: | ---: | ---: |
| P2-64 | 30 | 675,720 ms | 0.46% | 0.81% |
| P2-32 | 40 | 329,617 ms | 14.74% | 5.76% |

- P2-64: the dominant EP32 points (all_gatherv 8,264 B and reduce_scatterv 262,144 B, 627 s together) stay within 0.58–0.63% maximum deviation.
- P2-32: decode reduce_scatterv/ep/16 at 262,144 B (47,180 ms) has a 19.33% pass range and 17.25% maximum deviation; decode all_gatherv/ep/16 at 16,528 B (105,604 ms) has 7.38%.
- Unequal-SP rows use a separate exact table and were not joined: 21,392 ms (P2-32) and 88,751 ms (P2-64).

### 6. Extra-pass costs

Same timing model as the repeatability probe (`len(engines) × 486.104 s + repeats × (steps × 0.205 s + cases × 0.2 s)`):

| Workloads | Engines | 1 pass | 2 passes |
| --- | --- | ---: | ---: |
| 432 supplement-only | 0, 1 | 0.598 GPUh | 0.926 GPUh |
| 989 original | 0, 1, 2 | 0.980 GPUh | 1.555 GPUh |
| all 1,421 | 0, 1, 2 | 1.308 GPUh | 2.211 GPUh |

The remaining approximate G1h–G4h budget is 2.160175 GPUh. Two passes for the 432 supplement-only workloads would give every workload three runs (with G1h and the existing G2h-01/04) at about 0.93 GPUh. Two passes over all workloads (2.21 GPUh) exceed the budget.

## Conclusions

Observations:

1. The four failing attention scopes fail at the noise floor of their single-run targets. The error is in the targets, not in the estimator.
2. On the same 989 workloads, a median-of-three target makes three of the four failing scopes pass with unchanged folds and thresholds.
3. CSA FULL decode stays at 21–25% p90 for q_projection and index_compressor_fused_compress_store even with three runs. index_compressor_fused_compress_store still fails at scope level.
4. P2 uses exact collective points only. P2-64's points are repeatable within 1% time-weighted; P2-32 has one heavy decode point with 17% maximum pass deviation.

Inferences, not verified:

- Two supplement passes would likely make the full 1,421-workload population behave like `median3_shared`: three scopes pass and index_compressor still fails. The supplement workloads have not been measured more than once, so this is not established.
- CSA FULL decode index compression is not repeatable at the 10% level with the current measurement method. Its cause (cache state, clocks, graph replay timing, or instrumentation) is not isolated.

## Limits

- The G2h runs come from runner-replay packages whose sampling scope extends past `sample_tokens`. Attention identities and native hashes match G1h-02 and median ratios are 0.989–1.022, but gap values are 0.7–7% lower. Gap statistics do not enter the kernel-time error.
- Three runs exist only for 989 of 1,421 workloads.
- The collective per-pass medians pool all scenarios in each pass; unequal-SP rows were not evaluated.
- The share bound in section 4 covers attention-layer kernel time only, not decode-step time.
- No formal input, threshold or acceptance rule changed. Adopting a three-run recipe, a new table, or a different collective qualification rule needs a YC decision.
