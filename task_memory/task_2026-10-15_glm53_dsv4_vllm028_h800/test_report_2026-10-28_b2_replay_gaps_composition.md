# Test report: stage 1a assignment B2 (replay intervals, V4 gap reducer, A4 composition check)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead verification of B2 at `32afae01`, the A4 result on the P1h bridge data, the gap-model and gap-transfer diagnostics, and the remedy recommendation. |

## 1. Scope

- Change under test: commit `32afae01` "H3o-(b) stage 1a B2: replay intervals, V4 gap reducer, A4 composition check" (17 files). Codex request `0feb7dfd-4718-49ae-98ba-e448419f6bc1`, assignment [h3o_operator_costs/codex_assignment_B2_replay_gaps_composition_check.md](h3o_operator_costs/codex_assignment_B2_replay_gaps_composition_check.md), report `h3o_operator_costs/codex-implement-0feb7dfd-4718-49ae-98ba-e448419f6bc1/implementation.md`.
- Data: the P1h bridge run `exp-1028-031147-398843` (1 H200, `package_02`, source `548ec6c0`), downloaded to `/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_02/download/P1H_BRIDGE/bridge`. The raw files of that run predate B2, so they were rebuilt on CPU from the saved traces and capture records (`rebuild_bridge_raw.py`). No GPU was used for this report.
- Acceptance (design §5 stage 1a and [stage1a_replay_route.md](h3o_operator_costs/stage1a_replay_route.md) §4): the CPU suites pass with only the known baseline failure; the rebuilt call rows equal the worker's call rows; the bridge report still passes; the reducer writes a CSV only when every class has a host-ahead sample; A4 reports every group without pooling coordinates and stops the route when a group fails.

## 2. Environment

- Host: local development machine, CPU only.
- Python 3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python` (no `torch`, no `vllm`).
- `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration`, `PYTHONDONTWRITEBYTECODE=1`, `ulimit -v 31457280`, `timeout 600` per command.
- Scratch: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_02/` (`b2_rebuild/` rebuilt raw records and reports; `b2_final_checks/` pytest base and reducer output; `analysis/` scripts).
- Bridge config: `h3o_operator_costs/p1h_bridge/package_02/bridge_config.json`, copied to `b2_rebuild/bridge_config.json` with `output_root` set to `b2_rebuild`.

## 3. Lead corrections before the commit

1. **Ratio convention in A4.** The engine module records `compress_ratio = max(1, native ratio)` (`dsv4_attention_runner.py:280`), while the plan keeps the native `compress_ratios` entry, 0 for uncompressed layers (`dsv4_attention_rows.py:250`). Codex's strict inventory check therefore failed on every mode. `planner_roles` now compares against `max(1, layer.compress_ratio)`; the test `test_real_planner_uncompressed_ratio_zero_matches_engine_ratio_one` builds a real plan with ratio 0 and checks the roles.
2. **Graph-completion dependency of eager breaks in graph-mode steps.** In a PIECEWISE step, the replay nodes of a piecewise graph are reported on other Kineto streams (368-371), while the eager break that follows runs on the launch stream (7). The eager break's dependency was the end of the last same-stream activity, so it ignored the graph's still-running nodes and measured gaps that were too large. The reducer now adds the end of the whole preceding graph launch, `graph_ends.before(start)`, for a scope-owned eager activity when the step mode is not NONE, and rejects such an activity that precedes every graph activity. Effect: the `eager` class mean falls from 3.508 to 1.935 µs, and the timeline of a PIECEWISE prefill CSA call composes to its measured window (202.6 = 202.6 µs, `evidence/p1h_bridge/b2/a4_call_timeline_piecewise_prefill_csa.txt`). Two unit tests cover the rule and its rejection.

## 4. Commands and results

```bash
PY=/data/ycfeng/envs/frontier-py310/bin/python
B=/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_02/b2_rebuild
T=/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_02/b2_final_checks
```

| Check | Command | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| B2 suites | `$PY -m pytest tests/unit/test_dsv4_replay_gaps.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_graph_capture.py tests/unit/test_graph_replay_trace.py -q -p no:cacheprovider` | all pass | 217 passed (Codex 215 + 2 lead tests) | PASS |
| Composition suites | `$PY -m pytest tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsv4_composition.py -q` | all pass | 72 passed | PASS |
| Earlier profiling suites | `$PY -m pytest tests/unit/test_dsa_query_one_profiling.py tests/unit/test_dsv4_attention_profiling.py -q` | all pass | 184 passed | PASS |
| Kernel-gap suites | `$PY -m pytest tests/unit/test_kernel_gap_pricing.py tests/unit/test_kernel_gap_profiler.py -q` | baseline: 1 known failure | 29 passed, 1 failed (`test_every_collective_launches_one_kernel_by_default`, `OperatorSpec` has no `kernel_count`; failing at `0bc14e61` too) | PASS (baseline) |
| Import isolation | import the 5 changed modules and both scripts, then scan `sys.modules` | no `torch`, `vllm`, `cuda` | none loaded | PASS |
| File size | `wc -l` | ≤ 2,000 lines | largest changed file `dsa_vllm_runner.py` 839 | PASS |
| Rebuild | `rebuild_bridge_raw.py --source … --config … --output $B` | rebuilt call rows equal the worker rows | `call_rows_equal: true` for all 12 (mode, case) pairs (`rebuild_summary.json`) | PASS |
| Bridge report | `$PY tests/comparison/calibration/dsv4_graph_bridge_check.py --config $B/bridge_config.json --report` | PASS, unchanged kernel counts | `PASS: 3 modes, FULL=24, PIECEWISE=72`, exit 0 | PASS |
| Reducer | `$PY -m frontier.profiling.kernel_gap.dsv4_replay_gaps --raw_files $B/{NONE,PIECEWISE,FULL_AND_PIECEWISE}/raw/dsv4_raw_e0.jsonl --output_dir $T/gap_output --device h200 --model DeepSeek-V4-Flash --diagnostics_path $T/gap_diagnostics.json` | no CSV unless every class has a host-ahead sample | `eager_preparation: no host-ahead sample`, exit 1, no CSV; diagnostics byte-identical to the saved evidence | PASS (as designed) |
| A4 | `$PY tests/comparison/calibration/dsv4_composition_check.py --config $B/bridge_config.json` | verdict over all groups | `FAIL: 74 groups, {no_host_ahead_call: 38, fail: 28, pass: 8}`, exit 1; report byte-identical to the saved evidence | route stop rule triggered (section 6) |

## 5. Reducer values (host-ahead samples only, µs)

| Class | Samples | Host-ahead | Mean | Median | p10 | p90 | Min | Max | Negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda_graph` | 6,216 | 6,192 | 0.4029 | 0.4478 | 0.0322 | 0.9277 | −4.128 | 4.736 | 507 |
| `cuda_graph_piecewise` | 17,304 | 17,102 | 0.9618 | 0.4160 | 0.0 | 1.1519 | −1.728 | 24.449 | 1,692 |
| `eager` | 7,996 | 3,156 | 1.9345 | 1.2148 | −0.6084 | 4.8320 | −4.224 | 32.416 | 483 |
| `eager_preparation` | 8,698 | 0 | — | — | — | — | — | — | — |

`eager_preparation` has 8,602 host-bound samples and 96 without a dependency end. Observation: the bridge engine is host-bound outside graph replay, so no preparation sample is host-ahead. B3 adds a host-ahead hold so that this class gets samples.

## 6. A4 result

Every evaluated group has target TP 8, rows at or below the multi-stream threshold 1024, and `indexer_short_context = False`. Errors are relative to the measured window (composed − target) / target. "Diagnostic" composes each call with its own measured signed gaps instead of the class means.

| Engine mode / step mode | Case | Variant | Planner | Host-ahead / calls | Median | p90 | Signed median | Diagnostic median | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FULL_AND_PIECEWISE / FULL | decode-b1-p256 | CSA | T / F | 8/8, 8/8 | 1.57%, 1.68% | 1.99%, 2.26% | +1.55%, +1.68% | ≈1e-16 | pass, pass |
| FULL_AND_PIECEWISE / FULL | decode-b1-p256 | HCA | T / F | 8/8, 8/8 | 5.72%, 5.79% | 6.19%, 6.12% | +5.72%, +5.79% | ≈1e-16 | fail, fail |
| FULL_AND_PIECEWISE / FULL | decode-b1-p256 | uncompressed | T / F | 8/8, 8/8 | 5.11%, 6.92% | 5.47%, 7.40% | +5.11%, +6.92% | ≈1e-16 | fail, fail |
| FULL_AND_PIECEWISE / FULL | decode-b4-p2048 | CSA | T / F | 8/8, 8/8 | 1.62%, 1.72% | 2.22%, 2.45% | +0.43%, +0.80% | ≈1e-16 | pass, pass |
| FULL_AND_PIECEWISE / FULL | decode-b4-p2048 | HCA | T / F | 8/8, 8/8 | 5.43%, 5.35% | 6.07%, 5.79% | +5.43%, +5.35% | ≈1e-16 | fail, fail |
| FULL_AND_PIECEWISE / FULL | decode-b4-p2048 | uncompressed | T / F | 8/8, 8/8 | 4.43%, 5.94% | 4.74%, 6.58% | +4.43%, +5.94% | ≈1e-16 | pass, fail |
| FULL_AND_PIECEWISE / FULL | decode-b8-p2052 | CSA | T / F | 8/8, 8/8 | 1.84%, 2.18% | 2.19%, 2.33% | +1.84%, +2.18% | ≈1e-16 | pass, pass |
| FULL_AND_PIECEWISE / FULL | decode-b8-p2052 | HCA | T / F | 8/8, 8/8 | 5.54%, 5.46% | 5.84%, 5.85% | +5.54%, +5.46% | ≈1e-16 | fail, fail |
| FULL_AND_PIECEWISE / FULL | decode-b8-p2052 | uncompressed | T / F | 8/8, 8/8 | 4.11%, 5.94% | 4.85%, 6.34% | +4.11%, +5.94% | ≈1e-16 | pass, fail |
| FULL_AND_PIECEWISE / PIECEWISE | prefill-p4095 | CSA | T / F | 24/24, 24/24 | 5.76%, 5.96% | 7.48%, 7.53% | −5.76%, −5.96% | ≈1e-16 | fail, fail |
| FULL_AND_PIECEWISE / PIECEWISE | prefill-p4095 | HCA | T / F | 24/24, 24/24 | 6.27%, 7.07% | 6.71%, 7.39% | +6.27%, +7.07% | ≈1e-16 | fail, fail |
| FULL_AND_PIECEWISE / PIECEWISE | prefill-p4095 | uncompressed | F | 24/24 | 8.93% | 9.25% | +8.93% | ≈1e-16 | fail |
| PIECEWISE / PIECEWISE | decode-b4-p2048 | HCA | T / F | 8/8, 8/8 | 8.37%, 8.65% | 8.56%, 9.05% | +8.37%, +8.65% | ≈1e-16 | fail, fail |
| PIECEWISE / PIECEWISE | decode-b4-p2048 | uncompressed | F | 8/8 | 8.70% | 9.15% | +8.70% | ≈1e-16 | fail |
| PIECEWISE / PIECEWISE | decode-b8-p2052 | CSA | T / F | 7/8, 8/8 | 8.24%, 8.11% | 8.58%, 8.40% | +8.24%, +8.11% | ≈1e-16 | fail, fail |
| PIECEWISE / PIECEWISE | decode-b8-p2052 | HCA | T / F | 8/8, 8/8 | 8.65%, 8.17% | 8.96%, 8.56% | +8.65%, +8.17% | ≈1e-16 | fail, fail |
| PIECEWISE / PIECEWISE | decode-b8-p2052 | uncompressed | F | 8/8 | 8.81% | 9.12% | +8.81% | ≈1e-16 | fail |
| PIECEWISE / PIECEWISE | prefill-p4095 | CSA | T / F | 24/24, 24/24 | 6.65%, 6.81% | 8.44%, 8.92% | −6.65%, −6.81% | ≈1e-16 | fail, fail |
| PIECEWISE / PIECEWISE | prefill-p4095 | HCA | T / F | 24/24, 24/24 | 6.26%, 6.30% | 6.51%, 6.96% | +6.26%, +6.30% | ≈1e-16 | fail, fail |
| PIECEWISE / PIECEWISE | prefill-p4095 | uncompressed | F | 24/24 | 8.43% | 8.72% | +8.43% | ≈1e-16 | fail |

The 38 `no_host_ahead_call` groups are all 26 NONE groups, the 6 PIECEWISE b1 decode groups, and 6 further PIECEWISE groups (decode b4 3, decode b8 1, prefill 1 in each graph engine mode); every call in them has a host-bound owned activity.

Observations:

- The diagnostic error is at floating-point level in all 36 evaluated groups. The stream program of `compose_attention_phase` and the scope predecessors therefore reproduce every measured window once each call's own gaps are used. The whole A4 error comes from replacing those gaps with one value per class.
- The class values overprice uncompressed and HCA layers and PIECEWISE decode by 4-9%, and underprice PIECEWISE prefill CSA by 6-7%. The sign of the error is consistent within a group (signed median ≈ median), so the error is a bias, not noise.

Inference: one gap value per segment class is too coarse for V4 attention. The gaps depend on the scope (which kernel waits on which, and on which stream) and, in prefill, on SM contention between parallel branches (in the PIECEWISE prefill CSA call of `a4_call_timeline_piecewise_prefill_csa.txt`, the `fused_qkv` deep_gemm waits 21.4 µs and the index-compressor projection 20.3 µs after their dependencies).

## 7. Diagnostics on remedies (not acceptance)

### 7.1 Gap models (`evidence/p1h_bridge/b2/a4_gap_models.py`, output `a4_gap_models.txt`)

The script refits each model on the attention-owned gaps of the host-ahead calls only (the reducer's class means also include non-attention activities), then composes each call. "Fold" models add each call's measured intra-scope gaps to the scope cost and model only the scope entry gap. "Held-out" fits on three cases and evaluates the fourth.

| Model | Groups | Pass in-sample | Pass held-out case |
| --- | --- | --- | --- |
| `class` (one value per class, refit on attention calls) | 20 | 6 | 8 |
| `class_position` (class, entry or intra-scope) | 20 | 7 | 9 |
| `class_scope` (class, scope) | 20 | 11 | 13 |
| `fold_class_edge` (fold; entry per class and same/cross-stream) | 20 | 11 | 13 |
| `fold_class_scope` (fold; entry per class and scope) | 20 | 19 | 17 |

The worst held-out `fold_class_scope` group is PIECEWISE decode b8 CSA (+18.9%): its entry gaps were fitted on the prefill case, whose contention gaps do not apply to decode.

### 7.2 Per-scope gap transfer between decode shapes (`a4_scope_gap_transfer.py`, `a4_scope_gap_transfer.txt`)

Each call of a decode group is composed with the per-scope mean gaps (entry and intra-scope) of the nearest other decode shape (b1→b4, b4→b8, b8→b4) in the same engine mode, step mode and variant.

| Groups | Calls per group | Median error | p90 error | Signed median |
| --- | --- | --- | --- | --- |
| 13 FULL and PIECEWISE decode groups | 8-16 | 0.2-0.7% | 0.5-1.2% | −0.3% to +0.7% |

Prefill transfer cannot be tested: the bridge has one prefill case.

## 8. Decision and recommendation

Design §3.3: "One value per segment class is an approximation; A4 checks it per group, and a failing group stops the route." The pilots row and §9 item 2 send the decision to YC; no factor is added.

Recommendation (b): record each attention scope's measured signed gaps (its entry gap and its intra-scope kernel gaps, host-ahead samples only) with that scope's cost row at its shape, as a separate gap column, so the kernel-time meaning of the cost column is kept. The predictor learns the gap column like the duration. The composition weight of a scope becomes its cost plus its gap. The class values stay for non-attention operators and pre- and post-forward work. A4 then checks the per-call composition (the current diagnostic), and generalization across shapes moves to A6 on the rest-of-P1h grid. The change falls in entries 1, 3 and 10, which are not implemented yet. Under D-NO-INTERRUPT the lead adopts (b) by default and continues B3, which does not depend on it, unless YC decides otherwise.

## 9. Limits

- All A4 and remedy numbers are in-sample on one bridge run with four cases (three decode shapes and one prefill), 8 decode steps and 24 prefill steps per group.
- The bridge engine is host-bound in NONE and partly in PIECEWISE b1, so 38 groups have no host-ahead call, and `eager_preparation` has no value. B3's host-ahead hold is needed before those groups can be evaluated.
- The driver version was not recorded in the bridge run (B2 adds it for later runs).
- H200 only; it does not qualify H800.
