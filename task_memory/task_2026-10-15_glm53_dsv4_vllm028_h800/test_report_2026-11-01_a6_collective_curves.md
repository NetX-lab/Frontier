# A6 collective curve attribution

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Reproduced all production curves and attributed their retained point errors without modifying inputs. |

## Result and evidence

Tickets 05/06/09: all eight production curve summaries exactly reproduce the frozen A6 result (seven FAIL, one PASS). All source payload medians reconstruct the production values. The diagnostic leaves one complete payload point out, using the production `_Curve.predict_us` and the unchanged A6 summary/threshold code. No measurement, simulator input, population or acceptance rule changed.

| Curve | Points / interior points | A6 median / p90 error | Worst measured / predicted (us) | Absolute error (us) / relative error | Original verdict |
| --- | --- | --- | --- | --- | --- |
| all_gatherv/ep/16 | 5 / 3 | 12.2631% / 12.8350% | 177.793598 / 154.719591 | 23.074007 / 12.9780% | FAIL |
| all_gatherv/ep/32 | 5 / 3 | 9.0120% / 11.0099% | 339.780807 / 300.674089 | 39.106719 / 11.5094% | FAIL |
| all_reduce/tp/8 | 8 / 6 | 8.8278% / 18.7931% | 9.934400 / 11.944411 | 2.010011 / 20.2328% | FAIL |
| logits_all_gather/tp/8 | 24 / 22 | 5.3963% / 9.0221% | 48.376000 / 42.150559 | 6.225441 / 12.8689% | FAIL |
| reduce_scatterv/ep/16 | 5 / 3 | 2.1399% / 7.2202% | 63.265598 / 68.636992 | 5.371394 / 8.4902% | PASS |
| reduce_scatterv/ep/32 | 5 / 3 | 30.6440% / 39.6705% | 290.723193 / 168.831427 | 121.891766 / 41.9271% | FAIL |
| sp_all_gather/tp/8 | 9 / 7 | 3.5267% / 22.9781% | 10.531200 / 13.313505 | 2.782306 / 26.4196% | FAIL |
| sp_reduce_scatter/tp/8 | 8 / 6 | 5.2514% / 8.9289% | 10.390400 / 11.392461 | 1.002060 / 9.6441% | FAIL |

## Observations and interpretation

- EP32 reduce-scatter at 1,048,576 input bytes is the largest error: 290.723193 us measured versus 168.831427 us predicted. Its upper measured neighbor is 201,326,592 bytes (192 times larger). The three source medians span 6.9% of their pooled median. The retained kernel name is the same at all three neighboring points; the upper point is eager and the two lower points use graph replay. A kernel-name change does not explain this specific failure. The sparse grid and measurement-mode change are observed confounders, not isolated causes.
- All-reduce at 131,072 bytes has 20.2328% LOO error despite equal graph modes and kernel signatures at its neighbors. Its three pass medians span 19.9% of their median. Thus kernel partitioning alone cannot explain every failed point.
- SP all-gather at 131,072 bytes has 26.4196% LOO error. The next point is 6,291,456 bytes (48 times larger), changing from the vLLM Lamport kernel to NCCL and from graph to eager. Logarithmic interpolation bridges that interval. SP reduce-scatter also crosses a kernel/mode change near its worst point; its median criterion fails although its pooled p90 passes.
- Logits all-gather has 24 consecutive integer batch sizes. Its largest error is at batch 17 (549,440 input bytes): measured 48.376000 us, predicted 42.150559 us, absolute error 6.225441 us. Neighboring kernels/modes are equal and that point has only 3.1% pass spread. Its original median error is 5.3963%; p90 is 9.0221%. This is a median failure and cannot be reported as a p90 failure.

A diagnostic per-pass reconstruction uses exactly the same payload sizes and pooling within each pass. Six of the seven originally failing curves fail in all three passes. Logits fails in pass 1 and passes in passes 0/2. The originally passing EP16 reduce-scatter fails in each separate pass; one source point spans 72.6%. These comparisons do not add per-pass admission criteria and do not override the original pooled verdict. They show that selecting a favorable pass or re-pooling existing passes is not a general correction.

Decision under D-NO-INTERRUPT: retain the current formal measured input and all A6 failures. A generic smoothing change or extra averaging is not supported as a complete remedy. A subsequent correction must account for the actual discrete touched payloads, kernel/mode transitions, and measurement variation; the large prefill/decode gap alone does not establish that additional reachable intermediate workloads exist. Finish the already-running attention repeatability experiment and F comparisons before selecting new measurements. Stop expanding this collective analysis at this evidence boundary.

## Verification

CPU Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6; workspace `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; source `8b530816e43ac6b5b655af44c62a3c5998b616e9` with the new analysis script uncommitted during execution. Exact main command and environment are in `h3o_operator_costs/a6_collectives_20261101/command.json`. Exit 0, elapsed 1.453960 s; core analysis 0.135531 s. PASS for reproduction, not qualification.

```bash
env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_collectives_20261101/compare_passes.py
```

The per-pass command exited 0 and produced all 24 curve/pass summaries. Expected: each pass preserves the full production payload grid, and each summary uses the unchanged production LOO predictor and A6 limits. Actual: all payload-grid assertions passed. There was no training or GPU operation.

Artifacts: `tests/analysis/v4_collective_curve_errors.py`; task-local `h3o_operator_costs/a6_collectives_20261101/{command.json,run.log,result.json,compare_passes.py,pass_comparison.json}`. The result retains every point, raw CSV row, original and predicted value, signed/absolute/relative error, neighbors, modes and kernel signatures. All actual values are positive; no undefined relative error was replaced by zero.

Input: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/data/profiling/network/h200_dgx/summary.csv`, SHA-256 `1a7c701f2701aa94440f091fb6c5d03477b61f81ad7d43924f0a51008f3d4c65`. The original A6 result hash is `65132e28c3713462e194f544a773137a807810801bd096418130286edba55460`. The worker repeatability probe and three existing CPU replays remain independent and active.

Verification limits: this is a CPU reconstruction of retained measurement tables, not a new native timing experiment. Three passes do not establish noise causality, cache/clock behavior, or the correct interpolation family. Existing formal qualification, NIXL payload/memory limits, and per-reference E2E gates remain open.
