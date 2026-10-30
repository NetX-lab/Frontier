# Test report: A10 operator-path E2E replay of P2-32 and P2-64 on H200 (stage 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created: P2-64 client reading on the pre-supplement table (probe `p2_64_c`), the loss of the promoted-table runs to the full disk (I-003), and their restart. |

## Scope

A10 replays the two H200 P2 cases on the V4 operator path (D-OPERATOR-LEVEL-ROUTE, ADR 0002: no whole-step cost table) and reads the client metrics against the persisted native vLLM reference with `tests/comparison/e2e_gate/run_gate.py`. Each case runs twice: cold (empty predictor cache, the simulator trains every model) and warm (cache from a cold run, no fitting).

- Driver: `h3o_operator_costs/a10/a10_driver.py <case> <root>`, launched by `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10.sh <case> <name>` (root `stage4/a10/<name>`, `ulimit -v 62914560`, `timeout 86400`).
- Gate: `h3o_operator_costs/a10/a10_gate.py <case> <frontier run dir> <output dir>`, run from the `a10/` directory with `PYTHONPATH=$REPO:$A10:$REPO/tests/analysis`.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6); `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, BLAS/OMP/MKL threads 1.
- Gate rule: client gate, relative error limit 0.10 per metric. TTFT semantic `request_issue_to_first_output_token`. Native reference: three repeats of 128 requests (`vllm/dsv4-h200-p2-64-canonical-20261025-01-repeat01..03`). Routing status MATCH (R0 and A13 (b)).

## 1. P2-64 on the pre-supplement table (probe `p2_64_c`)

This run is not the official A10 reading. It used the attention table committed before the G1h supplement (main `5561d762`) and ran without the B21 observer. P2-64 had no replay coverage misses on that table, so the run completed.

- Command: `stage4/run_probe_c.sh p2_64 p2_64_c` (root `stage4/probe/p2_64_c`).
- Receipt: `status FINISHED`, `init_wall_s 606.4`, `run_wall_s 20,391.7` (5.66 h).
- Gate output: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10_gate/p2_64_c_old_table/` (`matrix.json`, `reading.json`, `rows/`); copied from the original location under `/tmp`.

| Metric | Native vLLM (3 repeats) | Frontier | Absolute error | Relative error | Limit | Result |
| --- | --- | --- | --- | --- | --- | --- |
| TTFT mean | 741.216 ms | 417.061 ms | −324.155 ms | 43.73% | 10% | FAIL |
| TPOT mean | 25.871 ms | 25.299 ms | −0.572 ms | 2.21% | 10% | PASS |
| Request E2E mean | 27,206.770 ms | 26,297.824 ms | −908.946 ms | 3.34% | 10% | PASS |
| Token throughput (window rate) | 5,963.206 tps | 5,985.814 tps | +22.608 tps | 0.38% | 10% | PASS |

Per native repeat (relative error of the same Frontier value):

| Repeat | TTFT native | TTFT error | TPOT native | TPOT error | E2E native | E2E error | Throughput native | Throughput error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeat01 | 779.89 ms | 46.52% | 25.43 ms | 0.52% | 26,794.64 ms | 1.85% | 5,948.03 tps | 0.64% |
| repeat02 | 725.49 ms | 42.51% | 26.26 ms | 3.64% | 27,584.86 ms | 4.67% | 5,976.59 tps | 0.15% |
| repeat03 | 718.27 ms | 41.94% | 25.93 ms | 2.42% | 27,240.80 ms | 3.46% | 5,965.07 tps | 0.35% |

Reading: overall FAIL, on TTFT only. The native TTFT spread across repeats is 718-780 ms, so the Frontier value of 417 ms lies far outside the native run-to-run variation. Frontier predicts the first token too early by about 324 ms on average; the decode-side metrics agree within 5%. The cause is not yet diagnosed.

## 2. Official runs on the promoted table

The promoted table is `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv` (7,105 rows, SHA-256 `01a9d51b4389db15c0d61eaa7347482e6a99df421540929abc7f0c55052365eb`, commit `fdb6e032`).

### 2.1 First attempt (lost)

Started on main `b8f6f506` on 2026-10-29: the cold runs (`stage4/a10/p2_32_cold_g1hs`, `p2_64_cold_g1hs`) started at 20:12, trained 1,758 cache files and started simulating at 20:23. The warm runs (`p2_32_warm_g1hs`, `p2_64_warm_g1hs`, caches hard-linked from the cold caches) started simulating at 20:27. All four failed with `OSError: [Errno 28] No space left on device` at 20:58-21:01, when `/data/ycfeng` filled (I-003). No metrics were written. The trained cold caches are complete (the cold runs had finished initialization) and are kept.

### 2.2 Restart (running)

Started 2026-10-30 04:39 on main `9a4f3471`; the simulator code and the promoted table are the same as in the first attempt.

| Run | Root | Cache |
| --- | --- | --- |
| P2-32 cold | `stage4/a10/p2_32_cold_g1hs_r2` | empty; trains |
| P2-64 cold | `stage4/a10/p2_64_cold_g1hs_r2` | empty; trains |
| P2-32 warm | `stage4/a10/p2_32_warm_g1hs_r2` | hard-link copy of `p2_32_cold_g1hs/cache` |
| P2-64 warm | `stage4/a10/p2_64_warm_g1hs_r2` | hard-link copy of `p2_64_cold_g1hs/cache` |

Deviation: each warm cache comes from the first-attempt cold run, not from the restarted cold run. Both cold runs train from the same commit's code and the same table, so the cache files are expected to be the same; the cold-warm comparison checks this through the receipts' cache inventories.

A mid-run cProfile window for the speed work (`stage4/run_profile3.sh p2_64 p2_64_mid2 60 1200`, cache hard-linked from `p2_64_cold_g1hs`) runs beside them; its first attempt was also lost to the full disk.

Results: pending.
