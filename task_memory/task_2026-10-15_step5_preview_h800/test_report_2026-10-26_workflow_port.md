# Test report: DECODE workflow port (calibration item 1)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: port checks (unit, dummy smoke, 77-case fidelity), the measured NIXL check fix, and the simulator cost profile. |

## Scope

Branch `feat/step5-d31-workflow` (worktree `/data/ycfeng/Frontier/.worktrees/step5-d31-workflow`),
from `feat/step5-preview` at `d034202c` (simulator code). Commits:

| Commit | Change |
| --- | --- |
| `1a6b4f8f` | glm53 `5f907f66`: NIXL READ completion at DECODE engine steps (`nixl_read_completion`, default `transfer_end`). |
| `a07518e0` | glm53 `96d9c1ef`: an all-idle DP group starts its next pass at the earliest posted READ end. |
| `f9c661ea` | glm53 `82390e10`: vLLM async scheduling on unified DECODE (`max_concurrent_batches=2`, output placeholders). |
| `0b769dc6` | The ported placeholder tests use a Step-5 TP8/DP2/EP16 lane instead of the DSV4 fixture. |
| `cf90408a` | `tests/integration/step5_pd/frontier_workload.py` selects `engine_step` and DECODE depth 2. |
| `331d646e` | The measured NIXL READ check accepts an engine-step end at or after start + measured READ time. |
| `6dd29176` | Comment no longer names the glm53 DSV4 step-cost path (D31.10). |

Environment: Python 3.10.6, `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025` (unit, workload
runs) and `/data/ycfeng/envs/frontier-py310` (fidelity matrix, smoke matrix, as in earlier reports).

## Results

| Check | Command / evidence | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Ported tests | `pytest tests/unit/test_nixl_read_completion.py tests/unit/test_decode_output_placeholders.py` | All pass | 20 passed | PASS |
| Full unit suite, base vs port | `pytest tests/unit -q --continue-on-collection-errors -rfE`, logs `/data/ycfeng/tmp/d31_wf/unit/{base,cand}.log` | No failure caused by the port | Base 279 failed / 41 errors; port 325 failed / 41 errors. No base failure passes or disappears. The 46 new failures (41 `test_collective_sim_cc_backend.py`, 3 `test_moe_routing_runtime.py`, 2 `test_op_trace_utils.py`) all raise `FileNotFoundError: Unable to locate collective-sim repository root`: the submodule is not initialized in the new worktree (`git submodule status` shows `-3dac53c`). | PASS (environment-only difference) |
| Dummy smoke matrix | `bash examples/architecture/run_dummy_smoke_matrix.sh`, log `/data/ycfeng/tmp/d31_wf/smoke_matrix.log` | 12 cases, rc 0 | "Completed 12 profiling-independent dummy smoke case(s)", rc 0 | PASS |
| Fidelity, default path unchanged | `/data/ycfeng/tmp/d31_fidelity_wf_run.sh` (`run_matrix.py run --label candidate --jobs 8`), compared with the `e275c83` baseline outputs (`/data/ycfeng/tmp/d31_fidelity_dfi/baseline`, identical to `d034202c` on 77/77) by `/data/ycfeng/tmp/d31_fidelity_wf_recompare2.py` | Only the new config field differs | 77/77 cases ran (rc 0). Every case differs only in `config.json` `/kv_cache_transfer_config/nixl_read_completion: missing in baseline`; no other artifact differs. Output `/data/ycfeng/tmp/d31_fidelity_wf_recompare2.json`. | PASS |
| Full 32-GPU run with the port, first attempt | `frontier_workload.py run --case h200_32 ... --output-dir /data/ycfeng/tmp/d31_wf/h200_32_wf01` | Completes | Stopped at the first READ end: "Measured NIXL READ-to-ready duration changed" (`KVCacheTransferEndEvent._validate_nixl_read` required end == start + measured READ time). Fixed in `331d646e`. | FAIL → fixed |
| Full 32-GPU run with the port, second attempt | same, `--output-dir /data/ycfeng/tmp/d31_wf/h200_32_wf02` | Completes; compared with step E | Running; passed the first READ end. | pending |

## Short-trace runs (dropped)

Measured runs on the 32 × 16 short trace (`/data/ycfeng/tmp/d31_frontier_b/requests_32x16.csv`,
all arrivals at 0 s) stopped on both base and port with `On-demand prediction feature 'num_tokens'
for share_expert_up_proj ... above declared maximum 6144.0: 12256.0`. Both PREFILL DP lanes join one
EP wave, and the lane batch takes the shared expert's token count from the merged batch
(`build_ep_lane_batch`). The canonical trace (1 s arrivals) does not reach this case in PREFILL.
The modeling question it shows is calibration item 2 in `plan.md`.

## Simulator cost profile

Command: base code (`step5-preview`, `d034202c`), one request (6144 input, 64 output tokens), the
step E predictor options, the trained cache copied from the prefix run, under `cProfile`
(`/data/ycfeng/tmp/d31_prof/`, `prof.out`). Wall time 575 s with the profiler.

- 5,720 EP waves (about 65 engine steps × 88 MoE layers) take 368.7 s of the 535.4 s
  `Simulator.run()`; initialization takes 35.5 s.
- Each wave prices all 16 EP lanes separately: 104,321 `predict_stage_execution_time` calls,
  391.7 s in total (3.75 ms each); 1,382,668 on-demand predictions.
- Inference: the full workload has about 3,100 engine steps, so a full run takes about 3 hours.
  Caching lane predictions with identical workloads (balanced routing gives equal lane loads)
  could shorten this several times. Deferred: it is a performance change in core modules and does
  not block calibration, because several runs fit in parallel on the 64-core host.

## Precision warning in the run logs (no effect on predictions)

Step E, this profile run and the port runs log `Profiling metadata mismatch for op=attn_pre_proj /
attn_post_proj / moe_grouped_gemm (precision mismatch (BF16 -> FP8)). Using approximation
factor=0.500`. The model config marks these operations FP8 (`MODEL_CONFIG_QUANT_OPS`), and the
file-level profile precision is BF16. The factor is applied only by
`QuantizationManager.adjust_compute_time`, which the dense layer path calls
(`_predict_dense_layer_execution_time`). The profile shows 0 calls to it on the Step-5 path, which
covered one PREFILL and 64 DECODE steps. Step-5 operators take their precision from
`step5_operator_binding` (the projection weight dtype, or the MoE position context), not from
`QuantizationManager` speedup factors. The warning is therefore misleading but changes no predicted time.

## Limits

- The port changes behavior only when `nixl_read_completion=engine_step` or DECODE
  `max_concurrent_batches > 1` is selected; the fidelity matrix covers neither.
- Accuracy against the native run is judged from the full run (`h200_32_wf02`) versus step E.
