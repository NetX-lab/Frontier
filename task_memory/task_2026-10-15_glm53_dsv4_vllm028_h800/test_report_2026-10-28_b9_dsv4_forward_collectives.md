# Test report: B9, DeepSeek-V4 forward collectives and the G5h launcher (CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Dispatch bytes corrected for int64 `topk_ids` (see [test_report_2026-10-28_b10_g5h16.md](test_report_2026-10-28_b10_g5h16.md)). |
| 2026-10-28 | Added the commit and the submission lock check. |
| 2026-10-28 | Created: lead review and rerun of Codex assignment B9 (request `c82022e6`). |

## Scope

Assignment [h3o_operator_costs/codex_assignment_B9_dsv4_forward_collectives.md](h3o_operator_costs/codex_assignment_B9_dsv4_forward_collectives.md); design [h3o_operator_costs/g5h/design.md](h3o_operator_costs/g5h/design.md). Codex report: `h3o_operator_costs/codex-implement-c82022e6-3b51-4b19-99f6-92df67bc6810/implementation.md`.

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `ulimit -v 31457280`, scratch `/data/ycfeng/tmp/b9_review`.

## Lead corrections

| Change | Reason |
| --- | --- |
| Both stage configs: `engine` keeps only the six fields that `ServingParallelism.from_engine` reads (`async_scheduling`, `data_parallel_size`, `dtype`, `enable_expert_parallel`, `pipeline_parallel_size`, `tensor_parallel_size`), as in the K5 stage configs. | Codex copied the whole P2 `engine.json`, including the P2 profiler output path, the KV-transfer ports and the worker extension. Nothing reads them, and the profiler path names another run's output directory. `engine_source` keeps the reference. |
| `stage_g5h32_tp8_dp4_ep32.json` `engine_source` now names `h200_mainline/p2_64_profile/package_01/engine.json`. | The DP4 engine is the P2-64 one; the two engines differ only in `data_parallel_size` and the profiler path (checked). |
| New `g5h/launch/launch_g5h.sh`. | The P1h launcher wrapper pattern (`p1h_bridge/package_03/launch/launch_glm_producers.sh`): proxy, `STEPMIND_BACKEND=rjob`, personal credentials from the restricted files, then `submit.py`. Only the header and the final `exec` line differ. |

## Checks

| # | Command (from the working root) | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `python -m frontier.profiling.collectives.vllm_collective_benchmark --config g5h/launch/stage_g5h16_tp8_dp2_ep16.json --list-cases` | `cases 82 passes 3 measurements 246`; equal to `dry_run/g5h16_tp8_dp2_ep16.txt` | as expected; `cmp` exit 0 | PASS |
| 2 | the same for `stage_g5h32_tp8_dp4_ep32.json` | `cases 46 passes 3 measurements 138`; equal to `dry_run/g5h32_tp8_dp4_ep32.txt` | as expected; `cmp` exit 0 | PASS |
| 3 | `--list-cases` on `w5_glm_comm_20261020/launch/stage_k5_tp8_dp2_ep16.json` and `stage_k5_tp8_dp4_ep32.json`, `cmp` with their `dry_run/` files | byte-identical | both `cmp` exit 0 | PASS |
| 4 | `pytest tests/unit/test_vllm_forward_collectives.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_collective_kernel_evidence.py tests/unit/test_vllm_dsv4_forward_collectives.py g5h/launch/tests/test_runtime_check.py -q -p no:cacheprovider` | all pass | 135 passed | PASS |
| 5 | `pytest tests/unit/test_kernel_gap_pricing.py -q -p no:cacheprovider` | only the known failure | 1 failed (`test_every_collective_launches_one_kernel_by_default`), 20 passed | PASS (known failure) |
| 6 | cold import of `vllm_collective_benchmark`, `vllm_dsv4_forward_collectives`, `nixl_kv_read_benchmark`, `ep_kv_read_overlap_benchmark`, asserting no `torch`, `vllm` or `cuda` module | exit 0 | all four exit 0 | PASS |
| 7 | `grep -rn --include=*.py _ceil_div frontier tests` | no match | exit 1, no output | PASS |
| 8 | `/data/ycfeng/tmp/unused_imports.py` on the 9 changed Python files | `[]` | `[]` for all | PASS |
| 9 | `bash -n g5h/launch/launch_g5h.sh` | syntax OK | OK | PASS |

The literal `grep -rn "_ceil_div" frontier tests` of the assignment also matches `frontier/profiling/collectives/__pycache__/vllm_forward_collectives.cpython-310.pyc`, an ignored bytecode file from 2026-10-21 that predates B9. It is not deleted (deletion needs approval and gains nothing); the source-only grep (check 7) settles the hygiene item.

## Dry-run bytes against design §4 (rank 0)

From the saved dry runs; they equal the design's table.

| Call | Graph `T` = 1, 2, 4, 8, 16, 24, 32 (input → output) | Eager uniform `T` = 6144; 1 |
| --- | --- | --- |
| embedding all-reduce | 8,192 … 262,144 → same | 50,331,648; 8,192 |
| SP attention all-gather | 8,192 → 65,536 (T 1-8); 16,384; 24,576; 32,768 → 262,144 | 6,291,456 → 50,331,648; 8,192 → 65,536 |
| SP attention reduce-scatter (unpadded key) | 8,192 … 262,144 → 8,192 … 32,768 | 50,331,648 → 6,291,456; 8,192 → 8,192 |
| SP final mHC all-gather | 32,768 → 262,144 … 131,072 → 1,048,576 | 25,165,824 → 201,326,592; 32,768 → 262,144 |
| EP16 dispatch (int64 ids, corrected after G5h-16 `_01`; the first run used 8,240 bytes per row) | 8,264 → 132,224 … 33,056 → 528,896 | 6,346,752 → 101,548,032; mixed `eager_t6144_1` 6,346,752 → 50,840,128 |
| EP16 combine | 131,072 → 8,192 … 524,288 → 32,768 | 100,663,296 → 6,291,456; mixed 50,397,184 |
| EP32 combine, one 1-token lane | — | 151,060,480 → 6,291,456 |
| logits all-gather | — | 32,320 × b → 258,560 × b, b = 1-24 |

## Code review notes

- `vllm_forward_collectives.py`: `sp_all_gather` and `sp_reduce_scatter` added to `OPS`; `call_bytes` returns the unpadded input and `ceil(T / TP)` output rows for the reduce-scatter; `_ceil_div` renamed `ceil_div`. The GLM plan is unchanged (check 3).
- `vllm_dsv4_forward_collectives.py` (145 lines): the seven calls in design order, `calls_per_forward` 1/43/43/1/43/43/1, captured follows the scenario, logits only with logits rows, no DP sync; `dsv4_plan` rejects unknown grid keys and calls; TP calls only on uniform eager vectors.
- `vllm_collective_benchmark.py`: the SP ops import vLLM's `sp_all_gather` / `sp_reduce_scatter` lazily and issue them on the unpadded rows; `communicators()` records the custom-collective state (the attribute names exist in the native `custom_all_reduce.py:97,232-242,301`); `ARCHITECTURE_PLANNERS` selects the parser and planner and rejects other architectures.
- `g5h/launch/runtime_check.py`: `build_lock` refuses uncommitted sources and records the Git HEAD, six source hashes, the P2 overlay NCCL hash and five native vLLM file hashes; `check_runtime` writes `runtime_checks/<host>.json`, and only a hard-gate failure writes `nodes/<host>/failed.json` and exits 1. `submit.py` follows the K5 submit, with the H200 / `step_main` and 64-GPU asserts and the lock receipt.

## Not verified here

No GPU run: MNNVL kernel selection, NCCL kernels and timings are verified by G5h-16 and B10.

## After the commit

Committed as `5083d036`. `runtime_check.build_lock` on both stage configs (preview copies in `/data/ycfeng/tmp/b9_review/stage_g5h{16,32}_*_lock_preview.json`; `submit.py` writes the real lock at submission): `workspace_head` `5083d0369a5f9b86c333b30fce78154a3364225f`, NCCL `/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2` → `618968ee277041751512e619190132a9bdf351c3ce7129c73484f6ef441ad692` (the P2 overlay value), 6 benchmark source hashes and 5 vLLM file hashes. PASS.
