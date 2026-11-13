# Foundation P2 immutable sequence repair

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record requested mutation regression RED→GREEN. |

TaskID: D29-PRICING-IMPLEMENTATION. Working directory:
`/data/ycfeng/Frontier/.worktrees/step5-position-pricing`.

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python3.10);
`PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp`.

RED command: `python -m pytest tests/unit/test_step5_moe_pricing.py::test_workload_and_result_freeze_caller_owned_sequences -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_p2_red`.
Expected: failure when lists remain aliased. Actual:1 FAIL; sizes mutation16→8192
and tensor-list clearing altered cache key/stages/endpoint geometry.
Log: `/data/ycfeng/tmp/d29_p2_red.log`.

GREEN command: `python -m pytest tests/unit/test_step5_moe_pricing.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_cost_export.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_p2_green`.
Expected: immutable workload/result values and unchanged validation/scalar behavior.
Actual:49 PASS/1 SKIP in11.13s. Caller mutations of sizes, tensors, nested groups,
local services and counts leave validated identity/duration/count evidence fixed.
Log: `/data/ycfeng/tmp/d29_p2_green.log`.

The affected suite executed real joint HTSIM in the finite-exporter fixture.
The skip is unavailable torch in py310; no GPU calibration is claimed.
Owned gross after current pending producer edits:2198/3956 across43 of44 paths,
including312 submodule lines. Imported CORE/native/root paths are excluded.
