# Physical pricing and reporting validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record third interface and producer checks. |

Workdir: /data/ycfeng/Frontier/.worktrees/step5-position-pricing.
Environment: frontier-py310 Python3.10; asyncopd-py312 Python3.12.13;
PYTHONDONTWRITEBYTECODE=1, OMP/OPENBLAS/MKL_NUM_THREADS=1, TMPDIR=/data/ycfeng/tmp.

Exact affected command:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_contract_reuse.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_package.py tests/unit/test_step5_comm_transport.py tests/unit/test_step5_dispatch_probe.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_final_owned_tests2
```
PASS335/SKIP71 in24.21s. /data/ycfeng/tmp/d29_final_owned_tests2.log.
Earlier fixture failures: dataclass.replace on namedtuple; assigning a copied
component's physical_frame. Corrected to _replace and ExecutionTime setter.
Expected real reporting: source16/gathered17, local-live0, no shuffle event,
one joint dispatch3ms, complete block29ms, count proofs in ledger. All PASS.

Submodule command (submodule cwd):
```sh
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-position-pricing:$PWD/python PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/test_grouped_plan_input.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_sub_final2
```
PASS7 in1.74s. /data/ycfeng/tmp/d29_sub_final2.log. Actual HTSIM completed
flows, bytes, dependencies, all eight rails and no-overlap composition checked.
The first invocation omitted parent PYTHONPATH and failed collection; corrected.

Native original Python methods were AST-isolated from retained installed source
at /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm and executed through
the actual NativeRoutedExperts class with CPU tensors using asyncopd-py312.
Modular expert→NoDPEP finalize; E8M0 OR valid-shape child selection; actual
GroupCoordinator(input_, output) delegation all PASS. Full pytest import in
py312 lacks fasteners, so no dependency installation or GPU runtime proof.

Limits: H04 test uses explicit cpu_fixture paired-call implementation, not the
unimported CORE real estimator. No measured local/protocol/TP coefficients,
backend/graph/G3 or E2E qualification. Root owns integrated acceptance.
