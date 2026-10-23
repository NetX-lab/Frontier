# Final-source acceptance checks (in progress)

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Close source re-review; freeze1be91bad, launch final gates and record101PASS environment supplement plus final budget adoption. |
| 2026-10-23 | Record independent parent verification of the prepare metadata repair before re-review/final freeze. |

## Prepare metadata repair: parent affected checks

Code change3d51b9b2; integrated review snapshot
1be91bad431d3bd886de7f5d80c19bd7e6c54a25 in
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-04`.
Python3.10.6, existing frontier-py310 environment; detached local submodule
3dac53c544e546574cb6e8078cd14053d0916113, cloned locally without network or build.

```bash
timeout 180s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_comm_operator_families.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_quantization_manager_registry.py tests/unit/test_quantization_module_precision.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_trace.py tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_prepare_metadata_parent09 --junitxml=/data/ycfeng/tmp/d29_prepare_metadata_parent09.xml > /data/ycfeng/tmp/d29_prepare_metadata_parent09.log 2>&1
```

Parent parsed JUnit:196 tests,196 PASS,0 FAIL/ERROR/SKIP,6.911s. The four actual
MetricsStore CSVs under the basetemp were separately read with csv.DictReader:

| Configured case | Expected and observed prepare row | Result |
| --- | --- | --- |
| Llama2, either signature | No prepare row | PASS |
| Step, matching signature | FP8,profiling,empty approximation_factor | PASS |
| Step, different signature | BF16,approximation,2.0 | PASS |

Llama2 CSV bytes also equal the retained D30 baseline at
`/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate/cases/dp_dense_online_lanes2_replicas2/metrics/meta_llama_llama_2_7b_hf/online_serving/dp_dense_online_lanes2_replicas2/op_precision_metadata.csv`.
Genuine grouped-GEMM approximation, unchanged lookup/hash/speedup state and warning
set independence are covered by the same actual-writer tests. This is parent
execution, not a sum with owner196 or earlier parent194/53 suites.

## Remaining gates

Independent re-review returned No findings and closed P2 before the final runs
below were launched. Frozen production source is1be91bad; integrated-04 stays
unchanged throughout those runs. Earlier60ba strict77 PASS remains tied to60ba;
main04 is diagnostic. Native/GPU/G3/E2E qualification remains separate.

## Final gate launches

All commands run from integrated-04 with the same common environment:
PYTHONDONTWRITEBYTECODE=1,TMPDIR=/data/ycfeng/tmp,OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1. PYTHONPATH=$PWD for fidelity and
$PWD:$PWD/tests/unit for the remaining commands. The unit command additionally
sets FRONTIER_TMP_ROOT=/data/ycfeng/tmp. The CLI unsets FRONTIER_TEST_CPU_WRITER_FILE
and FRONTIER_TEST_CPU_WRITER_SHA256.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root "$PWD" --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/d29_workflow_fidelity_20261023_final03 --jobs 4 --clean-cache --continue-on-failure > /data/ycfeng/tmp/d29_workflow_fidelity_final03_run.log 2>&1
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29_unit_parent03 --junitxml=/data/ycfeng/tmp/d29_unit_parent03.xml > /data/ycfeng/tmp/d29_unit_parent03.log 2>&1
timeout 5100s /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d29-parent-position-fixture03/fixture_manifest.json --output-root /data/ycfeng/tmp/d29-parent-position-main05 --mode all > /data/ycfeng/tmp/d29-parent-position-main05.log 2>&1
```

These three results are pending. The CLI's own timeout remains2400s per mode;
the outer5100s only permits both modes and receipt generation. Fixture03 remains
explicit CPU/unadmitted evidence. No original gate or baseline was changed.

## Final-source environment supplement

Python3.12.13 asyncopd-py312 with the same common environment plus
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm and
VLLM_TARGET_DEVICE=cpu:

```bash
timeout 180s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_position_cyclic.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_final_environment_parent02 --junitxml=/data/ycfeng/tmp/d29_final_environment_parent02.xml > /data/ycfeng/tmp/d29_final_environment_parent02.log 2>&1
```

Parent parsed101PASS,0FAIL/ERROR/SKIP,11.923s. This supplies the known aiohttp/Torch
coverage on the final source. Full-suite failure/error-set reconciliation must
still inspect the new raw XML and account for every difference explicitly.

## Final budget

Independent audit followed by root remeasurement confirms111 allocated/110 changed
paths,9062 source gross and9076 global (including12-line size note and2-line gitlink).
Root adopts proposal A through repairs/final_scope_reconciliation.json:46 allowance
adjustments totaling+1258/-1258,all within the same owner; no path/owner/global cap
increase. All paths fit, and all original step caps remain separate. Reused prior
step receipts are identified in the immutable raw proposal rather than claimed
as new test executions. Root additionally verified the actual integrated-04
submodule,8630 superproject gross,333 submodule gross and98 changed Python paths.

Artifacts:final_ledger_proposal03.json (raw audit),final_budget_parent_receipt.json
(root check),repairs/final_scope_reconciliation.json (effective allocation).
