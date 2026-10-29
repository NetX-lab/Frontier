# Post-key frozen governance environment supplement

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record52PASS with the unchanged approved py312 CPU extras and original no-binary namespace. |

TaskID D29-POST-KEY-FROZEN-REGRESSIONS. Source
46fe83ac55876c10f6dd8ff792705e87efd1905f/tree
bbbca98802261662e02d32a59c1a4d6ab20a12e5/submodule
3dac53c544e546574cb6e8078cd14053d0916113;all clean before/after.

## Why this existing supplement was necessary

Full-unit frontier-py310 has the same five governance failures as retained79,
all with `ModuleNotFoundError: No module named 'torch'`. Prior py312 parent52
had all five passing. No new source or broad suite is introduced. Same existing
module is rerun on frozen46fe with approved existing extras only;no installation.

## Exact command and environment

```bash
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/baseline_environment/run_suite.py governance52
```

Payload:
```bash
timeout 120s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_post_key_regressions_20261024/baseline_environment/governance52/basetemp --junitxml=/data/ycfeng/tmp/d29_post_key_regressions_20261024/baseline_environment/governance52/results.xml
```

Expanded argv/env/versions in evidence/governance52/execution.json. Original
read-only namespace,no native binary projection;runuserbrainpp,MemoryMax30G.
PYTHONPATH=repo:repo/tests/unit:/data/ycfeng/tmp/d29_py312_cpu_extras01/site;
PYTHONDONTWRITEBYTECODE=1,TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp,
OMP/OPENBLAS/MKL threads1,VLLM_TARGET_DEVICE=cpu. Source/temp paths explicit.

Python3.12.13,pytest9.0.2,numpy2.2.6,pandas3.0.1,scipy1.17.1,sklearn1.7.2,
plotly7.1.0,fasteners0.20,ddsketch3.0.1,Torch2.9.1,aiohttp3.13.3.
These are pre-existing dependencies;none installed for this assignment.

## Actual result and exact IDs

Expected same52 original parent IDs allPASS. Observed **52PASS/0FAIL/0ERROR/0SKIP**,
rawexit0,21.540s. No lost/new IDs or changed retained statuses. Raw XML SHA256
`1fb0fb13d185781ef44b01a959e8535efa8b46073029a61237661968923f5044`.

Covered five original dependency-failure IDs:

- test_linear_wrapper_result_carries_plan_typed_operator_contracts
- test_linear_wrapper_result_omits_unmeasured_typed_operator_contracts
- test_moe_wrapper_initializes_runtime_with_routed_width
- test_moe_wrapper_typed_result_round_trips_through_manager
- test_real_linear_producer_csv_round_trip_reaches_manager_loader

All share `tests.unit.test_profiling_governance_minimal_red::`. Exact old/new XML
digests,IDs/environment binding in evidence/governance52_receipt.json. Final
canonical full-unit comparison joins these same IDs mechanically;the raw full
unit result remains separate and its failures are not relabelledPASS. Old group
parent's unrelated original TorchSKIP staysSKIP. This is an environment supplement,
not full CLI/native/GPU/G3/T2 qualification or an extra aggregate test total.
