# D29 prepare metadata source/factor validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual CSV RED/GREEN,coherent selected evidence,genuine approximations and exact scope. |

TaskID D29-PREPARE-METADATA-REPAIR;workdir
`/data/ycfeng/Frontier/.worktrees/step5-d29-legacy-output`;source before repair
f49014e1,immutable worktree base9974acdb88f5c47166fee28dec564b7cc782ca1a.
Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`,existing pytest;
PYTHONPATH=$PWD,bytecode disabled,OMP/OPENBLAS/MKL threads1,TMPDIR=/data/ycfeng/tmp.
No peer import/edit,full CLI,GPU/platform/network or delegation.

## Diagnosis and actual seam

The actual selected registry returns prepare for Step position context/reached
FP8,deferred=False. Existing register_profiling_metadata registers FP8 but compares
it to get_precision fallback BF16,recording approximation2.0. Writer replaced only
precision with actual profiling FP8,not source/factor. Actual H04 position path
MoeOperatorTimes._predict_step5_position_operator_times calls paired exact
_predict_step5_ffn_operator without adjust_compute_time. Source inventory is empty
for this Step configured model. These distinguish the cause from timing or source
override:the metadata row mixed two contracts.

Ranked predictions before fix:1.writer inherits stale generic-registration source
and factor;2.exact predictor actually scales;3.source inventory overrides. Observed
CSV/registration cause1 confirmed;actual native exact caller excludes2;source
inventory empty excludes3. Fix retains actual per-op quant_match (not warning sets)
and changes only the emitted complete matching context-required row. Existing
registration scaling/speedups,precision lookup,hash and inventory stay unchanged.

Initial full-row RED2=1 PASS/1 FAIL,0 SKIP/ERROR,1.831s. Failure exact assertion
data_source approximation != profiling,prepare FP8,factor2.0. Logs/JUnit below.
First tool invocation was cancelled by harness shutdown and produced no file;
fresh actual command below generated the recorded RED,not inferred from cancellation.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_comm_operator_families.py::test_precision_csv_requires_actual_selected_prepare_inventory -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_prepare_metadata_red --junitxml=/data/ycfeng/tmp/d29_prepare_metadata_red.xml > /data/ycfeng/tmp/d29_prepare_metadata_red.log 2>&1
```

## Passing criteria and final observations

Four configured cases are model{Llama2,Step}×signature{match,mismatch}.
Llama2 never registers prepare and retains baseline catalogue. Step uses actual
selected registry plus real registration and MetricsStore writer,not a row stub.

| Case | Actual CSV row / preserved behavior | Status |
| --- | --- | --- |
| Matched Step prepare | FP8,profiling,empty approximation_factor | PASS |
| Different-signature Step prepare | BF16,approximation,2.0 | PASS |
| BF16-profiled vs FP8-target grouped GEMM,matching signature | FP8,approximation,0.5;10 input time→5 | PASS |
| Same GEMM precision+signature mismatch | FP8,approximation,0.5;10→5 | PASS |
| Generic precision lookup | prepare remains BF16 | PASS |
| CSV side effects | config hash and speedup dict equal before/after writer | PASS |
| Warning-set independence | clearing warned_approximations before writer does not alter rows | PASS |
| Legacy CSV | byte-identical original Llama2 baseline | PASS |

Final focused4 PASS/0 SKIP/FAIL/ERROR,1.887s;seven whole affected suites196 PASS,
0 SKIP/FAIL/ERROR,7.385s. Parsed actual JUnit,not successful exit alone. Focused4
included in196,not added twice. Original prior194 grows by two signature cases.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_comm_operator_families.py::test_precision_csv_requires_actual_selected_prepare_inventory -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_prepare_metadata_delivery_green --junitxml=/data/ycfeng/tmp/d29_prepare_metadata_delivery_green.xml > /data/ycfeng/tmp/d29_prepare_metadata_delivery_green.log 2>&1
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_comm_operator_families.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_quantization_manager_registry.py tests/unit/test_quantization_module_precision.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_trace.py tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_prepare_metadata_delivery_affected --junitxml=/data/ycfeng/tmp/d29_prepare_metadata_delivery_affected.xml > /data/ycfeng/tmp/d29_prepare_metadata_delivery_affected.log 2>&1
cmp /data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate/cases/dp_dense_online_lanes2_replicas2/metrics/meta_llama_llama_2_7b_hf/online_serving/dp_dense_online_lanes2_replicas2/op_precision_metadata.csv /data/ycfeng/tmp/d29_prepare_metadata_delivery_green/test_precision_csv_requires_ac0/op_precision_metadata.csv
```

cmp exit0/no output. Final matching row path
/data/ycfeng/tmp/d29_prepare_metadata_delivery_green/test_precision_csv_requires_ac1/op_precision_metadata.csv;
mismatch row same directory ac3. Original RED ac1 under d29_prepare_metadata_red
preserved. Iterations green4/affected196 and green_final4/affected_final196 are
quick observations,not additional unique tests. All tooling/cache/logs in approved
local temporary directory;no duration fixture claimed GPU evidence.

## Scope,budget and verification limits

git diff --check and AST both changed Python files PASS;independent sub clean,
gitlink unchanged. Only quantization_manager.py+10/-1=11 gross and existing
test_comm_operator_families.py+17/-3=20,total+27/-4=31<=40. Retained quant19/29,
test62/authorized up-to70. Test cap needs12 over old50;root authorized up-to20
zero-sum from step5_cost120→100 and owns overlay. No donor file/registry edited.
Previous retained3856+9+14=3879/3956,remaining77 (not3856+31). Owner45 changed/46
authorized paths/global111/10000 unchanged. Existing047 test/source delivery intact.

Parent-reported strict77 on60ba is77/77 IDENTICAL,but it predates this metadata
repair and is not asserted as rerun here. Source02CLI60ba worktree unchanged.
Parent owns final integration/re-review/global gates;no qualification or global
acceptance claim. No remaining owner defect found in scoped checks.

Immutable source commit3d51b9b27f154701ce9a79cba42c84e5afc32555:
two paths+27/-4=31 gross. AST comparison againstf49014e1 confirms get_precision,
adjust_compute_time,get_compute_speedup_factor,get_config_hash,_build_config_dict,
_get_approximation_scale and has_precision_mismatch unchanged. No predictor or
other production file edited.
