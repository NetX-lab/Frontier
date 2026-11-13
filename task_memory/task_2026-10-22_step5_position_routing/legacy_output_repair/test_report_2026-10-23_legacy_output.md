# D29 legacy output and fixture validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual seams,unchanged output,RED/GREEN and two scoped slices. |
| 2026-10-23 | Mark fixture edits/tests as uncommitted proposal after authority correction. |
| 2026-10-23 | Record explicit YC approval,fresh six/whole-file verification and separate test commit. |

TaskID D29-LEGACY-OUTPUT-REPAIR;all commands workdir
`/data/ycfeng/Frontier/.worktrees/step5-d29-legacy-output`,frozen
9974acdb88f5c47166fee28dec564b7cc782ca1a. Python frontier-py3103.10.6;
pytest existing environment;threads capped1,bytecode disabled,temporary files
only `/data/ycfeng/tmp`. No GPU/full CLI/network/platform/submodule build.

## Expectations and observations

Real MetricsStore._write_op_precision_metadata uses configured BaseModelConfig
Llama2 and Step models. No actual prepare inventory must yield no prepare row;
actually selected/registered FP8 prepare must yield FP8. Existing record_ep_wave
_reporting_case must omit absent proof keys across every legacy phase,including
zero work;actual nonempty dispatch/combine/post-combine costs remain recorded.
RED11=7 FAIL/4 PASS(no SKIP/ERROR),2.401s;GREEN11 PASS,2.256s.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_comm_operator_families.py::test_precision_csv_requires_actual_selected_prepare_inventory tests/unit/test_ep_wave_trace_context.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_legacy_output_red --junitxml=/data/ycfeng/tmp/d29_legacy_output_red.xml > /data/ycfeng/tmp/d29_legacy_output_red.log 2>&1
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_comm_operator_families.py::test_precision_csv_requires_actual_selected_prepare_inventory tests/unit/test_ep_wave_trace_context.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_legacy_output_green --junitxml=/data/ycfeng/tmp/d29_legacy_output_green.xml > /data/ycfeng/tmp/d29_legacy_output_green.log 2>&1
```

Strict parent evidence inspected:all77 candidate runs success,comparison77
mismatches only metadata/ledger types.Raw original
/data/ycfeng/tmp/d29_workflow_fidelity_20261023/{comparison.json,compare.log}
untouched. Real minimised Llama2 CSV after fix equals original baseline bytes:

```sh
cmp /data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate/cases/dp_dense_online_lanes2_replicas2/metrics/meta_llama_llama_2_7b_hf/online_serving/dp_dense_online_lanes2_replicas2/op_precision_metadata.csv /data/ycfeng/tmp/d29_legacy_output_green/test_precision_csv_requires_ac0/op_precision_metadata.csv
```

Exit0/no output PASS:no row-count literal/filter/baseline generation used.
This is one actual writer output comparison,NOT full strict77 acceptance.

## Six existing fixture contracts:initial proposal and authority pause

Tests and edits below ran under root's initial instruction before its authority
correction. Two proposed files remain isolated/uncommitted,not staged,with no
further edits permitted pending YC approval. Observations do not admit scope.
This paragraph records the historical pause;latest approval and fresh validation
are below. Two files are now explicitly approved and committed separately.

Exact six IDs RED6 FAIL,1.815s;GREEN6 PASS,1.548s,0 SKIP/ERROR.
The error evidence was stale four-member lists,unfiltered V4 expected set and
three partial SimpleNamespace families lacking operators,not numerical behavior.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_precision_metadata.py::test_inventory_uses_canonical_v4_scopes_without_generic_duplicate_owners tests/unit/test_moe_share_expert_operator_families.py::test_moe_family_declares_router_dispatch_and_grouped_gemm_ops tests/unit/test_moe_share_expert_operator_families.py::test_moe_family_declares_predictor_tp_and_ep_metadata tests/unit/test_moe_share_expert_operator_families.py::test_sklearn_moe_training_uses_moe_family_profiling_names tests/unit/test_moe_share_expert_operator_families.py::test_sklearn_moe_dataset_contract_filters_gating_context_from_operator_precision tests/unit/test_moe_share_expert_operator_families.py::test_quantization_default_registry_uses_moe_and_share_expert_family_names -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_family_fixture_red --junitxml=/data/ycfeng/tmp/d29_family_fixture_red.xml > /data/ycfeng/tmp/d29_family_fixture_red.log 2>&1
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_precision_metadata.py::test_inventory_uses_canonical_v4_scopes_without_generic_duplicate_owners tests/unit/test_moe_share_expert_operator_families.py::test_moe_family_declares_router_dispatch_and_grouped_gemm_ops tests/unit/test_moe_share_expert_operator_families.py::test_moe_family_declares_predictor_tp_and_ep_metadata tests/unit/test_moe_share_expert_operator_families.py::test_sklearn_moe_training_uses_moe_family_profiling_names tests/unit/test_moe_share_expert_operator_families.py::test_sklearn_moe_dataset_contract_filters_gating_context_from_operator_precision tests/unit/test_moe_share_expert_operator_families.py::test_quantization_default_registry_uses_moe_and_share_expert_family_names -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_family_fixture_green --junitxml=/data/ycfeng/tmp/d29_family_fixture_green.xml > /data/ycfeng/tmp/d29_family_fixture_green.log 2>&1
```

Both proposal whole files53 PASS,2.096s(no skips),observed before correction. Explicit new operator fields and real
interface assertions remain;V4 inventory uses actual applicability and excludes
Step prepare. No test deletion/skip or production mock support.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_moe_share_expert_operator_families.py tests/unit/test_dsv4_precision_metadata.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_family_fixture_whole --junitxml=/data/ycfeng/tmp/d29_family_fixture_whole.xml > /data/ycfeng/tmp/d29_family_fixture_whole.log 2>&1
```

## Original whole files and existing consumer/trace checks

Final194 PASS,5.469s(no SKIP/FAIL/ERROR),including actual constructor/current repo
CPU writer,four phases/identity negatives/native injection/numerical trace.
No external CPU writer reference needed:real integrated writer lives in9974 repo.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_comm_operator_families.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_quantization_manager_registry.py tests/unit/test_quantization_module_precision.py tests/unit/test_op_trace_utils.py tests/unit/test_ep_trace.py tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_legacy_output_affected_final --junitxml=/data/ycfeng/tmp/d29_legacy_output_affected_final.xml > /data/ycfeng/tmp/d29_legacy_output_affected_final.log 2>&1
```

Original attempt before local sub initialized had149=147 PASS/2 FAIL,3.630s,
FileNotFoundError resolving collective-sim for CPU trace cases. Command above
without final consumer file,basetemp/JUnit/log d29_legacy_output_affected reproduces
this prerequisite error. Required local init used existing pinned repo,not network:

```sh
git worktree add --detach /data/ycfeng/Frontier/.worktrees/step5-d29-legacy-output/frontier/cc_backend/backends/collective-sim 3dac53c544e546574cb6e8078cd14053d0916113
```

This command ran in isolated existing
/data/ycfeng/Frontier/.worktrees/step5-position-pricing/frontier/cc_backend/backends/collective-sim.
Target pinned by9974 gitlink;only new detached checkout created,no other source
edits,sub commit/build/gitlink change. Full consumer suite then PASS.

Original approved194 PASS/0 SKIP/FAIL/ERROR. Initial proposal53 was not accepted
during pause;latest approved rerun below independently verifies the same53.
Focused11/six already included in their corresponding runs,not counted twice.
Full parent strict77/global gate/GPU qualification not run.

## Scope verification

git diff --check PASS;AST parse all six changed Python files PASS;isolated sub
git status clean. Approved four-path step+36/-1=37<=80;retained increment35;
baseline3805+35=3840/3956,remaining116,43 changed/44 approved owner paths.
At original pause,fixture+14/-2=16 excluded;conditional3856 only after approval.
Latest YC approval now admits that exact16 delta;final3856 below. Original caps PASS.
Root's duplicate correction supersedes prior3846 estimate;no dispatch edit here.
Approved source commit b60dfddd;proposed files deliberately remain uncommitted.

## Latest explicit YC approval and fresh verification

Root reports exact YC answer "批准，类似问题都授权". Resume preserved two-file
delta<=60 (per-path50/10),owner3956/global10000 unchanged,effective111 paths.
Working diff verified identical to preserved patch. No production edits/imports.
Separate immutable fixture commit `047f7313`:two files+14/-2=16 gross.

Initial exact six RED remains `/data/ycfeng/tmp/d29_family_fixture_red.xml`:
6 tests,6 FAIL,0 SKIP/ERROR,1.815s. Fresh approved exact six GREEN:
6 tests,6 PASS,0 SKIP/FAIL/ERROR,1.495s. Both whole files:
53 tests,53 PASS,0 SKIP/FAIL/ERROR,2.752s. All results read from JUnit,not inferred
from exit code. Original assertions retained,MoE catalogue explicitly includes
prepare,V4 inventory derives actual applicability and excludes Step-only prepare.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_precision_metadata.py::test_inventory_uses_canonical_v4_scopes_without_generic_duplicate_owners tests/unit/test_moe_share_expert_operator_families.py::test_moe_family_declares_router_dispatch_and_grouped_gemm_ops tests/unit/test_moe_share_expert_operator_families.py::test_moe_family_declares_predictor_tp_and_ep_metadata tests/unit/test_moe_share_expert_operator_families.py::test_sklearn_moe_training_uses_moe_family_profiling_names tests/unit/test_moe_share_expert_operator_families.py::test_sklearn_moe_dataset_contract_filters_gating_context_from_operator_precision tests/unit/test_moe_share_expert_operator_families.py::test_quantization_default_registry_uses_moe_and_share_expert_family_names -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_family_fixture_approved_green --junitxml=/data/ycfeng/tmp/d29_family_fixture_approved_green.xml > /data/ycfeng/tmp/d29_family_fixture_approved_green.log 2>&1
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_moe_share_expert_operator_families.py tests/unit/test_dsv4_precision_metadata.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_family_fixture_approved_whole --junitxml=/data/ycfeng/tmp/d29_family_fixture_approved_whole.xml > /data/ycfeng/tmp/d29_family_fixture_approved_whole.log 2>&1
```

Both commands run concurrently in original isolated9974-derived worktree with
frontier-py310 Python3.10.6,existing pytest,thread caps1,no bytecode. Logs/JUnit
under approved paths above. AST parse both changed files and git diff --check PASS;
sub checkout clean/gitlink unchanged. No broad reruns after required53 PASS.

Final test path retained caps14/50 and2/10;fixture16<=60,combined original37+16=53
<=140. Original retained35 plus fixture16=51;root3805+51=3856/3956,remaining100,
45 changed/46 authorized owner paths,effective111 global paths. Existing original
four caps10/22/48/49 within29/60/50/65. No cap crossing/global ceiling expansion.
Original194 plus new whole53=247 distinct PASS;focused6/11 not counted again.
Source b60dfddd unchanged;test047f7313 only approved two files. Parent owns full
strict77/CLI/global review;no owner blocker/remaining fixture defect found.
