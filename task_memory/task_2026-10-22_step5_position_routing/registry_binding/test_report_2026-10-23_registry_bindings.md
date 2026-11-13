# D29 registry binding test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive standalone actual RED, 213-test GREEN, bundle receipts and exact budget. |

TaskID: D29-REGISTRY-SEAM-IMPLEMENTATION. Worktree:
`/data/ycfeng/Frontier/.worktrees/step5-registry-bindings`,
branch `feat/step5-registry-bindings`, base `d96b3279`.
Verified commit: `8f96408cf6fbb19c45a918b2464aea3ddff29f1a`.
Host: kun-workspace-vgen2. CPU-only; no platform/network/dependency operations.
Python `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6;
pytest9.1.1, sklearn1.7.2, pandas2.3.3, numpy2.2.6.

## Observed acceptance results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Baseline actual dual-binder RED | Old store loses one of two fitted FP32 gate bindings |1 retained, assertion1==2;1 failed,0 errors,9.51s | PASS: expected RED |
| Final actual dual-binder GREEN | Both groups survive; layer3/88 select matching context/group | Test passed; distinct hashes75601934/fd767cb1 | PASS |
| Same precision/context/source/backend/group | Correct selector resolves; ambiguous/missing source fails; unknown source/group unavailable | Gate and grouped-GEMM matrix passed, including JSON vs mapping | PASS |
| Family/cluster views and snapshots | All native bindings enumerated; no silent hash-label replacement | Both gate and expert groups included;30 bundle artifacts loaded; conflicting synthetic hash label fails | PASS |
| Physical row enumeration | Recover the actual source point before selecting one model | Layers3/88 queries and four measured collective plans passed | PASS |
| Legacy/default/typed/cache controls | Existing defaults, precisions, hashes, cache and legacy receipts unchanged | Selected regressions passed; bundle publication preserves pickle bytes; warm fit avoids refit | PASS |
| Existing shared and independent actual paths | Real fits, cache markers and existing binders verified | Shared dual binder and independent selected actual fit/cache tests passed | PASS with limit below |
| Affected test selection | No failed/error/skipped tests |213 passed,0 failures/errors/skips,171.16s | PASS |
| Whitespace and scope | Four authorized code/test paths only; clean diff | `git diff --check` empty; root-owned submodule not staged | PASS |

Independent fit/cache is verified; no independent public registry binder exists in
the inspected path. This report does not qualify that complete public chain.
CPU fixture timings are not native observed compute measurements.

## Exact final GREEN command

Run from the worktree above:

```bash
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_contract_reuse.py tests/unit/test_step5_model_identity_reuse.py tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_manager.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_predictor_cache_atomicity.py tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py tests/unit/test_step5_measured_training.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_registry_final02 --junitxml=/data/ycfeng/tmp/d29_registry_final02.xml > /data/ycfeng/tmp/d29_registry_final02.log 2>&1
```

The new persistent regression file contributes6 tests. Previous decisive
GREEN03 was1 passed/12.09s; selected final01 was184 passed/162.57s.
Final02 includes all of those tests plus the actual Step measured-training tests.

## Exact standalone baseline-store RED command

This reads only the baseline store method from Git, injects it in memory, and
runs the final persistent real two-group fixture. It never edits a peer worktree.
Exit1 is required, with the retained count1 assertion and no setup error.

```bash
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python - <<'PY' > /data/ycfeng/tmp/d29_registry_standalone_red03.log 2>&1
import ast
import subprocess
import pytest
import frontier.execution_time_predictor.prediction_model_registry as registry
baseline = subprocess.check_output([
    'git', 'show', 'd96b3279:frontier/execution_time_predictor/prediction_model_registry.py'
], text=True)
cls = next(node for node in ast.parse(baseline).body if isinstance(node, ast.ClassDef)
           and node.name == 'PredictionModelRegistry')
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
              and node.name == '_store_model_precision')
namespace = dict(vars(registry))
exec(compile(ast.Module(body=[method], type_ignores=[]), '<baseline-store>', 'exec'), namespace)
registry.PredictionModelRegistry._store_model_precision = namespace['_store_model_precision']
raise SystemExit(pytest.main([
    'tests/unit/test_prediction_measurement_bindings.py::test_actual_dual_binder_retains_fp32_gates_for_layers_3_and_88',
    '-q', '-p', 'no:cacheprovider', '--basetemp=/data/ycfeng/tmp/d29_registry_standalone_red03',
    '--junitxml=/data/ycfeng/tmp/d29_registry_standalone_red03.xml',
]))
PY
```

## Budget receipt

`git diff --numstat 1aee0af -- <path>`; repair measured against`d96b3279`.
The new file is absent from both bases; its174 lines are all additions.

| Path (execution_time_predictor prefix omitted) | Gross vs1aee0af | Cap | Incremental repair gross |
| --- | ---: | ---: | ---: |
| prediction_model_registry.py |82+11=93 |120 |93 |
| sklearn_moe_execution_time_predictor.py |96+17=113 |138 |17+2=19 |
| moe_routing_workload.py |130+6=136 |145 |14+2=16 |
| tests/unit/test_prediction_measurement_bindings.py |174 |180 |174 |

Whole repair302/360; existing callers35/60. All ceilings PASS. No compression
or omitted tests were used to fit allocation.

## Artifact receipts and limits

- Raw final log/XML: `/data/ycfeng/tmp/d29_registry_final02.{log,xml}`.
- Standalone RED log/XML: `/data/ycfeng/tmp/d29_registry_standalone_red03.{log,xml}`.
- Prior actual RED: `/data/ycfeng/tmp/d29_registry_actual_red02.{log,xml}`;
  original independent diagnosis: `/data/ycfeng/tmp/d29-core-registry-collision03.log`.
- Selected persistent evidence: `registry_binding/evidence/` in this task directory.
- Review diff: `registry_binding/evidence/registry_binding_8f96408c.patch`.
- Observed full bundle: `/data/ycfeng/tmp/d29_registry_final02/test_all_native_bindings_survi0/cache/bundle_3f2f6c45abef019ff59bdf8f66f49fd0c23880515dfde8235dbf7ca5f9aaca7a.json`.
- Gate3-87: `moe_gating_linear:75601934`; gate88-90: `moe_gating_linear:fd767cb1`.
- Expert3-87: `moe_grouped_gemm:4a7b2210`; expert88-90: `moe_grouped_gemm:0880496d`.

Warm trainer may rewrite duration/count pickle files under its existing behavior;
its stable hashes/exact lookup and no-refit behavior, not byte immutability, were
validated. Bundle publication alone was checked for byte immutability.
Root owns independent review, actual-main/strict77, GPU/global gates,
CPU backend consumer and native observed compute producer work.
