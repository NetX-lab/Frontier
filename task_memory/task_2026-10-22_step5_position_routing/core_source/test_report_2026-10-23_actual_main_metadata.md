# Actual-main metadata hook and C2 caller integration

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record authorized C2 import, real caller regression, actual CLI failure and selected metadata correction. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Worktree `/data/ycfeng/Frontier/.worktrees/step5-position-core`. Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`. Commands below use `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp`, `-B`, and pytest `-p no:cacheprovider`. No GPU/platform operation.

## Authorized integration and hypotheses

- Preserved owned changes and records in `404d555a`, then `git merge --no-ff --no-edit f41d1ee9` -> `591894f1`. C2 source use is authorized; root acceptance remains pending. No peer files were edited.
- Actual H04 caller initially failed because the older test built output-only dispatch plans and no complete buffer context. Replaced that test stub with the existing CPU fixture plan/count producer and actual `_step5_position_queries` binder. The next failure was the test replica's missing `router_topk`; an attempted replacement reused another SimpleNamespace, so it did not fix the failure. Final test uses formal `ReplicaConfig` TP8/DP2/EP16 and existing model-derived topk. No production default/guard added.
- CPU collective leaf remains explicitly isolated by the test's `predict_collective_plan` callable reading the exact CPU cost input. It does not establish production CPU importer/native support. Full buffer.compute_context, dispatch/output/scales/topk layouts and count identities reach the actual H04 method before any compute query.

## Exact commands and results

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py > /data/ycfeng/tmp/d29-core-c2-real-caller01.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_measured_ffn.py::test_position_real_estimator_reaches_execution_time_without_shuffle > /data/ycfeng/tmp/d29-core-c2-real-caller02.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py > /data/ycfeng/tmp/d29-core-c2-real-caller03.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py > /data/ycfeng/tmp/d29-core-c2-real-caller04.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_measured_ffn.py::test_position_real_estimator_reaches_execution_time_without_shuffle > /data/ycfeng/tmp/d29-core-actual-main-metadata-red01.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py > /data/ycfeng/tmp/d29-core-actual-main-metadata-green01.log 2>&1
```

- Caller01 FAIL: empty/foreign compute context rejected, expected after C2 import; other cases36 PASS/3 SKIP.
- Caller02 and03 FAIL before pricing: test-only SimpleNamespace lacks router_topk. Formal ReplicaConfig corrects it.
- Caller04 PASS:37 PASS/3 SKIP in77.61s. Complete fixture plan/count binder reaches H04 and ExecutionTime, positive empty expert pairs and exact source/shared count guard retained.
- Metadata RED FAIL: added a call to the real initialization metadata hook on the same fitted position CSV; `ValueError: Position MoE metadata requires selected operator_facts` reproduces actual CLI.
- Metadata GREEN PASS:37 PASS/3 SKIP in78.89s. Existing `position_moe_operator_facts` automatically enumerates actual groups/bindings in that hook, before legacy helpers. No alternate registry or metadata fallback.

## Actual CLI terminal evidence

`/data/ycfeng/tmp/d29-core-position-main-seam02/{command.json,requests.csv,run.log,execution.json}` is the actual `frontier.main` invocation emitted by `tests.integration.run_step5_measured_main.command`. Workload6144/1024; internal subprocess timeout2400s unchanged. Result: **exit1, timed_out=false, wall1295.815292s**. Log2144–2178 traces initialization through `sklearn_execution_time_predictor._register_profiling_metadata_from_files` to owned `moe_dataset_training._register_additional_profiling_metadata_from_files`, which lacked automatic position enumeration. This is an owned consumer defect, now fixed and persistently tested, not a registry/backend prerequisite.

Full CLI rerun still uses preserved complete CSV/cache inputs; they predate complete C2 producer fields. Native qualification and existing registry multi-binding/production CPU backend remain separate prerequisites. No offline/online metrics acceptance yet. Do not count overlapping tests twice.

## Accounting and limits

Grouped retained scope rows yield44 owned paths/4404 total. Current baseline gross **+3149/-155=3304**,1100 remaining; every per-file ceiling, Python AST and whitespace PASS. Metadata file162/164; updated real caller test105/108. No peer edits, extra agent, new backend/registry, invented source/image identity or CPU-to-native relabeling.
