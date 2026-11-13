# Complete compute/buffer binding and CPU fixture inputs

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record C2 strict binder, same-context CPU plan payload, regressions and runtime limits. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Working directory `/data/ycfeng/Frontier/.worktrees/step5-position-core`. Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`; native/timer focused checks use Python3.12.13 `/data/ycfeng/envs/asyncopd-py312/bin/python`. All commands set `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp`, use `-B`, and disable pytest cacheprovider.

## Commands and observed results

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py -k 'position or actual_main_command' > /data/ycfeng/tmp/d29-core-c2-fixture-binding01.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_measured_ffn.py tests/integration/test_step5_measured_main.py > /data/ycfeng/tmp/d29-core-c2-fixture-binding02.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_routing_runtime.py tests/integration/test_step5_measured_main.py -k 'position or actual_main_command' > /data/ycfeng/tmp/d29-core-c2-fixture-binding03.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_routing_runtime.py -k position_runtime > /data/ycfeng/tmp/d29-core-c2-buffer-final01.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_step5_fixed_probe.py tests/unit/test_step5_request_free_profiling.py tests/unit/test_step5_attention_profiling.py -k 'not routed_layer_groups and not native_step_router and not native_routed_builder and not native_fp8_prepare_constructor' > /data/ycfeng/tmp/d29-core-source-package-final01.log 2>&1
```

Observed: binding01 **5 PASS/3 SKIP**; binding02 **36 PASS/3 SKIP,80.93s**; binding03 **6 PASS/3 SKIP,48.37s**; final buffer **2 PASS**. Expected source/native point equals complete buffer.compute_context, buffer key equals retained receipt, no guessed fit grouping; missing and foreign FP8-only context rejected. Actual producer252-row/two-group shared+independent cold/warm fits remain paired; explicit CPU plan keys carry the same full context/layout, actual agreed graph mode and positive counts/cost composition. Position command forwards existing profiling_data_dir; balanced command unchanged. Three skips are torch wrapper tests under py310, not acceptance.

Source/package union **152 PASS/1 SKIP/4 FAIL,6.83s**. All four failures are retained baseline `ModuleNotFoundError: No module named 'torch'` in unmodified request-free tests at lines22,135,159,200. No source behavior change or environment installation. Earlier F1/provenance focused **20 PASS** and exact-source FFN/replay **27 PASS** are recorded in runtime_points/source_exact_pairs reports; do not aggregate overlapping runs.

## Actual-main evidence and limits

Fresh full generation `/data/ycfeng/tmp/d29-position-actual-main-fixture02` wrote37044 rows per MoE measurement and attention fits, but its600s auxiliary timeout stopped preflight before manifest. No full generator rerun/performance campaign is used to substitute acceptance.

Actual CLI seam01 uses `tests.integration.run_step5_measured_main.command()` with those complete CSV inputs and6144/1024 requests, preserving internal subprocess timeout2400s. Its outer120s tool timeout truncated execution before receipt; not a CLI2400s timeout. Exact argument vector: `/data/ycfeng/tmp/d29-core-position-main-seam01/command.json`; input context and source identity beside it; log `run.log`.

Seam02 reuses that exact vector, replacing only output directory, under a background launcher with no outer timeout and the same internal2400s. Artifacts `/data/ycfeng/tmp/d29-core-position-main-seam02/{command.json,requests.csv,run.log,execution.json}`. Result pending at this record. Both commands predate the newly committed complete CPU buffer inputs and are failure-boundary evidence, not full fixture/main acceptance. No synthetic sources are relabeled native, no native-only importer bypassed.

Remaining prerequisites: root existing registry multi-native-binding retention/selection, explicit existing-backend CPU cost consumption, and observed native compute-to-buffer association with repaired C1/C2 tip. F4b imported bfe6f2ae; no CORE normalizer. Strict77/full-unit/native/GPU/G3 remain root-owned.

## Ownership

Only44 approved paths plus core_source task records. Current gross +3150/-155=3305/4404 before final commit, all per-path ceilings/AST/whitespace PASS. `tests/integration/test_step5_measured_main.py`153/153 and runtime test59/60; no quota borrowed. Submodule3dac53c is isolated local development use; gitlink remains unstaged/root-owned. No peer writes, extra agent, Docker/GPU/platform/auth/cloud/SSH/push/deletion.
