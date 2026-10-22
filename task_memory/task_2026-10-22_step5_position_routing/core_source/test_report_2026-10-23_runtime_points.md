# Runtime source/point and selected physical costs

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record exact point/policy passthrough, actual estimator composition and source regressions. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Worktree step5-position-core, Python3.10.6 frontier-py310 and Python3.12.13 asyncopd-py312. All commands set PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PYTHONPATH=$PWD, TMPDIR=/data/ycfeng/tmp; no cacheprovider.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-runtime-final02 tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/integration/test_step5_measured_main.py tests/unit/test_moe_profiling_output_metadata.py > /data/ycfeng/tmp/d29-core-runtime-final02.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-attention-source01 tests/unit/test_step5_attention_profiling.py -k native_initialization > /data/ycfeng/tmp/d29-core-attention-source01.log 2>&1
VLLM_TARGET_DEVICE=cpu /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-router-final01 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router or position_adapter_binds or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' > /data/ycfeng/tmp/d29-core-router-final01.log 2>&1
```

Observed: **58 PASS/3 SKIP,86.65s**; **2 PASS/98 deselected**; **20 PASS/11 deselected**. Expected exact source/EP point selection, full retained policy/context/buffer/count key equality, actual BF16 estimator→H04→ExecutionTime positive empty expert scope and no shuffle, paired source/shared widths. Source constructor receives actual layer/bound-forward callables, checks sealed config and uses verifier-declared image. All observed PASS. Three skips are torch-native wrapper cases unavailable in py310, not acceptance.

The actual source/gathered MoEWrapper case additionally executes under py312 with CPU tensors and narrow native/timer leaves. Existing persistent integration FunctionDef and unchanged PRICING `STEP5_NATIVE_EXPERT_SCHEMAS`/`step5_native_expert_features` are AST import-isolated only to avoid py312's missing fasteners/sklearn package initialization. Compile those exact nodes from their existing paths, inject the feature module under its existing import name, then call the test with fresh MonkeyPatch.context for `(None,False)`, `(float8_e4m3fn,False)`, `(float8_e4m3fn,True)`. **3 PASS**; log `/data/ycfeng/tmp/d29-core-wrapper-final03.log`. This is actual wrapper/feature execution but CPU plumbing, not native kernel qualification.

Failure evidence retained: execution01 policy default conflicted with position CSV; execution02 requested gathered17 but agreed PW dummy makes32; hypotheses preceded each correction, native layouts never changed. execution03 **2 PASS**. runtime-final01 **57 PASS/3 SKIP/1 FAIL**: old minimal command fixture lacked new routing field; balanced default preserved, command-fixed01 **3 PASS**. wrapper-final01 imported CollectiveTensor through cc_backend and hit missing fasteners; direct existing dict metadata removes unnecessary profiling dependency. wrapper-final02 reached the original pricing feature import and hit the same environment prerequisite; exact existing feature function import isolation settled the test, not a production fallback.

Runtime source binder retains complete per-cell execution_policy/branches/buffers/counts and restores actual plans from the existing count key; it does not perform affine model grouping. F4a/F4b normalization remains PRICING-owned. Registry native-binding retention and explicit CPU plan cost CLI entry remain unowned root prerequisites; full actual-main/native/GPU/review/strict77 not claimed. Baseline environment failures remain separate in previous reports.
