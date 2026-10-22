# Foundation P2: actual bound router implementation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record reviewer finding, actual constructor RED→GREEN and affected source checks. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Finding: P2, reviewer ses_e9abd8364ffee5sQKdpiQh3yH4, frozenbd97ac5/02f061d. Production source306-307 admitted native router class and inherited select_experts but omitted the timed _select_experts bound method. A bound implementation compiled from foreign source with the same __module__ bypassed those checked facts. Scope is moe_impl.py and existing test_step5_moe_profiling.py; no verifier duplication/native semantics or peer edits.

Environment: /data/ycfeng/Frontier/.worktrees/step5-position-core; asyncopd-py312 Python3.12.13, frontier-py310 Python3.10.6. All checks use thread caps/bytecode disabled and /data/ycfeng/tmp. Fault-injected router/receipt tests are synthetic CPU regression evidence only, not image/runtime qualification.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-p2-red02 tests/unit/test_step5_moe_profiling.py -k only_replaced_bound_router > /data/ycfeng/tmp/d29-core-p2-red02.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-p2-green03 tests/unit/test_step5_moe_profiling.py -k 'only_replaced_bound_router or position_adapter_binds or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' > /data/ycfeng/tmp/d29-core-p2-green03.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-p2-verifier01 tests/unit/test_step5_fixed_probe.py tests/unit/test_moe_profiling_output_metadata.py > /data/ycfeng/tmp/d29-core-p2-verifier01.log 2>&1
```

RED02:1 FAIL/30 deselected,4.11s; exact failure `DID NOT RAISE <class 'ValueError'>`. Clean adapter construction succeeds; only returned router's _select_experts is replaced by MethodType compiled from foreign_router.py while retaining its __module__. Class, public method, module file/spec origin and hash are retained; factory callable stays original and its injected result changes. Expectation is constructor rejection before timing.

GREEN: existing callable tuple includes native_router._select_experts. Same constructor now raises `bound native callable code/source differs from its installed module`; affected adapter/timer/oracle/norm/source-context20 PASS. Unmodified public selector and source tensor/restore semantics still pass. Existing receipt negatives and metadata source suite37 PASS separately.

Initial setup runRED01 failed before the fault at `derived native source mismatch: vllm/forward_context.py`. Cause: changing the test-only fixture source left its derivation SHA stale. Refreshing that test-only seal allowed the decisive RED02; authentic published receipt was not used or modified. No fallback or suppressed exception introduced.

Quota:37/44 owned paths,+2116/-133=2249/4404 gross,2155 remain. Per-path gross: moe_impl53/172; test_step5_moe_profiling276/290. AST and git diff --check PASS. Full-unit/independent re-review/native runtime remain root-owned.
