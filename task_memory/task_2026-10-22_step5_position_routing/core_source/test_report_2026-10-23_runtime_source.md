# Runtime and explicit DP source validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record production source/cache/constructor and legacy checks. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Workdir: `/data/ycfeng/Frontier/.worktrees/step5-position-core`. Python: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6. Every command uses `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `-B`, pytest `-q -p no:cacheprovider`; temporary roots are under `/data/ycfeng/tmp`.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_ep_aggregate_admission.py tests/unit/test_step5_fixed_probe.py tests/unit/test_step5_attention_profiling.py tests/unit/test_step5_request_free_profiling.py tests/unit/test_step5_moe_profiling.py -k 'not native_forward and not table_extent and not variable_decode and not cache_mutation' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-source-consumers-10
```

Observed **215 PASS,19 SKIP,4 deselected**,7.51s. Passing conditions: actual Step model/replica dummy-predictor constructor resolves source policy before parent; unsupported scope fails before parent; no position ratio allocation; full immutable source cache separates lane reorder and rejects missing source; actual bound request-owner DP source batches reach shared materialization; D26 tail starts6128 and routes residues96..223; q1 starts6144 and routes224..231; physical17/live16/assignments128 are separate; all-padding physical rows retain width with0 assignments; full6144 counts equal P6128+D16 and chunk split counts; old ratio/Hamilton/runtime/admission checks unchanged. Expected and observed agree. This verifies source/count/cache plumbing, not measured operator prices or native execution.

Baseline focused suite before edits:45 PASS/8 SKIP (interface report). Independently preserve unavailable existing request-free tests, not suppress them:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_request_free_profiling.py -k 'native_forward or table_extent or variable_decode or cache_mutation' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-baseline-env-03
```

Observed **4 FAIL/15 deselected** solely `ModuleNotFoundError: No module named 'torch'` at direct imports in tests lines22,135,159,200. Test file is unmodified. These are environment-unavailable checks, not PASS; no installation, fallback or test skip change. asyncopd-py312 has torch but lacks fasteners for full simulator import (first-interface report). Root owns broader baseline/full-unit qualification.

`git diff --check`, Python AST parsing all changed Python source/test files and exact retained-path/per-file quota accounting: **PASS**. At verified boundary29/44 owned paths changed, +1442/-95=1537 gross/4404. Runtime follow-up comprises11 changed paths; exact staged accounting is recorded in progress. Disaggregation predictor3019->3015 lines (registry enum moved to existing sole owner), base scheduler1993->1995; no unallocated split/wrapper.
