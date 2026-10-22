# Source-local lane and explicit position CLI

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual scheduler lane width and explicit CLI/source input checks. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Workdir /data/ycfeng/Frontier/.worktrees/step5-position-core. frontier-py310 Python3.10.6. No dependency or platform operations.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-cli-lane-final02 tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_request_free_profiling.py -k 'not request_free_native_forward and not native_metadata_can_preserve and not variable_decode_prefix and not request_free_profile_rejects' > /data/ycfeng/tmp/d29-core-cli-lane-final02.log 2>&1
```

Observed48 PASS/1 SKIP/4 deselected. Expected and observed: source/count/cache/conservation/defaults preserved; actual BaseClusterScheduler lane constructor chooses physical source16 forEP0/7 and1 forEP8/15, while immutable gathered workload remains17 with live16. Shared mapper rejects mismatched contiguous DP/TP world. Position CLI requires the explicit input and approved Step/MoE-TP1/EP16/DP2, actual input produces36 source+EP/group planned cells and gathered17 even when legacy token grid is4096. Missing input/wrong geometry fail. The four deselections are previously recorded torch-unavailable baseline tests, not new failures suppressed by this change.

Initial lane test attempted object.__new__ on abstract BaseClusterScheduler and failed before the actual method. It was corrected to call the actual unbound method with its existing callback/state seam. No production workaround. Final repeat passes.

Checks:39/44 owned changed paths,+2201/-135=2336/4404 gross,2068 remain; AST/per-path ceilings and git diff --check PASS. Temporary WIP test/scheduler per-path excesses were corrected before commit by placing the same relevant lane test under its existing allocated conservation file and removing redundant re-admission inside an already admitted scheduler path. No borrowed scope or new file.

Limits: dry-run source/CLI and scheduler plumbing are CPU evidence; native wrapper cell invocation, selected schema/CSV/ranges, finite-model fixture actual-main and GPU/runtime qualification remain pending.
