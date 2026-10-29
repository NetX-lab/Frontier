# Native-final source and CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded native source/input controls,130PASS, builder baseline and distinct dispatch/routing counts. |

Task: W128-V4-BALANCED-NATIVE-FINAL. Workdir `/data/ycfeng/Frontier/.worktrees/v4-balanced-binding`; Python3.12.13 `/data/ycfeng/envs/asyncopd-py312/bin/python`, no conda activation. CPU only; no native/C26 edit, GPU/platform/cloud/SSH/Docker, package build/seal/submission or delegation.

## Exact passing commands

From `/data/ycfeng/tmp/opencode-v2-tmp/opencode`:

```bash
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/v4-balanced-binding/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import sys
sys.path.insert(0,'/data/ycfeng/Frontier/.worktrees/v4-balanced-callers')
import frontier.config
sys.path.insert(0,'/data/ycfeng/Frontier/.worktrees/v4-balanced-binding')
import tests.comparison.calibration.native_full_structure as collector
import pytest
print('actual_config',frontier.config.__file__)
print('actual_collector',collector.__file__)
raise SystemExit(pytest.main(['/data/ycfeng/Frontier/.worktrees/v4-balanced-binding/tests/unit/test_native_serving_inventory.py','/data/ycfeng/Frontier/.worktrees/v4-balanced-binding/tests/unit/test_native_v2_serving_inventory.py','/data/ycfeng/Frontier/.worktrees/v4-balanced-binding/tests/unit/test_native_reader_binding.py','-q','-p','no:cacheprovider','--basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-native-final07','--junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-native-final07.xml']))
PY
```

The command explicitly combines A's775-field actual config and B's current collector without editing either other worktree. Observed actual paths printed exactly as above;130PASS in71.64s. XML is the retained result; no failures/skips.

From this worktree, `git diff --check` returned clean. Builder check imported the maintained script without calling main, AST-selected only literal native source-pin loops, applied their hashes to actual template/prior config, then added checker.ROUTING_PINS:70 dispatch pins,74 final pins, candidate06 absent. No candidate was created. Baseline-preservation shell first failed because unqualified `python` is not installed; reran the exact absolute interpreter above successfully. The AST inspector initially literal_eval'ed a nonliteral CHECKER/R2 tuple; corrected only the inspector to select literal native tuples.

## Acceptance observations

| Passing condition | Observation | Verdict |
| --- | --- | --- |
| Explicit balanced environment and actual registry/module/code identity | Real strategy function with fake native host modules passes; missing identity/env rejects | PASS CPU |
| Known actual env/factory/simulator/router/MoERunner origins and native byte pins | Collector records source origins; checker rejects wrong origin/hash; callable shadow rejects in collector | PASS CPU |
| Winning same-name strategy module/package rejects | Existing PathFinder control and native reader shadow controls pass | PASS CPU |
| Actual full Frontier input is not replaced by abbreviated settings | Root775-field bytes, SHA747a08e5..., positive passes; missing/mismatched input hashes, selector/model/role/full-format negatives reject | PASS CPU |
| Frozen R2 authority and non-routing predicates unchanged | R2 hash29523700...; original provenance/inputs/replay/workers/lifecycle/terminal AST equality passes after validation moved to harness predicate | PASS |
| Observer failure remains soft | Existing host-return/error/observer tests pass; no device reads/fences/collectives added by routing inventory | PASS CPU |
| Distinct source scopes | Five existing files224 routing lines + builder29 = B253; builder dispatch3 separately,488+3=491<=500 | FAIL LOCAL250; reallocation pending |

## Failure diagnosis and limits

First real-input check imported B's old774-field SimulationConfig instead of A775; positive rejected, as it should. Preloading A config fixed that check. A broad mixed-worktree run then imported A's old collector, giving7 import failures/123PASS; explicitly preloading B collector fixed it, final130PASS. Neither correction relaxed runtime/source admission. The initial Frontier validation placement altered frozen inputs AST; moved to existing harness predicate and original-predicate equality passed.

Root input receipt source410f062 and actual trace SHA2d557d7d923b71fbf62bc7e552d1fb547a8ec88c1e54660c0c76cc2de3c49ec2 are retained in evidence/root_frontier_input_receipt.json. This input binds a future existing Simulator(config) call, not completed prediction fits or numerical/graph/pricing/memory qualification. Root rechecks effective P/D role materialization on A's final source.

No native CUDA routing/kernel/FP8/backend proof or package admission. Local3-line overrun, forthcoming root-supplied C integration negatives, final common review and root generation/runtime work remain unresolved.
