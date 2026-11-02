# Native dispatch CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded primary affected/merged R1 verification and independent targeted approval. |
| 2026-10-22 | Recorded W128 CPU observations and verification limits. |
| 2026-10-22 | Added R1 targeted P2 RED/GREEN evidence without broad suite reruns. |

Environment: Linux CPU; `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python
3.12.13, Clang22.1.3, PyTorch2.9.1+cu128. No CUDA/native package import is required
by AST extraction; the oracle uses only CPU torch tensors. No conda activation,
GPU/auth/platform/SSH/cloud operation, build or package.

Working directory: `/data/ycfeng/Frontier/.worktrees/v4-native-dispatch`.

```bash
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_v2_serving_inventory.py tests/unit/test_native_serving_inventory.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-cpu08
env PYTHONDONTWRITEBYTECODE=1 NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/v4-native-dispatch/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_reader_binding.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-reader03
git diff --check
```

| Expected passing condition | Observed | Result |
| --- | --- | --- |
| Observer lifecycle and original results/exceptions preserved | 72 tests PASS,4.40s | PASS |
| Actual prefill-tail q1 differs from ongoing decode q1 | Native classifier -> PIECEWISE1 vs FULL1 | PASS |
| Peer uniformity and unequal token padding change expected dispatch | PIECEWISE1 and FULL2 | PASS |
| Wrong/missing/circular/ambiguous evidence rejected | 12 negative controls incl boolean-as-token | PASS |
| Actual CPU DP snapshot before mutation, native transport unchanged | raw[1,2], padded[2,2]; original exception identity preserved | PASS |
| Failed forward never credited, replay-only failure cannot be hidden by aggregation | Explicit association error and successful forward count0 | PASS |
| Observer input/output/footer failures are soft | native result identity and totals preserved, errors recorded | PASS |
| Existing bound/counters reconcile | Capacity loss explicitly counted; retained sums match successful calls | PASS |
| Other predicates and reader guards preserved | Non-graph functions AST-identical;43 reader tests PASS,46.57s | PASS |
| Native05/generation05 do not become admitted | Missing dp_utils pin produces UNVERIFIED | PASS |

Reader03 was run before the final scalar-type-only oracle hardening; non-graph
AST equality and the affected observer/oracle suite were rerun after it. This
hardening does not change the reader branch. Root still owns final full checks.

Initial failures and fixes are recorded in progress.md. No performance assertion,
historical operand recovery, all32-rank fresh-run admission or four-metric
calibration is claimed. Physical row cardinality and observer overhead require the
authorized fresh run. The existing4096-row bound remains; loss must be reported
UNVERIFIED, never ignored or used to stop healthy native work.

## R1 query-bound regression

Same Python/environment/workdir as above. Before the one-line fix:

```bash
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest 'tests/unit/test_native_v2_serving_inventory.py::test_dispatch_successor_rejects_incomplete_or_wrong_operands[query_bound]' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-r1-red --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-r1-red.xml > /data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-r1-red.log 2>&1
```

Observed RED: `Failed: DID NOT RAISE any of (ValueError, KeyError)`;1 failed,
1.87s,exit1. Real classifier inputs1/2/1/prefillTrue were accepted despite a
source-impossible sum/len/max, while descriptor/vector otherwise matched.

After adding `num_tokens <= num_reqs * max_query_len`:

```bash
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_v2_serving_inventory.py -k 'test_actual_native_classification_and_dp_dispatch or test_dispatch_successor_rejects_incomplete_or_wrong_operands or test_frozen_r2_non_graph_predicates_and_reader_guards_remain_identical' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-r1-green --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-r1-green.xml > /data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-dispatch-r1-green.log 2>&1
```

Observed GREEN:18 passed,27 deselected,2.92s,exit0. The new impossible-operand
control rejects; four valid native dispatch controls still pass, including
classifier None and DP uniform0 for prefill/peer cases. Non-graph functions remain
AST-identical to frozen R2. No None/zero branch or native function was changed.
Root's reported115 PASS is separate earlier evidence, not this regression's proof.
Only affected tests were run; no overhead measurements or suites were introduced.

## Primary verification

Primary source inspection and independent targeted review accepted R1. Primary used the same Python3.12.13 interpreter and pinned native root as above:

```bash
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_v2_serving_inventory.py -k 'test_actual_native_classification_and_dp_dispatch or test_dispatch_successor_rejects_incomplete_or_wrong_operands or test_frozen_r2_non_graph_predicates_and_reader_guards_remain_identical' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/root-native-dispatch-r1 --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_native_dispatch_r1_20261022.xml
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_v2_serving_inventory.py tests/unit/test_native_serving_inventory.py tests/unit/test_native_reader_binding.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/root-native-dispatch-merged --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_native_dispatch_merged_20261022.xml
```

First workdir: v4-native-dispatch; second: merged main worktree. Expected: impossible input rejected, valid dispatch and reader/shadow boundaries preserved. Observed18PASS/27deselected in2.91s and116PASS in49.16s. Logs are beside the XML. No new performance, cardinality, GPU/native or calibration claim; the source slice remains within4files/488lines before the future builder binding.
