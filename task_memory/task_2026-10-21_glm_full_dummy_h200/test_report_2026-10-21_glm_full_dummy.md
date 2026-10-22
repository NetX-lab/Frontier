# W128-FULL CPU verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded final focused tests, native source contracts, offline plan and verification limits. |

## Outcome

**PASS for the scoped native FULL CUDA-event producer's source/CPU contracts.**
**UNVERIFIED on GPU. Q11 end-to-end coverage remains OPEN:** no kernel-only
producer/admission or measured H200 artifact is delivered. No simulator code,
consumer, existing profiling table, or shared interface changed.

Environment: Linux host; working directory
`/data/ycfeng/Frontier/.worktrees/glm-full-dummy-h200`; Python
`/data/ycfeng/envs/frontier-py310/bin/python`, version **3.10.6**. Torch is absent
from this CPU environment. Native source checks use git objects at
`2cf0a6915ce544dc493a0990f2ea38d81601128a` in
`/data/ycfeng/Frontier/.real-engine/vLLM-BS`. No GPU/platform command was run.

## Focused tests

Exact final command:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS \
  /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_dsa_full_dummy_profiling.py \
  tests/unit/test_dsa_attention_profiling.py \
  tests/unit/test_dsa_extend_profiling.py \
  tests/unit/test_dsa_query_one_profiling.py \
  tests/unit/test_dsa_predictor.py \
  -q -p no:cacheprovider \
  --junitxml=task_memory/task_2026-10-21_glm_full_dummy_h200/focused.xml
```

Expected: all checks pass, native source contract does not skip, existing regular
q1/context0-to-FULL dummy rejection remains intact. Actual: **150 PASS, 0 FAIL,
0 ERROR, 0 SKIP in 3.95 seconds**. Evidence: `focused.xml`.

New CPU contracts cover non-width-one capture sizes, both index variants,
external event persistence, capture-only Python hook execution, requested
replay counts, actual query-one native-padding witnesses, and explicit rejection
of invalid slot mappings, missing metadata restaging, incomplete capture scopes,
Python hook execution during alleged replay, duplicate output overwrites, and
ordinary q1 serving plans. Native source assertions read the pinned git objects,
not the native working tree. Existing DSA tests include
`test_regular_zero_context_measurements_cannot_qualify_full_dummy`.

The event fixture's `0.25 ms` is synthetic test input. It is **not** a GPU
measurement, prediction, performance result, or simulator calibration.

Quick iterations: the first implementation passed 145 checks in 4.74 s; adding
negative paths/native contracts produced 150 checks. The final command above
was rerun after identity/backend validation changes. No hidden failures were
suppressed or changed to passing expectations.

## Offline CLI

Exact executed command:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /data/ycfeng/envs/frontier-py310/bin/python \
  -m frontier.profiling.attention.dsa_full_dummy plan \
  --model_name zai-org/GLM-5.3 \
  --model_config_path data/config/models/zai-org__GLM-5.3.json \
  --target_tensor_parallel_size 8 --block_size 64 \
  --max_context_len 7168 --max_kv_tokens_per_lane 14336 \
  --max_num_seqs 8 --chunk_budgets 1024 --capture_sizes 2 4 8 \
  --source_revision 2cf0a6915ce544dc493a0990f2ea38d81601128a \
  --model_config_revision repository-config-at-9ed979e \
  --expected_attention_backend FLASHMLA_SPARSE \
  --output /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_full_plan_9ed979e.json
```

Expected/actual **PASS**: schema `dsa_full_dummy.v1`; widths `[2,4,8]`; eight
reduced layers, four per variant; `measurement_type=CUDA_EVENT`;
`kernel_only_qualified=false`, `consumer_admission=false`,
`gpu_execution_observed=false`; no `records` field and no measured timing.
The ordinary experimental-GLM-profile warning is retained and expected. These
example widths and capacities are not the frozen formal native configuration.

## Static and scope checks

- AST parse of both new Python files: **PASS**.
- Python source sizes: producer about 240 lines, test about 180 lines; below
  the 2,000-line module limit and 500 total executable/test added lines.
- No existing source file is edited. `git diff --cached --check` is required
  immediately before the scoped commit; task records are explicitly staged
  despite the repository's ignored `task_memory/` rule.
- No 77-case simulator matrix or complete unit-suite rerun: the user explicitly
  scopes this slice to a new GPU profiling entry point and focused contracts,
  with no simulator behavior changes. No simulator numerical-parity claim is
  inferred from these checks.

## Remaining limits

Actual external-event graph replay, native recapture/stream behavior and H200
timing stability require the root's GPU qualification. Per-kernel duration and
kernel count attribution need a separately qualified capture-node/scope mapping;
CUDA-event wall time is not KERNEL_ONLY. The current consumer's typed error is
unchanged. See `notes.md` for the exact isolated schema, native references,
root-only launch requirements, and bounded follow-up proposal.
