# Native balanced binding — primary verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded independent c491e6a verification, targeted review acceptance and final builder deadline/source binding. |

## Source correction acceptance

Root independently ran the following in `/data/ycfeng/Frontier/.worktrees/v4-balanced-binding` at source `c491e6a`:

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 \
  NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/v4-balanced-binding/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py \
  /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest \
  tests/unit/test_native_serving_inventory.py tests/unit/test_native_v2_serving_inventory.py \
  tests/unit/test_native_reader_binding.py -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/root_v4_native_r1_c491e6a \
  --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_v4_native_r1_c491e6a.xml \
  > /data/ycfeng/tmp/glm53_dsv4/root_v4_native_r1_c491e6a.log 2>&1
```

Expected: all affected source/observer/reader controls pass without skips. Actual XML: **133 tests, zero failures/errors/skips — PASS**. Original independent reviewer `W128-V4-BALANCED-NATIVE-R1-REVIEW` returned **APPROVE**, closing both schema-origin/immutable-source and effective-DP P2s, with no new actionable finding. Reviewer inspected source/receipts but did not rerun tests or Git; the command above is root's independent rerun. Native05 remains historically UNVERIFIED.

## Builder completion

Existing builder: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py`.

- Pass the existing full `commit` to `verify_balanced_frontier`; no new persisted config field.
- Set outer payload to3600s, replay to1800s, retain startup1800s, and align readiness cost to3600s/32GPUh.
- AST controls verified these exact values and the verifier call using frontier-py310 Python3.10.6; **PASS**. No build occurred during this check.
- Relative to preserved baseline SHA256 `21b5a7232e31df74052856d933c003b5af9caca7ce2fc6362fc1dbca36593560`, final builder is31 additions/7 deletions:3 dispatch lines plus35 routing lines. Receipt: `evidence/native_root_builder.json`.
- Final builder SHA256 `0b095d44aa5a53a73e3f6bad5e28dd1aa4e10c3c770bbd0d857162e7330fc4ac`.
- B cumulative routing is318 lines; A432+B318+C202=952 across23 files. Root used YC's delegated small-correction authority for the two additional replay-timeout lines beyond the950 ceiling. The original four-line reserve only covered payload and readiness, not replay. No resource budget changes.

## Remaining acceptance

Merge complete shared/native/prefill source, verify the combined actual config/package, freeze R3 with its unchanged R2 dependency, then build/seal and inspect the locked image before the authorized native06 run. This report accepts the corrected source slice and bounded builder edits, not native runtime, routing GPU behavior, reference stability, or E2E calibration.
