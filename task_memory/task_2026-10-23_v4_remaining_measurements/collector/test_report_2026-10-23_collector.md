# Collector launch-seam and baseline verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded CPU launch probes, separate requested baseline-suite results, source-integrity check and unapplied scope increment. |

Task: `W128-V4-NATIVE-SERVE-PROFILE`. Outcome: **scope BLOCKED; collector NOT IMPLEMENTED**.

## Environment

- Worktree: `/data/ycfeng/Frontier/.worktrees/v4-native-serve-profile`, clean baseline `993f05f59ef22bfab38433dd93e2fcf486fc113a`.
- Python: `/data/ycfeng/envs/asyncopd-py312/bin/python`, CPython **3.12.13**; pytest **9.0.2**.
- CPU only; no dependency installs, CUDA initialization, native import, GPU job, platform/SSH/cloud/native/C26 edits.
- Each pytest process used `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`, and `-p no:cacheprovider`.
- Source binding: `FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.

## Requested suites (separate, never combined)

Run from the worktree above:

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_native_serving_inventory.py -q -p no:cacheprovider
```

Expected: all existing serving-inventory contracts pass. Actual: **28 passed in 0.95s**, exit 0, **PASS**.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_native_v2_serving_inventory.py -q -p no:cacheprovider
```

Expected: all existing V2 serving and pinned native dispatch controls pass. Actual: **45 passed in 4.42s**, exit 0, **PASS**.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/v4-native-serve-profile/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_native_reader_binding.py -q -p no:cacheprovider
```

Expected: R4 reader controls pass against the explicitly bound worktree checker. Actual: **135 passed in 55.17s**, exit 0, **PASS**. Checker SHA256 is retained in `launch_seam_receipt.json`.

Collector stand-in suite: **NOT RUN**; collector is absent. Baseline PASS is not collector acceptance.

## Actual launch-path probe

The exact Python command body and observed JSON lines are retained in `launch_seam_receipt.json`. It imports the real `tests.comparison.*` launch utilities in this worktree and uses the existing native-engine fixture. The command builder's model metadata read was replaced with a CPU stand-in solely to inspect argument forwarding; no model file was written.

| Probe | Expected passing condition for completed H | Observed baseline | Verdict |
| --- | --- | --- | --- |
| New extension with balanced_topk | Accept the explicitly registered composed extension | `ValueError: Native balanced routing requires the registered worker extension` | BLOCKED |
| New extension worker PYTHONPATH | Existing harness directory forwarded to spawned workers | `pythonpath: null`; old extension forwards the current calibration directory | BLOCKED |
| Existing native profiler_config | CLI contains `--profiler-config` and JSON fields | `forwarded: false` | BLOCKED |

These three omissions require two additional file owners; they cannot be fixed solely in the allocated new collector and `native_pd_worker.py:269` branch without changing or bypassing the existing route.

## Proposed increment verification

`scope_increment.diff` is an **unapplied zero-context proposal**. Default `git apply --check --numstat` rejected it because zero-context patches require explicit permission; source files remained unchanged. Correct check:

```bash
git apply --check --numstat --unidiff-zero task_memory/task_2026-10-23_v4_remaining_measurements/collector/scope_increment.diff
```

Observed exit 0, `7 1 tests/comparison/dp_placement_pp/vllm_replay.py` and `2 2 tests/comparison/calibration/pd_replay.py`: **PASS applies to baseline; 12 changed lines total**, not authorization or behavioral verification of the proposed implementation.

```bash
git diff --exit-code 993f05f5 -- tests/comparison/calibration/native_pd_worker.py tests/comparison/calibration/native_full_structure.py tests/comparison/calibration/pd_replay.py tests/comparison/dp_placement_pp/vllm_replay.py
```

Observed exit 0, no output: **PASS these sources remain unchanged**.

## Optional locked image

Docker handbook read before this command:

```bash
docker image inspect hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635 --format '{{.Id}}'
```

Observed identical local image ID, exit 0. **PASS image available locally only**. No container was run; installed hook resolution was **NOT RUN** after the stop-on-scope instruction applied.

## Verification limits and next gates

No new stream/event/PDL observation exists. Capture/replay event survival and CUPTI restart are unknown; no timings, dependencies or GPU completeness are qualified by the CPU baseline results. Root must allocate the launch increment; then H must implement and verify the real collector, resolve installed hooks, preserve every inventory record, and pass its error-isolation controls. Root still owns the separate measured-phase driver bracket and explicit later GPU request.
