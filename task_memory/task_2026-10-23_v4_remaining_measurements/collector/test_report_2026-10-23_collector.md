# Collector launch-seam and baseline verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded CPU launch probes, separate requested baseline-suite results, source-integrity check and unapplied scope increment. |
| 2026-10-23 | Completed approved collector; recorded22 new tests, separate affected suites, GREEN launch and installed-image source/torch CPU controls. |

Task: `W128-V4-NATIVE-SERVE-PROFILE`. Latest outcome: **CPU collector PASS; GPU qualification UNVERIFIED**. Original pre-approval probes/results below are retained as historical evidence, not current status.

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

## Approved continuation: completed CPU collector

Same worktree/interpreter/environment as above. Root approved the exact12-line two-path addition, recorded in the main worktree requirements/plan; root files and commits were neither edited nor merged. No dependency install/GPU/SSH/cloud/native/C26 operation occurred. Docker CPU controls used the requested locked image, no pull/network/GPU, read-only source mount, and guarded/absent CUDA initialization.

Final source counts are in `completion_receipt.json` and `collector_source.diff`. All five source files are within individual allocations; one test file is exactly300 lines.

### New collector suite

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_native_serve_profile.py -q -p no:cacheprovider
```

Expected: all22 collector cases pass through the real module at this worktree's SOURCE path, including native stream function bodies loaded from pinned git objects. Actual: **22 passed in1.16s, exit0, PASS**. Not summed with baseline suites.

Coverage: nested range parents and separate parser prefixes; native has_prefill versus request warmup/formal identity; keyword-only passthrough and original exception identity; enter/exit/record errors isolated with traceback/UNVERIFIED; actual start/done record/wait generation/source/destination stream rows; no invented disabled/None-slot edges; capture/Python-replay/eager labels; PDL launch kwargs; FULL and PIECEWISE descriptor1/2/4/8/16 and padded local1/descriptor16; unchanged inventory RPC/counters/output; no extension selection and bounded overflow; missing CUPTI mapping preserves raw handles/null ID; saved attention alias; real launch fields.

### Required existing suites, rerun separately

Use the exact three commands already recorded above; observed after source changes:

| Suite | Actual | Verdict |
| --- | --- | --- |
| `test_native_serving_inventory.py` |28 passed in0.86s, exit0 | PASS |
| `test_native_v2_serving_inventory.py` |45 passed in5.65s, exit0 | PASS |
| `test_native_reader_binding.py` (worktree R4 checker env) |135 passed in55.51s, exit0 | PASS |

### Additional affected suites, separate

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_native_pd_worker.py -q -p no:cacheprovider
```
Expected: existing native lifecycle/provenance/staged-child defaults remain valid. Actual: **16 passed in3.34s, PASS**.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_dp_placement_harness.py -q -p no:cacheprovider
```
Actual: **75 passed in0.63s, PASS**.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_calibration_pd_harness.py -q -p no:cacheprovider
```
Actual: **68 passed in5.88s, PASS**.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_native_pd_manifest.py -q -p no:cacheprovider
```
Actual: **14 passed in1.34s, PASS**.

### Real launch GREEN probes

Exact command body is in `completion_receipt.json`. Imports real sibling `vllm_replay`, P/D module and native worker from this worktree. The runtime sibling mapping is the same object in all3 callers; importing the same Python file additionally by package-qualified name creates a second Python module object, so object-identity assertions across those two import names are not used.

Observed: `balanced_routing GREEN strategy balanced_topk`; `worker_pythonpath GREEN` points to this calibration directory; `profiler_config GREEN` contains the exact native torch JSON fields; `single_extension_mapping GREEN same_object true`. The earlier RED launch receipt is historical and unchanged.

### Direct CPU controls beyond persistent test budget

- Decorated sparse callback: real local collector wrap around a native-shaped outer eager decorator. Expected same native marker at capture and saved Python replay, with2 annotations; observed **original_calls2, annotation_count2, errors[], PASS**.
- Per-step local request/token1..11 versus synchronized16: real collector Worker wrapper and reused dispatch object. Observed **11 steps, local counts[1..11], synchronized_tokens16 for every step, errors[], PASS**. No collective/readback added by collector.

These are CPU controls, not native device/capture proofs. Exact executed bodies are retained in `completion_receipt.json`.

### Locked-image checks and failures

1. Attempted native imports of every hook with torch CUDA lazy-init prohibited. Container emitted `No module named vllm._C` and no CUDA-runtime warnings, then exceeded120s before final output. **INCOMPLETE**; only this CPU container `f71e92c7e6e6` was stopped with `docker stop -t2`. No native import/runtime resolution claim.
2. Read-only installed-source AST lookup using `importlib.util.find_spec('vllm')`, no torch import. Every15 class-method and9 importing-module free-function name found: **24/24 PASS**; installed root `/usr/local/lib/python3.12/dist-packages/vllm`, no CUDA import. This is source-name evidence, not executable class-method availability.
3. Real torch CPU profiler control initially failed because read-only container had no writable tempfile directory. Corrected only the control's scratch mount to approved `/data/ycfeng/tmp/opencode-v2-tmp/opencode:/scratch`, `TMPDIR=/scratch`. Observed torch **2.13.0+cu130**, `cuda_initialized:false`, collector annotations `frontier_native_outer|0|model_forward` and `frontier_native_component|1|input_projections`, errors[], record/wait methods callable: **PASS**. Native torch emitted `gpuGetDeviceCount failed code35` from CPU profiler device metadata; no CUDA context was initialized.

Common Docker flags: `--rm --pull=never --network=none --entrypoint python3 --memory=4g --read-only -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/workspace/tests/comparison/calibration -v /data/ycfeng/Frontier/.worktrees/v4-native-serve-profile:/workspace:ro`; image is the locked digest above. Python alarms45/60s bound controls. Exact source/torch command bodies and output facts are retained in receipt.

### Failure diagnosis and final limits

First collector test run:12 PASS/3 FAIL (stand-in context absent, PDL parameter hidden in variadic stand-in, incorrect dummy expectation). Corrected native-shaped fixtures, then22 PASS. Additional worker tests initially16 FAIL because their fake launch module lacks registry; moving the real mapping to a module-level sibling import fixes the cause, no compatibility fallback/second mapping/extra test file. Final16 PASS.

The collector always emits UNVERIFIED. Native result/exception remains untouched by instrumentation errors; record traceback errors/dropped capacity and leave stream mapping null if CUPTI unavailable. Native GPU stream IDs/PDL attributes/compile-breakable annotation and event survival/captured-segment ownership/CUPTI restart remain UNKNOWN. No timer events, costs or inferred GPU edges from source names. Root owns review, explicit capacity, driver and later GPU proposal.

Final integrity control: receipt JSON/per-file budgets/R4 checker SHA256/unchanged FullModelInventory source **PASS**. Source counts: collector405, native worker11, P/D environment4, command builder8, one test300 =728 changed lines. Diff artifact is zero-context (`--unified=0`) to avoid blank context lines being flagged as trailing whitespace in the committed evidence; consumers use `--unidiff-zero` if applying it.
