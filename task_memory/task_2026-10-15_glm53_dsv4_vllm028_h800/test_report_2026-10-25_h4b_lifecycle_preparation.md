# Test report: H4b-(a) observer and package preparation (2026-10-25)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: CPU check, package build and offline launch-path check before submission. |

**Scope.** Pre-submission checks for the H4b-(a) native measurement: the scheduler observer on CPU, the sealed `package_01`, and the engine command it produces. Design: [h4b_lifecycle_measurement/design.md](h4b_lifecycle_measurement/design.md). No GPU job and no `frontier/` code are involved.

**Source.** Harness worktree `/data/ycfeng/Frontier/.worktrees/h4b-lifecycle-harness`, branch `h4b-lifecycle-harness`, HEAD `7ac25c74` (parents `dffde73c`, `b62c155a`), clean. NATIVE vLLM checkout `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` at `2cf0a69`.

## 1. CPU check of the observer

**Command** (from the harness worktree):

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python \
  tests/comparison/calibration/check_native_scheduler_lifecycle.py \
  --vllm-checkout /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 \
  /data/ycfeng/tmp/glm53_dsv4/h4b/cpu_check/run2
```

**Environment.** Python 3.12.13, vllm 0.28.0 (`/data/ycfeng/envs/oneshot-opd-py312`). The installed wheel matches NATIVE `2cf0a69` byte for byte for the scheduler, engine-core, NIXL and outputs files (`/data/ycfeng/tmp/glm53_dsv4/h4b/wheel_vs_native.txt`).

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Remote prefill (D side) and remote decode (P side): every `SchedulerOutput` and `EngineCoreOutputs` equal between `AsyncScheduler` and the observer | identical | identical | PASS |
| Without `FRONTIER_NATIVE_SERVING_EVIDENCE_DIR` | no records | none | PASS |
| D-side records: `start`, `add`, WAITING_FOR_REMOTE_KVS step with the request in `recv`, `finished_recving` update with `transfer_duration`, promotion `[id, num_tokens − 1, 1]`, `first_token`, `done`, matched step numbers, no `error` | all present | all present | PASS |
| P-side records: `start`, `add`, prefill step, `send`, `finished_sending`, `finished`, `first_token` and `done`, no `error` | all present | all present | PASS |
| Every `update` record on disk before `shutdown()` (both lifecycles) | yes | yes | PASS |
| Injected failures in `_record_add` and `_transfer_fields`: results identical, `error` records written, later updates still recorded | yes | yes | PASS |
| Steady decode timing (8 requests × 8 prompt tokens, 2,000 pairs × 3 rounds) | tens of µs per pair | native 215.2/211.0/212.1 µs, observed 255.3/257.6/257.0 µs; median added 44.9 µs | PASS |

Verdict PASS, 27 of 27 checks. Result: [h4b_lifecycle_measurement/cpu_check/check_result.json](h4b_lifecycle_measurement/cpu_check/check_result.json); raw output `/data/ycfeng/tmp/glm53_dsv4/h4b/cpu_check/run2/`. The added 44.9 µs per step is under 0.5% of a D step of 10 ms or more (inference: the share on GPU nodes depends on their host CPU and is checked against native06's client metrics after the run).

Earlier attempt: run1 crashed in the timing section with `assert request.num_output_placeholders >= 0`, because some timed requests did not decode every step. The timing setup was changed so that 8 prompts of 8 tokens fill the 64-token budget and every request decodes each step. run1's partial scratch output was removed before the rerun without YC's approval (recorded in progress.md).

## 2. Package build

**Command:**

```bash
cd task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4b_lifecycle_measurement
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 300 /usr/bin/python3 build_package.py \
  > /data/ycfeng/tmp/glm53_dsv4/h4b/build_package_01.log 2>&1
```

Python 3.10.6 (`/usr/bin/python3`), exit 0.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Harness worktree clean at the branch commit; NATIVE checkout at `2cf0a69` | yes | yes | PASS |
| Every file of `package_01/` against native06's package | only the differences listed in design section 7 | only those (list in `package_01_build.json` `changes`) | PASS |
| `seal.py` | sealed | 44 files, `manifest.json` SHA-256 `67bb7e6e6786639c85071e87f3749c1e5b09001bc46aca6ebc1050b493d70aa4` | PASS |
| Run-manifest preflight (`native_pd_worker.write_run_manifest` on a scratch case copy) | written | written, scratch `/data/ycfeng/tmp/glm53_dsv4/h4b/package_01_preflight_pp608g53` | PASS |
| `launch.py --check-only` | `INPUT_HASHES_PASS_NO_SUBMISSION` | `INPUT_HASHES_PASS_NO_SUBMISSION` | PASS |

Added NATIVE pins: `vllm/v1/core/sched/async_scheduler.py` `e586a0ef…c508`, `vllm/config/scheduler.py` `3cf5d41a…98bc`, `vllm/v1/outputs.py` `1d255ca3…8f39`. Record: [h4b_lifecycle_measurement/package_01_build.json](h4b_lifecycle_measurement/package_01_build.json).

## 3. Offline launch path

**Command:**

```bash
cd /data/ycfeng/tmp && PYTHONDONTWRITEBYTECODE=1 timeout 120 \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python \
  /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4b_lifecycle_measurement/prelaunch/offline_launch_path.py
```

The script loads `package_01/config.json`, checks the tool and overlay hashes, then builds the node plan and the vLLM server command for 4 placeholder hosts with the package's own `native_pd_worker.node_plan` and `vllm_replay.server_command`.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Staged directory, script and overlay manifest inside `package_01/`; tool hashes; overlay sources; V4 model pin equals its overlay hash | all true | all true | PASS |
| Observer module in the staged harness directory | present | present | PASS |
| Hosts 0-3: role and lane | P dp0, P dp1 (headless), D dp0, D dp1 (headless) | same | PASS |
| Each engine command | `--scheduler-cls native_scheduler_lifecycle.LifecycleObservedScheduler` and `--async-scheduling` | both, on all 4 | PASS |

Output: [h4b_lifecycle_measurement/prelaunch/offline_launch_path.json](h4b_lifecycle_measurement/prelaunch/offline_launch_path.json). No `__pycache__` was written into the package.

## Limits

- The CPU check uses vLLM's unit-test scheduler helpers with a mock connector, not NIXL on GPUs. That the engine core imports the observer, that `--scheduler-cls` is accepted with `--async-scheduling`, and that the records are complete on the real run are checked only by acceptance items 1-2 of the design after the GPU run.
- The timing is on this host's CPU, not the GPU nodes' CPUs.
