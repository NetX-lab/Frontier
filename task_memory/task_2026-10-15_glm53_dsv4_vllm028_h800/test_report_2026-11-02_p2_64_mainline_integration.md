# Test report: P2-64 harness integration into the final task branch

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Recorded accepted harness integration, independent verification, the resolved mock failure, deferred CUPTI errors, and unchanged frozen inputs. |

## Scope and result

Ticket 18 source integration PASS. The five-file change accepted at `4861e680548a5666f3107945ec4881b7022c23fb` is integrated into `feat/glm53-dsv4-vllm028`, based on `39d710a4`. One existing test mock is updated for the explicit default `pair_warmup=False`. This does not admit the queued GPU reference or close tickets 18/19.

The lead reviewed the complete six-file diff. `native_pd_worker.py` retains `NATIVE_WORKER_EXTENSIONS` and existing profiling/shutdown behavior. `pd_replay.py` retains `profiler_config`, targeted warmups cover the P/D Cartesian product, and formal requests retain native routing. The new pair-warmup test is byte-identical to the accepted harness. No `frontier/` file changes relative to `0ee7f6c2`.

Changed files:

- `tests/comparison/calibration/native_pd_worker.py`: validate and retain the issued promotion receipt; record preemptibility; pass the warmup option.
- `tests/comparison/calibration/pd_proxy.py`: translate warmup-only P/D targets into each role's native rank header.
- `tests/comparison/calibration/pd_replay.py`: cover configured P/D pairs in canonical warmups; reject targets on formal requests.
- `tests/comparison/dp_placement_pp/vllm_replay.py`: send pair headers only for valid warmup records.
- `tests/unit/test_native_pair_warmup.py`: accepted HTTP, native routing and receipt regressions.
- `tests/unit/test_native_streaming_replay.py`: explicitly receive and assert the disabled pair-warmup default while preserving profiler-option assertions.

## Environment and exact commands

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
Python: `/data/ycfeng/envs/openmopd-py312/bin/python`, Python 3.12.13.
Environment: `PYTHONPATH` is the working directory; `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`; `TMPDIR=/data/ycfeng/tmp/glm53_dsv4/h200/p64_mainline_lead_20261102` for lead checks.

Exact argument arrays, start/end timestamps and environment are preserved under `h200_mainline/p2_64_formal/integration_20261102/`:

- `focused_command.json` / `focused.log`: seven relevant test files; expected all pass.
- `corrected_default_routing_command.json` / `corrected_default_routing.log`: the two affected parameterizations only.
- `native_profile_command.json` / `native_profile.log`: native profile source verification with `FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.
- `pi_verification.json`: exact delegated pytest commands and retained output; model `custom/deepseek-v4.1-flash-sys`, thinking `high`, session `7db34b2a-7536-4ef2-a139-40b1d1a40af6`.
- `frozen_check_only.json` / `frozen_check_only.log`: exact `python3 -B .../package_02/launch.py --check-only` command and frozen harness identity.

## Observed results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Pi initial five-file suite | Warmup, worker, manifest, canonical HTTP and calibration regressions pass | 136 passed in 16.03 s | PASS |
| Lead initial seven-file suite | Relevant behavior passes | 2 failed, 174 passed, 35 skipped in 80.34 s | FAIL, resolved below |
| Lead affected mock recheck | Both plain and profiler paths receive `pair_warmup=False` and retain original options | 2 passed in 1.17 s | PASS |
| Pi serve-profile/streaming recheck after lead adjustment | Relevant runtime harness assertions pass | 40 passed, 35 skipped in 62.18 s | PASS for executed tests; native-source skips resolved below |
| Lead native profile with retained source path | Execute all native-source cases | 36 passed in 3.15 s, no skips | PASS |
| Frozen package check-only | Input hashes pass without submission | `INPUT_HASHES_PASS_NO_SUBMISSION`, exit 0 | PASS |
| Accepted source and scope | New test identical; six tested file hashes unchanged; frozen harness clean at `4861e680`; empty core diff | All assertions pass; `git diff --check` exit 0 | PASS |

The two initial failures were exact `**options` assertions in `test_pd_main_dispatches_canonical_three_repeat_client_without_native_ownership`: the dispatcher now supplies `pair_warmup=False`. The lead added that explicit mock parameter and asserted it is false. The profiler assertions remain intact. The later Pi pass reflects this lead adjustment, not spontaneous recovery.

The 35 skips came from `test_native_v2_serving_inventory.py`'s missing `FRONTIER_VLLM_BS_ROOT`. The retained native source was available; setting that input yielded 36/36 for the whole profile file. The relevant checks therefore have passing evidence after the correction; no claim is made that one final all-suite invocation passed.

Pi also expanded to unrelated CUPTI tests: a run reported 2 failed, 198 passed, 35 skipped and 25 errors, with the errors in `test_cupti_scope_lifetime.py`. Only truncated error summaries were retained; the root cause is UNKNOWN and is not classified as a baseline or environment failure. A separate broader native-only run reported 172 passed, but was unnecessary to source acceptance. Repeated broad execution was stopped through `run_ds4.sh --stop p64-mainline-integration-1102c` (exit 130). No CUPTI code was changed. The limitation is recorded in `future.md`; it does not reopen the stopped investigation or waive any GPU admission criterion.

## Frozen inputs and verification limits

`source_verification.json` and `final_verification.json` bind the reviewed six files and preserve the accepted isolated harness. `package_02` remains the package submitted for `exp-1102-164536-160317`; its run manifest SHA256 remains `c22085987055d018efb8abc104081cfe56d001b22026bb3aa367564d46754cd7` as verified by the existing check-only/startup evidence.

`queue_status.json` copies the latest existing Pi status (`status_p64f_40.json`, mtime 2026-11-02T10:45:58Z): Starting, 8 pending, preemptible `yes`, creator `i-fengyicheng`. Launcher PID 3140124 is alive; the original GPU-monitor Pi has no exit record. These local reads make no new platform request. The earlier exact-job replica/events evidence establishes unscheduled co-scheduling membership; the specific scarce resource and ETA remain unknown.

The final staged whitespace check flagged spaces emitted by pytest in the preserved raw `focused.log` traceback. The raw log stays byte-identical. The staged check on all source, Markdown and JSON files passes; generated `.log` files are excluded from that style check.

CPU checks do not prove actual pair coverage, native runtime admission, four-metric accuracy or GPU usage. Terminal receipt, cloud copy, pair coverage, R4, all-request/warm readings and actual container GPU-hours remain dependent on the submitted job. No simulation replay was repeated.
