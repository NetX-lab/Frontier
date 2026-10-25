# Test report: H200 temporary mainline, wave 1 package preparation and submission (2026-10-25)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: P2-32r and P2-32p package builds, CPU checks, quota read and the wave 1 submission. |

**Scope.** CPU checks of [h200_mainline/design.md](h200_mainline/design.md) section 5 for `h200_mainline/p2_32_reference/package_01/` (P2-32r) and `h200_mainline/p2_32_profile/package_02/` (P2-32p), the quota read before submission, and the platform state just after submission. Authority: D-H200-TEMP-MAINLINE. Label: DIAGNOSTIC.

**Environment.** Local host. `/usr/bin/python3` 3.10.6, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`, `ulimit -v 31457280`. NATIVE checkout `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` at `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Harness worktrees, both clean:
- P2-32r: `/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model` at `b62c155abd981a71e12e62aa0c0fe3990741d2f8`;
- P2-32p: `/data/ycfeng/Frontier/.worktrees/v4-native-profile-run` at `a90c9dbbee53e2cabbfac73ba8490ffb958ce8eb` (branch `feat/v4-native-profile-run`, recreated with `git worktree add` for this work).

**Commands.**

```bash
cd task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 900 /usr/bin/python3 build_package.py p2_32_reference)
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 900 /usr/bin/python3 build_package.py p2_32_profile --package package_02)
```

Logs: `/data/ycfeng/tmp/glm53_dsv4/h200/build_p2_32_reference.log`, `build_p2_32_profile.log` (failed first build), `build_p2_32_profile_02.log`. Records: [h200_mainline/p2_32_reference/package_01_build.json](h200_mainline/p2_32_reference/package_01_build.json), [h200_mainline/p2_32_profile/package_02_build.json](h200_mainline/p2_32_profile/package_02_build.json).

## Results

| # | Check | Expected | P2-32r `package_01` | P2-32p `package_02` | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | Comparison with the source package | Only the leaves of design section 4 differ; every other file byte-identical | native06 file set; changed retained files `case/inputs/frontier.json`, `overlay.json`; `frontier.json` only the two `replica_config_device` leaves; `manifest.yaml` includes `H200_nonpreemptible_running_plus_queued_limit` 64 | profile02 file set; changed retained files `case/inputs/frontier.json`, `launch.py`, `overlay.json`; engine files only `profiler_config.torch_profiler_dir` | PASS |
| 2a | `seal.py` | seals the package | 43 files, `manifest.json` `9640996d…1055` | 45 files, `manifest.json` `690bd7d7…315d` | PASS |
| 2b | Run-manifest preflight on a scratch copy | harness commit, clean harness tree, NATIVE `2cf0a69`, run tag, H200, `step_main`, 32 GPUs | as expected; scratch `/data/ycfeng/tmp/glm53_dsv4/h200/p2_32_reference_preflight_t07u709v` | as expected, harness `a90c9dbb`; scratch `/data/ycfeng/tmp/glm53_dsv4/h200/p2_32_profile_preflight_k2uyxp8p` | PASS |
| 3 | Frontier reconstruction (R3 `verify_balanced_frontier` at the harness commit) | both roles device h200, TP8, `attn_dp` 2, EP16, PP1, world 16 | as expected; `moe_tensor_parallel_size` 1, routing `balanced` / `balanced_topk` | as expected | PASS |
| 4 | Offline launch path, 4 placeholder hosts | hosts 0-1 P lanes 0-1, 2-3 D lanes 0-1; heads 0 and 2; `--data-parallel-size 2`; profile: `--profiler-config` trace directory `<output_dir>/torch_traces/<role>/dp<lane>` | as expected; `profiler_config` null | as expected; trace directories under `.../dsv4-h200-p2-profile-20261025-01/torch_traces/{prefill,decode}/dp{0,1}` | PASS |
| 5 | `launch.py --check-only` | `INPUT_HASHES_PASS_NO_SUBMISSION` | as expected | as expected | PASS |
| 6 | Quota read before submission (`/data/ycfeng/tmp/glm53_dsv4/h200/monitor/quota_20261025T152240Z.json`) | at least 64 free non-preemptible H200 in `step_main` | `step_main` H200 total 512, non-preemptible use 218; 291 usable GPUs on nodes, 35 nodes with 8 usable GPUs | same read | PASS |
| 7 | Platform state after submission (`monitor/status_1.json` 15:25 UTC, `status_2.json` 15:27 UTC) | creator `i-fengyicheng`, preemptible `no`, `step_main` queue, 4 replicas, locked image, NFS mount of the package on this host | `exp-1025-152327-225209`: all as expected; phase Starting, then Running at 15:27 UTC | `exp-1025-152329-181604`: all as expected; phase Starting, then Running at 15:27 UTC | PASS so far |

**Failed first build of P2-32p.** `build_package.py p2_32_profile` (package `package_01`) completed checks 1 and 2a, then stopped at 2b with `ModuleNotFoundError: No module named 'vllm_replay'`. Cause: profile02's `native_pd_worker.py` imports `vllm_replay` at module level, and the builder put only the package root on `sys.path`, while `launch.py` also adds `tests/comparison/dp_placement_pp`. The builder now adds both paths. `p2_32_profile/package_01/` is left in place, incomplete and unused.

**Submission.** Wrappers `/data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_32_reference.sh` and `launch_p2_32_profile.sh` (handbook proxy, `STEPMIND_BACKEND=rjob`, `KUBEBRAIN_WORKSPACE_NAME=kun-workspace-vgen2`, `/data/ycfeng/tmp/stepmind-env/bin/python -u <package>/launch.py`) started under `setsid nohup` at 15:23 UTC. Launcher pids 503808 and 503908. Session H200 use: 64 of the 64 allowed.

## Limits

- Checks 1-5 are CPU checks of the packages and the launch commands. They do not show that the H200 hosts start vLLM with these commands or that R4 accepts an H200 run; those are design section 6 items and need the finished runs.
- Check 7 is the state just after scheduling. Worker `nvidia-smi`, terminal status and the run checkers are recorded when the jobs end.
- `replica_config_network_device` stays `h800_dgx` (design section 4).
