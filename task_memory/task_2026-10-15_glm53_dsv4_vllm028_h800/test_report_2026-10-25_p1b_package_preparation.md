# Test report: P1b H800-64 basic validation, package preparation (2026-10-25)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: package_01 build, comparison with native06, seal, run-manifest preflight, Frontier reconstruction, offline launch path, check-only, and the queue state. |

**Scope.** CPU checks of design section 5 ([p1b_basic_validation/design.md](p1b_basic_validation/design.md)) for `p1b_basic_validation/package_01/`. No GPU job was submitted. The label is DIAGNOSTIC `basic_validation`.

**Environment.** Local host. `/usr/bin/python3` 3.10.6 (PyYAML, pandas, numpy, scikit-learn), `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`, `ulimit -v 31457280`. Harness worktree `/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model` at `b62c155abd981a71e12e62aa0c0fe3990741d2f8`, clean. NATIVE checkout `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` at `2cf0a6915ce544dc493a0990f2ea38d81601128a`.

**Command.**

```bash
cd task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/p1b_basic_validation
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 900 /usr/bin/python3 build_package.py)
```

Log: `/data/ycfeng/tmp/glm53_dsv4/p1b/build_01.log`. Record: [p1b_basic_validation/package_01_build.json](p1b_basic_validation/package_01_build.json).

After the build, the launch-path check was extended to also build the `--kv-transfer-config` argument, as the worker does. It was rerun on the built package by importing `build_package.launch_path` (the builder refuses an existing `package_01/`, and the check writes nothing into the package); its result replaced `launch_path` in the record.

## Results

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Comparison with native06 | Same file set; JSON and YAML differ only at the leaves of design section 4; `launch.py` differs only in its two assertions; every other file byte-identical | Same 43-file set (44 after `seal.py` adds `manifest.json`). `config.json` and `case/inputs/native_config.json`: `replica`, `engine.data_parallel_size`, `run_tag`, `job_tag`, `output_dir`, `retained_files` for `case/inputs/frontier.json` and `overlay.json`, `tool_hashes` for `case/inputs/frontier.json`. Engine files: `data_parallel_size`. `frontier.json`: `replica_config_attn_dp` and both roles' `moe_expert_parallel_size`. `manifest.yaml`: the 11 listed leaves. `source_lock.json`: the four fields of the `frontier.json` entry. `evidence_readiness.json`: `purpose`, `target`, two cost fields. `overlay.json`: paths only | PASS |
| 2a | `seal.py` | seals the package | 43 files, `manifest.json` `b77768b6…3570` | PASS |
| 2b | Run-manifest preflight on a scratch copy of the case | harness `b62c155a`, clean harness tree, NATIVE `2cf0a69`, run tag, 64 GPUs | all as expected; scratch `/data/ycfeng/tmp/glm53_dsv4/p1b/package_01_preflight_tcck_8mq/` | PASS |
| 3 | Frontier reconstruction (R3 `verify_balanced_frontier`, harness `b62c155a`) | both roles: `attn_tensor_parallel_size` 8, `attn_dp` 4, `moe_expert_parallel_size` 32, `num_pipeline_stages` 1, world size 32 | both roles exactly these values; also `device` h800, `moe_tensor_parallel_size` 1, routing `balanced` / `balanced_topk` | PASS |
| 4 | Offline launch path for 8 placeholder hosts | hosts 0-3 P lanes 0-3, hosts 4-7 D lanes 0-3; heads 0 and 4; others headless with start rank = lane; `--data-parallel-size 4` | as expected. DP address 10.0.0.1 (P) and 10.0.0.5 (D); DP RPC port 24471 (P) and 24571 (D); `kv_transfer_config` accepted the NIXL side-channel windows 5600-5603 and 5700-5703 with DP4 (producer on P, consumer on D) | PASS |
| 5 | `launch.py --check-only` | `INPUT_HASHES_PASS_NO_SUBMISSION` | `INPUT_HASHES_PASS_NO_SUBMISSION` (also after the step 4 rerun) | PASS |
| — | Staged paths (as in the H4b prelaunch check) | overlay staged in the package; tool hashes match and lie inside the package; workspace is the package | all true. The only path still naming native06 is the intended provenance field `source_lock.retained_files["case/inputs/frontier.json"].source` | PASS |
| 6 | Quota read just before submission | at least 64 free H800, and H4b-(a) Running or finished | not reached: H4b-(a) `exp-1025-121318-827142` still Pending at 13:32 UTC, queue remaining H800=21 (`/data/ycfeng/tmp/glm53_dsv4/h4b/monitor/status_7.json`) | not run |

**Design correction.** Design section 2 said `pd_replay` checks RPC, HTTP and side-channel port *ranges*. The code (`pd_replay.kv_transfer_config`) checks only that the two NIXL side-channel windows `[port, port + dp)` are disjoint and exclude the role's HTTP and DP RPC ports. The design was corrected; check 4 now runs that function.

## Limits

- These are CPU checks of the package and the launch commands. They do not show that 8 hosts rendezvous, that vLLM starts DP4 with these commands, or that the R4 checker accepts a DP4 run. Those are the acceptance items of design section 6 and need the GPU run.
- `evidence_readiness.json` keeps native06's descriptive `native_limits` text ("all32rank forwards/replays"); its `target` and `cost` fields carry the P1 layout.
