# Profile03 CPU loop commands

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | 保留两次实际 RED/两次 matrix 的 exact argv、输出路径及 source hashes。 |

Task `W128-PROFILE03-RPC-LOOP`。工作目录 `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`；local Python3.10.6，未激活 conda/安装依赖/运行 Docker 或 GPU。
RED-capable 命令，已实际执行两次，每次 exit1、exact `RPC call to execute_model timed out.`，0.258/0.312秒：
```bash
/data/ycfeng/toolchain/bin/uv run --offline --no-project --python /data/ycfeng/envs/frontier-py310/bin/python -- python -B /data/ycfeng/tmp/glm53_dsv4/profile03_rpc_loop_native_20261024/loop.py --symptom
```
完整控制矩阵，已实际执行两次，每次 exit0、7 GREEN/2 intended RED，0.265/0.267秒：
```bash
/data/ycfeng/toolchain/bin/uv run --offline --no-project --python /data/ycfeng/envs/frontier-py310/bin/python -- python -B /data/ycfeng/tmp/glm53_dsv4/profile03_rpc_loop_native_20261024/loop.py
```
scratch `S=/data/ycfeng/tmp/glm53_dsv4/profile03_rpc_loop_native_20261024`：原输出 `S/{matrix,symptom}_{1,2}.json`，原 stderr 同名 `.stderr`；exact argv/exit/timing/output hashes 在 `rpc_loop.json.commands`。两次同类输出除 wall_seconds 完全一致。
真实 native import 的失败 `ModuleNotFoundError: No module named 'torch'`、`vllm._version` warning 保留于 `S/native_import_failure.log`；exact argv 在 `rpc_loop.json.native_import_probe`。未安装依赖，故采用明确标记的 native AST slices。
初始 harness `NameError: LONG_WAIT_TIME_LOG_MSG` 保留于 `S/initial_missing_log_constant.log`；从 native AST 绑定日志常量后重跑，无 native/source 行为修改。
native commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`；六个文件通过只读 `git show <commit>:<path>` SHA256 与本地比较。wrapper/multiproc/gpu_worker/envs 同时通过 package config 与实际 P1 source pins；queue/KV aggregator 无 run source pin，不声称实际镜像一致。
script SHA256 `5608b2ec83b0744fa756ca203f8d01963b33516a6018d7adab1f5c1c37db80d9`；218/220 gross lines。证据 JSON SHA256 `c06abc548549a35ad7e3411585870fc629aee8677deffda1aaeb3953058270b4`。
仅证明 source-bound 条件性 autostop/RPC 行为；不是 GPU、8-worker EngineDeadError 复现或已确认根因，未选择任何生产修复。
