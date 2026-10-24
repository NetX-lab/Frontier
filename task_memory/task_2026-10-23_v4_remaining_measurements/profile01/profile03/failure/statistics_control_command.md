# Statistics control commands

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | 保留两次实际fresh CPU命令、source binding、native tiny-file和旧证据不变验证。 |

Task `W128-PROFILE03-STATISTICS-CONTROL`；cwd `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`，local Python3.10.6，未安装依赖/运行Docker或GPU。
两条命令已实际执行，各exit0；每次3 intended RED/2 GREEN，matrix wall_seconds=0.016420/0.015765。这是控制矩阵内部实测运行时间，不是原CPU/GPU阶段性能。
```bash
/data/ycfeng/toolchain/bin/uv run --offline --no-project --python /data/ycfeng/envs/frontier-py310/bin/python -- python -B /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/adapter.py /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/run_1 > /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/run_1.json 2> /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/run_1.stderr
/data/ycfeng/toolchain/bin/uv run --offline --no-project --python /data/ycfeng/envs/frontier-py310/bin/python -- python -B /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/adapter.py /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/run_2 > /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/run_2.json 2> /data/ycfeng/tmp/glm53_dsv4/profile03_statistics_control_native_20261024/run_2.stderr
```
重复时使用同一adapter加一个尚不存在的新child output目录；run_1/run_2保留，不覆盖。原stdout为run_1.json/run_2.json，两个stderr为空；文件paths在输出内相对fresh目录记录，除wall_seconds全部字段完全相同。
新增adapter SHA256 `ed17c13990d1058d514f58358f52d09d1fb3ccaf58d0843cfd3977dfca0d9c1a`，139/180gross lines；旧loop SHA256 `5608b2ec83b0744fa756ca203f8d01963b33516a6018d7adab1f5c1c37db80d9`，218行不变。
旧loop AST只删除四个top-level nodes：Assign202、If203–213、Expr214–217、Expr218；compile保留此前所有imports/bindings/transport/case。扩展两个原生table方法，原四个Torch方法code objects相等；使用原URI helper、213行活动赋值和90行true默认值。没有native branch或timeout patch。
每个fresh目录保留两个endpoint×五case的mock_export_event.marker；true CUDA与CPU-only false各有两份native profiler_out_<rank>.txt。所有marker/table内容hash、kwargs、native stdout及调用次数在JSON中，实际文件未移到cloud或旧目录。
根source为commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`；七文件分别通过只读 `git show <commit>:<path>` 哈希比较。wrapper=`eb0db0fd1c0a95c320fbaf9bdd66b6222a2d26ba9c1ac97efbfe1b196b3fbf15`，multiproc=`d5ca86635611758c2bd543ffd1fe9ad200db62880ce1f47eda8e4c63847595ea`，config/profiler=`78af917ec2ea2ba3bacda15f58fda9782b8ff82c0b810fd5c9563f4203aa5ba8`。
根已验收原loop的receipt `/data/ycfeng/tmp/glm53_dsv4/profile03_rpc_root_jgnthas4/receipt.json` 仅只读并记录hash；旧九份durable和旧scratch十一文件共20个SHA256均保留。新增helper没有失败；旧torch import失败和初始harness失败未改。
生成证据 `statistics_control.json` SHA256 `e56d760e6e2655f2bac70489a4a580834618d7ac28fe8c5c4625d07d10cd9397`；exact argv/source pins/native ranges/limits见该文件。只证明受控native toggle行为，不证明原phase耗时、GPU保证或raw-trace identity。
