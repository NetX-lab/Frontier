# Profile03 native RPC CPU loop

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | 完成 W128-PROFILE03-RPC-LOOP；保留 source-bound CPU RED/GREEN、重复命令及限制。 |

## 1. 规划完成情况
**汇总：** PASS 3｜SKIPPED 1｜TODO 0。PASS 表示诊断交付完成，不是根因、修复或 native run 验收。
| 子任务 | 状态 | 依据 |
| --- | --- | --- |
| 读取约束并绑定 native source | PASS | 六个文件均匹配 native commit；四个同时匹配 package/实际 P1 source pins。 |
| 实际 native 方法 CPU loop 与重复执行 | PASS | 两次 matrix exit0；两次 symptom exit1；除计时外所有输出完全一致。 |
| 限定范围交付 | PASS | `rpc_loop.json`、本文、`rpc_loop_command.md`；scratch code 218/220 lines。 |
| 镜像/GPU/生产修复/进一步委派 | SKIPPED | 明确不属于本任务；没有 Docker、GPU、Git writes 或父文档编辑。 |

执行链为真实 AST 方法体：`MultiprocExecutor.execute_model -> collective_rpc -> FutureWrapper.result/get_response -> MessageQueue.dequeue/acquire_read/ReadTimeoutWithWarnings`；mock worker 内执行 `Worker.execute_model -> annotate_profile -> WorkerProfiler.step -> _call_stop -> TorchProfilerWrapper._stop`。
WorkerProfiler 使用真实 constructor/start 与 512 次 step，下一次 execute_model 内触发 autostop。Torch wrapper constructor、GPU decorators 和 Worker/Executor constructors 不执行；方法体不手抄。
| Case | 预期 = 实测 | virtual seconds / 关键读数 |
| --- | --- | --- |
| no_pause | GREEN | 0；两个回复就绪。 |
| within_budget | GREEN | 299。 |
| exact_deadline_ready | GREEN | 300；availability 先于 deadline 检查。 |
| autostop_exceeds_budget | RED | stop301；读到300时仍有待等回复，exact native TimeoutError。 |
| shared_deadline_remaining | RED | 第一个回复299；第二个 dequeue 剩余1秒，不重新获得300秒。 |
| all_ready_after_deadline | GREEN | stop301，302才 drain；两个 dequeue timeout=0 仍成功。 |
| no_aggregator_nonoutput_pause | GREEN | non-output stop301；native callsite 只选择 rank0 回复。 |
| future_fifo | GREEN | 请求后一个 Future 先 drain 前一个；顺序(call,rank)=0/0,0/1,1/0,1/1。 |
| future_fifo_lazy_ready | GREEN | 302 drain 两个 queued Futures；四次 timeout=0，保持 FIFO。 |
RED 原文：`TimeoutError: RPC call to execute_model timed out.`，direct cause 为 native queue `TimeoutError`；完整 native 路径 traceback 保存在 JSON。
已执行命令和原始日志见 [rpc_loop_command.md](rpc_loop_command.md)，总进程耗时0.258–0.312秒；不是等待300秒。

## 2. 偏离检测
**结论：** 未偏移，正常推进。**窗口：** 本次委派开始至交付；**前沿：** source-bound loop → repeated evidence → STOP。
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 仅实现要求的 autostop、shared deadline、lazy read 和 FIFO 控制。 |
| D2 细节反复重试 | CLEAR | 一次缺少 native 日志常量，绑定原常量后通过；原失败保留。 |
| D3 支线修补 | CLEAR | 没有修复生产/native/C26 或镜像依赖。 |
| D4 并行任务停滞 | CLEAR | 本委派禁止进一步 delegation；父任务仍由 root 协调。 |
| D5 超范围镀金 | CLEAR | 没有添加模型、8-worker EngineDeadError 或真实 transport 模拟。 |
| D6 分析空转 | CLEAR | 已交付0.3秒可重复 RED 命令。 |
| D7 路线违背 | CLEAR | local CPU only；输出仅在三文件与授权 scratch。 |
**文档检查：** S1/S2/S3/S4/S6 未发现本切片缺陷。S5：父 `plan.md:44`、`progress.md:44` 仍记 Starting；`timeline.md:23–28` 已记实际 fatal/terminal。本任务明确禁止改父文档。
**归因：** 无执行偏离；父状态同步由 root 负责。

## 3. 纠正建议
**分支：** B，父状态文档过期，本切片 scope/验收条件无需改变。
**文档更新（root 执行）：** 父 `plan.md:44`、`progress.md:44` 按 timeline 更新实际失败状态；可直接执行，不授权本 agent 编辑。
**需停止的偏离行为：** 无。
**接下去的路线：** 本切片交付后 STOP；root 接收证据，再决定实际 pending-reply 时序验证与修复。不得把本控制直接当作 profile03 根因结论。

## 4. 最值得关注的 3 点
1. 300秒是 pinned `envs.py` 的真实默认值。**超预算本身不保证 timeout**：drain 时所有回复已就绪仍能成功；原故障候选需要未就绪回复的实际时序证据。
2. 仅 local Python3.10.6，无镜像/binary qualification。worker-local stop 时长同步生成未来回复时间；不是并发进程。native queue 方法使用 mock buffer/wait/clock，spinloop 与 memory fence 未验证。Kineto/C++ parser、table dump、GPU、NCCL/NIXL、EngineDeadError 均未执行；仅 None outputs/PP1/两个 transport endpoints，未验证 connector payload merge 或原 run aggregator 激活。
3. 核心 source SHA256：wrapper=`eb0db0fd1c0a95c320fbaf9bdd66b6222a2d26ba9c1ac97efbfe1b196b3fbf15`；multiproc=`d5ca86635611758c2bd543ffd1fe9ad200db62880ce1f47eda8e4c63847595ea`。queue/KV aggregator 只匹配 native commit，不在 package/实际 source pin map 中。所有 hashes/AST ranges/变更边界见 JSON；原 import/初始 harness 失败原文保留。候选仍为 hypothesis，root owns repair decision。
