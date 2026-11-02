# Profile03 native statistics-toggle control

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | 完成 W128-PROFILE03-STATISTICS-CONTROL；保留受控 RED/GREEN 与限制，按最新profile04分配记录root后续路线。 |

## 1. 规划完成情况
**汇总：** PASS 3｜SKIPPED 1｜TODO 0。PASS 为本 CPU 控制交付，不是原故障根因或 GPU 验收。
| 子任务 | 状态 | 依据 |
| --- | --- | --- |
| 复用已验收 source-bound loop | PASS | 旧 loop SHA5608b2ec…80d9；AST 仅移除末尾 runner，旧文件不编辑。 |
| native statistics-toggle/负控制/FIFO | PASS | 两次 fresh matrix exit0，3 intended RED/2 GREEN；除 wall_seconds 全字段相同。 |
| 限定记录与旧证据保留 | PASS | adapter139/180行；九个旧 durable artifacts与旧 scratch十一文件，20 hashes均不变。 |
| 生产/native/package/GPU/父文档/Git/委派操作 | SKIPPED | 不在此分配范围，没有执行。 |

受控输入是目标 endpoint1 的 export222/statistics313秒，peer endpoint0两项成本0；**不是原 run CPU/GPU 性能测量**。Native RPC 默认300秒不变。
| Case | dump CUDA / activities | native结果 | 每个endpoint export/key_averages calls | native text tables |
| --- | --- | --- | --- | --- |
| cuda_default_true | 默认true / CPU+CUDA | RED at300 | 1/1 | 2；`self_cuda_time_total`，默认row limit。 |
| cuda_false | false / CPU+CUDA | GREEN at222 | 1/0 | 0。 |
| export_too_slow_false | false / CPU+CUDA；export301 | RED at300 | 1/0 | 0。 |
| cpu_only_false | false / CPU；export222/stat313 | RED at300 | 1/1 | 2；`self_cpu_time_total,row_limit=50`。 |
| cuda_false_fifo | false / CPU+CUDA；两个Futures | GREEN at222 | 1/0 | 0；原FIFO四读顺序通过。 |
全部 RED 是原 loop 的 exact native `TimeoutError: RPC call to execute_model timed out.`；direct cause 与完整 traceback 在 JSON。所有case、两个endpoint都写出相同内容的 **mock export/event marker**，statistics输入仅在export完成后调用。tiny text tables由真实 native `_write_profiler_table` 写入新 scratch；内容、native rank0 stdout和遗漏均被断言。
复用 `Worker.execute_model/annotate_profile`、512次step后第513次autostop，以及原 MultiprocExecutor/get_response/FutureWrapper/native queue方法；没有第二套timeout算法。
新增 source-bound wrapper方法仅 `_build_profiler_table:255–265`、`_write_profiler_table:267–275`。原 `_start/_stop/_profiler_step/_maybe_add_version_metadata` code objects全部相等；不改native flag branch。另执行原 constructor活动赋值 `wrapper.py:213`、原 `_is_uri_path` helper `config/profiler.py:24–35`；默认true来自同文件90行，不声称完整ProfilerConfig/image constructor已验证。

## 2. 偏离检测
**结论：** 未偏移，正常推进。**窗口：** 本次follow-up至交付；**当前前沿：** native toggle控制 → fresh equality/immutability → STOP。
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 仅请求的toggle、两项负控制与继承FIFO。 |
| D2 细节反复重试 | CLEAR | 两次fresh CPU执行直接通过；没有新增helper失败。 |
| D3 支线修补 | CLEAR | 不改原loop/native/package或安装依赖。 |
| D4 并行任务停滞 | CLEAR | 两个独立fresh命令并行；本分配不进一步委派。 |
| D5 超范围镀金 | CLEAR | 不增加timeout、模拟八worker或CPP统计parser。 |
| D6 分析空转 | CLEAR | 实际五case输出与原生tiny文件已交付。 |
| D7 路线违背 | CLEAR | 所有新写入仅本scratch与三个授权记录。 |
**文档检查：** S1–S6均未发现本切片明显缺陷；requirements D-V4-PROFILE03-STATISTICS-CONTROL 与 plan selected diagnostic dependency一致。**归因：** 无。

## 3. 纠正建议
**分支：** A，scope/验收合理，无需改父spec或tickets。**需停止的偏离行为：** 无。
**接下去的路线：** 本控制交付后STOP；root按最新 `D-V4-PROFILE04-PACKAGE` 分配推进fresh package及实际验证。该CPU控制不授权本agent修改plan/package或提交GPU；root独立镜像检查不算本agent执行结果。

## 4. 最值得关注的 3 点
1. CPU+CUDA下false确实跳过native text statistics且mock export仍执行；CPU-only自动统计不受CUDA flag关闭影响。export301+false仍RED，因此不是任何未来export≤300秒的保证。
2. 未测量原故障export/statistics阶段、实际pending-reply时序或GPU性能；mock marker不是raw trace，不能证明raw-trace identity、完整性或保存语义。没有Kineto/CPP parser、GPU、NCCL/NIXL、EngineDeadError执行；继承原loop的constructor/decorator、None输出、PP1、虚拟transport和两endpoint限制。
3. 七个native文件匹配commit2cf0a691；五个同时匹配package/实际P1 pins，queue/KV aggregator仍无run source pin。exact transforms、hashes、commands和原生文件hash在 [statistics_control.json](statistics_control.json)；原helper/import失败保留在未改旧证据中。root owns final selection/repair。
