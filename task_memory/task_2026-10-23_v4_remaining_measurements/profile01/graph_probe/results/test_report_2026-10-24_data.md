# W128-GRAPH-PAYLOAD-DATA local test report
## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded independent raw-data derivation, negative controls, process reproducibility, proof limits and phase checkpoint. |

## 1. 规划完成情况
**汇总：** PASS5｜BLOCKED/FAILED/SKIPPED0｜TODO0
| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 输入绑定 | PASS | 4原始文件53146B与copy receipt一致；31输入哈希、8文件+manifest闭包、source/seal/CPU/matrix绑定通过。 |
| 独立U1–U4推导 | PASS | 实际trace、callback字段、event handle与DAG通过；未调用producer判定函数。 |
| U4负fixture | PASS | 六种复制后修改的fixture被拒绝；原始数据未修改。 |
| 两个新进程复现 | PASS | 完整输出逐字节一致；31检查PASS、0FAIL。命令/版本/digest见command.md。 |
| 交付与预算检查 | PASS | 只读guard通过：三个指定文件、Markdown89行、脚本136行、四个原始文件哈希不变。 |

验收预期与观测（JSON数组索引均为0-based；以下不是按名称/顺序/低位数值推断身份）：
| 证明 | 预期 | 实际观测与原始字段 | 结果 |
| --- | --- | --- | --- |
| U1 | 唯一目标launch、5个正ID kernel | traceEvents[58] CPU annotation内唯一[76] cudaGraphLaunch；correlation14连接[66,68,70,72,74]，graph id2；全部trace共6个kernel。 | PASS |
| U2 | 严格9节点/5kernel/4event | report.dag.nodes完整、handle/full ID唯一且正值；8条双向一致DAG边、无环；source graph1、exec graph2。 | PASS |
| U3 | 完整唯一callback连接、每kernel一次、稳定source身份 | report.callback_capture.events：9条NODE_CLONED20覆盖全部9节点含4event，唯一EXEC_CREATED18；before=after=完整DAG IDs；5个replay clone ID各出现一次。 | PASS |
| U4 | 4独立事件边界、2scope各2kernel、1无scope kernel | report.event_handles与dag.nodes[].event_handle直接匹配；DAG可达性交集给出a={1,2}、b={5,6}，outside={8}。 | PASS |

全部source→clone uint64 ID由原始callback字段直接给出；handle及完整callback记录另见data_audit.json，不使用ID算术：
| DAG index | 类型/边界 | 原始full ID | clone full ID |
| --- | --- | --- | --- |
| 0 | event/a.start | 4294967296 | 8589934592 |
| 1 | kernel/a | 4294967297 | 8589934593 |
| 2 | kernel/a | 4294967298 | 8589934594 |
| 3 | event/a.end | 4294967299 | 8589934595 |
| 4 | event/b.start | 4294967300 | 8589934596 |
| 5 | kernel/b | 4294967301 | 8589934597 |
| 6 | kernel/b | 4294967302 | 8589934598 |
| 7 | event/b.end | 4294967303 | 8589934599 |
| 8 | kernel/unscoped | 4294967304 | 8589934600 |
事件handle：a.start262820000、a.end284948128、b.start284948720、b.end284949312；四个都唯一。EXEC_CREATED记录graph handle292034848→exec handle292280944，直接查询ID1→2。原始DAG的8条边为0→1→2→3→4→5→6→7→8；这是边字段观测，不用数组顺序推导。
负对照：traceEvents[60] correlation4、graph/node ID0，仅此1个kernel被排除；它与目标kernel可以同名，不影响直接ID连接。U4复制fixture分别测试缺边界、重复边界、环、不一致边、共享kernel scope、错误节点类型，6/6拒绝。早期交换相邻边界仍是合法scope，故最初fixture被接受；改为真实共享kernel拓扑后拒绝，未修改probe或原始数据。
诊断：52行sequence1..52、pid152、monotonic严格递增；EnableCallback enter/exit各2，其余具体stage各1。订阅/两次enable/unsubscribe的4个API result均0，callback errors=[]、report无error字段、fault.log0B。stages:33已unsubscribe，34读取retained graph，38进入Kineto，39 entered，42–43 target replay，46退出；source194–202/528–575说明callback释放与引用寿命。没有逐callback时戳或native内存转储，不能把stage边界扩展成未观测的逐query时间证明。
来源与环境：report.script_sha256与实际本地主源码/封包源码一致；ABI报告ResourceData24B/align8/offsets0,8,16，GraphData56B/align8。report/API提供torch2.13.0+cu130、CUDA bindings13.3.1、runtime13.0.96、CUPTI13.0.85；source204–229静态证明descriptor解引用和in-callback query顺序。报告与trace.deviceProperties[0]均NVIDIA H800，device0；driver13000，host与提供的platform receipt一致。
平台事实仅采用提供receipt：exp-1024-134447-148581 Succeeded、1replica/1H800、12s、exit0/restart0、i-fengyicheng/codesign/non-preemptible、local NFS100.96.128.195；image2286e853…c94b06635与seal/submission/config/report/imageID一致。本代理未查询平台；trace的GPU0..15 metadata标签不是分配16GPU的证据。
证明边界：callback/DAG是producer实际运行观测，本审计独立检查其一致性，不等于保留了原生descriptor字节。image与CPU预检、77-case比较是提供receipt+本地哈希绑定，不是再次执行；worker Python3.12.3仅预检提供，实际run未报告版本。worker二进制重哈希、numerical accuracy、重复replay时独立clone ID稳定性、general native captured ownership及full V4 cost仍UNKNOWN。旧crash原因仍为缺indirection的推测，无直接crash证明；performance_acceptance与consumer_admission不升级。

## 2. 偏离检测
**结论：** 未偏移，正常推进。
**检测窗口：** 本次委派开始至最终只读guard｜**当前前沿：** 本代理已完成；root acceptance和独立review不属本代理。
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 一次性脚本仅覆盖请求的独立判定/负fixture/复现。 |
| D2 细节反复重试 | CLEAR | 单次fixture设计纠正，合法边界交换不制造错误，改为真实重叠。 |
| D3 支线问题修补 | CLEAR | 没有修改probe/native/platform。 |
| D4 可并行任务停滞 | CLEAR | 请求限定独立串行本地审计，禁止进一步委派。 |
| D5 超范围镀金 | CLEAR | 仅三个指定交付文件和136行一次性脚本。 |
| D6 分析空转 | CLEAR | 实际原始字段已产出独立判定与确定性artifact。 |
| D7 路线违背 | CLEAR | 保留UNKNOWN、不调用producer函数、不提升成本/ownership结论。 |
**文档检查：** S1–S6均未发现明显缺陷；旧source任务的root-only TODO不是本次审计未完成项。
**归因：** 无。

## 3. 纠正建议
**分支：** A：spec/tickets合理，无需改动。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. 本代理只读guard已通过 → 返回三文件与digest，然后停止；不扩展调查。
2. Root完成actual-data acceptance与独立review；不据此开放general native ownership/full V4 cost。

## 4. 最值得关注的 3 点
1. 五个目标kernel均通过完整callback ID直接匹配，四个event也有clone witness；这只证明该tiny diagnostic。
2. 31检查与六负fixture通过；两个独立新进程的完整结果digest一致，具体命令/完整脚本已留档。
3. 原始native descriptor/binary重哈希/numerical accuracy未提供，旧crash推测和native/full-cost UNKNOWN保持原样。
