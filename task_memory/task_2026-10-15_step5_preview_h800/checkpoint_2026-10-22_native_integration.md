## 1. 规划完成情况

| Modification History | Change |
| --- | --- |
| 2026-10-22 | 集成145a2f5、P0 capture终态及主线自检。 |

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 12｜TODO 13

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 01 baseline | PASS | 已完成复用审计 |
| 02 reference | PASS | 已冻结 |
| 03 inventory | PASS | 已完成 |
| 04 rank-local | PASS | G1范围已完成 |
| 05 physical scopes | TODO | 完整准入仍缺 |
| 06 simulation scope | PASS | ticket06按D1已关闭该范围；原完整census仍按需 |
| 07 observer | BLOCKED | 原生验证未完成 |
| 08 G3 | TODO | reference/controls已实现，capture已Succeeded，原生判定进行中 |
| 09 runtime/clean | TODO | 四track基础请求完成；完整G4/clean仍缺 |
| 10 profiles | TODO | MRv2 producer已集成；H800完整画像未准入 |
| 11 routing | BLOCKED | 07依赖 |
| 12 communication | TODO | 已有质量失败及待判定产物 |
| 13 NIXL | BLOCKED | 08/12依赖 |
| 14 normalization | TODO | CPU comparator完成；原生准入未完成 |
| 15 model path | TODO | 结构完成；native reconciliation开放 |
| 16 measured compute | TODO | 已集成；native匹配与真实画像开放 |
| 17 measured PD | TODO | CPU生命周期已集成；G3及完整CLI开放 |
| 18 pressure/cancel | BLOCKED | 17依赖 |
| 19 graph/host | BLOCKED | 07/17验收依赖；现有CPU接口核对可先行 |
| 20 placement | TODO | 结构完成；native placement开放 |
| 21 routing/CC prediction | BLOCKED | 11/12/13/17依赖 |
| 22 bundle/replay | TODO | 四bundle已生成；完整CLI回执仍缺 |
| 23 freeze | TODO | common workload完成；各track runtime开放 |
| 24 diagnosis | SKIPPED | 并入28 |
| 25 fit freeze | SKIPPED | 并入28 |
| 26 old matrix | SKIPPED | D12退役 |
| 27 old H200 matrix | SKIPPED | D12退役 |
| 28 four-metric gate | TODO | CPU comparator完成；正式四项结果未产生 |
| 29 regression acceptance | BLOCKED | 28及共同HEAD验收依赖；集成回归已通过 |
| 30 support acceptance | BLOCKED | 05/29及必要原生证据 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。完整CLI的最新执行回执仍缺，不能把长期“进行中”当作实际推进。
**检测窗口：** checkpoint_2026-10-21_measured_merge.md至今｜**当前前沿：** 08/09原生判定、10/16/17/22 CPU接入、12既有画像判定

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | logger修复直接解除actual-main异常；685/465与77-case收口后停止加测，G3已实际提交并结束 |
| D2 细节反复重试 | CLEAR | D22已关闭；G3 reference失败隔离与MRv2修正均形成1171ab6/82f4452交付，而非重跑旧失败包 |
| D3 支线问题修补 | CLEAR | 修改归属08/09/10/16/17/22；p90补算是用户明确请求 |
| D4 可并行任务停滞 | NO DATA | G3/producer与CLI各有owner；前两项已交付，CLI最新完整结果未收到，不能证明它仍在执行 |
| D5 超范围镀金 | CLEAR | p90仅作请求的诊断；未增加正式指标或新运行时 |
| D6 分析空转 | CLEAR | 145a2f5集成；32H800 capture完成；实际descriptor消费端已分配接口核对 |
| D7 路线违背 | CLEAR | P0优先、32H200非抢占上限、30%解锁和10%最终门槛均保留 |

**文档检查：** S1：plan§4中11仍写complete06，与§5的06.simulation_scope不同。S5：plan§4的06/14/16/17/20/22状态未跟随各ticket；§6仍写旧P0 Running、最终fidelity待做及G3实现中。S2/S3/S4/S6未见可证实的新缺陷。
**归因：** 无执行HIT；S1/S5需修正，以免旧索引再次串行化CPU工作。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan§4：同步06.simulation_scope、11依赖和已交付CPU里程碑，保留native验收阻塞（可直接执行）。
2. plan§6、progress、公开状态及native_run：记录145a2f5、685/465PASS、77/77一致、capture Succeeded且原生判定未完成（可直接执行）。
3. requirements与metrics_check：记录用户要求的p90补算及四硬件/规模的8并发诊断，不改变正式四项口径（可直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 08/09：判定现有capture原始证据 → 完成标志：每个G3/G4条件都有PASS/FAIL/UNVERIFIED与缺口；从实际记录提取dummy descriptor，不再猜测。
2. 并行16/17/22：消费端接口核对后立即实现最小MRv2/combined-policy绑定，保留严格缺失错误；完整CLI由现有owner交回真实6144/1024、tail16、READ/release及metrics回执。
3. 10/12/22→09.clean→28：实际reached-shape输入准入后完成P0三次128请求的四项比较；64正式继续等待同硬件32四项<=30%。

## 4. 最值得关注的 3 点

1. 代码和回归已收口到145a2f5；新P0 native capture已Succeeded，下一步是消费它的证据与形状。
2. 正式128请求/QPS1运行尚未完成；刚报告的p90仅来自8并发基础样本。
3. 完整CLI最新回执与MRv2/combined-policy消费端仍是代码主线缺口；不得用更多CPU测试数代替这两项完成。
