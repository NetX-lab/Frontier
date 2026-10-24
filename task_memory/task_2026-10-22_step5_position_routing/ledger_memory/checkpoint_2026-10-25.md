## 1. 规划完成情况

### Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Check the implementation frontier and stop broad reruns while caller/setup prerequisites are open. |
| 2026-10-25 | Apply the route correction; commit component code and retain the exact2-gross caller blocker. |

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 2｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 基线和 allocation | PASS | 0ca54025；原套件87PASS/6SKIP |
| 两个真实 RED controls | PASS | red2：真实 row retention 和实际 EP read_text guard |
| write/read repair 和字节 controls | PASS | 6c2f179c；48PASS/3SKIP 控制，不含完整 caller admission |
| 完整 focused integration | FAILED | pinned-import-only：62PASS/4FAIL/6SKIP，checkout resolver 缺失 |
| exact local accounting 和证据归档 | PASS | 四文件120gross；39原始证据；不等于独立 composed acceptance |
| 原有 performance snapshot | BLOCKED | 开放写文件无法 pickle；最小 cap12→14；未应用 proposal |

## 2. 偏离检测

**结论：** 已偏离：命中 D1。
**检测窗口：** 完成 RED 至今｜**当前前沿：** focused evidence、caller allocation、提交。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | HIT | 已知四个 setup 失败后仍重跑完整52–56秒 integration；应只跑失败四项 |
| D2 细节反复重试 | CLEAR | 每次调整对应已记录的 globals/import/checkout 原因，没有扩大 timeout |
| D3 支线问题修补 | CLEAR | 只触及所需 ledger、validator、caller 和证据 |
| D4 可并行任务停滞 | CLEAR | 独立 baseline/candidate 和 EP 检查并行；用户禁止继续委派 |
| D5 超范围镀金 | CLEAR | 无 flags/API/proxy/cache 或数值行为扩展 |
| D6 分析空转 | CLEAR | 已交付真实 RED/GREEN 和四文件修改；新 snapshot 问题已定位 |
| D7 路线违背 | CLEAR | 未扩大 pricing/CLI/GPU/shape 运行；未越过任何代码 cap |

**文档检查：** S2 plan.md 未单列原有 save_state 对 MetricsStore 的依赖；其不能 pickle 开放 writer。S1、S3–S6 未发现明显缺陷。
**归因：** D1 执行偏离。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. ledger_memory/plan.md：归属原有 performance snapshot blocker，并记录提交前所需 root 零和重分配（可直接执行；不改 cap）。

**需停止的偏离行为：**
- D1：停止完整 integration 重跑；checkout 未就绪时只留 raw FAIL，不改 donor/source 或隐藏构造错误。

**接下去的路线：**
1. 小型 caller/字节/report control → 保留真实结果和完整命令，不运行 pricing wave。
2. 精确 cap 和 snapshot 最小 delta → root 明确分配后才改相关代码。
3. 独立可完成的 write/read/negative controls 和 scoped commits；最终 audit/review/freeze 由 primary 完成。

**纠正执行：** 已停止完整 integration 重跑；随后仅执行小型 caller/static accounting 和证据归档。已向 primary 返回 metrics38→36、performance12→14 的零和提案，当前 cap 未改。

## 4. 最值得关注的 3 点

1. 真实 retention 与实际 validator materialization 均已观察 RED/GREEN；不等于 native/RSS/runtime admission。
2. save_state 是真实 caller blocker，不能用新的 list proxy 或静默丢失状态解决。
3. 旧基线与当前 raw FAIL/SKIP 都必须保留；独立 clean audit/full regressions 仍未执行。
