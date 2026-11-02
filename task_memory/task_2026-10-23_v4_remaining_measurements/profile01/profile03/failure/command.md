# Local audit command and checkpoint

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Verified100 exact checks, reran once, and exercised a failing in-memory negative control. |

## 1. 规划完成情况
**汇总：** PASS3｜BLOCKED/FAILED/SKIPPED0｜TODO0（仅本地审计切片；GPU运行失败不变）
| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 固定输入、计数及身份 | PASS | 35本地文件哈希绑定；432行/144提交/137成功/7失败/288未提交；四主机32UUID。 |
| fatal链、profiler和transport记录 | PASS | timeline.json保留原始时钟、LF行号与UNKNOWN边界；timeline.md给出证据引用。 |
| 可执行一致性检查与复跑 | PASS | 最终脚本100检查PASS，复跑相同digest；negative control预期FAIL/exit1。 |
环境：当前worktree；`/usr/bin/python3`3.10.6；仅Python标准库，无conda、GPU初始化或runtime/package导入。
```bash
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_failure_timeline_audit.py --record
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_failure_timeline_audit.py
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_failure_timeline_audit.py --negative-control
```
第一条生成本人拥有的timeline.json；通常复核只运行第二条。预期/实测：前两条PASS/exit0、100checks/35inputs；第三条仅把派生warmup提交数16改成15，命中`exact_client_counts`，FAIL/exit1，不修改输入或timeline.json。
输入集合：28个selected_download文件、4个receipt绑定平台日志、receipt.json、selected_download.json及launcher日志。最终绑定SHA256：`30ab5c1de4fc87c2ea3ccf7ff64d22ed248d49fecf49807078960af2dd1744ad`。
首次及复跑canonical observation SHA256相同：`b5982e889b0fffa31f6e30f7801d4d96f2d966784e0efeafc59114c4337e2c6e`；timeline.json字节SHA256：`82406421d9c21ccda4b0b1191aef7c68d4bc058230a5519b2c23918ee14f1b5c`。
最终脚本SHA256：`a897e658295a79d955eae16a054dffa67fc1f0f83dd557f3ed2d80cc18cf4911`。初次记录发现Python文本读取将CR进度转换成换行；改用raw bytes按LF分行，最终引用与read/ripgrep一致，再记录和复跑。输入字节未改。
检查器FAIL表示绑定、精确观察或预期链条不一致；PASS不表示修复GPU故障，更不是GPU-hang reproduction。SDK终端进程exit code、三Stopped容器exit code、实际NET/IB选择和hang根因均未证实。

## 2. 偏离检测
**结论：** 未偏移，正常推进。**检测窗口：** 本切片启动至最终复跑｜**当前前沿：** 三份产物交回root。
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 仅建立请求要求的固定输入检查；记录、一次复跑、负控制后停止。 |
| D2 细节反复重试 | CLEAR | 一次LF行号读取纠正；无GPU重跑或参数试探。 |
| D3 支线问题修补 | CLEAR | 未修改runtime/package或修复missing-ip。 |
| D4 可并行任务停滞 | CLEAR | 本切片禁止subagents；完整数据导入与source loop由root独立负责。 |
| D5 超范围镀金 | CLEAR | 仅三份拥有文件与单个scratch脚本。 |
| D6 分析空转 | CLEAR | 每项检查落入可运行产物及证据引用。 |
| D7 路线违背 | CLEAR | 只报告观察，不推断跨主机先后或hang根因。 |
**文档检查：** S1/S2/S3/S4/S6未发现切片缺陷；S5：父plan.md:44及progress.md:44–46仍写Starting/launcher继续，与receipt的Failed/07:06:45Z冲突。父文档不在本人的写入授权内。
**归因：** 无执行偏离；父状态需root更新，不影响本地审计完成。

## 3. 纠正建议
**分支：** B：父状态文档存在过时记录，交root更新。
**文档更新（先执行）：**
1. root更新父plan.md:44、progress.md:44–46：保留旧Starting观察，追加Failed终态、exact counts和本产物引用；不改变任务范围或运行授权。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. 交回timeline.json/timeline.md/command.md及脚本路径，root独立检查并接受本切片。
2. 并行：root完整数据导入/质量核查 ‖ root实际source-backed反馈循环；不把本检查代替GPU故障重现。

## 4. 最值得关注的 3 点
1. 实际只提交repeat01：formal121成功/7失败；其余两轮288行均未提交。
2. P1TimeoutError与P0EngineDeadError同秒，marker传播可证；跨主机时间因果及根因不可证。
3. artifact一致性100PASS不等于运行成功；actual transport及三Stopped容器终态仍UNKNOWN。
