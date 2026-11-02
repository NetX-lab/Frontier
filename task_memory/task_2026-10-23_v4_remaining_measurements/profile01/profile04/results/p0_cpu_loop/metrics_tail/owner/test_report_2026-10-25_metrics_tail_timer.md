# V4-P0-METRICS-TAIL-TIMER owner report

Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Delivered the timer and the single allocated tiny control; Hong Kong local date. |

状态：owner 源码和证据已交付；63 checks PASS。这不代表 primary review 或完整 P0 diagnostic 已完成。
依据：[allocation](../primary/allocation.json) 和 requirements.md 的 D-V4-P0-AUDIT-METRICS-TAIL-DIAGNOSIS；primary 保留执行及验收责任。
源码：tests/analysis/v4_trace_audit_metrics_timing.py，98 gross lines，SHA256 `fdce5eca10e0cea28af12b87d25a94ed68f0ce0de2eda77c6b5fa2ccb5b541b5`。

精确命令（cwd `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`）：

```bash
timeout 60s /usr/bin/python3 -B task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/metrics_tail/owner/static_compile.py
timeout 60s /usr/bin/python3 -B task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/metrics_tail/owner/tiny_control.py
```

环境：`/usr/bin/python3`，Python 3.10.6 / GCC 11.3.0；仅 standard library，`-B`，未使用 conda。静态编译先完成，用时 0.005290464s；唯一 tiny control 内部计时 0.290513119s，exit 0，timeout 60s。

| 检查 | 预期 | 实际 / 结果 |
| --- | --- | --- |
| 静态编译及六项原始 source pins | 无源码执行；全部匹配 | PASS；adapter SHA256 `6dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a` |
| stripped function / complete generated AST | 完全等于 accepted AST；其他定义不变 | PASS；去掉 5 个外层 observations、1 个 nested wrapper，恢复 1 个原 sorted call |
| pipeline AST 和 composition | timer 在 accepted composition 之后；去掉插入后恢复原 pipeline | PASS；stage._obs / stage.run identities 保持 |
| empty、ties、repeated membership、original-order floats | 所有 count/sum/union 和 float hex 精确相同 | 五组 fixtures 全部 PASS；group/event objects 及顺序保持 |
| sparse boundaries / sort / counts | 八个边界；每组一次原 sorted；group-length totals | PASS；ties `[1,2,0,3]`；repeated fixture 9 groups / 31 memberships |
| float-order negative witness | 顺序改变会被精确比较发现 | PASS；原 sum `0x1.3a92a30553262p-11`，改序 `0x1.3a92a30553261p-11` |

[static_compile_result.json](static_compile_result.json) 和 [tiny_control_result.json](tiny_control_result.json) 保留命令、环境、全部原始 hashes、预期/实际数值和 PASS/FAIL；[ast_closure.json](ast_closure.json) 保留完整 accepted/transformed/stripped function、module、pipeline AST 及恢复的 sorted expression。控制源码分别为 [static_compile.py](static_compile.py) / [tiny_control.py](tiny_control.py)。
新增 observations 使用 `DEBUG-V4-METRICS-TAIL-01`；group construction 单独覆盖原 count/sum、states 和 membership 构造。wrapper 接收原 sorted binding 及原参数，恰好委托一次；union recurrence 原表达式保持。
限制：batch_start 在调用参数与 groups dictionary 构造之后；其前的调用方成本只体现为 kernel_pass_done 到 batch_start 的间隔。原 _obs 时钟舍入至毫秒并有日志开销；tiny control 不证明真实 P0 分段耗时或 non-perturbation。
额度：tiny control 已用 1/1，零 retries；owner 完整 trace / pipeline / affected / GPU 执行均为零。未写 shared task records，未 stage/commit；原失败保留。

1. Primary 审阅冻结源码、AST 和已有 tiny receipt；不重跑已耗尽的 tiny control。
2. Primary 按 fresh allocation 执行唯一完整 P0 timing diagnostic，结合原 GC / stage logs 记录完成或 censored segments；此交付不关闭 R5 或 ownership/dependency/cost/reference gates。
