# D29 next-session handoff prompt

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | YC deferred handoff;current-session progress now owns execution and this prompt is historical. |
| 2026-10-23 | Deliver reviewed code/records and exact continuation command after the bounded CLI repair. |

交接暂不执行：YC 已要求继续当前 session。请以 progress.md 的 Current-session
resumption 为准；下面是保留的交付时点提示，不代表任务仍在等待新 session。

原交接内容：

---

继续 Frontier Step D29。工作目录：
`/data/ycfeng/Frontier/.worktrees/step5-preview`，分支 `feat/step5-preview`。
先读本目录的 AGENTS.md，以及
`task_memory/task_2026-10-22_step5_position_routing/{requirements.md,plan.md,progress.md,summary.md}`。
当前文件就是交接入口；以 progress.md 的 Current frontier 为准，后面的旧状态是历史记录。

## 已完成

1. 三条实现流和源码修复已合并：代码 `4ed67d78`，T2 记录 `9c77b559`。
   `frontier/tests/data/Dockerfile` 与已复审的 `75ad26f9` 字节一致。
   submodule 已在当前工作树干净 checkout 到 `3dac53c544e546574cb6e8078cd14053d0916113`。
2. 最新 CLI placement 修复 `ae1fb721` 仅改现有 runner/test 两文件，40/65 gross。
   position 分支显式传既有全局参数
   `--collective_sim_cc_backend_config_placement_order TP,DP`。
   父端整文件 **57 PASS / 6 原有 SKIP**，独立复审 **No findings**。
   真实 command→config→Cluster/factory→CPU plan 在两模式、P/D 两角色均返回
   exact key/count receipt 和 0.03ms；未修改 backend gate 或数值行为。
3. 实测 111 allocated / 110 changed paths，native1350/1582、CORE3873/4404、
   PRICING3879/3956；source9102/9942、global9116/10000，所有逐文件额度通过。
   原 scope.json 保持不变，按顺序应用六个 overlay：registry、core_review、
   legacy_output、prepare_metadata、final_scope_reconciliation、cli_placement。
4. T2 记录已合并，63/63 checksums 通过，但仍明确绑定旧 `1be91bad`；
   新候选的 target staging/rebind/seal 尚未完成。

## 未完成与证据边界

- **完整 D29 workflow 未通过。下一步先跑修复后的完整 offline/online CLI。**
  已有 fixture03（7 CSV、76 cache、P/D bundles）可直接复用，避免重做昂贵准备。
- main05 在 `1be91bad` 上两模式均 exit1、没有超时，根因为缺 placement 参数。
  保留 `/data/ycfeng/tmp/d29-parent-position-main05`；不要覆盖或称它通过。
  scout 首次复现曾改写其中 offline 的
  `metrics/stepfun_ai_step_5_preview/offline_batch/step5_measured_cpu_main_offline/config.json`，
  该文件不得充当原始证据；command/log/execution receipts 仍可用。
- strict77 final03 **77/77**、full-unit parent03 **8015 PASS** 仅绑定 `1be91bad`。
  原始全量实际是 279 FAIL/32 ERROR/281 SKIP：历史267FAIL/31ERROR完全保留；
  新12 aiohttp依赖失败和1 Torch收集错误有同版本101PASS的Python3.12补验证据。
  不隐去 raw exit1、不把分段结果相加、不把旧结果改签成新 revision。
- 最终新源码的 required regression/full-unit comparison 和 T2 rebinding 仍需按计划收口。
- GPU/native profiling、G3、四指标 E2E 尚未验收。16 H800/两节点/30分钟/8GPUh
  只是未批准提案。不要提交 GPU job；待 workflow/package 准备好后向 YC 请求明确资源批准。
- 不新增 independent public consumer binder；不修复本次无关的 role-scoped placement
  override。该角色参数当前不生效，本次正确入口是全局 placement 参数。

## 下一步实际执行

先检查 `git status --short`，确认交付代码之后只有记录变更，记录新的冻结 HEAD。
`.cortexkit/` 与 `.pi/` 是未归属的 untracked 内容，保留、不 stage、不删除。
检查下面的新 output-root 未被使用；已存在则选新编号并记录，禁止覆盖旧证据。
保持 6144/1024、TP8/DP2/EP16、每模式2400s 等原判据。

```bash
timeout 5100s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 \
  PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/ycfeng/envs/frontier-py310/bin/python -B \
  tests/integration/run_step5_measured_main.py \
  --fixture-manifest /data/ycfeng/tmp/d29-parent-position-fixture03/fixture_manifest.json \
  --output-root /data/ycfeng/tmp/d29-parent-position-main06 --mode all \
  > /data/ycfeng/tmp/d29-parent-position-main06.log 2>&1
```

这是真实 CPU/synthetic-cost workflow，不是 native/GPU 性能验收；之前每模式仅初始化
就约26分钟。完成后读 summary.json、两模式 execution.json/validation.json、business
metrics/logs，不以 exit0 单独判定成功。若再失败，先建立秒级真实复现，定位后交回原
owner 在原额度内修复和复审，再冻结及重跑。不要放宽 gate/timeout 或用缩短 workload
代替正式验收。

其后按 plan.md 完成新 revision 的 required regressions/baseline reconciliation，
重绑并封存 T2：26 完整源码文件、63 checksum、2907 source/12 backend/53 pins。
worker proposal 中必须清除继承的 FRONTIER_STEP5_PROFILE_MANIFEST；保留 engine limits
7168/8192/128 与 published image f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee。
仅在上述 workflow/package 条件完成后进入 GPU 资源决策。

## 精确证据入口

本 task 下：
- `final_parent/{cli_placement_acceptance.json,review_2026-10-23_cli_placement.md,handoff_source_receipt.json,test_report_2026-10-23_handoff.md}`。
- `cli_placement_repair/{placement_repair.diff,budget.json,test_report_2026-10-23_cli_placement.md}`。
- `final_parent/{fidelity_final03_audit.json,unit_final03_comparison.json,t2_final_target_receipt.json}`。
  旧 suites/环境补验的精确命令与 baseline IDs 均在 final_parent 的原报告中。
- T2 packet：`../task_2026-10-22_step5_d29_native_balanced/t2_preparation/`。

当前委派任务均已完成，没有为交接留下运行中的完整 CLI 或 GPU 作业。
实现 owner 可复用 `ses_e9b3efeebffem2dcbXlVGNr8O2`（CORE）、
`ses_e9b3e99a6ffe3r2973GDLkFpEm`（PRICING）；它们只是已完成任务的会话标识。
遵守当前 AGENTS/owner quotas；类似必要修复路径调整已有 YC 授权，但不包含新增 GPU
额度或降低验收标准。最后提交代码和记录，继续保留所有未完成项。
