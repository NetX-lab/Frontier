# Prepared point repair: parent acceptance archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive accepted prepared component, existing parent tests/wave, separate review axes, raw-label limits and recomputed four-path retained accounting; no test/probe rerun. |

Task ID: **D29-PREPARED-PARENT-ARCHIVE**. Primary has accepted this CPU component
after independent Standards/Spec reviews and its own verification. This task only
archives that decision and inspects existing records/Git objects; it is not a new
test run, source review or final D29 workflow admission.

## Fixed source and record provenance

- Base: `caffccbb48011f0140359d391ad75b58906ff15b`.
- Code: `0be11762d0f83e9db77658a1447a672507166872`.
- Owner records / parent-executed tree: `b476907e9afe865476e0a3a6720f8eb4e3290c40`,
  `/data/ycfeng/Frontier/.worktrees/step5-d29-prepared-point`.
- Integrated tree: `eece87b2378bf86c6673b74f1a769f566bc5792a`.
- Records-only worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-prepared-archive`.
- Native gitlink: `3dac53c544e546574cb6e8078cd14053d0916113`, unchanged. This archive
  neither initializes nor checks out the submodule and needs no native dependency.

Read parent `requirements.md`, `plan.md`, `repairs/prepared_point_scope_amendment.json`
and `prepared_point_repair/{requirements.md,plan.md,progress.md,budget.json,
source_receipt.json,test_report_2026-10-23_prepared_point_reuse.md}`. All paths are
under `task_memory/task_2026-10-22_step5_position_routing/`. The actual component
contract is `prepared_point_repair/interface_receipt.md`; that component directory
has no `design.md`. The complete owner diff is
`prepared_point_repair/prepared_point_repair.diff`.

Separate review results are appended to `resumed_performance/review.md`.
`resumed_performance/prepared_parent_acceptance.json` retains exact artifact
paths/SHA256 values, raw statuses, source binding, derived accounting and limits.

## Parent environment and exact executed commands

Existing parent execution: `kun-workspace-vgen2`, CPU-only CPython3.10.6 at
`/data/ycfeng/envs/frontier-py310/bin/python`; cwd
`/data/ycfeng/Frontier/.worktrees/step5-d29-prepared-point`. No new environment or
conda activation is claimed. `sudo -n systemd-run --scope -p MemoryMax=30G`
executes as `brainpp`; bounds are240s for affected tests and60s for the trainer
selection/wave/comparison. Scratch/cache is under `/data/ycfeng/tmp`. Both pytest
commands unset the two external CPU-writer overrides, disable bytecode/cacheprovider,
set `PYTHONPATH` and scratch variables, and use one OpenMP/OpenBLAS/MKL thread.
Commands below are archived parent commands, **not executed by this archive**.

Six affected files:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 240s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_moe_position_routing.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_measured_ffn.py --basetemp=/data/ycfeng/tmp/d29_prepared_point_parent01_pytest --junitxml=/data/ycfeng/tmp/d29_prepared_point_parent01.xml > /data/ycfeng/tmp/d29_prepared_point_parent01.log 2>&1
```

Separate both-trainer/preflight and independent selected actual fit/cache checks:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_grouped_gemm_padded_features.py::test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers tests/unit/test_moe_grouped_gemm_padded_features.py::test_independent_selected_actual_fit_cache_binds_native_context --basetemp=/data/ycfeng/tmp/d29_prepared_point_parent_trainers01 --junitxml=/data/ycfeng/tmp/d29_prepared_point_parent_trainers01.xml > /data/ycfeng/tmp/d29_prepared_point_parent_trainers01.log 2>&1
```

Retained actual layer60 wave and captured-source comparison:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_prepared_point_repair/candidate_wave.py --output /data/ycfeng/tmp/d29_prepared_point_repair/parent-layer60-01 --captured-source /data/ycfeng/tmp/d29_main08_residual_repro/parent-layer60-source.json > /data/ycfeng/tmp/d29_prepared_point_repair/parent-layer60-01.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_prepared_point_repair/candidate_compare.py --replay parent-layer60-01 --layer 60 > /data/ycfeng/tmp/d29_prepared_point_repair/parent-layer60-01-comparison.log 2>&1
```

## Observed results and passing conditions

The archive reparsed the existing XML/JSON and hashed artifacts; it did not rerun
pytest, the wave, comparison, matrix or source/error checks.

| Evidence | Expected condition | Raw observed result | Disposition |
| --- | --- | --- | --- |
| Parent affected six files | No failure/error; retain existing Torch skip |83 cases:82 PASS,1 SKIP,0 FAIL,0 ERROR; XML132.253s/log132.25s |PASS with existing skip |
| Parent separate trainers | Both selected checks pass using current-repository CPU writer |2 PASS,0 FAIL/ERROR/SKIP; XML6.075s/log6.07s |PASS, kept separate from82 |
| Parent structural preparation | One row/key preparation, all current reads retained |One16464-row/168-key preparation; cold3258/warm1968 JSON decodes;80 reads per wave |PASS for allocated seam |
| Parent full layer60 payload | Both own-layer payloads match original |Both `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83` |PASS |
| Parent captured comparison | Every captured/replayed pair checked, no mismatch |208 captured/208 replay/208 checked,0 mismatch |PASS for complete captured wave |
| Owner matrix, not parent rerun | Original-query differential at disclosed fixture seam |36 positive/24 negative records in `matrix-final01/matrix_result.json` |Owner evidence only |
| Owner historical RED and final focused GREEN | Real seam initially fails then passes |RED1 FAIL/2 PASS; final GREEN3 PASS |Historical owner evidence, not parent execution |
| Source byte/retained scope inspection |Only allocated paths/caps; integration matches reviewed bytes |Three prepared files match0be; trace matches390;159/250 step;197/323,299/299,75/78 retained |PASS for this differential |

The existing skip is
`tests.unit.test_moe_position_routing::test_one_source_tensor_interface_preserves_masks_and_request_intervals`:
`could not import 'torch': No module named 'torch'`. It is not a test pass. Do not
sum82+2 or relabel the six-file selection as independent-trainer coverage. Owner
final affected82 PASS/1 SKIP and separate2 PASS are different raw executions in
the owner report, not the parent artifacts above. Primary's earlier prototype
matrix is also separate from the owner's committed36/24 matrix.

| Parent wave | Pricing seconds | Emit seconds | Flush seconds | Wave wall seconds | Events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold |1.7261437461711466 |0.16251190891489387 |0.13113739527761936 |2.01979305036366 |208 |
| Warm |1.0362420910969377 |0.05939718196168542 |0.13097241567447782 |1.226611688733101 |208 |

These are observed local timings, not a fixed speed promise or whole-workflow
prediction. The parent wave ran sourceb476 with query SHA256
`d9423f3f318cd355f2773ea6a65a9bf9d297cdc3ec7c8c46230b38456ccacbfc`, clean recorded
source status, `diagnostic_variable=none` and diagnostic preparation off. The
wrapper observes actual production preparation calls; it does not provide an
alternate preparation implementation. Both calibration fixture overrides are
removed by the retained actual-wave setup; selected CPU-fixture measurements
remain CPU-fixture evidence, not GPU ground truth.

**Raw descriptive-label limitation:**
`parent-layer60-01/result.json.scope` still says actual DECODE layer61 and
`parent-layer60-01-actual_comparison.json.scope` still says
`8MiB actual tail, subset of layer61 operators; no full op trace census`.
Both are inherited literals, not the executed filter. `candidate_compare.py:13`
rebinds `global_layer_ids==[61]` to the requested `--layer 60`;
captured-source geometry and `result.layer` are60. The retained417 JSONL records
contain416 events across two waves,208 unique EP/operator pairs, all at layer60.
The parent comparison checked208 pairs and retained a proof hash per pair with
zero mismatches. Preserve these raw artifacts and their hashes unchanged; the
receipt records `requested_layer=60` and accurate derived scope separately.

The actual comparison checks each `duration_ms` exactly and all five complete
proof fields: `collective_plan`, `collective_cost`, `operator_cost`,
`native_context`, `routing_source`. Those fields retain count/source/plan facts;
this is source-replay parity, not measured GPU accuracy. Timing errors against
GPU ground truth are not available and are not inferred.

**No integrated-union timing:** the executed prepared-point tree does not contain
trace390. Its trace SHA is
`f66f9df95422ad734707e723a38a516143e90fe80274c80dd169babf9a375939`, identical to79,
not the integrated trace SHA
`07d7cbbb081829f558b6a4c101201baf112e5387e2045ee244436853fb463a0d`.
The archive tree contains both reviewed changes, but no timing run was performed
there. Do not combine approximately0.131s prepared-wave flush with separate
trace speedups or claim the resulting union was measured.

## Git-derived retained accounting

Archive inspection uses CPython3.10.6 stdlib only and Git2.34.1. No Frontier module,
test, native package, network/platform/GPU operation or new agent is invoked.
The worktree was created with:

```bash
git worktree add -b docs/step5-d29-prepared-archive /data/ycfeng/Frontier/.worktrees/step5-d29-prepared-archive eece87b2378bf86c6673b74f1a769f566bc5792a
```

From the archive cwd, inspect the step, baseline-relative retained diff and only
changed production paths with:

```bash
git diff --numstat caffccbb48011f0140359d391ad75b58906ff15b 0be11762d0f83e9db77658a1447a672507166872 -- frontier tests data Dockerfile
git diff --numstat 1aee0af11cd58691e64603b99eca0ac459a5b350 eece87b2378bf86c6673b74f1a769f566bc5792a -- frontier/execution_time_predictor/moe_routing_workload.py tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_routing_runtime.py frontier/metrics/trace_store.py
git diff --name-only 79f1a3fd35d42b751cd1520c764ddac717b923d9 eece87b2378bf86c6673b74f1a769f566bc5792a -- frontier tests data Dockerfile
git ls-tree eece87b2378bf86c6673b74f1a769f566bc5792a -- frontier/cc_backend/backends/collective-sim
```

| Path | Prepared step additions/deletions/gross | Step cap |79 retained gross |Integrated retained additions/deletions/gross |Retained cap |Retained increase |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `frontier/execution_time_predictor/moe_routing_workload.py` |73/19/92 |180 |143 |191/6/197 |323 |54 |
| `tests/unit/test_prediction_measurement_bindings.py` |45/0/45 |45 |254 |299/0/299 |299 |45 |
| `tests/unit/test_moe_routing_runtime.py` |22/0/22 |25 |53 |75/0/75 |78 |22 |
| `frontier/metrics/trace_store.py` |Separate trace step |20 |0 |7/7/14 |20 |14 |

Prepared step is+140/-19=159 gross, not retained consumption159. The three
baseline-relative retained increases total121; add trace14 to obtain135.
The three recomputed79 rows match audit06. Git byte comparisons establish that
all three integrated prepared paths equal0be and integrated trace equals390.
Only these four paths changed under `frontier/tests/data/Dockerfile` over79;
all18 donor paths in the two amendments have zero change. The gitlink and
`docs/general/module-size-exceptions.md` bytes are unchanged. No donor or source
was edited by this archive.

Prior accepted audit06 records CORE3970/native1350/PRICING3920=source9240,
plus12 module-size-document lines and2 gitlink lines=global9254. Replacing the
four affected retained rows yields **CORE4105, source9375, global9389**, not
prior totals plus159+14. CORE4404/native1582/PRICING3956, source9942/global10000
ceilings remain unchanged. Untouched native/PRICING contributions are carried
from the accepted audit, not reaudited via a submodule checkout. This is bounded
four-path differential accounting; the final112-path union audit remains pending.

Archive-only checker correction: the first inline document-preservation check
raised `AssertionError` because it stripped the baseline's final newline but
not the appended document's prefix. Inspection showed both prefixes were7587
bytes and exactly equal, including that newline. The corrected equality check
passes; no review text, raw evidence or source was changed to satisfy it. This
was a document-checker error, not a parent pytest/probe failure.

Final records-only validation:19 recorded artifact SHA256 values and4 source
receipt hashes match; raw XML counts, JSON scope/arithmetic and unchanged
historical review-prefix checks PASS. No active executable commit hook exists.
`git diff --check` passes. These checks validate the archive, not another run of
the already-successful parent test/probe scenarios.

## Acceptance and remaining limits

Observed: existing raw test statuses/timings/counts/proof-check records and hashes;
Git source equality/diffs. Derived: complete captured layer60 scope and retained
totals from those records plus accepted audit06. Reported by primary: independent
reviews and component acceptance. Owner matrix, historical RED/GREEN and earlier
prototype results retain their own provenance.

No new complete CLI, fresh strict77, full-unit, final-source T2 seal, native
runtime, GPU, G3 or E2E pass is claimed. Prior79 gates are historical, not new-union
qualification. Full6144/1024 workload/2400s limits and original G3/four separate
E2E error<=10% gates remain. No numerical/whole-proof cache, public interface,
independent public-consumer mode or nested in-place receipt mutation support is
added. Main plan/progress/summary and next union review/freeze remain root-owned.

<details>
<summary>Records-only phase checkpoint (check-progress)</summary>

## 1. 规划完成情况

**汇总：** PASS 2｜BLOCKED/FAILED/SKIPPED 0｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 固定提交与证据读取 | PASS |eece worktree；现有 XML、JSON、Git 差异已读取 |
| 三份验收记录 | PASS |本报告、receipt、review 分轴追加已落地 |
| 文档检查与仅三文件提交 | TODO |提交前待完成记录一致性与范围检查 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** D29-PREPARED-PARENT-ARCHIVE 委派至本记录｜**当前前沿：** 文档范围检查与提交

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR |仅读取已有证据；未增加测试工具 |
| D2 细节反复重试 | CLEAR |未重跑成功检查或重复修复 |
| D3 支线问题修补 | CLEAR |过时标签只在授权记录披露；未修改脚本 |
| D4 可并行任务停滞 | CLEAR |唯一分配范围是三份记录；禁止新 agents |
| D5 超范围镀金 | CLEAR |不修改源码、主计划或其他记录 |
| D6 分析空转 | CLEAR |差异与证据读取直接形成验收记录 |
| D7 路线违背 | CLEAR |固定集成提交、原始证据与验收界限均保留 |

**文档检查：** S1–S4、S6 未发现明显缺陷；S5：冻结的父级 `plan.md:103–112` 仍标 prepared review/accounting 为 pending，晚于它的父级明确验收与 eece 集成已完成本组件；raw scope 两处仍为 layer61。原记录不重写，本报告明确当前范围与 provenance。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：spec/tickets 存在过时状态，先记录当前验收再继续。

**文档更新（先执行）：**
1. 本报告、`prepared_parent_acceptance.json`、`review.md`：追加最新组件验收、实际 layer60 范围与差异账目（可直接执行，已完成）。
2. 父级 `plan.md/progress.md/summary.md`：由 root 在整合时更新 prepared 已接受状态；保留全局 gates pending（root 所有，本任务不修改）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 本任务：校验记录 JSON、原始证据 hashes、仅三份文件变更并提交 → 完成标志：独立 commit 与干净 worktree。
2. Root：整合记录并继续 final union accounting/review/freeze → 完成标志：112-path 审计及下一冻结点；此处不把后续测试标 PASS。

## 4. 最值得关注的 3 点

1. 82 PASS／1 SKIP 与2 PASS 是独立选择；不相加，不升级 coverage。
2. 原始 layer61 标签保留；实际执行范围为 layer60 的完整208对，非 GPU ground truth。
3. 合并保留增量135、global9389；无 union timing 或全流程验收，最终审计仍待 root。

</details>
