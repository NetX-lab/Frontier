# D29 worker readiness and parent verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | 补录提交前五树/source/index只读复核；不重跑父端、记录checker或任何pricing/worker控制。 |
| 2026-10-24 UTC / 2026-10-25 HKT | 归档唯一worker终态、完整参考字节读写和父端验证；保留Pod403与完整workflow未通过的边界。 |

唯一job `exp-1025-050336-389487` 和父端验证均通过。本次结果覆盖worker CPU imports、完整serialized-reference append/read-back/cleanup、实际mount/source与launcher/container生命周期。D29完整workflow仍未验收。本次新增pricing、required-shape、fullCLI、native和GPU allocation均为0。

生产source `2ca5e57113b0bc99ab84e926e5bb492b3670072e`。Worker前后及父端records HEAD均为`dfe603edab128d2a4aacb1b7b64d898993d4ae09`；非task_memory文件等于生产2ca和冻结146。原116 paths/126 bindings审计和mandatory regressions已完成，本次不重跑。沿用CORE4377/4404、PRICING3955/3956、native1350/1582、source9682/9942、global9696/10000；本次没有重新composed accounting。

## 精确worker命令与环境

Submission host为`kun-workspace-vgen2`，cwd `/data/ycfeng/Frontier/.worktrees/step5-preview`。唯一原launcher命令：

```bash
bash /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/run_worker.sh
```

`launcher/execution.json`保存准确UTC时间/资源/image/source。2026-10-24T21:03:32.698151UTC开始，21:06:12.324292UTC结束；wrapper exit0、wall159.625913s，内部launcher158.872791s。Session87306已收集。原本地personal Python RJobBackend/spawn_tasks保持存活至终态，attempt1、retry0、无outer timeout。原backend atexit Job Stop在Succeeded后记录；未执行手工stop或重提，空stop annotations不单独排除外部actor。

实际requests/limits为4CPU/0GPU/64000Mi、1replica，creator=i-fengyicheng、codesign。Image `hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee`；本机NFS `100.96.128.195:/data/ycfeng:/data/ycfeng`。Worker `gpu-h800-0326.host.platform.shaipower.com` 的mountinfo确认该NFS和JuiceFS。Task cloud访问仅`/mnt/codesign-exp/ycfeng`；mandatory tenant mount未被业务读写。

Worker Python `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python`，3.10.6，-B，OMP/OPENBLAS/MKL/NUMEXPR均1，TMPDIR=/data/ycfeng/tmp，copied base prefix `/data/ycfeng/envs/frontier-runtime-base-py310-d29-20261025`。Package versions：`{"ddsketch": "3.0.1", "fasteners": "0.20", "numpy": "2.2.6", "pandas": "2.3.3", "plotly": "7.1.0", "pytest": "9.1.1", "scikit-learn": "1.7.2", "scipy": "1.15.3"}`；module origins和原numerical checks通过。两个child均配置32,212,254,720B address-space limit且未timeout。这些配置和平台64000Mi不建立全workload内存资格。

原child精确argv：

```json
{
  "runtime_probe": [
    "/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python",
    "-B",
    "/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/python_runtime_probe.py",
    "/data/ycfeng/envs/frontier-runtime-py310-d29-20261025"
  ],
  "append": [
    "/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python",
    "-B",
    "/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/worker_append.py",
    "--plan",
    "/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/worker_plan.json",
    "--receipt",
    "/data/ycfeng/tmp/d29_worker_readiness_20261025_01/worker/append.json"
  ]
}
```

完整execution receipts在`business/worker/runtime_probe.execution.json`和`business/worker/append.execution.json`，相对本报告的`evidence/worker_readiness_2026-10-25/`。

## 期望值与实际结果

| 检查 | Passing condition | 实际值 | 结果与边界 |
| --- | --- | --- | --- |
| RJob/replica/container终态 | Succeeded、container exit0/restart0 | RJob和replica Succeeded；container Completed/exit0/restart0；container finished21:06:06UTC，RJob finish21:06:08UTC | PASS |
| Creator/resource/image/NFS | 与reviewed plan相同 | i-fengyicheng/codesign/4CPU/0GPU/64000Mi/1replica/f037d958/本机NFS | PASS |
| Source/scripts/input | before/after/current identity一致 | 25source、6inputs、262metadata、9reviewed scripts/Git assets；2ca/146源码一致 | PASS；不是新composed audit |
| 原runtime probe | 原版本、origins、numerical严格条件 | PASS_CPU_IMPORTS_ONLY；2.771702s；RSS176,275,456B | PASS；仅imports |
| 完整append | ≤1GiB；完整wave、行值和顺序 |206完整waves、1,071,031,904B、3296ledger/42848operator rows、5000-event阈值 | PASS；reference only |
| Append/close/fsync | 真实字节与wall可重算 | append+close3.249062s；fsync0.038647s；329.64MB/s；含fsync325.77MB/s | PASS；含reference replay/digest，排除production pricing/emitter serialization |
| 两文件完整read-back | counts和全文件SHA一致 |1.325742s；两份完整digest匹配 | PASS |
| Append内存 | 只报告该进程实际peak |29,085,696B | OBSERVED；仅append child |
| 清理完成测试文件 | 完整校验后只删两份自身文件 |PASS_TWO_COMPLETED_TEST_FILES_REMOVED；空测试目录也删除 | PASS；保留active inputs和local evidence |
| 直接PodAPI读取 | 获得准确Pod对象 |403Forbidden原body保留；replica CR已有container终态 | FAIL_POD_READ_PRESERVED；不改权限/重提 |
| GPU runtime | 本轮未请求该检查 |nvidia-smi不可用，GPU请求0 | NOT_APPLICABLE；hostname不证明GPU runtime |
| 全workload/2400s | 完整6144/1024 main和输出/内存 | 本轮未运行 | UNVERIFIED |

| 已完成并删除的个人云测试文件 | 实际bytes | 完整read-back SHA256 |
| --- | ---: | --- |
| /mnt/codesign-exp/ycfeng/frontier_d29/worker_readiness_20261025_01/ledger.jsonl | 475762974 | 6af08420908ebfe79f7258e9dabe088023395662c1fde0e550034225e37a63d0 |
| /mnt/codesign-exp/ycfeng/frontier_d29/worker_readiness_20261025_01/operators.jsonl | 595268930 | 56f202c245427093750932be27d97033482915eb93651c2af3c71e8d3c31607b |

父端仅从保留完整参考字节重算累计digest，没有重新append、读回已删云文件、运行runtime或调用平台。469GB/mode是历史样本投影；195.4167MB/s来自469GB/2400s，不是测得完整工作量/RSS，也不是本测试的通过阈值。

## 父端准确命令、归档与结果

原script `/data/ycfeng/tmp/d29_worker_readiness_parent_20261025_01/parent_verification.py`；相同字节已归档为`evidence/worker_readiness_2026-10-25/parent_verification.py`，SHA256 `34f3ced2c0c73ca65a22bee65b7d4dfa05a5e13070905675888e906a34345f0b`。

```json
[
  "timeout",
  "60s",
  "sudo",
  "-n",
  "systemd-run",
  "--scope",
  "-p",
  "MemoryMax=30G",
  "runuser",
  "-u",
  "brainpp",
  "--",
  "env",
  "OMP_NUM_THREADS=1",
  "OPENBLAS_NUM_THREADS=1",
  "MKL_NUM_THREADS=1",
  "NUMEXPR_NUM_THREADS=1",
  "PYTHONDONTWRITEBYTECODE=1",
  "TMPDIR=/data/ycfeng/tmp",
  "/data/ycfeng/envs/frontier-py310/bin/python",
  "-B",
  "/data/ycfeng/tmp/d29_worker_readiness_parent_20261025_01/parent_verification.py"
]
```

cwd仍为当前工作树。Start `2026-10-24T21:18:37.209208+00:00`，end `2026-10-24T21:18:38.632489+00:00`，wall `1.423269422s`，exit0。Python3.10.6、-B、单线程math，60s/sudo-n/systemd-run/MemoryMax30G/runuser-brainpp；scope成功启动，effective cgroup peak未测。

Passing condition：原唯一attempt和plan/source/scripts与实际creator/mount/resources/runtime/完整reference字节/终态一致，五个当前工作树已整合；归档字节相同，selected evidence没有已知personal keys或URL credentials。

实际`PASS_SELECTED_RECEIPTS_SOURCE_AND_ARCHIVE_ONLY`。16份screened business/platform/log artifacts为21,528B；另有parent script/receipt和3份exact command capture为1,528B。Manifests记录准确path/bytes/SHA。Raw launcher logs仍在`/data/ycfeng/tmp/d29_worker_readiness_launcher_20261025_01/`受限目录，只有screened `.redacted` log副本归档。原preparation/failure/hash-bound worker_plan.json/joint_bindings.json保持原字节。第一次文档更新的表头定位错误保留在`record_update_first_failure.txt`；随后限定Modification History并继续，无runtime或verification重跑。

## 当前工作树与剩余验收

| 当前工作树 | 核验HEAD/branch | 状态 |
| --- | --- | --- |
| step5-preview |dfe603ed / feat/step5-preview | 唯一执行主线；生产2ca/146不变；本报告属于后续records commit |
| step5-d29-ledger-stream |1461573d / fix/d29-ledger-stream | 干净，已整合 |
| d29-ledger-final-1f6b4314 |1461573d / detached | 干净，保留mandatory验证冻结树 |
| d29-snapshot-copy-finish |b061f884 / d29-snapshot-copy-finish | 干净，已整合NO_ALLOC；0controls/launches |
| d29-init-preparation-finish |6335f581 / d29-init-preparation-finish | 干净，已整合ALLOC建议；0controls/allocated lines |

四个satellite均为当前主线祖先，不需要再次merge。原89行census与旧checkpoint保持；最新五树结果追加`resumed_execution/worktrees.json`。

`check_progress_2026-10-25_worker_readiness.md`识别S5当前worker状态过期并按原scope直接更新；没有D1-D7执行偏离。Mandatory strict77/full-unit/supplements已完成，本次不重跑。原full-unit raw8028PASS/279FAIL/32ERROR/281SKIP、exit1与历史presenceFAIL保留。

后续：两次普通D1/layer60 cold/warm joint pricing有限决定（目前Not Approved）→同源pricing/emission/flush和完整proof/key/order/parity→完整6144/1024 offline/online CLI，每模式frontier.main≤2400s，postvalidation在其外，测实际workload RSS→freshT2/source/package/native/G3/E2E。Required-shape6/6已耗尽，owner/source/global预算与E2E≤10%原门槛保持。

## 记录与归档的最终检查

首次记录检查 exit1、wall0.098173546s，因 execution_plan 的完成段引用报告但未写入具体 job ID 而失败。原脚本、准确命令、stdout/stderr和execution receipt均保留；补写已确认的同一 job ID，不改检查判据。

同一检查代码复跑通过，exit0、wall0.127923046s，开始2026-10-24T21:30:39.679997+00:00、结束2026-10-24T21:30:39.807932+00:00。精确argv在 records_verification_followup.execution.json，script SHA256为ef10ede733d10b4eecd2c4280abf50617ce63bdc1c27523652021908600a0c55。仍为60s/configured30GiB/Python3.10.6/-B/单线程math。

实际 PASS_WORKER_ARCHIVE_AND_CURRENT_RECORDS_ONLY：16份原业务归档21,528B、3份父端capture1,528B、9个reviewed scripts保持、89行旧census及此前checkpoint完全一致、5个当前工作树和8份Markdown记录通过。Selected record files的已知key/URL credentials扫描通过。没有新runtime/platform/pricing/shape/fullCLI调用。此后停止记录checker扩展，当前下一项仍是有限pricing决定。

## 记录提交现场复核

本次使用 `/data/ycfeng/envs/frontier-py310/bin/python -B`（3.10.6）的 stdlib subprocess 并行执行只读 Git 命令。未运行原 parent_verification.py 或 verify_records.py，未重跑任何 source regression、runtime、pricing、shape 或平台操作。本次是 Git 状态复核，没有新增 cgroup/RSS 或 composed accounting 测量。

精确 source 比较命令在当前工作树执行：

```bash
git diff --exit-code --name-only 2ca5e57113b0bc99ab84e926e5bb492b3670072e HEAD -- . ':(exclude)task_memory/**'
git diff --exit-code --name-only 1461573de388b72450189bc0c1b88b4ac32491f0 HEAD -- . ':(exclude)task_memory/**'
git diff --exit-code --name-only 2ca5e57113b0bc99ab84e926e5bb492b3670072e -- . ':(exclude)task_memory/**'
git diff --cached --name-only
git diff --check -- task_memory/task_2026-10-22_step5_position_routing
```

五个工作树分别执行 `git rev-parse HEAD`；四个保留树分别执行 `git status --porcelain=v1`，并在主线执行 `git merge-base --is-ancestor <保留树完整HEAD> HEAD`。精确目录/HEAD见上方工作树表。Pinned backend目录执行 `git rev-parse HEAD` 和 `git status --porcelain=v1`。

期望：HEAD仍为dfe603ed；三次非task source比较均无差异、exit0；index为空；四个保留树干净且为主线祖先；backend为3dac53c544e546574cb6e8078cd14053d0916113且干净。实际全部一致，输出 `PASS_CURRENT_GIT_SOURCE_AND_FOUR_RETAINED_WORKTREES`。Task diff whitespace检查exit0。原32份新证据和两个报告存在，仅七个已有任务记录被修改。S5修正已完成，有限两次pricing仍未批准。

Checkpoint补录的首个editor在写文件前返回exit1：`AssertionError`，触发于 `assert text.count('| --- | --- |\n') == 1`。只读检查发现该字符串有三处匹配：Modification History表，以及两个三列表格末尾的两列；三个待补录文件均未被写入。修正只在Modification History段内匹配完整分隔行，并在所有文档校验后才写入；不改变原raw失败证据或已有records checker。该editor失败没有独立wall测量，不作为验证失败或新控制次数。
