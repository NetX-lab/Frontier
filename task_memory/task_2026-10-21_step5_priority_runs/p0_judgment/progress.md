# D21-P0-NATIVE-JUDGMENT progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Completed read-only P0 evidence judgment and scoped checkpoint. |

- completed: read primary spec §0/D21 and tickets09/23; inspected actual staged/native/client evidence, rather than treating platform success as qualification.
- completed: exact24-request,6144/1024 token-ID/usage/hash joins,32 H800 UUIDs, four role-lanes, both idle boundaries,28 staged pins and3 metadata pins PASS; transfer/prompt totals reconciled through existing parser.
- completed: compact judgment and hashed receipt written only in this directory. Original runtime/reference UNVERIFIED outcomes preserved. Basic runtime PARTIAL; formal-ready NO.
- completed: milestone check-progress below; no production or simulator edits, no GPU/auth/network/SSH, no further delegation, no new harness.
- completed: executed the exact compact recheck below; PASS for all38 retained evidence hashes, exact24 requests, both idle boundaries and preserved UNVERIFIED outcomes.
- pending outside this delegation: parent-owned G3, actual replay/native readiness, measured-input admission and formal reference/comparison. These are not unfinished child implementation work.

## Exact compact recheck command

Run from `/data/ycfeng/Frontier/.worktrees/step5-preview`, local Python3.10.6; no conda environment or network:

```bash
python3 -B - <<'PY'
import json, pathlib, hashlib, collections, re
d = pathlib.Path('task_memory/task_2026-10-21_step5_priority_runs/p0_judgment')
receipt = json.loads((d / 'receipt.json').read_text())
r = pathlib.Path(receipt['evidence_root'])
assert all(hashlib.sha256((r / p).read_bytes()).hexdigest() == h for p, h in receipt['evidence_sha256'].items())
rows = [json.loads(x) for x in (r / 'stage/evidence/replay/client_requests.jsonl').read_text().splitlines()]
assert collections.Counter(x['segment'] for x in rows) == {'warmup': 16, 'concurrent': 8}
assert all(x['status'] == 'completed' and x['http_status'] == 200 and x['stream_completed'] and not x['validation_errors'] for x in rows)
assert all(x['accepted_prompt_token_count'] == x['usage']['prompt_tokens'] == 6144 and x['generated_token_count'] == x['usage']['completion_tokens'] == 1024 for x in rows)
assert all(sum(len(y['token_ids']) for y in x['token_id_deltas']) == 1024 and x['prompt_sha256'] == x['accepted_prompt_sha256'] for x in rows)
summary = json.loads((r / 'stage/evidence/replay/canonical_summary.json').read_text())
assert len(summary['idle_boundaries']) == 2 and all(b['verified'] for b in summary['idle_boundaries'])
assert all(v == 0 for b in summary['idle_boundaries'] for ob in b['observations'] for role in ob['roles'].values() for lane in role['lanes'].values() for v in lane.values())
logs = [(p, p.read_bytes().decode(errors='replace')) for p in (r / 'stage/evidence/nodes').glob('*/server.log')]
assert len(logs) == 4
assert sum(len(re.findall(r'Engine\s+\d{3}\b', text)) for _, text in logs) == 0
assert not any('FULL' in line and 'Capturing CUDA graphs' not in line and 'cudagraph_mode' not in line for _, text in logs for line in text.split('\n'))
assert summary['status'] == 'CLIENT_WORKLOAD_VERIFIED' and summary['runtime_qualification'] == summary['reference_admission'] == 'UNVERIFIED'
print('PASS: all retained evidence hashes, exact24 requests, both idle boundaries; actual replay/formal remain UNVERIFIED.')
PY
```

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 对齐 spec/D21/tickets09、23 | PASS | 当前规范和原始结局已核对 |
| 审核 native/request/idle/transfer/graph | PASS | 报告区分已观察事实与缺失证据 |
| 保存报告、hash receipt并交接 | PASS | 仅本目录；无生产编辑或新harness |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本次委派开始至今｜**当前前沿：** 完成证据判断并交回父任务

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只读原始证据，未新增观测工具 |
| D2 细节反复重试 | CLEAR | 两个本地读取错误各定位一次；未重跑native |
| D3 支线问题修补 | CLEAR | 未修复新问题 |
| D4 可并行任务停滞 | CLEAR | 委派明确限定单一判断；其他agents由parent管理 |
| D5 超范围镀金 | CLEAR | 仅报告、receipt及本进度记录 |
| D6 分析空转 | CLEAR | 分析就是委派交付；已产生明确PARTIAL/NO判断 |
| D7 路线违背 | CLEAR | 没有把basic升级为G3/G4，未调查simulator |

**文档检查：** S1–S6 均未发现明显缺陷；旧ticket状态由parent更新，本child不改其他记录。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 父任务接收本报告 → 完成标志：保留24请求/idle通过与runtime/reference未验证的不同范围。
2. Parent推进已有G3、实际replay及measured-input准入 → 完成标志：真实prerequisites满足才允许formal；不因startup gauge诊断重复GPU运行。

## 4. 最值得关注的 3 点

1. 24/24精确6144/1024完成，两次后续idle真实通过；首次缺失gauge只作诊断。
2. 观察到P6128、D6128外部+16本地；并发8请求有64个rank-read/8,952,774,656 bytes。
3. 保留日志没有Engine NNN/FULL runtime rows；G3、G4、profile准入和四误差未具备，formal-ready NO。
