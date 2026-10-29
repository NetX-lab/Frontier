# Native06 client verification test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded exact decisive command, actual results, retained supplementary failure and scoped phase checkpoint. |

Task `W128-V4-NATIVE06-CLIENT-ACCEPTANCE`. Environment `/usr/bin/python3`3.10.6, pandas2.0.1, no active conda, `PYTHONDONTWRITEBYTECODE=1`; foreground tool timeout60000ms. Only existing client/boundary APIs executed; no native run or persistent validator script.

## Verification results

| Check | Passing condition | Actual | Result |
| --- | --- | --- | --- |
| Complete frozen client helper | Exact populations/IDs/lengths/status/endpoints and clean timestamp provenance | 432/432; formal128×3; zero invalid requests; unmodified helper succeeds | PASS |
| Existing frozen boundary helper | Seven ordered cold/drain proofs, both roles/all lanes, fresh counts | `(True, '7 boundaries; both roles, all 2 DP lanes and fresh phase completion counts')` | PASS |
| Proxy correspondence | 432 corresponding unique rows, both upstreamHTTP200 | 432 rows / 864 upstream200; NIXL-pull6144/TP8 | PASS |
| Formal metrics | Stored aggregate exact; normalized API within1e-9 | Exact in all3; round2 TPOT sum difference approximately3.6e-15ms | PASS |
| Input preservation | All28 accessed pins identical before/after | SHA256/bytes/mtime_ns unchanged | PASS |
| Supplementary lane-equality assumption | This was not an existing acceptance condition | D145/143 after repeat2 formal; raw assertion failed | FAIL — unsupported assertion, corrected without rule/input edits |
| Full native/calibration admission | Root frozen R3 and independent acceptance | Outside this assignment; no claim | SKIPPED |

The comprehensive pass records10 categories in0.14313861494883895s. Its JSON is `/data/ycfeng/tmp/glm53_dsv4/native06_client_acceptance_20261023.json`,35377B, SHA256 `ccd397463588149aed54e899dc46453e5b8c275f77cfc0e5dc81386f5124c4f6`. The decisive replay below passed in0.3878673231229186s; output972B, SHA256 `50cf64f4b53e3fafe4eda64d292f0e784784b41d28d63ceda0dff88c6b56e4de`. Both successful stderr logs are empty (SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

The earlier supplementary attempt exited1 with `AssertionError: boundary_04_explicit_finish_counters_and_exact_totals`. Raw322B log: `/data/ycfeng/tmp/glm53_dsv4/native06_client_existing_helpers_20261023.stderr.log`, SHA256 `2138952e30eab6626d1d1bf908f4a122267e3d700829407a8568c74b9735cc97`. Its full traceback and cause remain in `client_receipt.json`. The correction removed only an invented50/50 DP-placement assumption; actual existing helpers and native data were unchanged.

The initial delivery-only check also failed an assertion that the shared acceptance directory contained exactly these three owned files. Concurrent root-owned `routing_rca.json` and `routing_rca.md` are present. Their presence is not an error in this task's output; the corrected read-only delivery check verifies only the three explicitly owned filenames and leaves the root files untouched. Raw tool exception: `Traceback (most recent call last): File "<stdin>", line 26, in <module>; AssertionError`. All metric, input-pin and raw-output checks preceding that directory assertion had passed.

## Exact decisive command executed

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY' > /data/ycfeng/tmp/glm53_dsv4/native06_client_decisive_replay_20261023.json 2> /data/ycfeng/tmp/glm53_dsv4/native06_client_decisive_replay_20261023.stderr.log
import hashlib, importlib.util, json, math, sys, time
from pathlib import Path
C = Path('/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023')
R = C / 'runs/groundtruth_clean/dsv4-h800-p0-canonical-20261021-06'
P = Path('/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_06')
K = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_native_binding/check_groundtruth_run_successor_r2.py')
start = time.perf_counter()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
M = json.loads((R/'run_manifest.json').read_text())
for name in ('client_metrics', 'e2e_metrics_gap', 'pd_replay'):
    relative = f'tests/comparison/calibration/{name}.py'
    assert sha(P/relative) == M['harness_sha256'][relative]
assert sha(K) == '29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a'
sys.path.insert(0, str(P))
from tests.comparison.calibration.client_metrics import native_client_rows
from tests.comparison.calibration.pd_replay import canonical_workload, dedicated_services, formal_metrics
from tests.comparison.calibration.e2e_metrics_gap import metric_values
W = canonical_workload(C/'inputs/trace')
F = native_client_rows(R, W, json.loads((C/'inputs/trace/request_ids.json').read_text()))
S = json.loads((R/'run/replay_summary.json').read_text())
B = json.loads((R/'run/canonical_repetitions.json').read_text())
A = [json.loads(x) for x in (R/'run/client_requests.jsonl').read_text().splitlines()]
Q = [json.loads(x) for x in (R/'run/proxy_requests.jsonl').read_text().splitlines()]
assert S['canonical_repetitions'] == B and S['status'] == 0 and not S['failed_request_ids']
assert json.loads((R/'replay_finished.json').read_text())['returncode'] == 0
D = dedicated_services(R, {role:S['instances'][role]['http_url'] for role in ('prefill','decode')}, 2)
spec = importlib.util.spec_from_file_location('frozen_boundary_helper', K)
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
assert k.native_idle_evidence(B, A, W[0], 2, D)[0]
assert len(Q) == len({r['request_id'] for r in Q}) == len(A) == 432
assert {r['request_id'] for r in Q} == {r['request_id'] for r in A}
assert all(r['prefill_status'] == r['decode_status'] == 200 for r in Q)
values = {}
for i in (1,2,3):
    rows = [r for r in A if r['repeat'] == i and r['segment'] == 'formal']
    m = formal_metrics(rows)
    assert m == B['repetitions'][i-1]['metrics']
    v = {name:float(n/d) for name,(n,d,_) in metric_values(F[f'repeat{i:02d}']).items()}
    assert all(math.isclose(v[n],x,rel_tol=0,abs_tol=1e-9) for n,x in dict(ttft_ms=m['ttft_ms']['mean'],tpot_ms=m['tpot_ms']['mean'],request_e2e_time_ms=m['e2e_ms']['mean'],token_throughput_tps=m['total_thpt_tokens_s']).items())
    values[f'repeat{i:02d}'] = v
print(json.dumps(dict(status='PASS_CLIENT_BUSINESS_ONLY',client_count=len(A),formal_counts={name:len(frame) for name,frame in F.items()},boundary_count=len(B['idle_boundaries']),metrics=values,elapsed_s=time.perf_counter()-start),indent=2))
PY
```

The command uses frozen imported APIs, including the full native client reader and boundary proof helper, rather than reimplementing those checks. Importing R2 does not invoke its `main()` or rank/native paths. Formal values, endpoints, boundary timestamps, all accessed hashes/mtimes and verification limits are pinned in `client_receipt.json`.

The following is the scoped `/check-progress` self-check after verified client delivery; PASS below denotes progress completion, not native admission.

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 1｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Client人口、完整helper与七boundary | PASS | 432/432；冻结client/idle helper通过。 |
| 三轮formal指标与现有client稳定规则 | PASS | 128×3与stored完全一致；未新增数值阈值。 |
| 限定三份报告/receipt交付 | PASS | 仅本任务拥有文件，保留原始异常与限制。 |
| root完整R3/rank/native准入 | SKIPPED | 明确归root并行；本任务没有代行。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本次派发至交付｜**当前前沿：** client完成；root完整准入独立继续。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用完整API与R2原有boundary helper，无新工具。 |
| D2 细节反复重试 | CLEAR | 一次定位错误50/50断言，保留失败；未反复修规则。 |
| D3 支线问题修补 | CLEAR | 仅派发的本地client验收。 |
| D4 可并行任务停滞 | CLEAR | root完整checker独立运行；本任务没有等待它或重复执行。 |
| D5 超范围镀金 | CLEAR | 仅三份拥有文档，无source/config功能扩展。 |
| D6 分析空转 | CLEAR | 证据直接形成client交付，无CPU overhead支线。 |
| D7 路线违背 | CLEAR | 保留mean、所有spikes、原始异常及native05 UNVERIFIED。 |

**文档检查：** S1–S4/S6未发现本任务范围内明显缺陷；S5：主requirements:153、plan:129、ticket18:71及progress:184仍为Running/unfinished，而本次派发与root `native06_platform_terminal.json`明确已终态。
**归因：** 无执行偏离；补充断言误差已纠正并完整留证。

## 3. 纠正建议

**分支：** B：主记录有终态时效缺口，由root更新其拥有文档。

**文档更新（先执行）：**
1. 主requirements:153、plan:129、ticket18:71、progress:184：root记录终态与本报告client PASS，另列完整R3/独立验收结论；不得用client PASS代替native准入。（可直接执行，root拥有）

**需停止的偏离行为：无**

**接下去的路线：**
1. root完成冻结R3与独立native acceptance → 完成标志：ticket18全rank/runtime/graph、shutdown及reference admission明确。
2. 并行继续已批准的DSV4 predictor工作；ticket19等待qualified完整cost与已准入18，GLM保持暂停。

## 4. 最值得关注的 3 点

1. 432/432、三个128formal与七boundary PASS；完整mtime provenance直接通过。
2. D217/215是实际请求分配，不是balanced expert routing失效；首轮warmup及formal spikes全部保留。
3. 完整native/calibration准入未由本任务证明，root须同步终态记录并保留独立gate。
