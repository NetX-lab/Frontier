## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded actual generation03 formal operator extraction and artifact checks. |

# Formal Operator Extraction Verification

Result: PASS for extraction and artifact integrity. This report does not establish Frontier/vLLM batch comparability or a TTFT error result.

## Execution

- Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
- Python: `/usr/bin/python`, `3.12.3`; conda environment: `not active`.
- Input: completed `h200-diagnostics-03` operator logs with the existing worker-identity validation marked PASS.
- Exact executed helper and arguments (paths expanded):

```bash
/usr/bin/python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_formal_operator_extract.py \
  --run /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-diagnostics-03/runtime/operators \
  --validation /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d007-operators-03-validation.json \
  --contract /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/operator_scope_contract.md \
  --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/formal-operators-03
```

The output directory must be fresh; preserve this completed generation. The helper streams binary lines, parses only selected batch prefixes, and stops after the third selected batch. The direct follow-up check reads only the extracted JSONL files and records `validation.json`.

## Criteria and evidence

- PASS: eight worker identities, each with the first three formal local batches. DP0: `3851, 3852, 3853`; DP1: `3863, 3864, 3865`. Both sets cover prefill, mixed, and decode, with matching TP siblings.
- PASS: 24 selected batches, eight per local phase; all selected request IDs map to formal client requests.
- PASS: 21,120 raw operator rows preserved byte-for-byte by the extractor; each worker contributes 2,640 rows. Selected raw files total 8,920,288 bytes.
- PASS: 48-layer scope counts, 96 gating rows, 96 dispatch/combine rows each, 95 nested add rows, and the phase-appropriate attention scopes in every selected batch; all observed durations are finite and positive.
- PASS: retained scope sums equal raw scope sums minus inner duplicate communication scopes and nested add scopes within absolute tolerance `1e-7 ms`. Gating rows are separately aggregated as linear and TopK; raw rows and inner/outer aggregates remain available.
- Execution scanned 26,722,952 binary lines / 14,227,231,982 bytes, parsing only 21,120 operator rows; elapsed time was 267.934446 seconds. No complete operator file was loaded into memory or parsed as JSON.
- Failure list: empty for this extraction and follow-up artifact check.

## Artifacts and limits

`receipt.json` preserves case ID, generation, source paths, source batch/request IDs, client IDs, DP/TP/PP identity, batch timestamps, extraction timestamps, raw file paths, and per-batch scope counts/sums. `validation.json` records the independent check outcomes. Eight `server.ops.dp*.tp*.pp*.jsonl` files contain the selected original records.

These are formal instrumented samples from the current generation. No historical data or warmup timing was used. Cross-DP collective-round identity is not inferred from local batch IDs. Scope sums remain a partial, instrumented forward decomposition and cannot be interpreted as CPU overhead or substituted for clean official request TTFT.
