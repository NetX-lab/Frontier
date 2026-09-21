## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded profiles-03 count failure, corrected profiles-04 GPU PASS, and independent timing/kernel evidence validation. |

# Linear timer/context diagnostic verification

## Execution

Source script:
`/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_linear_timing_context.py`.

Root executed the context-only profiles-04 worker after the profiles-03 context assertion failed. The profiles-03 communication result was retained rather than remeasured. The frozen GPU command, image, source receipts and worker log are in `analysis/h200-d019-profiles-04/{launch.sh,execution_manifest.json,issue26_h200_d019_profiles_worker.sh,runtime/timing-context.log}`. Command inside the approved image:

```bash
export PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907:/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908
export TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1
export VLLM_FRONTIER_INSTRUMENTATION=0
/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python \
  /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_linear_timing_context.py \
  --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-profiles-04/runtime/timing-context
```

GPU runtime: H200/step_main; Python3.10.16 conda `vllm-bs-0.10.2`; Torch2.8.0+cu128; BF16; tokens4096; TP4; hidden2048; Q32/KV4; QK norm on. This is a single-GPU local-compute diagnostic with TP4 shapes; no TP collective is executed. The script records the actual imported diagnostic vLLM path, source HEAD and profiling plan.

CPU verification: conda `dev-vidur-v03-hopper-e2e`, Python3.13.13. The verified script matches the frozen GPU-run snapshot byte for byte (`cmp` exited0). The following exact check ran from the worktree root:

```bash
cmp tests/performance/issue26_linear_timing_context.py task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-profiles-04/issue26_linear_timing_context.py
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import itertools, json, math, pathlib, statistics
base = pathlib.Path('task_memory/task_2026-09-07_issue26_ttft_h200/analysis')
run = base / 'h200-d019-profiles-04/runtime/timing-context'
receipt = json.loads((run / 'receipt.json').read_text())
assert receipt['status'] == 'COMPLETE_DIAGNOSTIC'
assert (receipt['gpu'], receipt['tokens'], receipt['tp'], receipt['dtype']) == ('NVIDIA H200', 4096, 4, 'torch.bfloat16')
rows = json.loads((run / 'event_samples.json').read_text())
assert len(rows) == 72
expected = set(itertools.product(range(2), ('synthetic', 'cold_drain', 'prefill_hot'), ('original', 'precreated')))
for op in {row['op'] for row in rows}:
    selected = [row for row in rows if row['op'] == op]
    assert {(row['round'], row['context'], row['timer']) for row in selected} == expected
    assert len({row['calls_per_iteration'] for row in selected}) == 1
    for row in selected:
        assert len(row['samples_ms']) == 20 * row['calls_per_iteration']
        assert all(math.isfinite(x) and x > 0 for x in row['samples_ms'])
        assert all(len(x) == 20 for x in row['samples_by_call_ms'])
        assert [value for sample in zip(*row['samples_by_call_ms']) for value in sample] == row['samples_ms']
        assert row['median_ms'] == statistics.median(row['samples_ms'])
    if op in ('attn_pre_proj', 'attn_rope', 'attn_post_proj'):
        assert selected[0]['calls_per_iteration'] == 1
coverage = json.loads((run / 'kernel_coverage.json').read_text())
assert coverage['kernel_launches'] == 608 and not coverage['missing_device_correlations']
results = json.loads((base / 'd019-linear-timing-context-results.json').read_text())
assert len(results['scope_instances']) == 36
for op, count in (('attn_pre_proj', 5), ('attn_rope', 1), ('attn_post_proj', 1)):
    instances = [x for x in results['scope_instances'] if x['scope'].endswith('/' + op)]
    assert len(instances) == 12 and all(x['kernel_count'] == count for x in instances)
    assert all(x['kernel_names'] == instances[0]['kernel_names'] for x in instances)
    for x in instances:
        assert math.isclose(x['active_union_ms'] + x['internal_device_gap_ms'], x['device_envelope_ms'], abs_tol=1e-12)
print('PASS: GPU receipt,72 event rows,all 12 conditions,multiplicity/finite samples,608 complete kernel correlations,36 target scopes')
PY
```

## Criteria and observed evidence

- **PASS:** six measured labels × three contexts × two timer variants × two rounds =72 rows. The three target ops each execute once per iteration and have20 positive finite samples per row. Embedding has two sequential calls per iteration, retained separately by call position. No samples are dropped.
- **PASS:**608 launch/device correlations, no missing kernel records;36 target scope instances with invariant ordered kernel identities across all conditions.
- **PASS:** script snapshot equality, Python AST/CLI preparation checks, and independent numeric/artifact checks.
- **Historical FAIL, resolved:** profiles-03 asserted every label had20 records. Actual `emb` had40 because `GPTModel.forward` executes it twice; traceback and raw values remain in its log. Corrected multiplicity is learned from stable warmup counts and still strictly checked against active measurements.

## Numeric comparison and practical limits

Values are milliseconds per operation. S is the mean of two event-sample medians. V is new low-density vLLM DP0/TP0 first-forward total divided by48; output projection uses its pure GEMM scope. These are diagnostic S/V values, not new simulator predictions or a clean TTFT gate.

| Operation | Synthetic S | Hot S | vLLM V | Synthetic absolute gap / relative gap | Hot absolute gap / signed relative gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV + Q/K norm | 0.133432 | 0.078192 | 0.080625 | 0.052807 / +65.496% | 0.002433 / −3.018% |
| RoPE | 0.031784 | 0.019216 | 0.019576 | 0.012208 / +62.362% | 0.000360 / −1.839% |
| Output projection | 0.046752 | 0.029824 | 0.030309 | 0.016443 / +54.253% | 0.000485 / −1.599% |

Precreating Event objects changes synthetic QKV by−3.561% on the mean-of-medians summary and hot QKV by0.000%; the much larger hot-context effect is−41.399%. RoPE and output projection show similarly large context effects and small timer effects. This excludes Event-object construction as the demonstrated dominant cause of the observed gap.

The separate profiler pass preserves kernel identities and similar active durations, but its hot QKV device envelope is0.149856ms versus event-only hot0.078192ms. It therefore perturbs the queue/host behavior under study. Its measured submission gaps support a mechanism only within that pass; they cannot be imported as an exact clean CPU overhead or subtracted from production timing. Full analysis and all comparison inputs are in `analysis/d019-linear-timing-context.md` and `analysis/d019-linear-timing-context-results.json`.

No production timer, generic profile CSV, predictor cache or workflow CPU term was changed. Operator-context integration and fresh first-forward/TTFT validation remain pending.

## Delivery record

Diagnostic script committed as `578a4b8d` (`test: diagnose linear CUDA timing and execution context`) after profiles-04 GPU and independent artifact checks passed. The commit contains only `tests/performance/issue26_linear_timing_context.py`. Reports and selected evidence remain persisted in the task directory under the existing Git ignore policy.
