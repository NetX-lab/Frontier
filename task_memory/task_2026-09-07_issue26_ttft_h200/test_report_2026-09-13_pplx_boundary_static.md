## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the static verification of the PPLX boundary-only source, worker and analyzer. |
| 2026-09-13 | Updated the recorded source pin to final commit `cf1ef5de` after instrumentation validation was added. |

# PPLX boundary-only harness static verification

## Execution

Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.

Pinned vLLM worktree: `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt`, based on `150fa4a1cf46500c22d5fa585ebc9801a43e5c2d` and finally committed at `cf1ef5de9c0c45aedec4cf9d22c8eb56caf8bf0b`. The initial boundary implementation was `f025cc30a9b8e59fe0304e7ce8b03c76a2fb5c8e`; `cf1ef5de` additionally rejects missing instrumentation in boundary mode. The worktree retains the ignored `vllm/vllm_flash_attn` runtime assets required by the H200 image. No GPU command or RJob was launched in this sub-step.

Commands:

```bash
python -m py_compile tests/e2e/issue26_pplx_boundary_analysis.py
bash -n tests/e2e/issue26_h200_pplx_boundary_worker.sh
python - <<'PY'
import ast
from pathlib import Path
p = Path('/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt/vllm/v1/worker/gpu_model_runner.py')
ast.parse(p.read_text())
text = p.read_text()
for token in ('VLLM_FRONTIER_BATCH_BOUNDARY_LOG_PATH',
              '_frontier_select_batch_boundary',
              'torch.cuda.synchronize()',
              'first_formal_4096_prefill'):
    assert token in text
print('PPLX_BOUNDARY_SOURCE_AST_PASS')
PY
python -m py_compile tests/e2e/issue26_pplx_boundary_analysis.py
SRC=/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt
test "$(git -C "$SRC" rev-parse HEAD)" = cf1ef5de9c0c45aedec4cf9d22c8eb56caf8bf0b
test -z "$(git -C "$SRC" status --porcelain)"
```

A synthetic analyzer artifact with 10 warmup rows of 100 requests, 100 formal rows and four DP0 TP0--TP3 boundary rows was also checked. The analyzer command was:

```bash
python tests/e2e/issue26_pplx_boundary_analysis.py \
  --run /data/ycfeng/tmp/issue26-pplx-boundary-analysis-smoke-20260913 \
  --output /data/ycfeng/tmp/issue26-pplx-boundary-analysis-smoke-20260913/report.json \
  --warmups 10
```

## Criteria and evidence

| Criterion | Evidence | Result |
| --- | --- | --- |
| Source parses and is pinned | `PPLX_BOUNDARY_SOURCE_AST_PASS`; worktree HEAD equals `cf1ef5de...`; `git status --porcelain` empty | PASS |
| Worker shell is syntactically valid | `bash -n tests/e2e/issue26_h200_pplx_boundary_worker.sh` | PASS |
| Analyzer imports and parses | `python -m py_compile tests/e2e/issue26_pplx_boundary_analysis.py` | PASS |
| Analyzer enforces the formal client gate | Synthetic run reports `client_rows=1100`, `formal_requests=100`, and exact warmup/formal identities | PASS |
| Analyzer enforces first-formal predicates | Synthetic run reports `cmpl-pf4096_dc1024:0-0`, batch size 1, 4096 prefill, 0 decode, DP0 TP0--TP3 | PASS |
| Rank statistics are computed | Synthetic durations 80.1/80.2/80.3/80.4 ms produce median 80.25 ms, P90 80.37 ms, max 80.4 ms, spread 0.3 ms | PASS |
| Clean runtime timing obtained | No GPU replay launched in this sub-step | PENDING |

## Scope and limits

The new source activates the boundary branch only after `VLLM_FRONTIER_TRACE_SKIP_WARMUP=1` and a strict request prefix plus pure-prefill predicates match. Non-selected forwards do not record the begin/end events, invoke synchronization, create `record_function` scopes, or write diagnostic logs. The selected forward records one start event and one end event, then synchronizes once after the end event to read `elapsed_time`.

The measured duration is a CUDA-event envelope on the model-forward stream. It can include queued device work on that stream and does not include host wall-clock gaps; it is neither a pure-kernel sum nor the existing full diagnostic outer span. The formal PPLX timing and comparison to the native/naive 78--79 ms clean baseline remain pending coordinator GPU scheduling.
