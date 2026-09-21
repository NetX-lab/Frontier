# PPLX boundary analyzer repair and artifact validation

## Execution

The analyzer was run without a GPU replay against the completed ten-warmup artifact:

```bash
python tests/e2e/issue26_pplx_boundary_analysis.py \
  --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-boundary-20260913-02/runtime \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-boundary-repaired-20260913-02.json \
  --warmups 10
```

The code change is commit `ad0f6f0d22c5b9fdc97c5955cf49b128a1f578d5`.

## RCA and correction

The original analyzer hard-coded `dp_rank == 0`. vLLM's DP load balancer assigned the first formal request `cmpl-pf4096_dc1024:0-0` to DP1 in this artifact, leaving DP0 files empty and causing an analyzer assertion even though all four DP1 TP workers recorded valid rows. The analyzer now validates the request and batch predicates first, derives the selected DP lane from the non-empty validated rows, and requires exactly one DP lane containing TP0--TP3. This preserves the shared four-TP gate and does not accept partial or cross-DP rows.

## Evidence

- Client rows: 1100 (10 warmups x 100 + 100 formal).
- Formal identity: 100 rows; prompt 4096, completion 1024.
- Selected first formal request: `cmpl-pf4096_dc1024:0-0`.
- Selected DP lane: DP1.
- TP0--TP3 CUDA event envelope (ms): 541.981445, 542.336853, 542.339172, 542.371643.
- Median: 542.338013 ms.
- P90: 542.361902 ms.
- Rank max: 542.371643 ms.
- Rank spread / outer span delta: 0.390198 ms.
- Backend: PPLX.
- Event semantics: model-forward CUDA event envelope; may include queued device work and excludes host wall-clock gaps.

This is a valid PPLX artifact and a corrected analyzer result. It is not a clean native 78--84 ms reference and contains no per-operator attribution; no Frontier production correction follows from it.
