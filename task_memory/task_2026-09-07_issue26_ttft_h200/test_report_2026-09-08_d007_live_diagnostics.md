## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified D007 on complete isolated operator and routing replays. |

# D007 exact-case live validation

Execution: RJob yc26-h200-diagnostics-20260908-03, H200 step_main, pinned image b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc. Source: independent vLLM instrumentation branch361d941c97fcec52e544f74b7ab91c54192de9c9; Python3.10.16. Exact worker /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h200_diagnostics_worker.sh; invocation `bash tests/e2e/issue26_h200_diagnostics_worker.sh /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-diagnostics-03/runtime`.

Validation command, once per MODE=ops/operators and MODE=routing/routing: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_diagnostic_identity_analysis.py --run /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-diagnostics-03/runtime/SUBMODE --mode MODE --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d007-SUBMODE-03-validation.json`. Analysis conda environment dev-vidur-v03-hopper-e2e, Python3.13.13. The two existing output paths are final receipts; repeat validation requires fresh output paths.

Criteria:400 unique full4096/1024 completions per submode,300 warmups excluded,100 formal IDs; exactly8 DP0..1/TP0..3/PP0 worker identities; per-worker unique batch/op or batch/layer joins; detail/batch metadata equality; every formal prefill seen exactly once by all four TP workers in one DP lane; first formal and at least3 formal batches per worker covered.

Evidence PASS: {"operators": {"workers": 8, "formal_requests": 100, "batch_rows": 42048, "detail_rows": 36389760}, "routing": {"workers": 8, "formal_requests": 100, "batch_rows": 48848, "detail_rows": 2344704}}. All requests completed; launcher emitted OPERATOR_AND_ROUTING_EXECUTION_COMPLETE. No metadata runtime failure recurred. Both streaming checkers returned0 and statusPASS; operator input totaled18.03GiB. Script/helper committede87b0b541a478eb9d2e422f9d92dfc689cff4949.

Limits: identity and local batch metadata pass does not establish cross-DP collective-round alignment, EP-global phase, operator-scope equivalence or numeric parity. DP dummy rounds are unlogged. Diagnostic client meanTTFT215.85371458ms differs materially from clean client120.50455297ms; clean official server115.982880592ms remains the primary target. Diagnostic measurements never close the E2E gate. No Frontier prediction or error is available yet.
