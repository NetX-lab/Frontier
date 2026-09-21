## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded fresh H200 profiling execution and input validation. |

# Fresh profiling generation02

Execution: H200 step_main, pinned image sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc, Python /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python3.10.16. Source vLLM46f7b179f. Exact launcher: /data/ycfeng/tmp/issue26-h200-network/launch-fresh-profiles-02.sh. Worker: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_h200_fresh_profiles_worker.sh. Full expanded profiling commands are retained in /data/ycfeng/tmp/issue26-h200-network/launcher-fresh-profiles-02.log.

Reproduce: `bash /data/ycfeng/tmp/issue26-h200-network/launch-fresh-profiles-02.sh` with a new job name and fresh output directory.

Criteria: all three phases complete; new CSVs contain BF16/CUDA_EVENT measurements, populated means finite/nonnegative; attention includes4096 prefill; complete linear registry partitions41points each.

Observed PASS: FRESH_PROFILING_EXECUTION_COMPLETE;82 linear,2026 attention,387 MoE rows. Linear timing counts41per active op, attention2026per exported op, MoE387per op. Attention contains272 target4096 prefill rows. Evidence: analysis/fresh_profiles_02_validation.json and runs/h200-fresh-profiles-02/runtime/{linear,attention,moe}.log.

Generation01 FAIL before measurements: token4112 exceeded implicit max_tokens4096. Worker now supplies linear max_tokens16384 and MoE max_tokens32768. Generation01 is retained.

Limits: this establishes fresh profile execution and basic input validity, not exact runtime shape/routing/operator parity. Frontier predicted TTFT, absolute error and relative error remain pending.


## Fresh MoE context supplement

The baseline training preflight required standalone_legacy in addition to prefill_hot. Exact execution is retained in runs/h200-fresh-frontier-01/worker.sh and runtime/merge-command.json. The worker ran in the same pinned H200 image with vllm-bs-0.10.2 Python 3.10.16, TP1/EP8, BF16/CUDA_EVENT, 43 token points, 3 load distributions, and 3 seeds. Supplement output: runs/h200-fresh-frontier-01/runtime/moe-standalone/compute/h200/qwen3-a3b-30b-moe/moe.csv. Complete merged input: supplements/moe-context-01/moe.csv.

PASS: 387 new standalone_legacy rows; all 387 current-task prefill_hot rows retained; 774 merged rows; dropped_rows=[]; each context filter returns 387. The existing merger's COMMITTED marker, payload byte sizes, and declared digests match both artifacts. Every populated timing mean is finite and nonnegative, with BF16/CUDA_EVENT metadata. Validation used local system Python 3.12 with csv/json/hashlib/math and a read-only sudo invocation because the merge receipts were mode0600 owned by the GPU worker's root user. The initial ordinary read failed with PermissionError; no file permission or data alteration was required.

The subsequent simulator import failed on missing plotly before training. This does not invalidate the new profile measurements; simulator runtime recovery is pending in generation02. Numerical operator comparability and routing alignment remain separate checks.
