## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified serial collective/context supplement and context-only recovery. |

# D019 supplement phase execution

## Execution

Worker: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_h200_d019_profiles_worker.sh`.
Launch commands on CPU master:

```bash
bash /data/ycfeng/tmp/issue26-h200-network/launch-d019-profiles-03.sh
bash /data/ycfeng/tmp/issue26-h200-network/launch-d019-profiles-04.sh
```

These exact launchers were run by persistent user systemd services named `yc26-h200-d019-profiles-20260908-03` and `yc26-h200-d019-profiles-20260908-04`; their full copies, worker snapshots and execution manifests are under `analysis/h200-d019-profiles-03` and `analysis/h200-d019-profiles-04`. They use fixed image digest b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc, eight H200 in step_main, company proxy, Python3.10.16 / conda vllm-bs-0.10.2 at `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Torch2.8.0+cu128, reviewed diagnostic vLLM4bc1bc026. No production protocol/timer change.

## Criteria

Existing worker must select only the requested independent phase, serialize GPU operations, retain source and environment provenance, and surface errors. Validate positive samples/counts, eight-rank collective correctness and size/kernel identity; preserve all legitimate repeated scope calls. The recovered worker must match its frozen executed snapshot.

## Evidence

- profiles03: communication sweep completed and independently validated8ranks/80rows/560samples/40kernel joins. Subsequent context phase FAIL with `AssertionError: ('emb', [...40 values...])`: GPTModel executes two embeddings per synthetic iteration, so the diagnostic's blanket20-pair assertion was incorrect. Raw failure is retained in `analysis/h200-d019-profiles-03/runtime/timing-context.log`. Whole job exit1; completed communication measurements remain separately qualified.
- profiles04: context-only phase PASS,72event rows and608kernel launches with zero missing correlations; `D019_PROFILES_EXECUTION_COMPLETE`, exit0. Stable warmup multiplicity determines expected counts; target QKV/RoPE/output-projection each remain1call/iteration. All raw event pairs and call positions retained. No repeated communication collection.
- Direct worker checks PASS: `bash -n tests/performance/issue26_h200_d019_profiles_worker.sh`; `git diff --check -- tests/performance/issue26_h200_d019_profiles_worker.sh`; executed snapshot bytes exactly equal final worker source.

Execution and measurement validation are complete for this worker sub-step. Context causality and integrated first-forward numerical calibration are separate analyses; successful process exit is not CUDA/TTFT acceptance.
