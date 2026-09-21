## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Verified the completed ten-warmup H800 bounded run and formal first-forward CUDA-event span. |

# H800 normal: ten warmups, formal CUDA span

## Test script and exact commands

Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.

Worker: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h800_ar_worker.sh`.
Client: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_token_id_client.py`.
Analyzer: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_bounded_warmup_span_analysis.py`.

Submitted launch recipe (already completed; creates a named GPU job):
`bash /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-ar-176000-normal-10warm-01/launch.sh`

Reproduce analysis without GPU execution:
```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
/usr/bin/python tests/e2e/issue26_bounded_warmup_span_analysis.py --run /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-ar-176000-normal-10warm-01/runtime/batch --warmups 10
/usr/bin/python tests/e2e/issue26_bounded_warmup_span_analysis.py --run /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-ar-176000-normal-01/runtime/batch --warmups 3
bash -n tests/e2e/issue26_h800_ar_worker.sh
```

GPU environment: conda `vllm-bs-0.10.2`, Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0. CPU analysis: `/usr/bin/python`, Python 3.12.3, standard library only.
Job `yc26-h800-ar-normal-10warm-20260910-01`; node `gpu-h800-0165`; codesign/h800, 8 GPU/64 CPU/400 GiB, KV override 176000, block size 16, normal AR, eager, prefix cache OFF, TP4/DP2. Source receipt: `0d633a946e6600c77a251bef8b5553ec7f43f7e7`. Fixed image and full arguments are preserved in launch.sh and runtime/server logs.

## Validation criteria and evidence

- PASS: exact ordered client identities, ten drained warmups with one request per replay, followed by one formal request; all eleven requests have 4096 prompt and 1024 completion tokens.
- PASS: client.log phase starts follow preceding phase ends; previous client completion precedes next arrival; formal summary reports one unique completion.
- PASS: eight worker identities present; exactly four matching formal prefill records, TP0-3, same DP lane and batch ID.
- PASS: first formal request `cmpl-pf4096_dc1024:0-0`, batch size 1, request token list [4096], total/prefill tokens 4096, decode tokens 0. Selected by request identity and predicates.
- Completion marker: `DIAGNOSTIC_EXECUTION_COMPLETE selection=batch` in `/data/ycfeng/tmp/issue26-h800-ar-normal-10warm-launch.log`; server.log ends with application shutdown complete.
- FAIL normal-scale reproduction: formal CUDA-event spans remain above the user's 70-110 ms reference range.
- Full standard-suite gate: NOT MET. This bounded worker ran ten single-request warmups, not three 100-request replays plus 100 formal requests. The existing standard identity validator hardcodes the latter contract; this result uses the bounded analyzer and does not claim that validator passed.

## Formal first-forward result

DP0, batch_id 5120, DP token counts [4096, 1].

| TP rank | CUDA-event span (ms) |
| --- | ---: |
| TP0 | 519.799133301 |
| TP1 | 522.410522461 |
| TP2 | 522.551147461 |
| TP3 | 522.017639160 |

| Metric | 3 warmups (previous, DP1) | 10 warmups (current, DP0) |
| --- | ---: | ---: |
| Rank median (ms) | 675.207061768 | 522.214080811 |
| Rank P90 (ms) | 682.078924561 | 522.508959961 |
| Rank maximum (ms) | 684.848632812 | 522.551147461 |
| Rank spread (ms) | 10.769958496 | 2.752014160 |

Rank-max difference: -162.297485352 ms (-23.698300%). Rank P90 uses linear interpolation over four ranks, not repeated-run statistics. Rank maximum summarizes four rank-local outer spans; it is not a measured common cross-rank start/end envelope.

Formal client TTFT 535.550271 ms; E2E 467389.809238 ms. These are separate client metrics.

## Interpretation and limits

Ten drained warmups did not recover the reference scale. The result does not establish that warmup count alone caused the reduction: the prior run used gpu-h800-0067 and DP1; this run used gpu-h800-0165 and DP0. No repeated-run confidence or causal warmup effect is claimed.

Source `vllm/v1/worker/gpu_model_runner.py:2474` computes `batch_execution_time_ms` using CUDA event elapsed_time, following synchronization. Earlier wording suggesting this was merely host execution duration is corrected: it is an instrumented CUDA-event span, potentially including host-induced device gaps, not pure kernel time.

The boundary logger (`vllm/model_executor/layers/fused_moe/layer.py:1632`) records host timestamps and opens/writes/closes a file at each boundary. These records cannot substitute for CUDA completion. Logging is an evidenced execution-path difference; its numeric contribution remains unmeasured.

Recorded mode_manifest.json incorrectly states formal_requests=100. Actual worker CLI requests=1 and client completion records=1 are authoritative. Raw manifest is preserved; worker metadata was corrected to 1 for future runs. Warmup metadata=10 is correct. Previous 3-warmup median/P90 arithmetic is corrected in the table above.

The bounded batch worker does not explicitly enable VLLM_MOE_UNIFORM_ROUTING for batch selection and the recorded manifest contains no such flag. Uniform-routing parity with the standard replay is therefore unqualified. No setting was altered in the completed run.

No scalar/skip GPU run or production correction is part of this ten-warmup check. Main pending work: recover a qualified standard-path normal baseline, then resume same-cluster normal/skip comparison and secondary scalar diagnosis. CUDA attribution and overall calibration remain open.
