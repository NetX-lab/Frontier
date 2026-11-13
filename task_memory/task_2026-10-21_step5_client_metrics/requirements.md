# Step canonical client metrics requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record D20-CLIENT implementation scope and acceptance. |

## [Original Request]

Implement executable Step canonical client-metric comparison in `feat/step5-client-metrics`, based on `6db2250`. D20 explicitly approves CPU implementation and extending Q10 shared interfaces to Step, superseding D19's implementation wait. Primary normative records are `../task_2026-10-15_step5_preview_h800/{requirements,spec,plan}.md` in the primary worktree.

Reuse the five-file shared client-metric delta `9f58dbe^1..9f58dbe` (final fix `1400f12`) and the shared `pd_replay.canonical_workload` reader. Preserve the D18 manifest/checksum pins in the Step consumer. Do not merge the whole GLM branch.

Owned code: `tests/comparison/calibration/{client_metrics,pd_replay}.py`, `tests/comparison/e2e_gate/`, optionally `tests/integration/step5_pd/canonical_replay.py` to reuse the reader, and associated tests. No scheduler, predictor, profiling, R2, frozen workload/stage changes; no GPU, network, authentication or push. Commit verified changes.

Use native source-format configuration, not model-name branching. Step actually writes `canonical_summary.json`, top-level `repetitions`, status `CLIENT_WORKLOAD_VERIFIED`/`REFERENCE_UNSTABLE`, adapter label `step5`, `stream_completed` and `generated_token_count`. Do not claim vLLM 0.28.0 or invent clean Git/native run-check evidence. Runtime identity must bind the case/source/image declaration to immutable actual receipts through the existing admission mechanism. Missing real provenance/runtime evidence produces `INSUFFICIENT_EVIDENCE`.

Frozen workload: 128 formal requests, 6144 prompt/1024 output tokens, QPS 1, 16 separate same-shape warmups per repetition, three clean repetitions. Client TTFT uses actual first output token, TPOT is the mean of each request's `(E2E-TTFT)/1023`, THPT is `917504/(last formal output-first actual formal dispatch)`, E2E is per request. Compare every repetition independently; retain pooled diagnostics without hiding a bad repetition. Four independent relative-error limits remain 10%. Internal/legacy TTFT semantics remain unchanged.

Acceptance: existing normalize and gate CLI executed end to end for both Step and GLM schemas; correct four metrics; missing/count/duplicate/clock/runtime-origin/repetition negatives reject; existing internal/legacy suites remain green. Document exact commands/results and remaining real inputs. Fixture results never qualify native ground truth.
