## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Frozen handoff while the already-started predictor training continues; export verified, actual first-forward acceptance pending. |

## Final handoff update — 2026-09-08 14:23 UTC

The previously live query completed naturally. Root validated the actual receipt:54.914898417309455ms,17successful RF.fit calls,13unique query entries/2592returns. Three target operators match the unique candidate rows within1e−15ms; the other8independent compute models retain exactly the prior values, features, branches and call counts. Boundary reconciliation error is below1e−10ms. `validation.json`, `query_receipt.json` and `query.log` are persistent here. Candidate ingestion PASS, CUDAgateFAIL versus both references. The pending-process/checklist text below is historical and superseded. No activeCPU/GPU job remains. Canonical inputs and production context selection were not changed. Exporter committedfb3ed797.

# Frozen handoff: linear context diagnostic candidate

Status: EXPORT PASS; ACTUAL BOUNDED QUERY STILL RUNNING at handoff. No claim of CUDA closure. The parent requested transfer to another session and no further experiment launches. Preserve the current process, candidate, source samples and caches.

## Running process and exact paths

- Audit PID: **616810**; parent shell PID: **616795**; tool session: **63084**.
- Last direct status: `Sl`, elapsed `05:12`, wait channel `hrtimer_nanosleep`. Latest log at 22:15:57 reports `attn_prefill_mixed` trained with 1890 mixed-batch samples. No `query_receipt.json` existed at this observation.
- Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
- Log: `/data/ycfeng/tmp/issue26-d019-linear-context-candidate-query-01.log`.
- Output: `/data/ycfeng/tmp/issue26-d019-linear-context-candidate-query-01`.
- Expected actual receipt: `/data/ycfeng/tmp/issue26-d019-linear-context-candidate-query-01/query_receipt.json`.
- Predictor cache: `/data/ycfeng/tmp/issue26-d019-moe-integration-01/predictor-cache` (current-task generation, retained).
- Collective cache: `/data/ycfeng/tmp/issue26-d019-first-forward-integrated-01/collective-cache` (unchanged calibrated ideal communication).
- Config: `analysis/d019-linear-context-candidate/config.json` in this task.

Exact command, already running; do not launch a duplicate:

```bash
PYTHONPATH=. WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 TMPDIR=/data/ycfeng/tmp \
  /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_predictor_query_audit.py \
  --allow-fit \
  --config task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-linear-context-candidate/config.json \
  --output /data/ycfeng/tmp/issue26-d019-linear-context-candidate-query-01 \
  > /data/ycfeng/tmp/issue26-d019-linear-context-candidate-query-01.log 2>&1
```

CPU environment: conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13. The first minute included a transient NFS `wait_on_page_bit_common`; initialization resumed without intervention. Existing config-based model cache keys mean the changed linear path also retrains some unchanged targets. Do not claim only three fits; inspect the final actual fit count.

## Completed evidence

The standalone exporter is `tests/e2e/issue26_linear_context_candidate.py` (84 lines, new and currently uncommitted). Source and config are frozen. No production file was edited. Other agents have unrelated untracked files; preserve them.

The real export command and all acceptance criteria are recorded in `../../test_report_2026-09-08_d019_linear_context_candidate.md`. The exporter actually passed assertions: 82 source rows; exactly one M4096/TP4 row; BF16/CUDA_EVENT dimensions and all three typed contracts match the producer receipt; all 40 samples per op from event-only/original timer/prefill_hot are pooled; no minimum or context cherry-picking. Only 18 CSV cells differ, six statistics per op. All other cells preserve source strings. The candidate config changes exactly one field: its linear input path. `export_receipt.json` retains raw samples, aggregation, old/new values and full source receipt.

| Operation | Prior ms/layer | Candidate ms/layer | Candidate 48-layer ms | vLLM TP0 48-layer diagnostic ms | Signed op gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| attn_pre_proj | 0.137167997658253 | 0.078143998980522 | 3.750911951065 | 3.870016008615 | -3.0776% |
| attn_rope | 0.032336000353098 | 0.019216000102460 | 0.922368004918 | 0.939648004249 | -1.8390% |
| attn_post_proj | 0.046784000471234 | 0.029856000095606 | 1.433088004589 | 1.454815996811 | -1.4935% |

These are candidate data and an arithmetic diagnostic comparison, not yet actual predictor acceptance. vLLM op values come from `../d019-compute-results/p_s_v_linear_attention.csv`; the instrumented run has known probe perturbation and is not the clean full-forward reference.

The prior actual integrated first boundary is 59.1903543839013 ms (`../d019-first-forward-integrated/query_receipt.json`). Replacing these three medians implies a delta of **-4.275455966591835 ms** and an independently expected boundary of **54.914898417309466 ms**. This would increase the remaining total gap against the 79.30735778808594 / 78.11878204345703 ms batch-only anchors, despite improving matched projection timings. Do not substitute this expectation for a real observed endpoint.

## Next session: required checks in order

1. Observe the existing PID/log/output. If it is still running, preserve it; if it exited, inspect its actual log and receipt before starting anything else. A successful process exit alone is not numerical acceptance. Preserve any new failure in this report and diagnose its source.
2. When present, require receipt status `PASS_BOUNDED_PREDICTOR_QUERY_AUDIT`, request IDs `[0]`, and actual `stop_boundary.time_s`. Compare each `(model, method, feature_key, branch, call_count)` with the prior integrated receipt. The three projections must be `measured_exact`, with one non-missing matched row and the selected 4096 feature, reading the candidate medians. Other eight distinct compute models must retain identical features/values/branches; account separately for repeated/delegated query entries. Inspect `matching_filtered_rows` and the actual CSV target to establish provenance.
3. Reconcile actual first-boundary milliseconds against 54.914898417309466 within numerical roundoff. Record signed and absolute errors against both batch-only anchors, with percentage errors. Record actual `rf_fit_policy.successful_calls_in_process`; capture whether unchanged model refitting leaves all unchanged compute outputs identical. Do not treat a new RF fit as proof that RF was the original error source.
4. Preserve the real query receipt and selected log in this candidate directory and write `validation.json`. Update the test report and this report from pending to observed PASS/FAIL. Retain the explicit diagnostic-only and no-CUDA-closure limitations.
5. Once focused acceptance passes, stage and commit **only** `tests/e2e/issue26_linear_context_candidate.py`. Task docs are ignored and should not be forced into Git. Preserve other agents' edits. If numerical verification fails, report it and correct only the established in-scope cause before committing the relevant completed sub-step.

No new shared runtime-context schema or production selector is authorized by this candidate. No pending source edit is required for the exporter. Pending tasks are the five verification/recording steps above; newly discovered unresolved failures are none at handoff. The design proposal remains in `../d019-linear-context-proposal.md` and explicitly recommends this bounded option before considering a formal feature.
