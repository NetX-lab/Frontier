## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded two reproductions of mixed GDN admission on reviewed HEAD. |

# R01 reproduction report

## Execution

Production HEAD: `d43ae93240444bd4eff9bd99f296d2e751370514`; production unchanged. Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

Interpreter: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, existing uv virtual environment; no conda activation. CPU execution with synthetic profiling data; no GPU command or native accuracy claim.

Regression script: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/integration/test_gdn_phase_admission.py`.

```bash
PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/integration/test_gdn_phase_admission.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r01-red > /data/ycfeng/tmp/pr33-r01-red.log 2>&1

mkdir -p /data/ycfeng/tmp/pr33-r01-observed
PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python tests/integration/test_gdn_phase_admission.py /data/ycfeng/tmp/pr33-r01-observed > /data/ycfeng/tmp/pr33-r01-observed.log 2>&1
```

The pytest driver sets OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 for its Simulator subprocess. The second execution adds retained batch observation; it does not change production scheduling.

## Criteria

Capacity three, online arrival times 0 / 0.002 / 0.002 seconds, input/output lengths 16/12, 33/3, 16/3. Real Simulator, non-dummy training/loading and scheduler/event dispatch. Expected successful supported-policy execution: all requests complete, every dispatched batch phase-pure, a one-token final prefill chunk remains prefill, all KV allocations and GDN slots released. Ownership stability/reuse coverage will be strengthened after the scheduling policy is selected.

## Evidence — FAIL, defect reproduced

First execution: **1 failed in 9.16 s**. Second execution independently raises the same error:

```text
replica_stage_schedule_event.py -> predict_stage_execution_time
  -> predict_attention_layer_time -> GDNPredictor.predict_attention_time
  -> GDNBatchFeatures.from_batch
ValueError: GDN predictor does not support same-batch prefill+decode mixing
```

Observed batches, represented as `(request ID, is_prefill_complete, scheduled tokens)`:

```text
Batch 0: [(0, false, 16)]
Batch 1: [(1, false, 31), (0, true, 1)]
```

The second batch contains 31 prefill tokens and one decode token; it proves the actual mixed dispatch independently of interpreting the exception. Raw observation: `/data/ycfeng/tmp/pr33-r01-observed/phase_admission_evidence.json`. The later request is already in the created batch when prediction rejects it. Final success/leak/one-token assertions are not reached, so this report does not claim those criteria pass.

No numeric prediction-versus-hardware error is measured. This is a deterministic functional failure, not a latency calibration result.

## Pending

Select the supported phase policy recorded in `review_remediation_decisions.md`, apply the admission correction, strengthen lifecycle coverage, run the regression to green and complete applicable fidelity checks. No fix has been committed or declared complete.
