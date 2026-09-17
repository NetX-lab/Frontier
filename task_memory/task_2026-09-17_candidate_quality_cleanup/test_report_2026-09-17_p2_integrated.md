## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded source-stable reporting regression and frozen-candidate artifact parity. |

# P2 integrated verification

Candidate `8e67fc25` with no concurrent production edits; baseline `c288a19f`. Dedicated `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, no conda; CPU synthetic trained profiles, not native timings.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_metrics_stage_execution_time.py tests/unit/test_execution_time_metrics_ownership.py tests/unit/test_execution_time_op_times.py tests/unit/test_metrics_full_stage_scope.py tests/unit/test_stage_reporting_contract.py tests/unit/test_op_trace_utils.py tests/unit/test_attention_trace_mapping.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_decode_ep_wave_materialization.py tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p2-integrated
```

PASS: **159 passed in 69.62 s**; log `/data/ycfeng/tmp/quality-p2-integrated.log`. Criteria include source isolation, actual dense layer IDs, residual owner-once behavior, lane/barrier metadata, and normal Simulator/profile-loader numerical assertions.

Artifact inputs: `/data/ycfeng/tmp/quality-pre-nondummy/test_nondummy_simulator_accept{0..7}/run` versus `/data/ycfeng/tmp/quality-p2-integrated/test_nondummy_simulator_accept{0..7}/run`. Existing `tests.integration.test_pr33_nondummy_acceptance.compare_baseline` reports **90/90 PASS**. The set of all files in each request-metrics directory is equal in both directions (**106 files** total). Each side must contain exactly one `acceptance_evidence.json` and exactly one `frontier_stage_batch_ledger_summary.json` per case; the existing fidelity comparator reports equality for all **16 pairs**. Log: `/data/ycfeng/tmp/quality-p2-artifact-comparison.log`.

No numeric expected value or tolerance was changed. Positive residual traces are the separately reproduced S1 correctness restoration, not covered by a claim that the broken frozen path succeeds. Full integrated final fidelity and full-unit regression remain P5 requirements.
