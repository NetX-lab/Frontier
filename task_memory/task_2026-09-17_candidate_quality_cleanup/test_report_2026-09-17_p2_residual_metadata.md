## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded residual metadata RCA and passing correction. |

# P2 S1 residual trace metadata

RCA and exact red/control commands: `metrics_review.md`, S1 appendix. Frozen candidate fails all six positive residual scenarios with `ValueError: Unsupported operation 'decode_draft_proposer'` or `'mtp_terminal_overshoot'`. Native main passes three scalar controls; main does not implement StageExecutionTime. No Stage parity with main is claimed.

The local emitter now distinguishes complete `resolved_meta` from additive `extra_meta`. It retains caller-owned residual metadata, derives ordinary tensor metadata as before, and preserves physical-layer identity. Related-wait explicit metadata uses the same contract. No operator registry change, duration change, broad catch, or expected-output adjustment.

Environment: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, dedicated venv/no conda, CPU. Executed from the feature worktree after the S1 patch:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_metrics_stage_execution_time.py tests/unit/test_execution_time_metrics_ownership.py tests/unit/test_execution_time_op_times.py tests/unit/test_metrics_full_stage_scope.py tests/unit/test_stage_reporting_contract.py tests/unit/test_op_trace_utils.py tests/unit/test_attention_trace_mapping.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_decode_ep_wave_materialization.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-s1-after
```

PASS: **139 passed in 7.30s**; log `/data/ycfeng/tmp/quality-s1-after.log`. Assertions preserve residual names/tags, durations 3 ms / 5 ms, ordering and start-time offsets, one stage owner even when another layer contains 99 ms, and physical layer IDs 7/9 with their tensor metadata. Broader tests cover operation metrics, stage scope and EP traces. This restores a broken supported reporting path; it is not a claim of identical behavior to the frozen candidate's exception.
