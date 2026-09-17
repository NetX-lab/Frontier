## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded fixed physical-layer interfaces and focused preservation checks. |

# Private layer prediction contract

The only callers of the private MoE/disaggregation layer predictors supplied `num_layers=1`. Removed that argument and its unreachable aggregate validation. The disaggregation operator diagnostic interface still receives the locally established count of one. Real `stage_num_layers` for MTP and public stage widths remain unchanged.

Environment: `/data/ycfeng/tmp/quality-review-env`, Python 3.12.3, uv venv, no conda.

```bash
PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_moe_predictor_layer_id_semantics.py tests/unit/test_dense_execution_time_layer_scaling.py tests/unit/test_pd_decode_moe_layer_accounting.py tests/unit/test_typed_ep_predictor_contract.py tests/unit/test_attention_query_cache.py tests/unit/test_mtp_terminal_overshoot_ep_replay.py -q -p no:cacheprovider
```

Criteria: unchanged layer/stage values, typed EP admission, per-layer routing, MTP ownership and snapshot counts. **105 PASS in 7.92 s**; `/data/ycfeng/tmp/quality-p1-private-layer.log`. Existing expectations and explicit per-layer versus optimized stage oracles were not changed. The prior full baseline includes these tests; final E2E rerun remains required.
