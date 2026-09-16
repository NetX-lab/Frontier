## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded routing and layer-state preservation. |

# P1d validation

Environment: Python 3.12.3, `/data/ycfeng/tmp/quality-review-env`, no conda. Put this venv on PATH so nested example shells select the same interpreter.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_moe_predictor_layer_id_semantics.py tests/unit/test_dense_execution_time_layer_scaling.py tests/unit/test_execution_time_op_times.py tests/unit/test_stage_execution_time.py tests/unit/test_stage_finalized_contract.py tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p1d
```

Criteria: preserve actual replica IDs, per-layer global expert ratios, lane routing, layer IDs, stage-owned work and all non-dummy artifacts. Observed **116 passed in 64.41s**. Compare the eight `run` directories against `/data/ycfeng/tmp/quality-pre-nondummy` with the established comparator: **90/90 PASS**; bidirectional metrics file inventories agree (106 files). Acceptance evidence and stage summaries also compare equal. No candidate numerical difference.

Logs: `/data/ycfeng/tmp/quality-p1d.log`. Full baseline with PATH corrected: `/data/ycfeng/tmp/quality-pre-unit-path.log` / `.xml`, **3587 passed, 18 failed, 25 skipped**. Reproducing the 18 failed node IDs in pinned main gives **18 failed**, logged in `/data/ycfeng/tmp/quality-main-failures.log` / `.xml`. These are pre-existing missing files/document expectations, not cleanup regressions. Synthetic profiles remain distinct from hardware parity.
