## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Migrated three unit fixtures to final communication/reporting interfaces; preserved independent lifecycle and timing assertions. |

# W10 focused fixture integration

**PASS: 161 tests in 3.42 s**. Initial focused reproduction was **15 failed, 146 passed in 4.39 s**. Changes are limited to the three assigned test files; no production edits or commits were made.

## Environment and execution

Worktree `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`; `/usr/bin/python` 3.12.3, no conda environment, CPU only. Exact final command:

```bash
python -m pytest tests/unit/test_mixed_layer_decode_ffn_scheduling.py tests/unit/test_pd_decode_moe_layer_accounting.py tests/unit/test_pdaf_prefill_model_time.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w10-unit-owned-final.log 2>&1
```

Initial evidence: `/data/ycfeng/tmp/pr33-w10-unit-owned-initial.log`. First repaired run: `/data/ycfeng/tmp/pr33-w10-unit-owned-second.log`, 161 PASS in 3.46 s. After adding the exact heterogeneous reporting-stage assertion, final evidence: `/data/ycfeng/tmp/pr33-w10-unit-owned-final.log`, 161 PASS in 3.42 s. `git diff --check` passed for these files and the non-dummy campaign.

## Causes, changes, and retained criteria

| File | Initial failures | Established cause | Fixture change and retained oracle |
| --- | --- | --- | --- |
| `tests/unit/test_mixed_layer_decode_ffn_scheduling.py` | 7 | Stub `_get_communication_time` rejected the final `include_stage_owned` keyword | Accept explicit keyword; preserve real normally constructed predictor and all layer branch/norm exclusion assertions |
| `tests/unit/test_pd_decode_moe_layer_accounting.py` | 6 | Scheduler stub exposed the retired corrected-metrics adapter name/signature | Supply `_create_prefill_corrected_execution_time_for_metrics(batch, stage_id, original, actual, start)` and return original timing; retain 94-layer PP2 progress, EP1/2 progress, duplicate active request once-only, completed request exclusion, and replay fail-fast assertions |
| `tests/unit/test_pdaf_prefill_model_time.py` | 2 | Predictor stub asserted `num_layers == 1` although W08 now requests a complete stage for final reporting | Build a real ordered `StageExecutionTime` with independent physical attention values for full-stage requests; preserve exact 0.481 s component-model-time assertion, timestamp-residue and synchronization-wait checks. Newly assert reported stage IDs `(0,1)` and physical attention durations `[1.25,3.75]` ms |

The prefill full-stage reporting fixture excludes FFN because the call requests `include_ffn=False`; its existing component ledger continues to own the independently supplied completed work. Numerical assertions were not loosened, deleted, or replaced by predictor-derived golden values.

## Limits and remaining ownership

This focused run validates the affected interfaces and original regression criteria. It does not establish the entire CPU suite is green. The parent owns the complete suite/fidelity results and baseline-failure audit. The non-dummy campaign already passed after production freeze (8 PASS in 36.78 s) and requires no rerun for these fixture-only changes.
