## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded Increment 5 per-layer timing, simulator parity, and wall-time evidence. |

# Increment 5 Verification Report — Unified Per-Layer ExecutionTime

## Execution

Environment: CPU host `ycfeng`, `/usr/bin/python` (Python 3.12.3), NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, PyTorch 2.5.1+cu124. No AMD/MI355X device was available.

Focused unit and contract checks:

```bash
python -m pytest \
  tests/unit/test_dense_execution_time_layer_scaling.py \
  tests/unit/test_execution_time_op_times.py \
  tests/unit/test_attention_simulation_output_equivalence.py \
  tests/unit/test_stage_execution_time.py \
  tests/unit/test_attention_query_cache.py \
  tests/unit/test_moe_predictor_layer_id_semantics.py \
  -q -p no:cacheprovider
```

Static checks:

```bash
python -m compileall -q frontier tests
git diff --check
```

The fresh focused command passed 89 tests. Earlier targeted regression groups recorded in the progress ledger passed 45 tests after the metrics-adapter fix, 93 disaggregation/EP/PD-AF tests, and 29 communication tests. Compilation and whitespace checks passed.

Two-worktree fidelity used a clean Increment 4 baseline at `/data/ycfeng/tmp/pr31-inc5-baseline` (SHA `57da36835051d4ec5b512abaa28103e54fd4e380`) and the candidate worktree. The command was:

```bash
python tests/integration/run_scheduler_refactor_fidelity.py \
  --baseline /data/ycfeng/tmp/pr31-inc5-baseline \
  --candidate "$PWD" \
  --output /data/ycfeng/tmp/pr31-inc5-fidelity-20260914-final-1789374105 \
  --workers 1 \
  --case co-location_offline_dense_model_basic_short \
  --case pdd_offline_dense_model_basic_short \
  --case pd-af-disagg_offline_dense_model_basic_short \
  --case co-location_offline_moe_model_basic_short \
  --case pd-af-disagg_offline_moe_model_basic_short
```

Wall-time runs used `/data/ycfeng/tmp/pr31_inc5_walltime_run.py` with three paired attempts per side, the same host and environment, online sequential PDD, two dense requests, prefill 8, decode 4, PP2, attention TP4, and total GPUs 16. Immutable results are under `/data/ycfeng/tmp/pr31-inc5-walltime/{baseline,candidate}/attempt-{0,1,2}/result.json`.

## Criteria and evidence

| Criterion | Expected result | Observed result |
|---|---|---|
| Layer representation | Identity-bearing timings represent one real layer; stage totals are owned by `StageExecutionTime` | PASS; focused layer/stage tests passed and public predictors return `StageExecutionTime` |
| Existing numerical behavior | Request/stage/event/order/timing artifacts match the Increment 4 baseline | PASS; fidelity reported `5 passed, 0 failed` (fresh rerun artifact: `/data/ycfeng/tmp/pr31-inc5-fidelity-20260914-final-1789374105`) with `rel_tol=1e-12`, `abs_tol=1e-9` |
| Dense, MoE, PDD, and PD-AF paths | Requests complete and timing/transfer fields remain structurally valid | PASS; all five fidelity cases completed |
| Attention query reuse | Immutable equal queries reuse numeric `AttentionTime`; layer identity and MoE work remain distinct | PASS; query-cache tests passed |
| Metrics adapter | A stage payload converted to legacy single-layer metrics uses first-layer raw fields and stage-owned communication once | PASS after fixing a detected MoE attention-all-reduce double-counting regression |
| Wall-time run health | All paired attempts complete with identical request/event counts | PASS; baseline and candidate each had 3/3 successful attempts, 2/2 requests complete, 104 events |

Paired wall-time medians (candidate relative to baseline):

| Metric | Baseline median | Candidate median | Candidate/baseline |
|---|---:|---:|---:|
| `init_s` | 2.191065608 | 2.190165941 | 0.999589 |
| `sim_wallclock_s` | 0.009217162 | 0.086823318 | 9.419745 |
| `total_proc_s` | 2.203441358 | 2.280853388 | 1.035132 |
| `event_count` | 104 | 104 | 1.000000 |

The event-loop wall time increased by 0.077606156 s in this small CPU case while initialization changed by -0.000899667 s and total process time changed by +0.077412030 s. These are observed measurements from three paired attempts, not a benchmark gate; no universal performance threshold was introduced.

## Regression and issue resolution

The first fidelity run found one real regression in `co-location_offline_moe_model_basic_short`: the metrics adapter wrote `attention_all_reduce_time=32.0` instead of the baseline `1.0`. The cause was `StageExecutionTime.__getattr__()` summing private per-layer fields before the legacy adapter converted the stage result to a single-layer metrics payload. The fix returns first-layer raw values for private per-layer compatibility names while retaining explicit stage aggregates (`model_time`, public aggregate properties, and `op_times`). The corrected MoE case and the complete five-case fidelity rerun passed.

## Hardware boundary

`SKIP: AMD/MI355X hardware unavailable` — no ROCm execution, real MI355X profiling, `DEVICE_EVENT` writer execution, or benchmark/ground-truth parity is claimed by this report.
