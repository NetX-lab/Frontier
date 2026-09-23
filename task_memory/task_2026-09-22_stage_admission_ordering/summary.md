# Stage admission ordering under pipeline parallelism — Summary

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Created at P4: fix, tests, P0–P3 and the vLLM comparison complete under D-9. |

## Overview

W9-01: with `attn_dp > 1` and `num_pipeline_stages > 1`, a busy lane's queued
ticket at the head of a stage's ready FIFO refused another lane's runnable
batch. MoE runs drained with requests unfinished (admission deadlock). Dense
runs completed but started the lanes one forward apart.

The fix (plan D-1, option B) changes one predicate in
`StageExecutionContext.try_acquire`. A full-stage ticket is refused only by an
EP wave queued ahead of it. EP waves keep the strict FIFO-head rule. The
admitted ticket leaves the FIFO by `remove(ticket)`.

## Deliverables

| Item | Path / commit |
| --- | --- |
| Rule and P2 tests | `dac4e69`: `frontier/scheduler/replica_stage_scheduler/stage_execution_context.py`, `tests/unit/test_stage_execution_context.py`, `tests/unit/test_shared_forward_group_admission.py`, `tests/unit/test_mixed_layer_decode_ffn_scheduling.py`, `tests/integration/test_stage_admission_pipeline_lanes.py` |
| Case matrix | `tests/e2e/stage_admission_matrix.py` (`a054d87`, `5ade853`, `aeeca93`) |
| vLLM comparison | `tests/comparison/stage_admission_pp/{vllm_burst_driver.py,run_vllm_worker.sh,compare_lanes.py}` (`799ccb4`, `a1b9819`, `aeeca93`) |
| Test report | `test_report_2026-09-23_stage_admission_ordering.md` |
| Calibration case | `calibration/stage_admission_case_001/` (manifest, inputs incl. `groundtruth_overlay.patch`, two vLLM runs, `analysis/`) |
| Evidence | `evidence/` (base negative controls, G2 comparisons, path-T explanation, Step 9 probe, co-execution decomposition script) |
| Branch / PR | `fix/stage-admission-ordering`, draft PR https://github.com/NetX-lab/Frontier/pull/36 |

## Validation (observed)

| Criterion | Result |
| --- | --- |
| C1 | 18 base admission deadlocks (G3a 10, G3b 6, G7 2) complete with requests and tokens conserved |
| C2 | 50/50 unchanged cases byte-identical (30 release recipes, every `PP=1` cell, G5) |
| C3 | 6 T cases identical. The other 8 change start times only: same batches, same component durations, no self-overlap, `peak_lanes ≤ attn_dp`. All 4 witnesses have a strictly larger co-execution fraction (D-9). |
| C4 | `tests/unit` and `tests/integration`: no regression, no new failure, skips and collection errors unchanged |
| C5 | One predicate plus docstrings; no flag, field, fallback, wake-up or special case |
| C6 | Step 9 probe shape MoE `attn_dp=2, moe_ep=2, PP=2` completes 6/6 (base drains) |
| C7 | vLLM DP=2/PP=2 on 4×H800, run `sa-pp-20260923b`: 50 MATCH, 0 MISMATCH, 2 INFORMATIONAL (dense V5, D-9). MoE: 26/26 rows MATCH. Dense: V1–V4 MATCH in every round. The base fails its negative controls: MoE deadlock, dense pairing and co-start. |

Decisions taken during execution:

| Id | Decision |
| --- | --- |
| R-7 | The vLLM ground truth uses the four-argument `topk_softmax`, applied as a recorded overlay patch. The checkout is unchanged. |
| R-8 / D-9 | Witnesses are judged by co-execution fraction. V5 gates MoE only. |

## Open and deferred work

- PR 35 (`fix/issue26-correctness-pr`) merges this branch forward after it lands and reruns G3b as the composition check with W3. Only then does Step 9 resume (C6).
- `PP=3` with `attn_dp=2` stays rejected by the node-size rule on the default backends (W9-02), outside this fix.
- vLLM-BS: the fork's Python `topk_softmax` still passes five arguments. So does its test `tests/model_executor/test_enabled_custom_ops.py::test_topk_softmax_wrapper_forwards_renormalize`. The four-argument form was applied only as this case's overlay patch.
- Dense per-rank duration variance, which vLLM shows and the dummy predictor lacks, is an execution-time-model topic. It is not part of admission.
