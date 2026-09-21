## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Independently recomputed the communication fit, checked all raw samples and kernel joins, and verified the existing backend field scope. |

# Independent communication fit review

Reviewer: `/root/moe_profile_repair`. Verdict: **PASS for the bounded first-forward parameter experiment**, not full-case communication calibration. No backend or production configuration was edited by this reviewer. Root may run the prepared first-stage hard stop with the existing case-local field and a fresh collective cache.

## Claims and direct evidence

| Claim | Label | Independently inspected evidence | Result / limit |
| --- | --- | --- | --- |
| Timing data are complete | Evidence | `h200-d019-profiles-03/runtime/communication/rank_0.json` through `rank_7.json` | All 560 event samples positive finite; stored median/min/max and block/48 conversion reproduce exactly; correctness receipts PASS. |
| Real-lane runtime uses the fitted NCCL family | Evidence | Corresponding eight kernel-signature traces | 40 unique correlation-to-launch and launch-to-user-range joins; real ranks 0–3 RING_LL, dummy ranks 4–7 custom one-stage. Signature durations excluded from timing inputs. |
| 16 MiB is held out | Evidence | Independent raw-data fit uses only [8,12,24,32] MiB direct PyNCCL rank 0–3 medians | No 16 MiB sample enters the coefficient fit. |
| Existing AR field can represent the scoped fit | Evidence | Actual `estimate_intra_server_ms` call, TP4 and 16 MiB | 0.3729050666666667 -> 0.09921379930445354 ms per call. |
| EP A2A is unchanged | Evidence | Actual same estimator call, EP8 and 16 MiB; source kind branch | Exactly 0.044277955555555554 ms before and after. |
| The fixed term is CPU launch overhead | Unknown | Eager CUDA events include device work, rank/submission gaps and potentially exposed host gaps | Cannot identify CPU separately; dividing by six is an existing parameterization, not six observed CPU launches. |
| The full forward communication gap is closed | Unknown | Existing in-context intervals remain 14–17% above this scoped prediction | Primitive holdout PASS does not close in-context or full-forward gates. |

The independent result is `d019-communication-fit-review.json`. It reproduces slope 4.369066666666667 us/MiB, aggregate floor 29.30873263778686 us, and per-step field 4.384788772964477 us. Training maximum absolute error is 4.866699147884317%; heldout prediction 99.21379930445353 us versus measured 99.85700249671936 us gives absolute error 0.64320319226583 us and signed error -0.6441242738955248%.

## Source and first-forward scope

Inspected `frontier/cc_backend/backends/collective_sim_cc_backend.py` `_build_scenario`, which forwards the existing field, and collective-sim `python/collective_sim_core/intra_server_model.py` `estimate_intra_server_ms`, `_estimate_steps`, and `_estimate_intra_bytes_per_rank`. For intra-server TP4 AR, six steps and 1.5 times the logical input bytes reproduce the published equation. The AR and A2A fixed-cost variables occupy separate kind branches. Fixing efficiency at 0.8 therefore preserves the existing ideal EP abstraction; changing efficiency instead would alter A2A and is not qualified by this TP4 sweep.

The current config is one replica, H200, attention TP4/DP2, MoE TP1/EP8, one pipeline stage, CUDA graphs disabled. The first source batch is request 0 with 4096 prefill tokens, giving 4096 * 2048 * 2 = 16 MiB. The current-task first-forward source ledger (`/data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime/frontier_stage_batch_ledger.jsonl`) records attention AR as the only nonzero AR field in that source batch; DP input/output, MLP and shared-expert AR are zero. This ledger is used to inspect reached operation identity, not as fresh measured ground truth. `BaseExecutionTimePredictor.predict_dp_moe_allreduce_times` returns explicit zeros for the retired DP communication domain. The reached shared MoE path has MoE TP1 and no shared expert; its dispatch/combine are ideal EP collectives.

`SklearnExecutionTimePredictor._should_strip_collective_sim_allreduce_launch_overhead` requires KERNEL_ONLY and captured decode. This pure prefill CUDA_EVENT case does not strip the calibrated field. The expected 48 attention AR calls total 4.76226236661377 ms, down from 17.8994432 ms. This is an operator-model change expectation; the independent bounded simulator rerun remains root-owned.

A counterexample prevents a broad approval: later small custom-AR decode calls also receive the same backend field, while their kernel family and size regime were not fitted. The value must remain a first-forward experiment until later reached shapes/families are checked. Current ideal-versus-naive EP protocol differences remain an explicitly deferred approximation under D020.

## Verification execution and failure record

CPU environment: conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13, executable `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`; `PYTHONDONTWRITEBYTECODE=1`. Checks ran as direct Python heredocs from the active worktree. The exact independent fit can be reproduced with the command below. The scope result additionally calls the actual `collective_sim_core.intra_server_model.estimate_intra_server_ms` using TP4/allreduce and EP8/alltoall scenarios with identical 16 MiB payload, 450 GB/s, efficiency 0.8, latency 0.5 us, and the two field values 50 and 4.384788772964477.

The initial review checker incorrectly searched only `cuda_runtime` launch records and failed an assertion. Raw trace inspection showed NCCL `cuLaunchKernelEx` records under `cuda_driver`; the checker was corrected to accept both CUDA API categories and all 40 joins passed. This was a checker assumption, not a source or measurement failure. No raw artifact was changed.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PYCODE'
import json, statistics
from pathlib import Path
p = Path('task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-profiles-03/runtime/communication')
ranks = [json.loads((p / f'rank_{i}.json').read_text()) for i in range(4)]
values = {m: statistics.median(s['per_call_ms'] * 1000 for r in ranks for row in r['measurements'] if row['operation'] == f'tp_pynccl_{m * 2**20}_bytes' for s in row['samples']) for m in [8, 12, 16, 24, 32]}
slope = 1.5 * 2**20 / (450000 * 0.8)
floor = statistics.mean(values[m] - slope * m for m in [8, 12, 24, 32])
print(floor, (floor - 3) / 6, 100 * (floor + slope * 16 - values[16]) / values[16])
PYCODE
```

Pending: root-owned first-forward rerun and in-context gap analysis. Newly discovered production issues: none. Broader decode/general-group-size qualification remains outside this bounded review.
