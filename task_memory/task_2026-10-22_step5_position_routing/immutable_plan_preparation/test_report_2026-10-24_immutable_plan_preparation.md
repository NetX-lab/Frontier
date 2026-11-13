# Immutable PLAN preparation diagnosis

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record frozen-source RED, paired controls and live replacement checks. |

**Task:** D29-IMMUTABLE-PLAN-PREPARATION

**Session:** `ses_e963098ecffebqCOtdysKGY6Xe`

## Scope and command evidence

Read-only source: `/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes`, HEAD `56529fcd`, code `2f2d9339`. Records worktree starts at `276e6520`. No production/test edits, delegation, training, full CLI, GPU, network or trace-encoding investigation.

Executed seconds-scale RED command:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_immutable_plan_preparation/probe.py red > /data/ycfeng/tmp/d29_immutable_plan_preparation/red-executed.log 2>&1
```

Exit1 is the intended assertion: 192 warm key serializations, eight distinct keys,184 repetitions. It is not a missing-method/correctness failure.

All exact control commands, environment, outcomes and hashes are in `evidence/evidence.json`: `compare.py {current-a3,key-a,current-b,key-b}`, `live_checks.py {live-current4,live-key}`, then `verify.py`. Scratch root is `/data/ycfeng/tmp/d29_immutable_plan_preparation`; runner snapshots and logs accompany this report. Python3.10.6, existing `frontier-py310`; no conda invocation.

Every pricing arm uses ONE explicitly labeled single-role diagnostic fixed-file-byte adapter:80 byte requests/wave, zero warm physical reads. This is not the P/D/run snapshot implementation. Relocation precedes CLI configuration parsing; source, original fixture and cache bytes/metadata remain unchanged.

## Observations

| Warm work/wave | Current | Derived-key reuse |
| --- | ---: | ---: |
| Plan / Tensor constructors | 0 / 0 | 0 / 0 |
| Fresh CollectivePlanCost constructors | 64 | 64 |
| Key encodes / key calls | 192 / 192 | 0 / 192 |
| JSON dumps / loads | 992 / 1520 | 800 / 1136 |
| Paired duration/count lookups | 288 | 288 |

Cold construction:8 plans,14 tensors,64 costs. Existing preparation already retains eight complete plans; H1 does not justify another plan cache.

Two independent pairs,12 unprofiled warm waves/arm: medians **0.167724→0.128676s** and **0.167579→0.129510s**, saving **0.039048/0.038068s (23.28/22.72%)**. Cold0.908–1.006s; setup/import/loading, profiling, acceptance/emission and mutation checks are excluded.

All208 events, ledger bytes, proofs, costs, ordering and full keys equal the retained reference. Payload SHA256: `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`. Event/ledger hashes and complete-byte comparisons are recorded. All144 operator proofs remain fresh. Ten live checks agree exactly across arms: row/estimator/identity replacement, duration/count changes, full context/buffer replacement, fresh layer61 proofs and measured source7000 selection. Public dictionaries remain detached; ordinary frozen-field mutation fails; `replace()` produces a fresh full key.

## Recommendation and limits

H2 is material: lazily retain only the existing full serialized string on `CollectivePlan`; keep `to_dict()` fresh. Fields are immutable typed tensors, tuples and JSON strings, but sizes/modality/context/buffers vary between plans: never reuse by phase or layer alone.

Source refs: `collective_plan.py:24–74,86–92`; `moe_routing_workload.py:155–246`; backend `collective_sim_cc_backend.py:803–880`; fresh numerical/proof work `sklearn_moe_execution_time_predictor.py:135–205`. Candidate regression seam: `tests/unit/test_step5_moe_pricing.py:58–91`. Estimate only: source8–12 gross/4–6 retained lines; tests35–55 gross/retained. **No allocation assumed.**

This is the CPU-fixture collective branch with reconstructed cached manager/exact tables, not native qualification or constructor wiring. Scratch harness failures are classified separately in evidence. Remaining live work still dominates. Rebase and repeat after snapshot delivery because input shape may change; no full-run improvement is claimed.
