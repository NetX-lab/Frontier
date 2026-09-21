## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated the multi-size H200 primitive sweep, kernel identities, heldout prediction, and existing backend parameter scope. |

# D019 communication sweep verification

**PASS:** 8 H200 ranks, 80 operation rows, 560 event samples and 40 unique kernel→launch→message-range joins. All expected group, byte, implementation and numerical-sum checks pass. Every real-size signature is RING_LL; every dummy signature is the separate custom one-stage kernel. Profiler times are excluded from fitting. The subsequent independent timing-context phase failed; no claim of an overall job PASS is made.

Execution: root-managed `yc26-h200-d019-profiles-20260908-03`, H200 `step_main`, conda `vllm-bs-0.10.2`, Python 3.10.16, Torch 2.8.0+cu128. The exact image, complete GPU command, input sizes, repetitions and CPU fitting command are recorded in `analysis/d019-communication-sweep.md`; raw artifacts and frozen executable scripts are in `analysis/h200-d019-profiles-03`. Analysis used `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, Python 3.13.13, with `PYTHONDONTWRITEBYTECODE=1` from the active worktree.

Criteria: exact TP/DP/EP identity; positive finite samples and consistent summaries; no missing kernel correlation; 16 MiB excluded from fitting; heldout error below 10%; explicit parameter scope and no application outside authorized work. The real 8-MiB signature has a different block thread count from larger sizes, disclosed in the report; it retains the same NCCL RING_LL family.

Observed fixed-bandwidth single-parameter fit: aggregate floor 29.3087326378 us; existing AR launch-field candidate 4.3847887730 us/step. Training sizes are 8/12/24/32 MiB. Training maximum absolute error is 4.866699%, leave-one-training-size-out maximum 6.488932%. Heldout 16 MiB: predicted 99.2137993045 us, actual 99.8570024967 us, absolute error 0.6432031923 us, signed relative error -0.644124%. This passes primitive calibration only.

The same candidate was passed directly to the current `collective_sim_core.intra_server_model.estimate_intra_server_ms` through a replaced immutable scenario config, without modifying any production file. Actual estimator outputs:

| Scope | Current ms/48 layers | Candidate ms/48 layers | Result |
| --- | ---: | ---: | --- |
| TP4 attention allreduce, 16-MiB tensor | 17.8994432000 | 4.7622623666 | Matches the independent fitted prediction |
| Ideal EP8 all-to-all, one phase | 2.1253418667 | 2.1253418667 | Exactly unchanged |

Receipt: `analysis/d019-communication-parameter-scope.json`. The source/config path and graph-strip conditions are documented in the main sweep report. This field does not affect all-to-all; the bandwidth/efficiency values remain unchanged.

Committed verification: `882ac05b` contains only the benchmark sweep increment. `git show 882ac05b:tests/performance/issue26_h200_collective_microbenchmark.py` was compared byte-for-byte with `analysis/h200-d019-profiles-03/issue26_h200_collective_microbenchmark.py`; PASS. The final direct check printed `PASS committed GPU snapshot; actual estimator candidate; ideal A2A unchanged`. Other workers' edits were preserved.

Remaining numerical limit: the candidate 4.762262-ms equivalent is 13.996–16.580% below the older low-density in-context attention AR totals. That separate communication-context 10% gate remains **FAIL** and was not fitted away. The fixed-TP4 estimate is an amortized eager collective floor, not independently identified CPU launch time, and does not qualify smaller custom-AR batches or other group sizes. Root's fresh first-forward validation and independent review remain pending; no production parameter was changed by this lane.
