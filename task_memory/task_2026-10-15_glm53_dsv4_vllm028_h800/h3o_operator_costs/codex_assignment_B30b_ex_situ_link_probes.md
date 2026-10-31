# Task: GPU probes and fit tools for the ex-situ variants F1b, F2 and F3 (stage 4 assignment B30b; build and dry-run only, no submission)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Created from decision 46 (F1-F4 comparison tests) and the ex-situ audit, section 5. |

## Goal

YC asked to test the F1-F4 repair proposals of the ex-situ input audit as diagnostic A10 variants and compare their errors (decision 46 in `requirements.md`). The formal inputs stay as they are; these variants are diagnostics. Three variants need H200 measurements that do not depend on the target cluster size or on the workload's lane mix:

- **F1b, collectives from link parameters.** Price the decode and prefill collectives with `astra_sim_analytical`, whose intra-server and inter-server latency and bandwidth are fitted from (a) one 8-GPU node and (b) a two-rank test with one rank on each of two nodes. The G5h tables (measured on the 16 and 32 GPU target pods) are held-out data only.
- **F2, the dummy-forward run-ahead depth K.** K = 19 was observed in the native P2-64 run; the hypothesized cause is the CUDA launch queue. Measure on one GPU the depth at which a CUDA Graph launch blocks while the GPU is busy, as a function of the graph's kernel count, and predict K from the native per-forward kernel count.
- **F3, KV transfer.** The NIXL fit in use (126.2 Gbit/s, 64.7 us per request) was measured on H800 with the GLM KV layout. Measure NIXL READs on two H200 nodes with the V4 KV layout and refit.

Deliver the probe code, the launch plans, the CPU fit and analysis tools, and a dry run. The lead submits the jobs through the platform after YC approves the GPU cost.

## Approach (lead decisions)

- **D1, one two-node job for F1b and F3** (2 replicas x 8 H200, `step_main`, non-preemptible), with phases run in order inside the job:
  - **Phase A, one node:** each node runs, independently and at the same time, the V4 forward collectives at TP8 DP1 EP8 with the existing `frontier.profiling.collectives.vllm_collective_benchmark`. The grid is generic powers of two, not the workload's token counts: graph tokens per lane 1-64, eager lane tokens 1-8192, logits rows 1-32 (exact lists are yours; keep them powers of two and cover the native ranges in `h3o_operator_costs/c1_coverage.md`).
  - **Phase B, two nodes, one rank per node:** the same benchmark at TP1 DP2 EP2 on the same generic grid, for two rank placements: local GPU 0 on both nodes (same rail), and local GPU 7 on node 0 with local GPU 0 on node 1 (the pair that a flat ring crosses at the server boundary).
  - **Phase C, two nodes:** `frontier.profiling.collectives.nixl_kv_read_benchmark` with eight TP reader pairs and one pair, using the V4 KV region layout of the native H200 P2 runs. Take the regions (count and bytes per block per region) from the native evidence the task already holds, cite the source in the plan, and fail the planner if the layout cannot be established. Grid like the H800 K5 stage (`w5_glm_comm_20261020/launch/stage_k5_nixl_read_tp8.json`): prompt lengths 64-8192 tokens, active pairs 1 and 8, requests in flight that the block pool admits.
- **D2, one one-GPU job for F2.** A probe that captures CUDA Graphs of k kernels on a power-of-two grid (for example 1-4096), holds the stream with a GPU sleep longer than the test, launches each graph repeatedly without synchronizing, and records the host time of every launch. The blocking depth is the number of graph launches accepted before the host launch time rises to the order of the sleep. Also measure plain kernel launches for reference. Decide from the data whether the limit counts graph launches or kernel nodes; do not assume either.
- **D3, reuse.** Reuse the G5h launch and runtime-check mechanism (`h3o_operator_costs/g5h/launch/`: `submit.py`, `runtime_check.py`, the stage JSON format, the pinned image and native checkout) and the existing benchmarks. Add only what a phase runner, a rank placement and the launch-depth probe need. Do not copy those modules; import or extend them. Keep the K5 and G5h stages reproducing their recorded plans byte-identically.
- **D4, fit tools (CPU, under `h3o_operator_costs/b30b/`):**
  - `astra` fit: intra-server (latency, bandwidth) from the phase A rows, inter-server (latency, bandwidth) from the phase B rows, by least squares on the backend's own pricing of each row (payload = the largest rank's input bytes, as the measured backend and `f_variants/f1_astra_per_call.py` use it). Report the fitted values, the residuals per curve, and write them as flat field assignments that `f_variants/run_f1_per_call.sh` and the A10 driver's `--field` option accept. Note: the native V4 dispatch is three grouped `all_gatherv` calls and Frontier prices one call over the summed bytes, and native per-step costs differ between `all_gatherv` and `reduce_scatterv`. Report how well one latency per link fits both, and do not add a per-operation correction factor.
  - NIXL fit: latency + bytes / bandwidth on the eight-pair rows, by the method of `w7_glm_comm_model_20261020/design.md` section 2.8, compared with 126.2 Gbit/s and 64.7 us.
  - Launch-depth analysis: depth against kernels per graph; the native per-forward kernel count of the P2-64 DP dummy forward from the B10 tool (`tests/analysis/v4_collective_kernels.py`) or the runtime census; the predicted K against 19.
- **D5, planning output.** A CPU dry run lists every case of both jobs, estimates each job's wall time from the case counts, repeats and the measured per-call times already in the G5h and K5 tables, plus start-up time, and prints the GPU-hours. Use soft runtime checks where a failure would discard completed measurements.

## Scope and context

- Working root: `/data/ycfeng/Frontier/.worktrees/b30b-ex-situ-link-probes`, branch `b30b-ex-situ-link-probes`, HEAD `b0e568f5`. Mainline is `glm53-dsv4-vllm028`.
- Task records: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/requirements.md` (decisions 43-46), `plan.md`, `test_report_2026-10-31_ex_situ_input_audit.md` (sections 2-5), `h3o_operator_costs/g5h/design.md`, `w7_glm_comm_model_20261020/design.md` section 2.8, `h200_comm_20261021/` (an unsubmitted H200 NIXL candidate with the GLM layout; its launch instructions are retired).
- Platform rules (for the plans, not for you to run): StepMind Python `RJobBackend` with `i-fengyicheng` auth, `step_main` H200 non-preemptible, cloud output under `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/`, the local workspace mounted as the code source. Do not submit, query or stop any job, and do not access the cloud volume.
- Writable paths:
  - `frontier/profiling`
  - `tests/unit`
  - `tests/analysis`
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30b`
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch`
- Exclusions: no change to simulator behavior (`frontier/` outside `frontier/profiling`), to the measured backend, or to the formal A10 inputs. Do not commit; the lead reviews and commits.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `PYTHONPATH=$PWD`; CPU torch through `PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages` when a test imports torch. No GPU is available here. Run commands with `ulimit -v 31457280` and a `timeout`. Use `/usr/bin/grep`. Downloads of earlier runs: G5h `/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_02/download/`, `g5h32_01/download/`; K5 curated curves `/data/ycfeng/tmp/glm53_dsv4/w5_comm_k5/` (read only; do not search `/data/ycfeng/tmp/glm53_dsv4/` recursively).

## Tests

- Unit tests for the new planner parts, the rank placement, the probe's analysis logic, and each fit on synthetic rows with known parameters (the fit recovers them).
- The existing G5h launch tests (`h3o_operator_costs/g5h/launch/tests/`) and the profiling unit tests that cover the benchmarks you touch still pass, and the existing G5h and K5 plans are unchanged (compare plan listings before and after).
- The dry run of both jobs.

## Expected return

- Report language: English.
- Changed and new paths; for each job the stage files, the case list size, the wall-time and GPU-hour estimate with its arithmetic, and the outputs each phase writes.
- The V4 KV region layout used for phase C and its source.
- Each check with its command, expected result and actual result.
- Remaining risks, for example a phase B placement the platform cannot guarantee, and what the job does then.
