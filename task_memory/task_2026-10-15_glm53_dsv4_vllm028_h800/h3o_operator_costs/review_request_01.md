# Review request: H3o-(a) proposal for operator-level DeepSeek-V4-Flash costs in Frontier

Report language: Chinese, with English technical terms and identifiers kept as is.

## Working root and scope

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (branch of the Frontier LLM inference simulator), HEAD `0ce31851`.
- Primary target (design review, not a diff review): `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md`.
- Task records: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/requirements.md` (items 26 and 27 are the governing decisions; read items 20-27 for context), `plan.md` (row H3o and the dependency line under "Current execution route"), `progress.md` (H3o rows and the 2026-10-26 records), `tickets/05-dsv4-flash-family.md`, `tickets/06-communication-measurement.md`, `tickets/09-predictor-bundle.md`, `tickets/14-decision-d-routing.md`.
- Repository rules: `AGENTS.md` (Development Gates, Module Size, vLLM parallel semantics) and `docs/adr/0002-operator-costs-not-measured-step-tables.md`.

## Background the reviewer needs

The project owner ruled (requirements item 26, D-NO-STEP-COST-SHORTCUT) that a table of whole steps measured in real vLLM serving runs, looked up by batch composition (`dsv4_step_cost.json`, read by `frontier/execution_time_predictor/dsv4_step_cost_input.py`), is not a cost model. It is diagnostic only and can never show that Frontier predicts. Frontier must compose each step's time from operator profiling CSVs (through trained predictors) plus the communication backend. Item 27 (D-OPERATOR-LEVEL-ROUTE) makes this operator-level route the next step, and requires a CPU-only proposal (scope, file list, GPU jobs, time) for review before any implementation or GPU job. The design under review is that proposal.

Target cases: P0 = H800, PD disaggregation, attention TP8, DP2, EP16 (16 prefill GPUs + 16 decode GPUs), prompts 6144 tokens, outputs up to 1024 tokens. P2-32 = H200 with the same topology. P2-64 = H200, DP4, EP32 (32 + 32 GPUs). Native step times (diagnostic): decode-only step about 19.3-24.7 ms; a first-tail step (one request with its last prompt token, after the KV transfer) with decodes about 21.4-26.0 ms; a first-tail step alone about 61-69 ms.

## Code and data the design relies on (please check its claims against them)

- V4 execution path: `frontier/execution_time_predictor/deepseek_v4.py` (`timing_operator_times`, `predict_source_execution`, `predict_structural_execution`, `predict_step_cost_execution`, `read_isolated_scope`).
- Dispatch: `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` (about lines 95-115 and 1380-1410), `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` (about lines 590-680, 1437-1450, 3570-3580, 4050-4056, 7560-7700).
- Template predictor: `frontier/execution_time_predictor/dsa_predictor.py`, `frontier/attention/dsa_features.py`, `frontier/execution_time_predictor/gdn_predictor.py`.
- Kernel gap: `frontier/execution_time_predictor/kernel_gap.py`, `frontier/profiling/kernel_gap/`.
- Communication: `frontier/cc_backend/backends/measured_cc_backend.py`, `frontier/operators/families.py` (V4 comm operator specs, about lines 760-830), `data/profiling/network/h800_dgx/summary.csv` and `glm_p0_ep16_20261021b/summary.csv`.
- V4 profiler: `frontier/profiling/attention/dsv4_attention_plan.py`, `dsv4_attention_runner.py`, `dsv4_attention_rows.py`, `dsv4_attention_main.py`; MoE profiler `frontier/profiling/moe/`.
- Existing data (readable): `/data/ycfeng/tmp/glm53_dsv4/w6_dsv4_profiles/attention_kernel_only.csv`; `/data/ycfeng/tmp/glm53_dsv4/v4_h800_q1_b36c2b9_results/Q1_6143_7166/compute/h800/deepseek-ai/DeepSeek-V4-Flash/attention_kernel_only.csv`; `/data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144/compute/h200/deepseek-ai/DeepSeek-V4-Flash/`; MoE acceptance `task_memory/task_2026-10-23_v4_remaining_measurements/moe/results/primary_acceptance.json` and the design `task_memory/task_2026-10-23_v4_remaining_measurements/moe_grid_design.md`; earlier proposal `task_memory/task_2026-10-23_v4_remaining_measurements/proposal.md`.
- Native cases: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic/h4a4_driver.py` (topologies, KV blocks, maximum concurrency).

## Questions to answer

1. **Method.** Does the proposal actually compose step time from operator profiling, with no path that reintroduces measured whole steps (or measured per-layer totals that hide the batch dependency) as the cost model? Is the use of the step-cost table limited to a diagnostic cross-check?
2. **Operator inventory.** Is the inventory in section 3 complete for a V4 forward on these roles (PREFILL and unified DECODE), including MTP if vLLM runs it in these cases, the hash-routing layers, the first and last layer, and the sequence-parallel (SP) sharding? Are any operators double counted (for example `moe_shuffling` versus `moe_grouped_gemm`, or the per-scope attention sum versus the layer scope)?
3. **Overlap and graph replay.** Is pricing a layer from a KERNEL_ONLY `dsv4_attention_layer` scope computed as the union of kernel intervals sound for the auxiliary-stream overlap? Is KERNEL_ONLY plus `kernel_gap.csv` a sound model of breakable CUDA Graph decode with an eager attention segment? What should be measured instead, or in addition?
4. **Eager and first-tail steps.** Is CUDA_EVENT per-scope data an adequate basis for eager steps that may be host-launch bound (the 61-69 ms first-tail alone)? Should the CPU-overhead tables be in scope now?
5. **Data adequacy and GPU jobs.** Are the G1-G5 / G1h-G5h grids sufficient for the serving batches (decode batch up to about 18 on H800 and about 38 on H200, contexts 6143-7168, prefill 6144 per DP rank, SP local rows, EP gathered rows), and are the GPU-hour estimates plausible? Is anything measured that is not needed?
6. **File plan.** Is the file list (section 5) the smallest complete change consistent with existing mechanisms (DSA/GDN family predictors, typed operator contracts in `linear_op.csv`, `moe.csv`, measured CC backend, kernel gap)? Is "no new config field" actually achievable? Is deleting `read_isolated_scope` correct?
7. **Acceptance.** Are the acceptance criteria in section 1 (held-out median 5% / p90 10% per scope, typed coverage errors, unchanged other paths, diagnostic cross-check) the right gates for this step? What is missing?
8. **Risks and sequencing.** Anything in section 8 (order) or section 9 (risks) that is wrong or missing, and anything that should be decided by the project owner before implementation.

Please give concrete corrections to the design where you find defects, so that the author can revise the document. Do not edit any file.
