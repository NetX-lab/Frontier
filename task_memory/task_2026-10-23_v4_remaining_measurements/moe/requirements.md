# W128-V4-MOE-PACKAGE requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded the root-owned corrected four-point design and bounded CPU package allocation. |

## Original Request

[Original Request] Implement and CPU-verify a fresh immutable single-H800 local-MoE measurement package in `v4-moe-local-run`, branch `feat/v4-moe-local-run`, base `56e9405d`. Stop before sealing or submission. No delegation, GPU/platform/auth/quota/cloud/network/Docker operations, parent edits, native/C26 changes, core changes, collector/consumer/admission changes, or original-package changes.

[Original Request] Own only `build_package.py` and `profile_plan.json` as maintained non-core source/config, at most two files and 300 gross lines by git diff --numstat. Generated immutable inputs and task records are counted separately. Reuse existing producer, launch, worker and seal mechanisms; preserve runtime source bytes.

[Original Request] Measure `deepseek-ai/DeepSeek-V4-Flash`, physical TP1, EP16/rank0, local16/global256 experts, M=12288/6152/32/16, max_tokens16384, balanced, one distribution sample, balanced_topk, disable_ray, num_gpus1; separate cuda_event and record_function invocations and outputs. One H800/codesign/non-preemptible,16 CPU,mem_gb128,1800 seconds, at most0.5 GPUh; locked native0.28 image sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635.

[Decision] Read the actual root corrected requirements.md, plan.md and moe_grid_design.md in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/`. The separate local_measurement_plan.md is historical. Root decisions D-V4-QUEUE-PARALLEL-WORK and D-V4-H800-RJOB-AUTHORITY cover preparation; root owns actual image/SDK gates, sealing and submission.

[Decision] Only the existing grouped-GEMM envelope is a candidate cost scope. Retain all original columns and traces. The gate receives M rather than native L_d, shuffle overlaps the grouped-GEMM envelope,6152 has mixed-prefill/dummy provenance, and rank0/random packed weights do not prove rank/time/weight/graph/serving equivalence. No admission or simulator consumption.

[Source observation] Existing SDK ResourceConfig.mem_gb=128 maps to memory_in_mb=128000 (decimal MB), not128 GiB; preemptible defaults False. Root must verify the actual SDK/platform allocation.

## Acceptance

CPU exact four rows per method; supported selectors and inspected backend semantics; fresh generated launcher/worker entry without dependency preloads; source snapshot and model configuration pins; native source/symbol and inherited image-binary pins; byte-identical runtime originals; immutable manifest and unchanged-original hashes; relevant existing tests; explicit source-bound existing77-case matrix. Report stand-in/source verification separately from root's actual image/GPU gates. Freeze source in a scoped commit after CPU verification.
