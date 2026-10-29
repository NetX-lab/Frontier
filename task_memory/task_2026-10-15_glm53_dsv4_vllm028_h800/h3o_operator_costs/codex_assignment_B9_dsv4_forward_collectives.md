# Task: DeepSeek-V4 forward collectives for the H200 communication benchmark — the V4 call list, native sequence-parallel ops, the V4 grid, and the G5h launcher with runtime checks (stage 2 assignment B9, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B9 of stage 2 (G5h, [g5h/design.md](g5h/design.md)). |

## Goal

H200 has no collective curves for DeepSeek-V4. G5h measures them with the existing vLLM collective benchmark (`frontier/profiling/collectives/vllm_collective_benchmark.py`) on real serving process groups. Today the benchmark plans only the GLM-5.3 (DeepseekV2-family) forward: `ModelShape.from_hf_config` requires `first_k_dense_replace`, and its `all_gather` / `reduce_scatter` ops call `GroupCoordinator` directly, which never takes vLLM's custom MNNVL kernels that V4 uses.

B9 does five things, all CPU-tested; the lead runs the GPU jobs afterwards:

1. **V4 call list:** a pure planner module for the V4 forward collectives, beside the GLM one.
2. **Native SP ops:** two new op kinds, `sp_all_gather` and `sp_reduce_scatter`, which the benchmark issues through vLLM's own `sp_all_gather` / `sp_reduce_scatter` functions.
3. **V4 grid:** the measured-case selection of [g5h/design.md](g5h/design.md) §4.
4. **Architecture table in `main()`:** the model config's `architectures[0]` selects the shape parser and the plan builder. The GLM plan stays byte-identical.
5. **G5h launch:** the G5h-16 and G5h-32 stage configs, an H200 launcher, and a per-node runtime check with hard and soft gate modes.

## Approach and decisions

### Evidence (lead)

Read [g5h/design.md](g5h/design.md) first. §2 has the native evidence with line references; §3 has the call list; §4 has the grid, the expected bytes and the case counts. Native source: `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm` (commit `2cf0a69`, read-only). Key facts:

- `models/common/ops/sequence_parallel.py`: `sp_all_gather(x)` tries `get_tp_group().device_communicator.custom_all_gather(x)`, then falls back to `tensor_model_parallel_all_gather(x, 0)`. `sp_reduce_scatter(x)` pads the rows to a multiple of TP with `F.pad`, tries `custom_reduce_scatter`, then falls back to `tensor_model_parallel_reduce_scatter(x, 0)`.
- `distributed/device_communicators/custom_all_reduce.py:399-500`: the custom kernels apply up to 2 MiB of all-gather input and 16 MiB of reduce-scatter input at 8 ranks (MNNVL multicast buffer). The MNNVL buffer is created when `CustomAllreduce` is constructed (`_init_mnnvl_buffer`), so `init_worker_distributed_environment` already sets it up in the benchmark. The custom gathers do not depend on CUDA-graph capture state.
- `models/deepseek_v4/nvidia/model.py:804-812` (`_use_sequence_parallel`: PP 1, EP, TP > 1, DP > 1), `:962-967` (per-layer gather before attention and reduce-scatter after it), `:1156-1157` (`sp_shard`, a slice without a collective), `:1196-1197` (the final gather of the mHC tensor `[rows, hc_mult, hidden]`, bf16: `mhc_pre_broadcast_tilelang` asserts a bf16 residual).
- `distributed/device_communicators/all2all.py` `AgRsAll2AllManager.dispatch` / `combine`: one `all_gatherv` of `[hidden bf16, topk_weights f32, topk_ids int32]` and one `reduce_scatterv` over the EP group, with per-EP-rank sizes `ceil(T / TP)` of each lane repeated TP times. This is the shape the GLM `moe_dispatch_all_gatherv` / `moe_combine_reduce_scatterv` calls already use.
- `model_executor/layers/logits_processor.py:118-129` with `platforms/interface.py:1102-1106`: logits use `tensor_model_parallel_all_gather(logits)` (dim −1).

### Planner changes in `frontier/profiling/collectives/vllm_forward_collectives.py`

- Add `sp_all_gather` and `sp_reduce_scatter` to `OPS`, with a comment that they call vLLM's sequence-parallel functions, which try the custom kernels first.
- `call_bytes`:
  - `sp_all_gather` returns the same as `all_gather`;
  - `sp_reduce_scatter` returns `(own rows × row_bytes, ceil(own rows / group size) × row_bytes)`. Its `shard_rows` hold the **unpadded input rows** `T` of each rank, because `T` cannot be recovered from `ceil(T / TP)`.

  Update the `CollectiveCall` docstring for this exception.
- Rename `_ceil_div` to `ceil_div` so the V4 module can import it; update every call site.
- No other change to this module. The GLM plan must stay byte-identical (Acceptance 2).

### New module `frontier/profiling/collectives/vllm_dsv4_forward_collectives.py`

A pure planner, like the GLM module. It reuses `CollectiveCall`, `ForwardScenario`, `ServingParallelism` (including `validate_sequence_parallel_moe`, whose conditions match `_use_sequence_parallel` with the default all2all backend), `ceil_div`, `DTYPE_BYTES`, `TOPK_WEIGHTS_DTYPE`, `TOPK_IDS_DTYPE` and `VOCAB_PADDING_SIZE`. The module docstring cites the native sources above.

- **`DeepseekV4Shape.from_hf_config(config)`**: `hidden_size`, `num_hidden_layers`, `num_experts_per_tok`, `vocab_size` and `hc_mult`. Every V4 layer is MoE (the config has `n_routed_experts` and no `first_k_dense_replace`). Raise when `n_routed_experts` is missing, as the GLM parser does.
- **`dsv4_forward_collectives(model, parallel, scenario)`**: the full forward of one step, in the order of the call list in [g5h/design.md](g5h/design.md) §3. `T` is the lane's tokens and `r = ceil(T / TP)`.

  | Call | Op | Group | Columns, dtypes | `shard_rows` per rank | Calls per forward |
  | --- | --- | --- | --- | --- | --- |
  | `embedding_all_reduce` | `all_reduce` | `tp` | (hidden,), act | `T` | 1 |
  | `sp_attention_all_gather` | `sp_all_gather` | `tp` | (hidden,), act | `r` | layers |
  | `sp_attention_reduce_scatter` | `sp_reduce_scatter` | `tp` | (hidden,), act | `T` (unpadded input) | layers |
  | `sp_final_mhc_all_gather` | `sp_all_gather` | `tp` | (hc_mult × hidden,), act | `r` | 1 |
  | `moe_dispatch_all_gatherv` | `all_gatherv` | `ep` | (hidden, k, k), (act, f32, int32) | `r` | layers |
  | `moe_combine_reduce_scatterv` | `reduce_scatterv` | `ep` | (hidden,), act | `r` | layers |
  | `logits_all_gather` | `all_gather`, dim −1 | `tp` | (padded vocab / TP,), act | logits rows | 1, only when some lane has logits rows |

  Captured flags follow the GLM rule: the six forward calls take `scenario.cuda_graph`; logits are never captured. There is **no DP-sync call**: D2 (c) prices V4 host time as an explicit zero ([g5h/design.md](g5h/design.md) §2). Apply the GLM module's scenario checks: one value per DP lane, at least one token per lane, and uniform tokens under CUDA graphs.

  The final gather's native tensor is `[r, hc_mult, hidden]`. The benchmark issues the bytes-equal 2-D `[r, hc_mult × hidden]` tensor: the custom and NCCL paths depend only on bytes, dtype and contiguity. State this in a comment.
- **`dsv4_plan(model, parallel, grid)`** returns the measured `(scenario, call)` list. The grid keys are:
  - `graph_tokens_per_lane`, e.g. `[1, 2, 4, 8, 16, 24, 32]`: one scenario `graph_t{T}` per value, uniform on every lane, `cuda_graph=True`, no logits rows. Every forward call is measured.
  - `eager_lane_tokens`, e.g. `[6144, 1]`: one scenario `eager_t{a}_{b}…` per **ordered** vector in `itertools.product(values, repeat=DP)`, eager, no logits rows.
    - On a uniform vector, every forward call is measured.
    - On a mixed vector, only the `ep` calls are measured. A TP call's cost depends only on its own lane, and the summary's critical-path maximum over all ranks would mix lanes.
  - `logits_rows_per_lane`, e.g. `1..24` as a list: one scenario `logits_b{b}`, eager, `b` tokens and `b` logits rows on every lane. Only `logits_all_gather` is measured.
  - `calls` (optional): a list of call names. When given, only those calls are kept, in every scenario.

  Raise on unknown grid keys. Do not add other selection modes.

### Benchmark changes in `frontier/profiling/collectives/vllm_collective_benchmark.py`

- **`_issue` ops:**
  - `sp_all_gather`: one tensor of `rows` rows; `lambda: sp_all_gather(tensor)`.
  - `sp_reduce_scatter`: one tensor of `rows` (the unpadded `T`) rows; `lambda: sp_reduce_scatter(tensor)`, so the pad copy is measured as native runs it.

  Import the two functions inside `_issue` (or inside the runner's `__init__`) from `vllm.models.common.ops.sequence_parallel`, never at module load. They use vLLM's TP group, which is the call's group.
- **`communicators()`**: for each group, add the custom-collective state of `device_communicator.ca_comm` when it exists: whether `mnnvl_multicast_ptr` is non-zero, `max_mnnvl_all_gather_size`, `max_mnnvl_reduce_scatter_size`, `max_all_gather_size`, `max_reduce_scatter_size` and `fully_connected`. The lead reads this in `plan.json` to confirm the MNNVL path.
- **`main()` architecture table:** one table maps `architectures[0]` of the model config to the shape parser and the plan builder:
  - `GlmMoeDsaForCausalLM`: `ModelShape.from_hf_config`, with the plan `[(s, c) for s in scenarios_from_grid(parallel, grid) for c in forward_collectives(model, parallel, s)]`;
  - `DeepseekV4ForCausalLM`: `DeepseekV4Shape.from_hf_config` and `dsv4_plan`.

  An unknown architecture raises with the supported names. Everything after the plan is unchanged, including the `--list-cases` print format.
- `summarize`, `forward_totals`, `network_all_reduce_rows` and the timing code stay as they are.

### G5h launch in `task_memory/.../h3o_operator_costs/g5h/launch/`

The pattern is the K5 launch `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/w5_glm_comm_20261020/launch/` (`submit.py`, `node_entry.py`, `stage_k5_tp8_dp2_ep16.json`); its H200 counterparts are the P2 launch fields in `h200_mainline/p2_32_profile/package_02/{config.json, submit_native_diagnostic.py}`.

- **Stage configs:**
  - `stage_g5h16_tp8_dp2_ep16.json`: replica 2, DP2, the full grid of [g5h/design.md](g5h/design.md) §4. Job tag `dsv4-comm-g5h16-01`; output `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/g5h16_tp8_dp2_ep16_01`; `runtime_check_gate: "hard"`.
  - `stage_g5h32_tp8_dp4_ep32.json`: replica 4, DP4, `graph_tokens_per_lane` and `eager_lane_tokens` as in G5h-16, `logits_rows_per_lane: []`, and `calls: ["moe_dispatch_all_gatherv", "moe_combine_reduce_scatterv"]`. Job tag `dsv4-comm-g5h32-01`; output `.../h200/g5h32_tp8_dp4_ep32_01`; `runtime_check_gate: "soft"`.

  Both use the K5 fields with these values:
  - `gpu_type "H200"`, `charged_group "step_main"`;
  - the pinned image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`;
  - `cuda_compat_path "/usr/local/cuda-13.0/compat"`, cpu 64, gpu 8, mem_gb 512, `gpus_per_node` 8;
  - `envs {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}`, `set_ucx_net_devices false`;
  - warmup 5, repeats 30, calls per repeat 10, passes 3, `shuffle_seed 20261028`, `timeout_seconds 2700`, `peer_wait_seconds 1200`, `final_wait_seconds 600`;
  - `model_config`: the absolute path of `h200_mainline/p2_32_profile/package_02/model/config.json`;
  - `engine` from `h200_mainline/p2_32_profile/package_02/engine.json`: TP8, DP2 or DP4, PP1, EP on, bfloat16, async scheduling, default all2all backend;
  - `engine_source` and `grid_source` strings naming those files, C1 §4.5 and g5h/design.md §4;
  - `p2_overlay`: the absolute path of `package_02/overlay.json`;
  - `native_checkout`: the `2cf0a69` checkout path above.
- **`runtime_check.py`** (no steptron import), with two parts.
  - `build_lock(config, config_path)` is called by the launcher at submission. It returns:
    - the Git HEAD of the workspace;
    - the sha256 of each benchmark source: the three collective modules, `runtime_check.py`, the K5 `node_entry.py` and the stage config;
    - the expected `libnccl.so.2` sha256: the `overlay.json` `binary_sha256` entry for `/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2`;
    - the checkout sha256 of `models/common/ops/sequence_parallel.py`, `distributed/device_communicators/{cuda_communicator,custom_all_reduce,all2all}.py` and `distributed/parallel_state.py`.

    It raises when a tracked benchmark source has uncommitted changes (`git status --porcelain -- <paths>`).
  - The node-side check (`python3 runtime_check.py --config C --lock L`) writes `<output_dir>/runtime_checks/<host>.json`, one entry per check: name, path, expected, actual, pass and gate. The checks:
    1. the image's `libnccl.so.2` against the lock;
    2. the mounted benchmark sources and stage config against the lock;
    3. the image's five vLLM files under `/usr/local/lib/python3.12/dist-packages/vllm/` against the checkout hashes. This check is always soft.

    With `runtime_check_gate: "hard"`, a failure of check 1 or 2 writes `<output_dir>/nodes/<host>/failed.json` and exits 1, so the K5 `await_nodes` of the peers stops at once. With `"soft"`, every result is recorded and the exit is 0.
- **Worker command:** one `timeout <timeout_seconds>` covers `runtime_check.py … && python3 -u <K5 launch>/node_entry.py --config <stage config>`. Reuse the K5 `node_entry.py` unchanged. It already does the rendezvous, the node ranks, `torchrun`, the environment record and the final barrier, and its UCX step is off with `set_ucx_net_devices false`. Export `LD_LIBRARY_PATH` (compatibility path first) and `PYTHONPATH` (workspace) as the K5 `worker_command` does.
- **`submit.py`:** the K5 `submit.py` with these differences:
  - assert `(gpu_type, charged_group) == ("H200", "step_main")`, and `gpu × replica ≤ 64` (D-H200-MAINLINE-CONTINUE);
  - write the lock to `receipts/<job_tag>_lock.json` before `spawn_tasks` and pass its path to `runtime_check.py`;
  - otherwise the same `ResourceConfig`: `positive_tags=[gpu_type]`, `extra_requirements=[]`, the cloud mount, the envs of the K5 launcher, `DISTRIBUTED_JOB=true` and `task_anti_affinities=["hard/default"]` for replica > 1, default `custom_resources`, non-preemptible, `spawn_tasks(..., use_image=True, code_mount_point=workspace)`, `EXP_ID` = job tag, and receipts at start and exit.

  Codex does not run `submit.py`.
- **Dry runs:** save `python -m frontier.profiling.collectives.vllm_collective_benchmark --config <stage> --list-cases` output as `g5h/launch/dry_run/g5h16_tp8_dp2_ep16.txt` and `g5h32_tp8_dp4_ep32.txt`.
- **Launcher tests:** `g5h/launch/tests/test_runtime_check.py`, run with pytest from the working root. They use temporary files:
  - a lock built from a temporary Git repository (its own `git init` with `-C <tmp>` and no exported `GIT_DIR`), including the uncommitted-source refusal;
  - hash match and mismatch for each check;
  - a hard failure writes `failed.json` and exits 1; a soft failure exits 0; the vLLM file check never fails the node.

### Tests in `tests/unit/`

- **New `test_vllm_dsv4_forward_collectives.py`.** Use a V4 config dict with the real values: hidden 4096, 43 layers, `n_routed_experts` 256, k 6, vocab 129280, `hc_mult` 4.
  - The planned calls and their order.
  - Calls per forward 1/43/43/1/43/43/1.
  - Captured flags.
  - The absence of a DP-sync call.
  - The case counts 82 (TP8 DP2 with the G5h-16 grid) and 46 (TP8 DP4 with the G5h-32 grid).
  - Per-lane `call_bytes` against these literal values (they restate [g5h/design.md](g5h/design.md) §4). Check lane 0 (rank 0) and lane 1 (rank 8):
    - embedding all-reduce: graph `T × 8192` for each graph `T`; eager 8,192 and 50,331,648;
    - SP attention all-gather: input 8,192 (T 1-8), 16,384, 24,576, 32,768; eager 8,192 and 6,291,456; output 8 × input;
    - SP attention reduce-scatter: input `T × 8192` (eager 50,331,648 and 8,192), output `r × 8192`. Also the padded input bytes = output × 8 (65,536 for T = 1), which the test computes from the output;
    - SP final mHC all-gather: input 32,768, 65,536, 98,304, 131,072; eager 32,768 and 25,165,824;
    - logits all-gather: input 32,320 × b, output 258,560 × b;
    - EP dispatch: input `r × 8240`; on `eager_t6144_1`, rank 0 input 6,328,320 and output 50,692,480, rank 8 input 8,240 and the same output;
    - EP combine: graph input 131,072 × r (EP16) and 262,144 × r (EP32); eager 100,663,296 (6144/6144), 50,397,184 (6144/1, both lanes), 151,060,480 (6144/6144/6144/1 in EP32), output `r × 8192` of the own lane.
  - The rejections: a missing `n_routed_experts`, an unknown grid key, an unknown `calls` name, non-uniform graph tokens.
- **Existing `test_vllm_forward_collectives.py`.** Add `call_bytes` cases for the two new ops.
- **`_issue` of the SP ops, in the existing benchmark test module that fakes the runner** (`test_collective_benchmark_helpers.py` or `test_collective_kernel_evidence.py`; follow its fakes). Fake `vllm.models.common.ops.sequence_parallel` in `sys.modules` and check:
  - `sp_all_gather` receives an `[r, columns]` tensor;
  - `sp_reduce_scatter` receives the unpadded `[T, columns]` tensor;
  - neither goes through `group.all_gather` / `group.reduce_scatter`.
- **Architecture table.** A unit test that `main()` (or the function it calls) selects each table entry, and raises for an unknown architecture.

### Not in scope (the lead records these deferrals)

- B10 (the native per-forward kernel table and the comparator against `kernels.jsonl`).
- Simulator entry 8: the grouped dispatch, the SP op mapping with the unpadded reduce-scatter key, PDD admission, and endpoint CommOperatorSpecs.
- Any GPU run; any change to the K5 launcher, the GLM plan, or the timing method.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `0faf1aae`).
  - Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
  - The lead edits task records under `task_memory/`. Codex writes only inside `h3o_operator_costs/g5h/launch/` there.
- **Writable paths:**
  - `frontier/profiling/collectives/`: `vllm_forward_collectives.py`, `vllm_collective_benchmark.py`, and the new `vllm_dsv4_forward_collectives.py`;
  - `tests/unit/`: the new V4 test module and the existing collective test modules named above;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/`: the stage configs, `runtime_check.py`, `submit.py`, `dry_run/`, `tests/`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b9/`.
- **Restrictions:**
  - No Git commands that change state in this worktree; the lead commits. `git diff`, `git status`, `git rev-parse` and `git archive` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001); a test that needs a repository creates its own in a temporary directory.
  - No GPU, network, job submission or package installs. Do not run `submit.py`. No deletions or moves.
  - Python 3.10 syntax for the simulator modules and tests; the launcher files run under the image's Python 3.12 and must also import under 3.10. No torch, vllm, cuda or steptron imports at module load, except `steptron` in `submit.py`.
  - Keep modules at or below 2,000 lines.
  - No magic literals in code paths: values that vary by model, hardware or stage come from the model config, the engine or the stage config. No guards for states the code cannot reach; no second path for something the code already does.
- **Read-only data:**
  - the native checkout above;
  - the K5 launch directory and its `dry_run/` files;
  - `h200_mainline/p2_32_profile/package_02/` (`engine.json`, `config.json`, `overlay.json`, `model/config.json`, `submit_native_diagnostic.py`);
  - [c1_coverage.md](c1_coverage.md) §4.5;
  - [g5h/design.md](g5h/design.md).
- **Environment:**
  - `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vllm);
  - `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`;
  - `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`;
  - `ulimit -v 31457280` and a `timeout` on every command.

## Acceptance

1. **V4 plan equals the design.**
   - `tests/unit/test_vllm_dsv4_forward_collectives.py` passes, with the literal byte values above.
   - The two V4 dry runs print `cases 82 passes 3 measurements 246` and `cases 46 passes 3 measurements 138`.
   - Report a per-call table of the dry-run bytes (rank 0) next to [g5h/design.md](g5h/design.md) §4.
2. **GLM plan unchanged.** `--list-cases` on `w5_glm_comm_20261020/launch/stage_k5_tp8_dp2_ep16.json` and `stage_k5_tp8_dp4_ep32.json` is byte-identical (`cmp`) to `w5_glm_comm_20261020/launch/dry_run/k5_tp8_dp2_ep16.txt` and `k5_tp8_dp4_ep32.txt`.
3. **Native ops.**
   - The fake-runner test shows that the SP ops call the native functions with the tensors above.
   - The `call_bytes` cases pass.
   - A cold import of `frontier.profiling.collectives.vllm_collective_benchmark` and `vllm_dsv4_forward_collectives` exits 0 without torch, vllm or cuda modules in `sys.modules`.
4. **Launcher.**
   - `g5h/launch/tests/test_runtime_check.py` passes.
   - `build_lock` on the real G5h-16 config refuses on the working tree, because the B9 sources are not committed yet; report the message. The test with the temporary repository shows the lock contents: every expected path and the `overlay.json` NCCL hash. The lead runs `build_lock` on the real configs after the commit and expects the hash `618968ee277041751512e619190132a9bdf351c3ce7129c73484f6ef441ad692`.
   - `python3 -c "import ast; ast.parse(open('submit.py').read())"` passes for `submit.py`.
   - Report the worker command that `submit.py` would build for G5h-16 (from a function that does not need steptron).
5. **Regression suites, all passing:**
   - `tests/unit/test_vllm_forward_collectives.py`, `test_collective_benchmark_helpers.py`, `test_collective_kernel_evidence.py` and the new module;
   - cold imports of `frontier.profiling.collectives.nixl_kv_read_benchmark` and `ep_kv_read_overlap_benchmark`;
   - `tests/unit/test_kernel_gap_pricing.py`, where only the known failure `test_every_collective_launches_one_kernel_by_default` remains.

   Report the counts.
6. **Hygiene:**
   - line counts of the changed modules;
   - no unused imports in changed files (`/data/ycfeng/tmp/unused_imports.py`);
   - `/usr/bin/grep -rn "_ceil_div" frontier tests` finds nothing.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- Executed checks with their output tails; the Acceptance 1 byte table, the Acceptance 2 `cmp` results and the Acceptance 4 worker command.
- Scratch paths, deviations and their reasons, and limits.
- If a decision above cannot be implemented as written (for example a native detail that contradicts [g5h/design.md](g5h/design.md)), stop that part and report the evidence and options instead of choosing a different design.
