# Task: native V4 gate and routed id dtype in the MoE producer, and the B12 review fixes (stage 2 assignment B12b, file entry 16, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: follow-up of B12 (`68caa2f4`) after the lead's review. |

## Goal

B12 (request `68caa2f4-8400-407e-a297-e9e5e66ce13e`, report `codex-implement-68caa2f4-8400-407e-a297-e9e5e66ce13e/implementation.md`) added the rank-class planner, the gate on SP-local rows L, the row identity and the G3h package plan. Its gate-parity finding (Acceptance 7) and the lead's review show that G3h would not measure two native V4 kernels:

1. **Gate.** Native V4 builds `GateLinear(out_dtype=torch.float32)` (`vllm/models/deepseek_v4/nvidia/model.py:571-577`; router logits float32 at `:706`). The producer's `balanced_topk` path builds `ReplicatedLinear(disable_tp=True)` with model-dtype output (`moe_impl.py:162-173`).
2. **Routed id dtype.** The native V4 runs route with the pinned `balanced_topk`, which returns `torch.long` ids when `indices_type` is None. The naive DP/EP prepare step leaves it None (`prepare_finalize/naive_dp_ep.py:103-104`). The native census shows `void vllm::moe::moe_align_block_size_kernel<long>` (`h3o_operator_costs/evidence/runtime_census/p2_32.txt:228`, `p2_64.txt:248`, `p0.txt:229`). The producer's functional kernel path casts the ids to int32 (`moe_vllm_kernel.py:947-950`) for every routing path, so it measures the `<int>` kernels.

The same int64 finding corrected the G5h dispatch bytes (`f342a01b`, [../test_report_2026-10-28_b10_g5h16.md](../test_report_2026-10-28_b10_g5h16.md)).

Lead decision (D-NO-INTERRUPT): B12's instruction "do not change the gate implementation" is withdrawn. B12b makes the `balanced_topk` producer path build the native V4 gate and keep the native id dtype, because G3h must measure the native kernels (D-OPERATOR-LEVEL-ROUTE). The GLM and other routing paths stay unchanged.

The B12 changes are uncommitted in the working tree. B12b edits them in place. The lead commits B12 and B12b together after review.

## Changes

1. **Native gate on the `balanced_topk` path.**
   - Build the gate as native V4 does: `GateLinear(input_size=hidden, output_size=num_experts, bias=False, out_dtype=torch.float32)`, with the model's parameter dtype, and record the backend as `vllm_gate_linear`, as the existing `grouped_topk_sigmoid` branch does (`moe_impl.py:143-158`).
   - Reuse that branch's construction rather than adding a second one. Take each value (output dtype, parameter dtype) from the model or router configuration where the existing branch does. Where V4 has no configuration field for a value that native V4 hard-codes, cite the native line.
   - The gate input stays the model dtype (bfloat16 for V4) on L rows. Check the wrapper's gate weight and input setup (`moe_wrapper.py:154`, `:217`, `:269-275`) and change only what the native gate needs.
   - Compare the dispatch inside native `router/gate_linear.py:183-241` (small-row kernels, BF16 to FP32 GEMM) with what the producer now calls, and report any remaining difference with file:line evidence.
2. **Routed id dtype per routing path.**
   - Keep the native id dtype in one place: the routing path table in `get_routing_runtime_metadata` (`moe_impl.py:258`), or a field beside it. `balanced_topk` gives int64, matching the native evidence above; every other path keeps int32, as today.
   - The functional kernel path casts the ids to the path's dtype instead of the fixed int32.
   - Do not add a new CSV column for the dtype: the existing `routing_runtime_path` column identifies it. Rows of the other paths keep their columns and values.
   - Trace every consumer of the routed ids in the measured scopes (shuffling, alignment, Marlin GEMMs, reduction) and list the kernels whose template or arguments depend on the id dtype. Compare them, by source and by name, with the `<long>` kernels in `evidence/runtime_census/p2_32.txt`. Report the list.
3. **`uniform_topk` unchanged.** B12 replaced the body of `uniform_topk` (`moe_impl.py:77-91`) with a call to `balanced_topk`. That function is also timed in the `uniform_topk` routing path (`moe_impl.py:226-231`), so the change alters the measured kernels of that path. Restore its HEAD body. The routed sample of the `balanced` load distribution (`load_distribution.py:86-89`, not timed) calls `balanced_topk` with offset 0 instead, which gives the same ids.
4. **Row identity only where it is defined.** `_build_profile_result` adds `ep_rank`, the three class columns and `gating_num_tokens` to every row, but the class columns describe balanced routing only. Add these columns to rows of the `balanced` load distribution only. Rows of other distributions keep exactly their HEAD keys and values.
5. **Single checks.**
   - `validate_lane_rows_args` runs in `parse_args`; remove the repeated calls in `_lane_rank_tasks` and `_dry_run`.
   - The balanced strategy hash is checked in the wrapper constructor and spread into rows through `routing_runtime_metadata`. Remove the bare `balanced_routing_columns()` calls in the routed-sample setup, in `_build_profile_result` and in `_dry_run`; keep the hash in the dry-run JSON.
   - `moe_profile_identity.py` imports the attention constant inside its function. Move the import to the module top unless that creates an import cycle; if it does, say which.
6. **G3h plan.** Update `g3h/package_plan.json` and its builder notes: the gate is the native `GateLinear`, the routed ids are int64, and the gate qualification that B12 recorded is replaced by the remaining differences you report in items 1 and 2. Leave `source_revision` at its B12 value; the lead replaces it after the commit. Do not build the package.

## Accepted as is (no change)

- The fixture repairs in `tests/unit/test_measurement_family_selector.py`: the lead reproduced the three HEAD failures (`_uses_output_placeholders`, 3 failed / 34 passed).
- The per-rank grouping in `profile_model`, the planner, the class algebra and the gate on L rows.
- The gate routing step with offset 0: its kernels are the native ones; only the id values differ, and the routed sample uses the gathered ids.

## Acceptance

1. **B12 checks rerun.** B12 Acceptance 1 (H800 grid dry runs equal to `moe/h800_moe_local_01/plans/{cuda_event,record_function}.json`), 2, 3, 4 (`prepare_package.py --check-plan`: 13 EP16 and 20 EP32 tasks per method, equal to `expected_tasks.json`), 5 and 9 pass, with the same commands.
2. **Other paths unchanged.**
   - `git diff HEAD -- frontier/profiling/moe/moe_impl.py` shows no change to `uniform_topk` or to the `grouped_topk_sigmoid`, `standard_fused_topk` and `uniform_topk` gate construction.
   - A CPU test builds a row of a non-balanced distribution and checks its keys and values against the HEAD row builder (or a frozen copy).
   - A CPU test checks that the functional kernel path receives int32 ids for `grouped_topk_sigmoid`, `standard_fused_topk` and `uniform_topk`, and int64 for `balanced_topk`.
3. **Native gate.** A CPU test with a mocked `GateLinear` checks the `balanced_topk` gate construction arguments (output float32, model parameter dtype, no bias) and `linear_backend == "vllm_gate_linear"`.
4. **Tests.** The B12 required suite and the importing-test command pass (report counts and skips). Report the new test count.
5. **Hygiene.** Line counts, no unused imports (`/data/ycfeng/tmp/unused_imports.py`), cold imports without `torch`, `vllm` or `cuda` in the minimal environment, and `git diff --check`.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `eaee3550`). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Concurrent work:** assignment B11b (request `ebbfa1e0`) is editing `frontier/profiling/attention/`, `tests/analysis/v4_attention_coverage.py`, `tests/unit/test_v4_attention_coverage.py` and `h3o_operator_costs/g1h/`. Do not edit those files. `frontier/profiling/attention/dsv4_attention_rows.py` is read-only for you.
- **Task records:** [../requirements.md](../requirements.md), [design.md](design.md) §3.5, §3.7 and entry 16, [c1_coverage.md](c1_coverage.md) §4.3, [codex_assignment_B12_g3h_moe_rank_classes.md](codex_assignment_B12_g3h_moe_rank_classes.md) (still valid except where this file changes it).
- **Writable paths:**
  - `frontier/profiling/moe/` except the frozen `balanced_routing.py` (SHA-256 `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`);
  - `frontier/moe_ep_workload.py` (the B12 additions only);
  - `tests/unit/` (the B12 test files and new tests; not the B11b files above);
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b12b/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No deletions or moves of files.
  - Python 3.10. Modules at or below 2,000 lines. No magic literals in code paths. No guards for states the code cannot reach. No second way to do what the code already does.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
- **Environment:** as B12: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6); the CPU torch environment adds `/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages` to `PYTHONPATH` and sets `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Expected return

- Report language: English.
- Changed paths, each with a summary of the change.
- The native gate comparison (item 1) and the id-dtype kernel list (item 2), with file:line evidence.
- The results of Acceptance 1-5, with the commands, the expected and actual values, and the counts.
- Scratch paths, deviations and their reasons, and limits.
- If the native source contradicts a decision above (for example, native V4 ids are int32 on a path the census does not show), stop that part and report the evidence and the options.
