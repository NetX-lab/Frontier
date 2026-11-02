# Test report: B15, the V4 measured collective consumer (entries 8 and 9)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: the Codex outcome, the lead decision on the combine fixture and the lead corrections, the CPU checks, the diagnostics and the open items. |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B15_v4_collective_consumer.md](h3o_operator_costs/codex_assignment_B15_v4_collective_consumer.md); design notes in [h3o_operator_costs/design.md](h3o_operator_costs/design.md) §6.
- Codex request `5efbdad3`, report `h3o_operator_costs/codex-implement-5efbdad3-a727-43de-873b-f6928d5eb4e1/implementation.md`, scratch `/data/ycfeng/tmp/codex-implement/5efbdad3-a727-43de-873b-f6928d5eb4e1/b15`.
- Input table: the H200 collective table of commit `9dac0271`. Base revision `cc808960`.
- Lead review scratch: `/data/ycfeng/tmp/b15_review/`.

## Codex outcome

Codex reported **partial**. It implemented:

- the `CollectiveImplementation` enum (`GROUP_COORDINATOR_DIM0` default, `VLLM_SEQUENCE_PARALLEL`, `GROUP_COORDINATOR_LAST_DIM`) and the spec fields `collective_implementation` and `sp_rows_builder`;
- one measured pricing hook, `measured_collective_time` in `frontier/operators/sequence_parallel_moe.py`, which calls `MeasuredCCBackend.predict_operator_collective(alias, implementation, size, num_devices, comm_domain, sp_rows)`;
- curve selection by (alias, domain, implementation), the equal-lane rule for uniform curves, exact unequal combine entries and holdout reporting;
- one grouped V4 dispatch, `dsv4_ep_dispatch_allgather`, with the ids dtype from `ModelArchitectureProfile.gathered_topk_ids_dtype`;
- the ordered SP width carrier `lane_batch.moe_sequence_parallel_token_widths` → `sp_token_widths` → `CommPayloadContext.sp_token_widths` → `ep_widths(ctx)`;
- the unpadded SP reduce-scatter, the embedding and logits endpoints as communication specs, and the PDD admission of the V4 communication payloads only;
- a new test file with 118 cases.

Codex stopped before moving the V4 combine payload onto the shared `ep_workload`, under the assignment's stop rule. The existing fixture in `tests/unit/test_dsv4_structural_execution.py::destination()` built a 40-row routing workload for TP8/EP64 (eight DP lanes). Eight lanes of 17 tokens give widths `(3,3,3,3,3,3,3,3)`, so the native wave routes 192 gathered rows, not 40. The old and new payloads therefore could not be equal on that fixture.

## Lead decision and corrections (D-NO-INTERRUPT)

- **Decision.** The 40-row fixture is not physical. `prepare_moe_wave_from_inputs` routes every gathered SP row, so the fixture now uses `routing_token_count=widths.gathered_tokens`. The V4 combine payload is the shared `ep_workload` gathered rows × 8192 bytes. The lane workload stays for routed compute and workload records.
- **Corrections.**
  - `frontier/operators/families.py`: the V4 EP specs use `_dsv4_ep_payload(ctx, name)` for every EP collective, and `sp_rows_builder` only on the combine; `_dsv4_ep_output_bytes` is deleted; `_dsv4_tp_payload` passes `ctx.selected_logits_rows` with no fallback.
  - `frontier/execution_time_predictor/deepseek_v4.py`: one `CommPayloadContext` carries `sp_token_widths` and `selected_logits_rows`; the per-workload combine context with `lane_workload` is removed.
  - `frontier/cc_backend/backends/measured_cc_backend.py`: `predict_unequal_reduce_scatter` is deleted, because `predict_operator_collective` is the only pricing entry.
  - Tests: the measured-backend, GLM P0 and GLM SP tests call `predict_operator_collective`; three guards that tested only the deleted method are removed; one ATTN_TP case is added; the consumer test drops its combine helper with `lane_workload`.
  - `vllm_forward_collectives.py` and `vllm_dsv4_forward_collectives.py`: import order only.
  - `tests/integration/run_dsv4_structural.py`: one comment.
- The `alltoall` → `all_to_all` rename in `_curve_key` stays. The GLM replay compares its error key for the legacy all-to-all operator, and the rename keeps that output byte-identical.

## Environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM).
- `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `ulimit -v 31457280`.
- The broad run also adds `/usr/local/lib/python3.10/dist-packages`, `CUDA_VISIBLE_DEVICES=""`, `VLLM_BS_SOURCE_ROOT` and `FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` (script `/data/ycfeng/tmp/b15_review/broad.sh`).

## Checks

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Focused tests: `pytest -q tests/unit/test_measured_cc_backend.py test_glm_p0_input_binding.py test_glm_sp_moe_comm.py test_comm_operator_families.py test_dsv4_dp_sync.py test_dsv4_graph_measured.py test_dsv4_source_execution.py test_dsv4_structural_execution.py test_v4_measured_collective_consumer.py` | All pass | 467 passed in 8.72 s | PASS |
| 2 | H800 table (`/data/ycfeng/tmp/b15_review/h800_after.json`, compared by `compare_baselines.py` with Codex's baseline) | The 5 curves and 9 exact entries are identical; only the logits curve is added | 5 curves identical, 9 exact entries identical; added `logits_all_gather/tp/8` | PASS |
| 3 | GLM communication replay (`tests/integration/run_glm53_comm_replay.py`, output `glm_after.json`) | Byte-identical to the baseline | SHA-256 `f2116303e30ca97853157a385248c3a72113a706676296afb16d075f080e6e15`, equal | PASS |
| 4 | GLM and V4 forward collective plans (`plans_after.json` against `plans_before.json`) | Byte-identical | SHA-256 `10c87f8dc1675d274658e86a8e2cdebdb774ba786f7787c5d597cac5e92e38dd`, equal; plan counts GLM 36 / 36, V4 82 / 46 | PASS |
| 5 | Structural driver `tests/integration/run_dsv4_structural.py`, offline and online (`/data/ycfeng/tmp/b15_review/structural/`) | Exit 0; stage and phase ledgers reconcile | Exit 0 in both; maximum stage error 2.66e-15 ms, phase error 6.94e-18 ms; one grouped dispatch per layer | PASS |
| 6 | H200 table load | 8 curves, 16 exact combine entries, the unequal dispatch rows as holdouts | Curves `all_gatherv/ep/16`, `all_gatherv/ep/32`, `all_reduce/tp/8`, `logits_all_gather/tp/8`, `reduce_scatterv/ep/16`, `reduce_scatterv/ep/32`, `sp_all_gather/tp/8`, `sp_reduce_scatter/tp/8`; 16 exact entries; 48 holdouts, all `moe_dispatch_all_gatherv` | PASS |
| 7 | Broad unit run (115 files, `broad_files.txt`; one known kernel_count test deselected) | No failure caused by B15 | 3,595 passed, 22 skipped, 29 failed, 42 errors. All are outside B15: `test_native_groundtruth_check.py` 42 errors (Python 3.10.6 `tarfile` has no `filter` argument); `test_colocation_release_review_contracts.py` 10 (module `frontier.config_optimizer` is absent); `test_dp_placement_harness.py` 6 (no `aiohttp`); `test_native_observer_bootstrap.py` 12 and `test_gdn_hybrid_e2e_increment14ab.py` 1 fail only after other tests load torch and pass in isolation (`iso1/`, `iso2/`) | PASS |
| 8 | Hygiene | Cold imports work; no new unused import; `git diff --check` clean | Cold imports OK; no unused import in the touched files (5 older ones in the sklearn predictor unchanged); `git diff --check` clean | PASS |
| 9 | Deleted paths | No caller of the removed functions | `_dsv4_ep_output_bytes`, `predict_unequal_reduce_scatter` and the combine `lane_workload` context have no remaining reference | PASS |

## Diagnostics from Codex (reported, not gated)

- H800 dispatch holdouts: 282 rows, median |error| 3.691 %, maximum 37.606 %.
- H200 dispatch holdouts: 48 rows, median |error| 3.055 %, maximum 5.875 %.
- H200 graph against eager: the eager SP reduce-scatter is 43.63 % slower than graph at 8,192 bytes. The loader still pools graph and eager rows into one curve.
- C1 coverage: the unequal combine vectors of P2-32 and P2-64 are fully covered by exact entries; 2 vectors of P0/H800 are not.
- The V4 descriptor width reader and the V1 reader differ on 3 of 14 fixtures; V4 keeps the descriptor reader.
- `tests/integration/test_vllm_dp_placement_runtime.py` has a scheduler late-join failure that Codex reproduced on the base sources; it is not caused by B15.

## Open items

- The PD rejections for attention stay with entry 4: `frontier/attention/deepseek_v4.py:155,218`, `frontier/execution_time_predictor/deepseek_v4.py:559,666` and `param_counter.py:607`.
- The graph and eager pooling of the H200 SP curves is a known bias; splitting it needs a decision on the curve key.
- Whether the GLM logits gather should also use the new logits curve.

## Limits

- All checks run on CPU. No V4 latency is compared with a native run here.
- The broad run shared the worktree with Codex B13s, which was editing `frontier/profiling/attention/` and its tests at the time. None of the failures are in those files.
