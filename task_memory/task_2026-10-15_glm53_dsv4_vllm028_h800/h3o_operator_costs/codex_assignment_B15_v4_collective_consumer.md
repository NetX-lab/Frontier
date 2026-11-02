# Task: the V4 collective consumer — collective implementations in the communication registry, the H200 measured curves, one grouped EP dispatch, ordered SP widths for V4, and the endpoint collectives (stage 2 assignment B15, file entries 8 and 9, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: Codex assignment B15 of stage 2 ([design.md](design.md) §3.5, §4 and §6 entries 8 and 9; [g5h/design.md](g5h/design.md) §2-4; input table committed in `9dac0271`). |

## Goal

G5h measured every V4 collective on H200 (`exp-1028-231342-951824` for TP8/DP2/EP16, `exp-1028-231408-632060` for TP8/DP4/EP32). The table is committed as `data/profiling/network/h200_dgx/summary.csv` (384 rows, CRLF, provenance in `summary_provenance.json`, commit `9dac0271`). Frontier cannot use it yet:

- `MeasuredCCBackend` (`frontier/cc_backend/backends/measured_cc_backend.py`, 251 lines) rejects the `sp_all_gather` and `sp_reduce_scatter` rows ("unsupported curve", line 145), so loading the H200 table fails.
- Its uniform curves would also take the H200 unequal EP rows. In the H800 table every unequal row is marked `single_active` and is skipped; in the H200 table the 48 unequal combine rows and 48 unequal dispatch rows are not marked (scenario names have no `single_active`), so the current rule pools them into the uniform curves.
- The V4 operators price native work with the wrong shapes: three separate EP dispatch gathers instead of one grouped `all_gatherv`; ids priced as int32 instead of int64; the SP reduce-scatter keyed by the padded size; the embedding all-reduce and the logits gather as dummy placeholders (`deepseek_v4.py:337-338`, `:373-379`).
- The exact unequal-combine lookup applies only to the GLM operator name `expert_parallel_alltoall_combine` (`sequence_parallel_moe.py:139-153`), and V4 has no ordered SP widths, so a V4 combine with unequal lanes is priced from the uniform curve.

B15 fixes these on CPU. The stage 3 consumers (entries 1-12 beyond these parts) stay out of scope.

## Approach and decisions

### Native facts (from [g5h/design.md](g5h/design.md) §2-3; re-check where the code depends on them)

`T` = tokens of one DP lane (the native descriptor width), `r = ceil(T / TP)`, hidden 4,096 bf16 (8,192 bytes per row), `k = 6`, 43 layers.

| Call (table `call`) | Table `op`, group | Per-rank input bytes (the key) | Frontier operator |
| --- | --- | --- | --- |
| `embedding_all_reduce` | `all_reduce`, tp | `T × 8,192` | `dsv4_embedding_allreduce` |
| `sp_attention_all_gather` | `sp_all_gather`, tp | `r × 8,192` | `dsv4_sp_attention_allgather` |
| `sp_attention_reduce_scatter` | `sp_reduce_scatter`, tp | `T × 8,192` (unpadded; the pad copy is inside the measured call) | `dsv4_sp_attention_reduce_scatter` |
| `sp_final_mhc_all_gather` | `sp_all_gather`, tp | `r × 4 × 8,192` | `dsv4_sp_final_mhc_allgather` |
| `moe_dispatch_all_gatherv` | `all_gatherv`, ep | `max(r_d) × (8,192 + 6 × 4 + 6 × 8)` = `max(r_d) × 8,264` (bf16 hidden, f32 weights, int64 ids) | new `dsv4_ep_dispatch_allgather` |
| `moe_combine_reduce_scatterv` | `reduce_scatterv`, ep | `TP × Σ_d r_d × 8,192` (gathered rows) | `dsv4_ep_output_reduce_scatter` |
| `logits_all_gather` | `all_gather` (dim −1), tp | `b × 16,160 × 2 = 32,320 × b`, `b` = selected logits rows | `dsv4_logits_gather` |

- The SP calls are vLLM `sp_all_gather` / `sp_reduce_scatter` (custom MNNVL Lamport kernels first, NCCL fallback), not `GroupCoordinator.all_gather` / `reduce_scatter`. They are a different collective implementation from the GLM `sp_moe_*` calls, which go through GroupCoordinator and are measured as `all_gather` / `reduce_scatter` in the H800 table.
- The logits gather is GroupCoordinator `all_gather` on the last dimension: NCCL all-gather plus a layout copy (`direct_copy_kernel_cuda` in the `kernels` column). The table's only evidence of this implementation is the call `logits_all_gather`.
- The dispatch ids are int64 under the pinned balanced routing (G5h §2 finding 2). The GLM producer gathers int32 ids (`frontier/profiling/collectives/vllm_forward_collectives.py:32`); the V4 producer gathers int64 (`vllm_dsv4_forward_collectives.py:40`).

Row classes of the two tables (lead's count from the files; `eq` = equal `tokens_per_dp_rank` over lanes):

| Table | Not `single_active`, eq | Not `single_active`, unequal | `single_active` |
| --- | --- | --- | --- |
| H200 (384) | 288 (graph and eager, every call) | 96: combine EP16 6, EP32 42; dispatch EP16 6, EP32 42 | none |
| H800 (971) | 620 (including 68 `logits_all_gather` and 69 `dp` rows) | none | 351 (3 eq and 36 unequal per call, 9 calls) |

### Existing mechanism (read first)

- `CommOperatorSpec` and `CommPayloadContext` (`frontier/operators/spec.py`, 378 lines): payload, domain and cardinality builders; `ZeroPayloadPolicy`; `apply_allreduce_launch_overhead_strip`.
- The communication registry in `frontier/operators/families.py` (964 lines): the V4 SP specs (about 765-792), the V4 EP specs (807-821), the endpoint placeholders as plain `OperatorSpec` (823-831), the GLM EP phase specs (`_ep_phase_alias`, `_ep_phase_payload`, 523-537, 720-747) and the GLM source TP specs (`_source_tp_collective_specs`, 540-552). The V4 payload builders are `_dsv4_sp_local_hidden_bytes` and friends (422-503).
- `frontier/operators/sequence_parallel_moe.py` (153 lines): `tp_workload`, `tp_payload`, `ep_widths`, `ep_workload` (the GLM geometry, shared by the producer comparisons and the GLM replay), and the hook `measured_unequal_combine_time`.
- The predictor entry `_predict_comm_operator_with_context` (`sklearn_execution_time_predictor.py:6284-6420`): payload, EXACT_NOOP, num_devices/domain/alias, the hook (6301-6305), then dispatch by alias to `predict_*_time`, which delegate to `self._cc_backend.predict_*` (8011-8170) with the dummy fallback when there is no backend. The all-reduce strip (`_strip_collective_sim_allreduce_launch_overhead_if_needed`, line 5024) is a no-op unless the backend has `estimate_intra_server_allreduce_launch_overhead_ms`; `MeasuredCCBackend` does not. The predictor module has 8,666 lines; change only the hook import and call there.
- V4 structural execution (`frontier/execution_time_predictor/deepseek_v4.py`, 804 lines): the phase map `timing_operator_times` (19-72), `predict_source_execution` with `communication(name)` (211-214), `predict_structural_execution` (290-457: the placeholders at 337-338 and 373-379, the EP loop at 365-372, the workload records at 380-404), `serving_context` (553-572), `source_physical_rows` (586-598). V4 lane rows come from `compute_rows` (`frontier/attention/deepseek_v4.py:179-200`), which reads the native descriptor width through the batch's graph metadata.
- EP wave (`frontier/scheduler/utils/ep_wave.py:105-126`): the V4 branch sets `routing_token_count` from `source_physical_rows` and leaves `sp_token_widths = None`; the GLM branch builds `sequence_parallel_moe_token_widths` from `padded_lane_token_counts` and passes it on. `predict_ep_wave_phase_times` (`expert_parallel.py:95-160`) sets the widths on each `lane_batch`; its V4 branch prices the source batch `source_batches[ep_id // tp]` with `lane_workload=lane_batch.lane_workload`.
- Users of the hook today: the predictor (6301-6305), `tests/integration/run_glm53_comm_replay.py:21, 69`, `tests/unit/test_measured_cc_backend.py`, `test_glm_p0_input_binding.py`, `test_glm_sp_moe_comm.py`.

### Decisions

1. **Collective implementation in the registry.**
   - Add an enum, for example `CollectiveImplementation`, as a field of `CommOperatorSpec` (for example `collective_implementation`). Members: GroupCoordinator collectives on dim 0 (the default, so every existing spec keeps its meaning), vLLM sequence-parallel collectives (`sp_all_gather` / `sp_reduce_scatter`), and the last-dimension GroupCoordinator all-gather. Name the members after what they are; avoid filler names.
   - Set the SP member on the three `dsv4_sp_*` specs and the last-dimension member on `dsv4_logits_gather`. Every GLM spec keeps the default, including GLM `logits_allgather` (the recorded GLM decision; whether GLM should later use the logits curve stays open and is not part of B15).
   - Other backends ignore the field: their V4 numbers change only through the payload changes of decisions 4-6, never through the implementation field.
2. **Measured backend: one curve table and one row-classification table.**
   - Replace `_COLLECTIVES` with one table keyed by (alias, domain, implementation) that gives the measured curve. Add the SP rows and a curve for the logits rows. The curve key of every existing H800 curve stays exactly as today, so `describe_curves()` for the H800 table is identical except for one added logits curve.
   - One table maps a table row to its curve: by `op` and `group`, except that the call `logits_all_gather` goes to the logits curve. This is the only call-name entry, because the `op` column cannot tell the two all-gathers apart; say so in a short comment.
   - Row classes in `_load_curves`: (a) uniform curve rows: not `single_active` and equal `tokens_per_dp_rank`; (b) exact combine rows: the `reduce_scatterv`/`ep` rows that are `single_active` or unequal and whose SP widths are unequal (the existing `_retained_combine_rows` checks unchanged); (c) holdouts: every other row except the `dp` group. Holdouts are not priced. Expose their prediction errors against the uniform curve through a separate method (for example `describe_holdouts()`), so `describe_curves()` stays unchanged; the log line at construction may add the holdout summary.
   - The H800 exact-combine table must be identical to today's (`_unequal_combine_times`, same keys and values).
3. **One measured pricing entry.**
   - Replace `measured_unequal_combine_time` with one hook in `frontier/operators/` (in `sequence_parallel_moe.py` or a new small module; say which), for example `measured_collective_time(operator, ctx, backend, size)`. It returns `None` for backends other than `MeasuredCCBackend`, so their path is unchanged. For `MeasuredCCBackend` it prices every `CommOperatorSpec` through one backend method, for example `predict_operator_collective(alias, implementation, size, num_devices, comm_domain, sp_rows)`.
   - The exact unequal lookup applies when the resolved curve is the EP `reduce_scatterv` curve and the ordered SP rows are unequal. No operator name is compared anywhere. The SP rows come from the spec (a builder field, for example `sp_rows_builder`, set on the GLM combine and the V4 combine; it returns the ordered per-lane SP rows from `ep_widths(ctx)`).
   - The legacy `predict_allreduce` / `predict_allgather` / `predict_reduce_scatter` / ... methods of `MeasuredCCBackend` keep working through the same table with the default implementation (other callers use them).
   - The predictor change is the import and the call (6301-6305). Its alias dispatch after the hook stays for the other backends. Show from the code that bypassing it for the measured backend gives the same numbers as today for every GLM operator (the `predict_*_time` methods only delegate; the strip is a no-op for this backend).
4. **One grouped V4 dispatch.**
   - Replace `dsv4_ep_activation_allgather`, `dsv4_ep_weights_allgather` and `dsv4_ep_ids_allgather` with one `dsv4_ep_dispatch_allgather` (`allgather`, `MOE_EP`) in the registry, the phase map (`deepseek_v4.py:52-58`), `metrics/constants.py` (93-96) and every other user (find them; the lead's list: `deepseek_v4.py`, `dsv4_composition.py` if any, `tests/unit/test_dsv4_structural_execution.py`, `test_dsv4_graph_measured.py`, `test_dsv4_dp_sync.py`, `tests/integration/run_dsv4_structural.py`; do not grep `task_memory/`). Remove the three old operators in the same change.
   - Payload: `max(r_d) × (hidden × 2 + k × 4 + k × ids_bytes)`. Share the geometry with GLM by generalizing `ep_workload` so both models use one function; the GLM values must stay identical.
   - **One source for the ids dtype.** Add a `ModelArchitectureProfile` field for the gathered top-k ids dtype (GLM `glm_moe_dsa` int32, `deepseek_v4` int64). The two producers (`TOPK_IDS_DTYPE`, `DSV4_TOPK_IDS_DTYPE`) and the consumer geometry read it, and so does the `topk_ids_dtype` label of the V4 `moe_gating_routing_topk` workload record (`deepseek_v4.py:356`, int32 today). A profile that never builds the SP EP dispatch needs no value; fail with a clear message if the dispatch is built without one. The GLM producer plans (the K5 dry runs) and the V4 producer plans must stay byte-identical.
5. **Ordered SP widths for V4.**
   - In the V4 branch of `ep_wave.py`, compute the widths once per wave: `sequence_parallel_moe_token_widths([compute_rows(source_batches[d]) for d in range(attn_dp)], tp)`, with the validations `source_physical_rows` performs today. Set `routing_token_count = widths.gathered_tokens` (equal to today's `source_physical_rows` value) and pass the widths to `predict_ep_wave_phase_times` as the GLM branch does. Replace `source_physical_rows` by the width function and update its tests (`test_dsv4_structural_execution.py:333-355`, `test_dsv4_graph_measured.py:112-132`, `test_dsv4_dp_sync.py:220`).
   - Keep V4 on `compute_rows` (the V2 descriptor width). Do not switch V4 to `padded_ep_wave_lane_token_counts` (the GLM V1 rule). Report, on the V4 fixtures of those tests, whether the two give the same lane widths.
   - The V4 dispatch and combine payloads read the widths through `ep_widths(ctx)`, the single reader. Choose the carrier from the V4 lane loop to the payload context: today the V4 context batch is the DP source batch, which other code reads and which lives for the whole step. Do not set a new attribute on the source batches. Prefer a carrier that GLM can use unchanged (GLM keeps `lane_batch.moe_sequence_parallel_token_widths`). Report the carrier, why, and every place it is read.
   - The combine payload becomes `widths.gathered_tokens × hidden × 2` through the shared geometry. Show in tests that it equals today's `_dsv4_ep_output_bytes` value (`sum(lane.global_token_counts) // router_topk × hidden × 2`) on the existing fixtures; then the combine no longer needs `lane_workload` for its payload. Keep `lane_workload` where the routed compute and the workload records need it.
6. **Other V4 payload changes.**
   - SP rows: the V4 SP payload builders read the lane rows from `compute_rows(ctx.batch)`, the same source as attention and the EP widths, instead of `get_effective_total_tokens_rounded(ctx.cluster_type)` (equal for `MONOLITHIC`; they differ for `PREFILL`, which has no graph-metadata branch in `batch.py:834-880`).
   - Reduce-scatter key: unpadded `T × hidden × 2` (G5h §2 decision). Today it is `local_rows × hidden × 2 × tp` (`families.py:475-477`). Equal when TP divides T.
   - Endpoints: `dsv4_embedding_allreduce` and `dsv4_logits_gather` become `CommOperatorSpec`s that reuse the `tp_workload` geometry (embedding `[T, hidden]` all-reduce on `ATTN_TP`; logits `[selected rows, vocab / TP]` with the last-dimension implementation). Remove the `base` placeholders (`deepseek_v4.py:337-338`, `373-374`) and price them through `_predict_comm_operator` as the SP operators are. Keep where each is charged: the embedding all-reduce on layer 0 of every forward, including DP dummy forwards; the logits gather on the last layer of forwards with selected logits rows only (keep today's presence rule; `test_dsv4_structural_execution.py:352` expects no logits gather on a dummy forward). Keep the shape fields of the workload records and replace the "unmeasured synthetic dummy duration" text.
   - V4 payload values must equal the G5h §4 bytes on the G5h grid points (graph T = 1, 2, 4, 8, 16, 24, 32; eager T = 1, 6,144; mixed lanes 6,144/1 and 1/6,144; logits b = 1-24).
7. **PDD roles in the V4 communication payloads only.**
   - `_dsv4_sp_local_hidden_bytes` (and the V4 dispatch/combine geometry) admits `PREFILL` and unified `DECODE` besides `MONOLITHIC`, and still rejects `DECODE_ATTN` and `DECODE_FFN`. Tests show that the payloads for a PREFILL-role and a DECODE-role batch equal the MONOLITHIC payload of the same batch.
   - The attention admission (`attention/deepseek_v4.py:217`) and the PD gate of `serving_context` (`deepseek_v4.py:557-558`, "PD phase/graph equivalence remains unqualified") stay as they are: they belong to entry 4 and need their own evidence. List every other rejection of PD roles on the V4 path that you find (file:line and message), without changing them.
8. **Diagnostics (report only, no runtime code).** With scratch scripts:
   - the holdout errors for both tables (call, group size, tokens, bytes, measured, curve value, relative error);
   - per call of the H200 table, the graph and eager medians at the sizes where both exist, and their difference (the loader pools them; this measures the pooling bias);
   - the ordered SP vectors that the C1 serving windows produce for the combine ([c1_coverage.md](c1_coverage.md) §4.5 and the G5h grid), and for each whether the exact table has it.

## Out of scope

- Any GPU, platform or network action.
- The stage 3 consumers (entries 1-7 and 10-12 apart from what decisions 4-7 name), the attention admission, `serving_context`, the V4 kernel-gap values, and the step-cost path (diagnostic only, ADR 0002) except renaming the dispatch operator where it is listed.
- The measured tables themselves (`data/profiling/network/*/summary.csv` and the provenance file): read only.
- The producers' measurement behavior (only the ids-dtype source of decision 4 changes there).
- GLM pricing changes of any kind.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree; HEAD includes `9dac0271`). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md) (§3.5, §3.7, §4, §6 entries 8 and 9), [g5h/design.md](g5h/design.md), [c1_coverage.md](c1_coverage.md) §4.5, and the G5h acceptance report [../test_report_2026-10-28_g5h_collective_kernel_comparison.md](../test_report_2026-10-28_g5h_collective_kernel_comparison.md).
- **Writable paths:**
  - `frontier/cc_backend/backends/measured_cc_backend.py`;
  - `frontier/operators/` (`families.py`, `spec.py`, `sequence_parallel_moe.py`, and one new module if the hook moves);
  - `frontier/execution_time_predictor/deepseek_v4.py`, `frontier/execution_time_predictor/dsv4_composition.py`, and in `sklearn_execution_time_predictor.py` only the hook import and call;
  - `frontier/scheduler/utils/ep_wave.py`, `frontier/scheduler/utils/expert_parallel.py`;
  - `frontier/model_architectures.py` (the ids-dtype field);
  - `frontier/profiling/collectives/vllm_forward_collectives.py`, `frontier/profiling/collectives/vllm_dsv4_forward_collectives.py` (reading the ids dtype only);
  - `frontier/metrics/constants.py`;
  - `tests/unit/`, `tests/integration/run_glm53_comm_replay.py`, `tests/integration/run_dsv4_structural.py`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b15/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No deletions or moves of files.
  - Python 3.10. No magic literals in code paths: sizes from the model config, dtypes from the profile, curves from the table. No guards or defaults for states the code cannot reach. No second implementation of a vLLM function. When a new path supersedes an old one, delete the old one in the same change.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. Do not grep or find recursively in `task_memory/` (it holds large copies of source packages) or in the vLLM tree; name the files.
  - Keep every module at or below 2,000 lines; `sklearn_execution_time_predictor.py` (8,666) and other modules already above the limit get no growth beyond the hook lines.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`. Run pytest from the repository root.
  - Tests that need torch (CPU) use `PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages` with the same interpreter and `CUDA_VISIBLE_DEVICES=""`. Native registry checks also need `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.
  - The pinned vLLM source `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`) is read-only.

## Acceptance

1. **H800 unchanged.** `describe_curves()` on `data/profiling/network/h800_dgx/summary.csv` before and after: every existing entry identical (field by field), plus exactly one logits curve. The exact-combine table identical. The lead's baseline on HEAD `9dac0271`: `/data/ycfeng/tmp/b15_baseline/h800_curves_clean.json` (5 curves: `all_gather`/tp/8, `all_gatherv`/ep/32, `all_reduce`/tp/8, `reduce_scatter`/tp/8, `reduce_scatterv`/ep/32; 9 exact-combine entries), made by `/data/ycfeng/tmp/b15_baseline/h800_curves.py` (the logger writes to stdout; the JSON starts at the first line that begins with `{`). Command and diff in the report.
2. **GLM unchanged.** `tests/integration/run_glm53_comm_replay.py --output-json` after the change is byte-identical to the lead's baseline on HEAD `9dac0271`: `/data/ycfeng/tmp/b15_baseline/glm53_comm_replay.json` (sha256 `f2116303e30ca97853157a385248c3a72113a706676296afb16d075f080e6e15`). The GLM K5 producer plans (dry run) and the V4 producer plans byte-identical. `test_measured_cc_backend.py`, `test_glm_sp_moe_comm.py`, `test_glm_p0_input_binding.py`, `test_glm_sp_moe_contract.py`, `test_comm_operator_families.py` and the other tests of the changed modules pass.
3. **H200 loads.** `MeasuredCCBackend` on the H200 table: the curve list (key, sizes, calls), the exact-combine entries (expected 16 ordered vectors, each with 3 passes pooled by the median: EP16 `1/6144` and `6144/1`; EP32 the 14 unequal vectors over {1, 6,144}), the holdout count (expected 48: the unequal dispatch rows), and no error.
4. **Pricing contract.** Unit tests: each V4 operator resolves the intended curve (SP, logits, GroupCoordinator, EP) on the measured backend; unequal V4 and GLM combines take the exact entry without a name comparison; a vector or size outside the measurement raises `MeasuredCCOutOfRangeError` with the curve key; a non-measured backend gives today's values for every GLM operator and today's alias calls for V4.
5. **V4 payloads.** A table of the V4 payloads at the G5h §4 grid points against the expected bytes (all equal). The combine payload equal to today's lane-workload value on the existing fixtures. The grouped dispatch payload with int64 ids.
6. **Widths.** `routing_token_count` unchanged on the existing V4 fixtures; the widths reach the V4 dispatch and combine contexts; source batches carry no new attribute after a wave.
7. **PDD roles.** PREFILL and DECODE payloads equal MONOLITHIC payloads for the same batch; DECODE_ATTN and DECODE_FFN rejected; the list of remaining PD rejections on the V4 path.
8. **Endpoints.** The embedding all-reduce priced on layer 0 of real and dummy forwards; the logits gather on the last layer of forwards with selected rows only; both through the backend; `run_dsv4_structural.py` expectations updated with the reason (run it if it fits the memory limit; report if it does not).
9. **Existing tests pass:** every test file that imports a changed module (the lead's list is in "Existing mechanism"; the full list from `grep -l` over `tests/` for the changed modules and the renamed operators). The known failure `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` is deselected. Report skipped tests and why.
10. **New unit tests** for decisions 1-7. Report the count.
11. **Diagnostics** of decision 8, with the scratch paths.
12. **Hygiene:** line counts before and after; no unused imports (`/data/ycfeng/tmp/unused_imports.py`); a cold import of each changed module exits 0 and loads no `torch`, `vllm` or `cuda` module unless it already did.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- The results of Acceptance 1-12, with the commands, the expected and actual values, and the counts.
- The enum and field names, the curve keys added, the hook location and signature, the SP-width carrier, and the ids-dtype field.
- Scratch paths, deviations and their reasons, and limits.
- If the code or the native source contradicts a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Examples: a GLM number changes; an H800 curve key must change; the V4 combine payload from the widths differs from the lane-workload value; the H200 unequal rows fail the `_retained_combine_rows` checks; the ids dtype cannot reach a producer without changing its plan.
