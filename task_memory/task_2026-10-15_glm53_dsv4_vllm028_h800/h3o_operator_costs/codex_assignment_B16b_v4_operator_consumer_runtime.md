# Task: finish the V4 operator consumer (stage 3 assignment B16b: the runtime part of B16, with the admission prerequisites, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: follow-up to B16 (Codex `54e2e32d`, partial). Adds the three admission prerequisites to the scope and corrects the loader rule for the mHC and endpoint contracts. |

## Goal

B16 part 1 is committed (`f61bd244`): the promotion script, the four H200 V4 tables in `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/` and the round-trip gap reader. The runtime part of B16 is not done: the V4 training function, the trained pricing path, the composed attention phase, the segment gaps, the measurement type, the host-time zero and the runner coverage error.

Codex stopped because three admission guards block a non-dummy V4 predictor, and their files were outside the B16 writable paths:

1. `frontier/operators/families.py:314-318`: `DEEPSEEK_V4_MHC_FAMILY` has `execution_enabled=False` ("source inventory only"). `bind_operator_query` and `validate_typed_operator_contracts` raise `NotImplementedError` for its operators, so 0 of the 41 rows of the promoted `linear_op_kernel_only.csv` validate.
2. The B16 assignment (decision 4) told the loader to validate every typed contract through `profile.resolve_layer_contract`. That is wrong for the mHC and endpoint operators: the producer writes them with the non-layer contract of `frontier/profiling/linear_op/profiling_plan.py:153` (`_non_layer_contract`, `layer_kind: null`), and the `deepseek_v4` profile has no layer contract for this family. The shared expert and the MoE rows do have layer contracts (15 and 132 contracts passed in B16).
3. `frontier/attention/families.py:640-656`: `DEEPSEEK_V4_ATTENTION_FAMILY` has `execution_enabled=False`. `base_execution_time_predictor.py:44-47` admits it only in dummy mode or with a bound step-cost table, so a non-dummy TP8/DP2/EP16 predictor fails at construction.

B16b delivers everything B16 promised for the runtime: after B16b, a non-dummy MONOLITHIC V4 run prices every operator except the runner operators from the trained H200 tables and reads no `_dummy_execution_time` (gate A1).

## Approach and decisions

### Read first

- The B16 assignment: [codex_assignment_B16_v4_operator_consumer.md](codex_assignment_B16_v4_operator_consumer.md). Its "Existing mechanism" section, its decisions 2, 3 and 5-11, its Out-of-scope list and its Acceptance 2-13 apply unchanged, except where this file corrects them.
- The B16 Codex report: [codex-implement-54e2e32d-92b5-49b8-b578-4119b950ab6b/implementation.md](codex-implement-54e2e32d-92b5-49b8-b578-4119b950ab6b/implementation.md), including its prerequisite evidence and the scratch at `/data/ycfeng/tmp/codex-implement/54e2e32d-92b5-49b8-b578-4119b950ab6b/b16/` (read-only; `prerequisites.json`, `asset_validation.json`, `validate_assets.py`).
- The lead's part 1 report: [../test_report_2026-10-29_b16_v4_operator_consumer.md](../test_report_2026-10-29_b16_v4_operator_consumer.md).
- The non-layer contract path:
  - the producer: `frontier/profiling/linear_op/deepseek_v4_plan.py` (`build_operator_plans`; the selected TP is the attention TP for `ATTENTION_TP` operators and 1 otherwise, line 103);
  - the runtime validator: `frontier/operators/typed_contracts.py:405` (`validate_typed_operator_contracts`, which checks each operator's metadata against the registry);
  - the runtime TP path for operators without a layer contract: `sklearn_execution_time_predictor.py:2905-2950` (`_resolve_typed_linear_contract` returns `None` when no profile layer contract owns the family). Note that its `ATTENTION_TP` branch goes through `resolve_effective_attention_tp_size`, which depends on `num_kv_heads`; check what it gives for V4 before using it.
- The sentinel tests that assert the disabled state today: `tests/unit/test_dsv4_standalone_operator_plans.py:206` (`test_canonical_admission_is_blocked_by_disabled_family`), `tests/unit/test_glm53_attention_cost_contract.py:317`, and the family tests found by `/usr/bin/grep -rln "execution_enabled\|not enabled for execution" --include='*.py' tests/unit`.

### Inputs

The committed tables of `f61bd244` and `h3o_operator_costs/h200_tables/manifest.json`. They are fixed: B16b does not rebuild or edit them. If a table contradicts a decision, stop that part and report.

### Decisions

1. **Enable the mHC family.** Set `DEEPSEEK_V4_MHC_FAMILY.execution_enabled=True` and remove its `disabled_reason`. The operators keep `predictor_target=False`, so the generic trainers do not pick them up. Report every consumer of the flag whose behavior changes (`bind_operator_query`, `binding.py:367`, and any other you find).
2. **Validate the mHC and endpoint rows through their non-layer contract** (replaces B16 decision 4's first bullet for these operators).
   - The loader validates each row's `typed_operator_contracts` with the existing `validate_typed_operator_contracts`, with the profile and the model config.
   - It then checks the row's operator entry against the registry and the deployment: the family id, the `tensor_parallel_mode` equal to the operator's `tp_mode`, and the `selected_tensor_parallel_size` equal to the TP the deployment uses for that mode.
   - The rule "attention TP for `ATTENTION_TP`, 1 for `REPLICATED`" exists once today, in the producer (`deepseek_v4_plan.py:103`). Move it into one torch-free function (for example beside `DEEPSEEK_V4_OPERATOR_AXES` in `frontier/operators/deepseek_v4_axes.py`), and call it from both the producer and the consumer. The producer's plans must stay byte-identical (its existing tests and `--check-plan`-style checks).
   - The shared expert and the MoE rows keep `profile.resolve_layer_contract`, as in B16 decision 4.
   - No new layer contract in `model_architectures.py`.
3. **Enable the V4 attention family.** Set `DEEPSEEK_V4_ATTENTION_FAMILY.execution_enabled=True` and remove its `disabled_reason`.
   - The base predictor's admission then passes for the trained path. Keep the dummy and step-cost paths working as they are.
   - Enabling it adds the family to `iter_execution_enabled_families()`. Report what this changes for its consumers (`quantization_manager._attention_compute_operations`, `profiling_mapping.validate_attention_catalog_alignment`, `time_components`, and any other), and confirm that no other model's predictions or metrics change. If the metrics catalog lacks V4 attention names, add them to `frontier/metrics/constants.py`.
   - The attention admission rules stay where they are: `attention/deepseek_v4.py` still admits only MONOLITHIC, and `serving_context` still rejects the PD path. B17 opens PREFILL and unified DECODE.
4. **Update the sentinel tests** that assert the disabled families. Each one becomes a positive test of the new admission (for example, the mHC contracts of a producer plan now validate), or is removed if the new tests cover it. List each change.
5. **The rest of B16:** decisions 2, 3 and 5-11 of the B16 assignment, unchanged. In particular: one V4 training function for both training paths; exact lookup only for operators and MoE, with the trainer's existing mean reduction for duplicate coordinates (B16 confirmed it at `sklearn_execution_time_predictor.py:191`); `moe_shuffling` removed from the V4 routed phase; the composed attention phase in the trained path; the segment-class gap table; KERNEL_ONLY in every role; host time as the explicit zero and the typed error for `skip_cpu_overhead_modeling=False`; `kv_cache_block_zero` raising `DeepSeekV4CoverageError` in non-dummy mode; registration lines only in `sklearn_execution_time_predictor.py` and `metrics_store.py`; `read_isolated_scope` and `_read_isolated_component` deleted after their tests move to the new loader.

## Out of scope

- B17: the runner operators' rows and phases (`dsv4_input_preparation`, `dsv4_block_table_write`, `dsv4_sampling`, and the `kv_cache_block_zero` rows), the PREFILL and unified DECODE admission, the V2 dummy workload (entry 4), `runtime_limitations` (entry 11), the module-size record (entry 12) and the A3 ledger sentinel.
- Changes to the tables, the promotion script, the GPU producers' measured behavior, `balanced_routing.py` and the collective consumer (B15).
- End-to-end replays (A10), the diagnostic cross-check (A11), the 77-case fidelity matrix (the lead runs it after the commit).
- Any GPU, platform or network action.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `f61bd244` or later). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md) (§1 gates A1, A5-A7, A9 and A12; §3; §4; §6 entries 1, 3, 5-8, 10 and the implementation notes), [c1_coverage.md](c1_coverage.md) (§4).
- **Writable paths:**
  - everything B16 allowed: `frontier/execution_time_predictor/` (`dsv4_predictor.py`, `deepseek_v4.py`, `dsv4_composition.py`, `kernel_gap.py`, `sklearn_moe_execution_time_predictor.py`, `moe_dataset_training.py`, `shared_prediction_model_manager.py`, `prediction_family_trainers.py`, new modules beside them; `sklearn_execution_time_predictor.py` registration and routing lines only), `frontier/scheduler/utils/expert_parallel.py`, `frontier/metrics/metrics_store.py` (only if decision 6 of B16 needs it), `frontier/profiling/kernel_gap/schema.py` (constants only), `tests/unit/`;
  - added for B16b: `frontier/operators/families.py` (the mHC family flag), `frontier/operators/deepseek_v4_axes.py` (the TP selection function), `frontier/profiling/linear_op/deepseek_v4_plan.py` (only to call that function), `frontier/attention/families.py` (the V4 attention family flag), `frontier/execution_time_predictor/base_execution_time_predictor.py` (only if the admission needs a change beyond the flag), `frontier/metrics/constants.py` (only V4 catalog names, if decision 3 needs them).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b16b/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No moves. Delete only `read_isolated_scope`, `_read_isolated_component` and code this change supersedes; list each deletion.
  - Python 3.10. Keep the touched modules at or below 2,000 lines, apart from `sklearn_execution_time_predictor.py` (8,626 lines) and `metrics_store.py` (5,768 lines), which take registration lines only. Report their line counts before and after.
  - No magic literals in code paths: the backend names, row axes, TP modes and gap values come from the profile, the operator registry and the tables. No guards or defaults for states the code cannot reach, and no fallback that hides a missing row. No second implementation of the class algebra, the composition, the row-axis rule or the TP selection rule.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
  - Search with `/usr/bin/grep` and quote `--include='*.py'`. Do not search recursively under `/data/ycfeng/tmp/glm53_dsv4/`.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Run pytest from the repository root.
  - The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Acceptance

1. **Prerequisites.**
   - Every row of the committed `linear_op_kernel_only.csv` passes the new loader: the mHC and endpoint rows through the non-layer contract check, the shared-expert rows through `resolve_layer_contract`. Every row of `moe_kernel_only.csv` passes. Report the counts.
   - A row whose `selected_tensor_parallel_size` or `tensor_parallel_mode` disagrees with the registry and the deployment raises `DeepSeekV4CoverageError` naming the operator.
   - The producer's plans built by `build_operator_plans` are equal before and after the TP-function move (compare the serialized plans).
   - A MONOLITHIC non-dummy V4 predictor on the committed tables (TP8, DP2, EP16, `skip_cpu_overhead_modeling=True`) constructs without a step-cost table.
   - The consumers of the two flags: what changed, and the evidence that other models are unchanged.
2. **B16 Acceptance 2-13**, unchanged (A1 with the stage components of the decode FULL and prefill NONE forwards; exact lookup and the coverage errors; the repeat report may cite B16's; rank class; composition and gaps; both training paths; A9 cache; host time; the `kv_cache_block_zero` coverage error; other paths with the placement late-join failure listed before and after and the dummy smoke matrix; new unit tests; hygiene).
3. **The sentinel test changes** of decision 4, listed with their before and after.

## Expected return

- Report language: English.
- Changed, new and deleted paths, each with a summary.
- The results of Acceptance 1-3, with the commands, the expected and actual values, and the counts.
- Scratch paths, deviations and their reasons, and limits.
- If the code or the tables contradict a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Continue with the parts that do not depend on it.
