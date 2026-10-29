**1. Assignment and outcome**

Implemented B16b: the runtime portion of B16 plus its corrected admission prerequisites. A MONOLITHIC, non-dummy H200 V4 predictor with TP8/DP2/EP16 constructs on the committed tables without a step-cost table. FULL decode and NONE prefill pricing pass a sentinel that rejects every `_dummy_execution_time` read. Final acceptance remains with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Final observed HEAD: `c428295c1799b2a2b03325c99d665cf26bdcccda`, later than the committed B16 part 1 revision. No Git state changes were performed.

Writable scope was `frontier/execution_time_predictor/`, `tests/unit/`, and the explicitly listed attention/operator family, TP-axis, producer-plan, metrics, gap-schema, and EP utility files. Changes to `sklearn_execution_time_predictor.py` are imports and routing only. Neither metrics file nor the gap schema required changes. All edits stayed within the approved scope; scratch holds caches, bytecode-free test outputs, and diagnostic artifacts.

Scratch root, abbreviated **S** below: `/data/ycfeng/tmp/codex-implement/86155181-0e36-4ecf-9c67-6e5ed04b9aa2/b16b`.

The B16/B16b assignments, prior Codex report and prerequisite evidence, lead part 1 report, applicable repository instructions, requirements/plan, design gates, and coverage records were inspected. Explicit decisions were preserved: mHC/endpoints use non-layer contracts; shared expert/MoE use resolved layer contracts; operators/MoE use measured coordinates only; duplicate coordinates use the existing trainer mean; composition and rank algebra are reused; runner rows/phases and PD admission remain B17. No table, promotion script, GPU producer, collective consumer, or task record was edited. No GPU/network/platform actions, package installations, delegation, publishing, or commits were performed.

No tracked pre-existing source edits were encountered at the initial inspection. The existing `.cortexkit/`, `.pi/`, and `remaining_measurements/` directories were preserved.

**2. Changes**

All paths below are relative to the working root. Counts are physical Python lines before → after; “new” identifies new files.

| Changed source file | Lines | Behavior and reason |
|---|---:|---|
| `frontier/attention/families.py` | 710 → 706 | Enabled `DEEPSEEK_V4_ATTENTION_FAMILY` and removed its disabled reason. Structural execution and operator target declarations remain unchanged. |
| `frontier/operators/families.py` | 969 → 968 | Enabled `DEEPSEEK_V4_MHC_FAMILY` and removed its disabled reason. Its operators retain `predictor_target=False`. |
| `frontier/operators/deepseek_v4_axes.py` | 19 → 30 | Added torch-free `select_deepseek_v4_operator_tp`: attention TP for `ATTENTION_TP`, 1 for `REPLICATED`. Producer and consumer share this rule. |
| `frontier/profiling/linear_op/deepseek_v4_plan.py` | 123 → 124 | `build_operator_plans` calls the shared TP selector. Serialized producer plans remain identical. |
| `frontier/execution_time_predictor/dsv4_operator_predictor.py` | new, 257 | Added complete operator/MoE loaders, contract/backend/identity/target validation, exact time/count lookup, runtime row-axis binding, and measured balanced-lane validation. |
| `frontier/execution_time_predictor/dsv4_training.py` | new, 51 | Added one `train_dsv4_models` entry used by independent and shared training. Caller-owned fitting, caching, duplicate reduction, and paired kernel-count training are retained. |
| `frontier/execution_time_predictor/dsv4_runtime.py` | new, 56 | Added the segment-owner table, the trained attention composition caller, and immutable operator-gap publication. Attention scope gaps are not charged again by generic `add_kernel_gap`. |
| `frontier/execution_time_predictor/dsv4_measurement.py` | new, 20 | Added family-keyed KERNEL_ONLY selection and the typed `skip_cpu_overhead_modeling=False` configuration error. |
| `frontier/execution_time_predictor/base_execution_time_predictor.py` | 496 → 498 | Calls the V4 host-time validation at construction. No admission bypass was introduced. |
| `frontier/execution_time_predictor/deepseek_v4.py` | 807 → 676 | Standalone/shared/gate/GEMM scopes use trained predictions and counts in non-dummy mode. Uses composed attention, explicit zero host fields, and a typed `kv_cache_block_zero` coverage error. Removed V4 shuffling ownership and the isolated readers. |
| `frontier/execution_time_predictor/dsv4_predictor.py` | 333 → 338 | Balanced-routing validation can validate MoE identity columns without requiring attention’s routing descriptor blob. Attention retains its existing strict descriptor validation. |
| `frontier/execution_time_predictor/kernel_gap.py` | 209 → 226 | Loads all four mandatory V4 segment gaps at attention TP. Preserves dummy/step-cost handling. The V4 layer wrapper publishes segment gaps once instead of applying generic layer gaps. |
| `frontier/execution_time_predictor/moe_dataset_training.py` | 502 → 504 | Skips generic MoE training after the common V4 trainer has supplied its models. |
| `frontier/execution_time_predictor/profiling_metadata.py` | 123 → 132 | Added the torch-free `CHECKPOINT_MXFP4_BACKEND_IDENTITIES` qualification table beside the existing linear backend registry. |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 620 → 632 | Routes every V4 role to KERNEL_ONLY and the common V4 trainer; skips generic compute/FFN/residual/CPU trainers. |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8,626 → 8,640 | Imports and early routing for measurement selection, common training, and prediction registration only. |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 1,621 → 1,614 | Removed the blanket rejection of trained structural V4 pricing. Runtime coverage is checked by the measured consumers. |
| `frontier/scheduler/utils/expert_parallel.py` | 832 → 833 | V4 EP lane pricing now passes through `_predict_layer_with_kernel_gap`. |

The backend qualification table is a local design choice within the assignment: it centralizes the accepted MARLIN/mxfp4 identity rather than embedding backend literals in loader control flow. The existing linear backend registry contains the required FP8 DeepGEMM identity. No model architecture layer contract was added.

The plan-level `num_tensor_parallel_workers` must not equal every per-operator TP: descriptor plans mix embedding TP8 with replicated head/norm TP1. The loader checks each typed operator entry against the deployment and registry. A direct probe also confirmed V4 has `num_kv_heads=1` and the existing generic attention-TP resolver returns 8 for the requested TP8; the new standalone route uses the producer’s explicit shared rule.

| Changed test file | Lines | Before → after / coverage |
|---|---:|---|
| `tests/unit/test_attention_execution_enabled_families.py` | 72 → 72 | Disabled-family exclusion becomes positive enabled-iterator/catalog coverage. |
| `tests/unit/test_operator_registry.py` | 262 → 263 | Enabled registry view now includes V4 attention; unrelated disabled-family coverage remains. |
| `tests/unit/test_dsv4_standalone_operator_plans.py` | 210 → 209 | Disabled mHC assertion becomes enabled admission; producer non-layer contracts now validate. |
| `tests/unit/test_dsv4_operator_manifest.py` | 200 → 199 | Canonical family queries now pass; inventory, target flags, and frozen V4 layer views remain covered. |
| `tests/unit/test_dsv4_model_config.py` | 231 → 230 | Closed attention admission becomes positive execution admission. |
| `tests/unit/test_dsv4_precision_metadata.py` | 427 → 427 | Both canonical families are enabled; precision inventory assertions remain. |
| `tests/unit/test_dsv4_parameter_memory.py` | 324 → 323 | Enabled admission replaces the old rejection assertion; memory totals remain covered. |
| `tests/unit/test_comm_operator_families.py` | 1,099 → 1,098 | Normal and structural V4 family admission both pass; communication timing checks remain. |
| `tests/unit/test_dsv4_source_execution.py` | 337 → 340 | Closed admission becomes a missing-measured-model coverage test naming the embedding operator. Dummy coverage remains. |
| `tests/unit/test_dsv4_attention_workload.py` | 365 → 365 | Positive enabled/structural admission assertion; renamed the stale closed-admission test. |
| `tests/unit/test_dsv4_cache_layout.py` | 274 → 268 | Removed obsolete family-guard bypasses/rejections. Positive capacity/dummy tests remain; real trained construction is covered by the new consumer fixture. |
| `tests/unit/test_glm53_model_config.py` | 276 → 251 | Removed an incomplete V4 constructor fixture that only exercised the old disabled guard; the real trained constructor supersedes it. Updated positive family admission. |
| `tests/unit/test_glm53_attention_cost_contract.py` | 345 → 343 | V4 loader admission passes and missing files still raise. Standalone `AttentionTrainer` construction now passes; its actual training is not claimed as verified. |
| `tests/unit/test_dsv4_graph_measured.py` | 215 → 130 | Removed isolated-reader tests/fixture after migration to the new loaders. Geometry and PD guard tests remain. |
| `tests/unit/test_kernel_gap_pricing.py` | 336 → 337 | Completed a non-V4 model stub with attention head counts. |
| `tests/unit/test_measurement_family_selector.py` | 1,006 → 1,011 | Completed selector/model fixtures with valid model topology. |
| `tests/unit/test_dp_dummy_forward_pricing.py` | 203 → 205 | Added the actual GLM model configuration to selector-only fixtures. |
| `tests/unit/test_device_timer_contract.py` | 161 → 165 | Added a valid dense model configuration to the ROCm measurement-selector fixture. |
| `tests/unit/test_dsv4_operator_consumer.py` | new, 328 | 19 cases: admitted tables, unchanged plans, exact coordinates/counts, all-lane class lookup, A1 forwards, composition/gaps, both trainers/cache parity, measurement roles, host/missing-gap/runner errors, and EP wrapper use. |
| `tests/unit/test_dsv4_operator_admission.py` | new, 73 | 14 cases: missing/invalid contracts, identities, targets/counts, EP/class/backend coverage, shared projection backend, and layer TP mismatch. |

Deleted symbols: `read_isolated_scope` and `_read_isolated_component` from `deepseek_v4.py`. Deleted superseded code also includes their isolated-reader tests/fixture, the obsolete disabled-admission constructor test, disabled-family assertions/bypasses, and `moe_shuffling` in the V4 routed phase. No files were deleted. V4 no longer creates or prices shuffling workloads; its work remains inside the grouped-GEMM envelope.

`frontier/metrics/metrics_store.py` remained unchanged at 5,768 lines. `frontier/metrics/constants.py` already contains every V4 attention/standalone/collective name, so no catalog additions were necessary. All touched source modules other than the explicitly permitted sklearn file remain below 2,000 lines.

**Flag consumers and other-model evidence**

- mHC: `bind_operator_query` and typed-contract validation now admit its operators. The enabled operator-registry view includes mHC. Its generic training target flags remain false.
- Attention: the enabled attention iterator includes V4; the operator registry’s shared attention-family entry also becomes enabled. Base predictor/cache admission passes, while supported cache tuple checks remain active.
- `quantization_manager._attention_compute_operations` and `profiling_mapping.validate_attention_catalog_alignment` now visit V4, but its profiling/E2E target lists are empty. Before/after operation and required metric sets are identical: 15 names each.
- `time_components` already used `include_structural=True`, so its canonical names and timing attributes remain identical: 88 names and 39 attribute entries. V4’s `LayerAttentionSpec.is_full_attention` remains false because its layout is COMPRESSED_KV.
- `binding.py:367` computes generic per-layer frozen state from the flag. V4 takes the earlier specialized layer-manifest path; its frozen inventory views remain unchanged. Other families’ flags and generic manifest behavior are unchanged.
- Profiling mapping’s enabled-family guards and standalone `AttentionTrainer` construction now pass for V4. This report verifies the assigned runtime trainer, not an expanded standalone training workflow.

A fresh-process before/after flag-consumer probe using HEAD source copies confirmed these set comparisons; evidence is `S/flag_consumers.json` and `S/flag_consumers_probe.py`. The CPU regression and dummy matrix found no additional prediction/metric regression in the exercised other-model paths. The full fidelity matrix remains a lead check.

**3. Verification**

All commands ran from the repository root with Python **PY** = `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6). `S/env.sh` establishes `ulimit -v 31457280`, the required PYTHONPATH, `PYTHONDONTWRITEBYTECODE=1`, disabled W&B, scratch TMPDIR/FRONTIER_TMP_ROOT, single-thread BLAS/OpenMP, and the Python environment’s bin directory in PATH. Broad and final consumer runs used `LOKY_MAX_CPU_COUNT=1`. Every tool command had a timeout.

| Check | Executed command / exact argv evidence | Expected and observed |
|---|---|---|
| Final trained consumer | `timeout 900s env LOKY_MAX_CPU_COUNT=1 "$PY" -m pytest tests/unit/test_dsv4_operator_consumer.py -q --tb=short -p no:cacheprovider --basetemp "$S/consumer_final_tmp"` | **PASS: 19 passed**, 217.27 s. Log: `consumer_final.log`. |
| Admission cases | Standalone pytest of `test_dsv4_operator_admission.py`, scratch `admission_final_tmp`, then included in the final broad run | **PASS: 14 passed** standalone, 1.77 s; all 14 passed again in the final broad run after the fractional-count fixture dtype correction. |
| Corrected selector fixtures | `timeout 90s` with `"$PY" -m pytest tests/unit/test_dp_dummy_forward_pricing.py tests/unit/test_device_timer_contract.py tests/unit/test_dsv4_attention_workload.py -q --tb=short -p no:cacheprovider --basetemp "$S/fixture_final_tmp"` | **PASS: 80 passed**, 1.87 s. Log: `fixture_final.log`. |
| Final affected-module regression | `subprocess.run(["timeout", "330s", *argv])`, with complete exact argv in `S/affected_rerun_command.json` | **FAIL with documented existing prerequisites/artifacts: 2,517 passed, 18 failed, 3 skipped, 1 deselected**, 150.24 s. Log: `affected_rerun.log`. 107 test files plus `predictor_cache_fixtures.py`. |
| Placement before | `timeout` pytest of `tests/integration/test_vllm_dp_placement_runtime.py`, `--basetemp "$S/placement_before_tmp"` | **18 passed, 1 failed**, 70.00 s. Log: `placement_before.log`. |
| Placement after | Same placement suite, `--basetemp "$S/placement_after_tmp"` | **18 passed, same 1 failed**, 84.44 s. Log: `placement_after.log`. |
| Dummy smoke matrix | `timeout 600s env METRICS_OUTPUT_DIR="$S/dummy_matrix_outputs" bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir "$S/dummy_matrix_cache"` with PYTHON_BIN from the common environment | **PASS: all 12 cases**, exit 0. Log: `dummy_matrix.log`. |
| Complete asset admission | `timeout 25s "$PY" -` calling all three new/current loaders and the four-class gap reader | **PASS:** attention 4,945; linear 41 source rows → 80 operator records; MoE 33 rows. Evidence: `final_asset_admission.json`. |
| Fixed asset hashes | `timeout 30s "$PY" -` rehashing every manifest output | **PASS:** all four hashes equal the committed manifest. Evidence: `final_asset_hashes.json`. |
| Flag-consumer comparison | Fresh child processes run `flag_consumers_probe.py before/after` under timeout | **PASS:** only the two intended family objects change admission; catalog/quantization/timing sets remain identical. |
| Cold imports | Fresh child interpreters importing each of `dsv4_predictor`, `dsv4_operator_predictor`, `dsv4_training`, `dsv4_runtime`, `dsv4_measurement`, and `profiling_metadata` | **PASS:** each exit 0, blocked-module list `[]`; no torch, vllm, or cuda module loaded. Exact child commands: `cold_imports_final.json`. |
| Unused imports | `/data/ycfeng/tmp/unused_imports.py` on every changed/new Python file, compared with HEAD copies | **PASS for introduced code:** zero new findings. Raw scanner findings already existed in base/sklearn/sklearn-MoE modules and the kernel-gap test fixture; retained within the narrow cleanup scope. Evidence: `unused_final.txt`, `final_hygiene.json`, `baseline_source/`. |
| Whitespace | `timeout 20s git diff --check` | **PASS**, exit 0, no output. |

The known default collective-count test was deselected as required: `tests/unit/test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default`.

The separate collection attempt for `test_mla_native_profiling_wrapper.py` could not run because it imports Torch, which is absent from the prescribed CPU environment. It was excluded from the final CPU collection; no dependency was installed.

**Acceptance results**

- **Prerequisites:** all 41 linear source rows validate, comprising 65 mHC/endpoint non-layer contracts and 15 shared-expert layer contracts. All 33 MoE rows validate, comprising 132 contracts. Selected TP/mode mismatches raise `DeepSeekV4CoverageError` naming the operator. The serialized producer-plan SHA-256 remains `184bdcfc08f720b25b60a35a14aaa875691a0156329824513dfff3f36f919833`, matching `S/plans_before.json`.
- **A1:** real non-dummy construction and every-layer/endpoints pricing pass with `_dummy_execution_time` reads forbidden. Attention uses the public stage method for all 43 layers. FFN uses the actual layer gap wrapper and stage assembly for all 43 layers; an additional EP wave test verifies that every one of the 16 lanes crosses the wrapper.
- **Exact lookup/rank:** all measured standalone, gate, and grouped-GEMM coordinates/counts pass, covering EP16 and EP32 lane bindings. L=5, uncovered MoE population, and a skewed vector with the same total raise coverage errors. The existing arithmetic-mean duplicate reduction at `sklearn_execution_time_predictor.py:193` is reused. Floating-point reduction comparisons use relative tolerance 1e-14; single-row coordinates compare exactly.
- **Composition/gaps:** sentinel checks pass below and above the multi-stream threshold under FULL, PIECEWISE, and NONE. LM head/logits gather take eager_preparation. Attention owns its trained scope gaps and adds no generic layer gap. Missing mandatory `kernel_gap.csv` raises. All seven V4 collective specs have kernel_count=1; positive collectives receive one gap in their segment, zero-cost collectives receive none. EP dispatch/combine gaps stay with those physical phase operators.
- **Both trainers/A9:** independent, warm-cache, and shared routes have identical model names, exact prediction maps, paired count maps, and tested attention stage ledgers. Cold/warm cache filename identities are unchanged. There are 568 registered base models: 554 attention time/gap models and 14 standalone/gate/GEMM models. Every time model receives its paired count model. No generic compute/FFN/residual/CPU model is registered. Full names are in `consumer_final_tmp/v4_models0/model_names.json`.
- **Host/runner:** all five host fields are zero in both non-dummy forwards. `skip_cpu_overhead_modeling=False` raises `DeepSeekV4CoverageError` during construction. A prefill with new cache blocks raises that typed error naming `kv_cache_block_zero`.
- **New tests:** 33 passing parametrized cases across the two new files.

The A1 stage components below are milliseconds from the CPU acceptance harness. Communication uses the deterministic `RecordingCCBackend`; these totals are acceptance evidence, not native E2E latency qualification. FFN components represent the assembled lane-0 stage. Separate tests cover the other lane classes and EP wrapper routing.

| Forward | Attention elapsed | pre_dispatch | dispatch | routed_compute | combine | post_combine |
|---|---:|---:|---:|---:|---:|---:|
| FULL decode | 354.7674725491188 | 1.5558070676215736 | 129.02140256813524 | 4.494559476945921 | 215.02140256813507 | 6.047904670794464 |
| NONE prefill | 448.86340438021466 | 4.8573773724124445 | 129.02341801326236 | 56.979869092837475 | 215.02341801326273 | 6.256668654062127 |

FULL uses 16 one-token decode requests at context 6,144. NONE uses one 6,144-token prefill request with context zero and no new blocks. Raw per-layer/operator ledgers are `S/consumer_final_tmp/v4_models0/a1_FULL.json` and `a1_NONE.json`. `schedule_time`, `sampler_e2e_time`, `prepare_inputs_e2e_time`, `process_model_outputs_time`, and `ray_comm_time` are zero.

**Part 1/repeat evidence carried forward, not rerun:** promotion determinism, the 4,945 attention identity comparison, and the 227 KERNEL_ONLY/CUDA_EVENT coordinate comparisons were inspected in the B16 report and `/data/ycfeng/tmp/codex-implement/54e2e32d-92b5-49b8-b578-4119b950ab6b/b16/asset_validation.json`. The per-operator/L EP16/EP32 repeat table remains there; medians agree within 1%, with the largest reported difference at shared up-projection L=1, 8.8485 versus 8.7685 µs. B16b did not rebuild fixed assets. Their current counts/hashes match the manifest at `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/manifest.json`.

**Regression failures and diagnosis**

The final broad run’s 18 failures consist of:

- Ten tests in `test_colocation_release_review_contracts.py` referencing the absent `frontier.config_optimizer` module or absent legacy `tests/debug/e2e-level/monolith_mode/scripts/` files. `git show HEAD:<path>` confirms those sources are absent from HEAD as well as the filesystem.
- Two MHA/MQA stage-2 CLI tests whose subprocesses report `flashinfer-python is not installed in the active Python env`. The exact diagnostic subprocess commands and stderr are in `S/stage2_diagnosis.json`.
- Five tests in `test_profiling_governance_minimal_red.py` importing the real linear/MoE profiling wrappers, which fail with `ModuleNotFoundError: No module named 'torch'`.
- `test_dsv4_runner_operators.py::test_package_check_writes_no_destination`, which asserts `package_03` while HEAD’s runner plan specifies `package_04`. This mismatch is present in both HEAD files. Evidence: `S/existing_failures_evidence.json`.

These are observed prerequisites/artifact mismatches; their independence from this implementation follows from unchanged source paths, missing installed dependencies, and HEAD evidence. They were not repaired outside the assignment.

The unchanged placement failure before and after is `tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`: lane 2 starts as `(0, False)` while the test expects `(0, True)`.

An earlier broad run had 26 failures. The eight additional selector/assertion problems were corrected within the test scope; the final run has only the 18 groups documented above. Initial harness construction also lacked `ReplicaConfig.cluster_num_replicas`; the fixture was corrected to the canonical value 1, without adding a source fallback. Earlier stale cache tests attempted generic training after family admission opened; those obsolete closed-admission expectations were superseded by the real trained fixture. No unsuccessful result is counted as a pass.

**4. Remaining work and decisions for Claude**

No assigned B16b runtime implementation remains pending. B17 still owns runner rows/phases, PREFILL/unified-DECODE admission, runtime-limitations text, the dummy workload entry, and the ledger/module-size records. The existing attention/serving guards still reject PD execution. E2E replays, the diagnostic cross-check, and the 77-case fidelity matrix were not run and remain with the lead.

The repository’s existing V4 runtime-limitations warning and some structural provenance wording are now stale relative to this trained route; those records were explicitly outside B16b. The pandas attention CSV reader still emits its existing mixed-type warning for columns 49/50, without rejecting rows.

Recommended independent follow-ups are to reconcile the runner package assertion with the committed package_04 plan, decide how to handle tests for removed legacy sources, and run profiling-dependent tests in the appropriate existing environment. Absolute scanner-zero cleanup would also require addressing the inherited unused/type-annotation findings; this change introduces none.

**5. Review handoff**

Review the new loader’s non-layer versus layer validation, shared TP selector, exact lookup/no-fallback behavior, balanced-vector validation, and backend qualification table. Review the common trainer registrations and the segment gap/composition ownership, especially the EP wave wrapper. Confirm that sklearn changes remain routing-only and that the updated sentinels retain topology/cache/PD protections.

Rerun the two new test files with the prescribed CPU environment and scratch basetemp, the affected regression argv, and the dummy smoke matrix if review changes runtime code. Preserve the known placement failure comparison. The complete broad failure log and evidence allow review without assuming all repository tests pass.

Source changes, checks, and evidence are ready for Claude’s review and commit. This report does not claim acceptance, native E2E fidelity, or completion of B17.
