## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Audited every changed predictor hunk against 0515589a; fixed construction ownership, MTP range and homogeneous snapshot duplication; recorded bounded module split decisions. |

# Predictor changed-hunk review

Final source supplement: the ledger includes the profile-supported non-MoE disaggregation homogeneous reuse and GDN missing-artifact fail-fast. The former reuses finalized numerics only for non-MoE, no explicit MoE, zero-GDN models; four role-specific exact oracle regressions prove stage values/operator maps unchanged. The latter resolves model-owned layer identity before requiring loaded GDN artifacts and cannot enter dense lookups when those artifacts are absent. Final related validation: **104 passed in 14.52 s**, recorded in `test_report_2026-09-16_predictor_owner_once.md`. Earlier performance observations predate these edits; final acceptance remains pending the root-owned paired run.

The separately pending mutable-source microbenchmark has now completed on frozen commit `f9099f85` in the exclusive performance window. `w04_performance_review.md` records all seven component medians and raw samples: one snapshot 17.412628 microseconds, stage with eight mutable layers 137.845563 microseconds, stage with eight finalized layers 5.185594 microseconds. This closes the component-cost supplement only; root owns final paired Simulator timing.

Post-pairing source follow-up: root authorized reusing the existing `StageExecutionTime.from_execution_time` from `_assemble_stage` only for repeated references to the same numerical object and uniform family/variant specs. Every model spec is still resolved, preserving range validation; mixed identities and independently predicted MoE values keep per-layer construction. The five base-module hunk ranges below were refreshed for this change. Focused evidence is **132 passed in 5.59 s** at `/data/ycfeng/tmp/pr33-homogeneous-aggregate-final.log`; the new spies prove one block calculation for dense/PDD homogeneous stages and unchanged invalid-end rejection. This additional change is relative to frozen `f9099f85`; final source/commit and paired timing remain root-owned.

Reviewer: `/root/w02_acceptance_tests`. Source baseline: `0515589a`. Candidate: `5a1cc2af280e5ae2d51caae55fe016c80bc3e2d2` plus current working diff. This report covers all 179 zero-context production hunks in the 10 files below. The source snapshot was read with `git diff --unified=0 0515589a -- frontier/execution_time_predictor`, surrounding function reads and actual caller searches. Raw diff: `/data/ycfeng/tmp/pr33-predictor-hunks.diff`. This is a predictor-scope report, not complete branch acceptance. No agent commit; integration owner records the eventual commit SHA.

## Findings and disposition

- **Fixed:** stage-owned values were charged once but constructed repeatedly in MoE/disaggregation multi-layer prediction. First-layer-only construction now skips later overhead/PP/proposer/terminal lookups. Disaggregation additionally repeated its current-stage handoff lookup inside one owner layer; the existing overhead constructor now supplies that value. Consumer residual lookups for the previous PP stage remain distinct and required.
- **Fixed:** MoE terminal replay combined every physical offset with the configured full-stage count, causing a valid eight-layer replay at offset1 to request invalid layer8. The public stage passes its actual count to the first owner only. Full eight-layer and offset5/count3 regressions execute the real terminal replay with valid metadata.
- **Fixed:** homogeneous dense stage assembly repeatedly snapshotted the same mutable numerical payload. One finalized snapshot is now shared across distinct physical identities. Source mutation isolation remains the entity contract.
- **Cleaned:** stale aggregate ExecutionTime docs/return annotation, unused MoE imports and incomplete-model/manager mock accommodations. Production assumes required model layer specs and the declared manager API; existing numerical tests use normal constructors or declared scoped hooks.
- **Retained boundary:** no new request cancellation, routing mutation/version API, global numerical attention cache, diagnostic execution IR, or source runtime mode flag. None is required by reachable supported callers in this scope.
- **Not claimed:** native AMD/NVIDIA numerical parity, whole-branch PASS, final performance acceptance, exhaustive review of unchanged giant-module bodies, or guarantees for concurrent mutation during a synchronous stage prediction.

## Actual caller and contract references

`frontier/simulator.py` creates predictors with the shared manager and actual replica IDs; `frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py` calls the public stage API. Terminal MTP helpers in sklearn call that same API with explicit stage/layer range. The public stage producers return `StageExecutionTime`; `ExecutionTime` means one physical layer. The model config owns `get_layer_attention_spec`; layer identity is never inferred from timing numbers. GDN native operation metadata is loaded through the standalone manifest boundary before numerical prediction.

| Review key | Disposition / requirement | Required/optional data, ownership, reused mechanism and rationale | Real caller chain | Regression or evidence |
| --- | --- | --- | --- | --- |
| TP | RETAIN; R01/W01 | Architecture-profile attention-linear op set; canonical profile field is required. No model-name branching. | `get_attention_linear_tp_policy_ops → training/profiling TP policy consumers` | test_attention_tp_effective_mapping.py |
| STAGE | REFACTOR/FIX; R05/R06/W03 | Physical numerical payloads; required model layer spec and explicit global IDs. Stage assembly owns stage scope; count never changes an ExecutionTime getter. | `replica_stage_schduler.schedule → predict_stage_execution_time → _assemble_stage` | test_stage_execution_time.py; test_dense_execution_time_layer_scaling.py; test_attention_query_cache.py |
| ATOMIC | REFACTOR; W06/W09 | One same-directory atomic destination for JSON/pickle; temporary cleanup on error. Dataset fingerprint identifies expensive trained-artifact reuse. | `GDNTrainer manifest/artifact publication; shared model cache writes` | test_gdn_artifact_boundary.py; test_gdn_training_predictor_increment8.py |
| GDN | RETAIN; R01/R07/W06 | Standalone pretrained manifest owner. Required exact artifact identity/task/features/bounds, finite nonnegative output. Explicit inactive phase zeros; no fitting, TP clipping or phase substitution. | `ExecutionTimePredictionModelManager._load_gdn_predictor_for_cluster → GDNPredictor.from_directory; predict_attention_layer_time → predict_attention_time` | test_gdn_artifact_boundary.py; test_gdn_training_predictor_increment8.py; test_gdn_hybrid_e2e_increment14ab.py |
| PATH | EXTRACT/REFACTOR; R07/N03/W06 | Shared immutable six-field resolution result; preserve explicit empty paths and legacy aliases at boundary, configured overrides, role paths, and public dictionary/tuple projections. Device metadata selects event family and malformed supplied metadata fails. | `manager get_training_file_paths/_resolve_measurement_input_files_for_config and sklearn _initialize_file_paths/_get_input_files` | test_measurement_path_precedence.py; test_measurement_family_selector.py; test_attention_measurement_csv_equivalence.py |
| FAMILY | RETAIN/REFACTOR; R01/R07/W01/W06 | CUDA_EVENT and DEVICE_EVENT retain separated family maps, precision/model registries, active-family validation and topology keys. Valid ReplicaConfig supplies device metadata; kernel-only policy remains role/config dependent. | `Simulator.__init__ → shared manager construction/train → predictor initialization/measurement selection → get_model` | test_measurement_family_selector.py; test_attention_measurement_csv_equivalence.py; test_gdn_hybrid_e2e_increment14ab.py |
| ROUTING | REFACTOR; R08/W04 | Canonical expert allocation generator replaces duplicate ratios. Real replica IDs required. Immutable exact per-layer EP workload cache is constructor-owned, bounded to256, and separate from numerical attention cache. | `Simulator passes actual_replica_ids; _get_moe_tokens_input → _materialize_layer_ep_workload → resolve_ep_lane_workload` | test_typed_ep_predictor_contract.py; test_moe_routing_conservation.py; test_moe_predictor_layer_id_semantics.py |
| CACHE | REFACTOR; R08/N04/W04 | Plain dict lifetime is one synchronous public stage; key is required model-owned family/variant. Batch, phase/context/padding/state, topology, quantization and loaded artifacts are fixed during that call. Direct helper without cache bypasses reuse. Distinct physical routing remains per layer. | `SklearnMoEExecutionTimePredictor.predict_stage_execution_time → private physical helper → attention cache helper` | test_attention_query_cache.py; w04_cache_review.md; w04_performance_review.md |
| OWNER | FIX; R05/N02/W03 | First physical layer alone constructs CPU/PP/proposer/terminal fields. Later layers skip stage lookups but retain layer TP/EP/MoE numerics. Actual public range controls MTP replay. Dummy fields follow same owner. | `public MoE/disagg stage loops → direct private layer producer; MTP terminal row → public stage; StageExecutionTime reads owner once` | test_attention_query_cache.py; test_mtp_terminal_overshoot_ep_replay.py; test_sklearn_disaggregation_execution_time_predictor.py |
| DENSE | FIX/REFACTOR; R05/N02/W03 | Predict homogeneous numerical payload once, finalize mutable components once, attach explicit distinct identities, assemble once. No sentinel passthrough or implicit multiplier. Dense MTP uses actual requested range. | `SklearnExecutionTimePredictor.predict_stage_execution_time; scheduler stage path; terminal replay public call` | test_dense_execution_time_layer_scaling.py; test_mla_predictor_runtime_operator_times.py |

A key in the ledger below binds that exact hunk to the caller, field/ownership contract, reused mechanism, rationale, requirement and regression listed above. Module/import hunks inherit the responsibility they enable; they do not create separate runtime owners. Range notation is `start,count`, matching Git; count0 means insertion/deletion anchor. All listed hunks have been source-reviewed. Verification is bounded by the explicit evidence section, rather than inferred from an exit code or hunk label.

## Critical-module size and responsibility decision

| File | Baseline LOC | Candidate LOC | Decision |
| --- | ---: | ---: | --- |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 4570 | 4646 | Canonical path precedence/substitution is extracted to measurement_input_paths.py; GDN artifact validation/load is delegated to GDNPredictor. Keep measurement-family registries and topology selection with their existing model-manager owner. Further unrelated training-family splits are deferred; no facade module added. |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | 3058 | 2985 | Remove recursive stage wrapping and duplicated allocation math; preserve role-specific operation filtering and EP contracts here. Stage assembly and allocation generator are existing shared seams. A further role split would move intertwined inherited PP/EP policy without removing a distinct new responsibility in this step. |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8315 | 8263 | Extract path resolution and GDN artifact prediction into cohesive modules; remove recursive stage conversion and duplicate finalization. Dense physical composition stays here with measurement/quantization hooks. Further operator-training separation is outside the bounded active repair. |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 3523 | 3584 | Reuse canonical routing materialization, stage assembler and bounded immutable EP cache; remove recursive layer stages. Preserve independent physical-layer routing here. A broad MoE extraction would expand constructor/topology interfaces; no correctness requirement remains that needs it. |

The size threshold triggers a responsibility decision, not mechanical file chopping. `measurement_input_paths.py` owns one reusable path policy (163 LOC); `gdn_predictor.py` owns one artifact-backed numerical boundary (318 LOC). Both replace coherent responsibilities rather than hiding reflective checks.

## Exact hunk ledger

### frontier/execution_time_predictor/attention_tp_policy.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 33,2 | 33,2 | `get_attention_linear_tp_policy_ops` | TP: RETAIN |

### frontier/execution_time_predictor/base_execution_time_predictor.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 10,1 | 10,1 | `<module/class>` | STAGE: REFACTOR/FIX |
| 02 | 54,0 | 55,34 | `_assemble_stage` | STAGE: REFACTOR/FIX |
| 03 | 76,1 | 110,1 | `_get_dummy_execution_time` | STAGE: REFACTOR/FIX |
| 04 | 406,1 | 440,1 | `predict_stage_execution_time` | STAGE: REFACTOR/FIX |
| 05 | 432,1 | 466,1 | `predict_stage_execution_time` | STAGE: REFACTOR/FIX |

### frontier/execution_time_predictor/cache_io.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 4,0 | 5,3 | `<module/class>` | ATOMIC: REFACTOR |
| 02 | 12,2 | 15,3 | `<module/class>` | ATOMIC: REFACTOR |
| 03 | 28,2 | 32,2 | `_atomic_destination` | ATOMIC: REFACTOR |
| 04 | 40,0 | 45,24 | `<module/class>` | ATOMIC: REFACTOR |

### frontier/execution_time_predictor/gdn_predictor.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 0,0 | 1,318 | `<module/class>` | GDN: RETAIN |

### frontier/execution_time_predictor/measurement_input_paths.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 0,0 | 1,163 | `<module/class>` | PATH: EXTRACT/REFACTOR |

### frontier/execution_time_predictor/random_forrest_execution_time_predictor.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 88,3 | 88,10 | `__new__` | ROUTING: REFACTOR |

### frontier/execution_time_predictor/shared_prediction_model_manager.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 38,0 | 39,5 | `<module/class>` | FAMILY: RETAIN/REFACTOR |
| 02 | 420,0 | 426,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 03 | 426,0 | 433,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 04 | 428,0 | 436,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 05 | 430,0 | 439,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 06 | 434,0 | 444,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 07 | 436,0 | 447,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 08 | 534,0 | 546,2 | `_measurement_family_name` | FAMILY: RETAIN/REFACTOR |
| 09 | 538,0 | 552,6 | `<module/class>` | FAMILY: RETAIN/REFACTOR |
| 10 | 566,1 | 585,1 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 11 | 567,0 | 587,5 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 12 | 570,1 | 594,1 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 13 | 572,1 | 596,1 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 14 | 576,1 | 600,1 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 15 | 580,1 | 604,1 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 16 | 588,1 | 612,1 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 17 | 591,2 | 615,2 | `_get_measurement_types_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 18 | 598,43 | 622,15 | `_resolve_measurement_input_files_for_config` | PATH: EXTRACT/REFACTOR |
| 19 | 731,1 | 727,3 | `_train_all_required_models` | FAMILY: RETAIN/REFACTOR |
| 20 | 732,0 | 731,6 | `_train_all_required_models` | FAMILY: RETAIN/REFACTOR |
| 21 | 749,1 | 753,9 | `_train_all_required_models` | FAMILY: RETAIN/REFACTOR |
| 22 | 833,0 | 846,55 | `_load_gdn_predictor_for_cluster` | GDN: RETAIN |
| 23 | 2122,1 | 2189,1 | `_train_attn_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 24 | 2220,1 | 2287,1 | `_train_attn_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 25 | 2252,1 | 2319,1 | `_train_attn_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 26 | 2291,0 | 2359,6 | `_is_mla_family` | FAMILY: RETAIN/REFACTOR |
| 27 | 4101,0 | 4175,1 | `_contract_model_registry` | FAMILY: RETAIN/REFACTOR |
| 28 | 4116,0 | 4191,1 | `_contract_precision_registry` | FAMILY: RETAIN/REFACTOR |
| 29 | 4129,0 | 4205,1 | `_legacy_model_registry` | FAMILY: RETAIN/REFACTOR |
| 30 | 4144,0 | 4221,1 | `_legacy_precision_registry` | FAMILY: RETAIN/REFACTOR |
| 31 | 4357,1 | 4434,1 | `get_model` | FAMILY: RETAIN/REFACTOR |
| 32 | 4372,0 | 4450,4 | `get_model` | FAMILY: RETAIN/REFACTOR |
| 33 | 4380,0 | 4462,4 | `get_model` | FAMILY: RETAIN/REFACTOR |
| 34 | 4440,1 | 4525,1 | `get_models` | FAMILY: RETAIN/REFACTOR |
| 35 | 4443,0 | 4529,12 | `get_models` | FAMILY: RETAIN/REFACTOR |
| 36 | 4449,0 | 4547,7 | `get_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 37 | 4451,2 | 4555,2 | `get_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 38 | 4454,0 | 4559,1 | `get_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 39 | 4460,2 | 4565,2 | `get_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 40 | 4465,0 | 4571,1 | `get_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 41 | 4468,1 | 4574,1 | `get_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 42 | 4484,1 | 4590,1 | `get_models_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 43 | 4502,36 | 4608,6 | `get_training_file_paths` | PATH: EXTRACT/REFACTOR |

### frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 4,1 | 3,0 | `<module/class>` | STAGE: REFACTOR/FIX |
| 02 | 24,1 | 23,1 | `<module/class>` | STAGE: REFACTOR/FIX |
| 03 | 39,1 | 38,5 | `<module/class>` | STAGE: REFACTOR/FIX |
| 04 | 506,32 | 509,9 | `_generate_expert_allocations` | ROUTING: REFACTOR |
| 05 | 604,0 | 585,1 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 06 | 690,1 | 671,1 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 07 | 749,1 | 730,1 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 08 | 772,1 | 753,1 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 09 | 776,5 | 757,5 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 10 | 798,1 | 779,1 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 11 | 821,1 | 802,1 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 12 | 825,5 | 806,5 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 13 | 867,5 | 848,5 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 14 | 905,5 | 886,5 | `_get_dummy_execution_time_for_cluster` | OWNER: FIX |
| 15 | 985,1 | 966,1 | `_get_zero_decode_ffn_ep_barrier_execution_time` | OWNER: FIX |
| 16 | 1042,0 | 1024,1 | `_get_communication_time` | OWNER: FIX |
| 17 | 1070,1 | 1052,1 | `_get_communication_time` | OWNER: FIX |
| 18 | 1242,0 | 1225,1 | `_predict_attention_only_stage_execution_time` | OWNER: FIX |
| 19 | 1259,1 | 1242,5 | `_predict_attention_only_stage_execution_time` | OWNER: FIX |
| 20 | 1261,1 | 1247,0 | `_predict_attention_only_stage_execution_time` | OWNER: FIX |
| 21 | 1264,1 | 1250,1 | `_predict_attention_only_stage_execution_time` | OWNER: FIX |
| 22 | 1377,0 | 1364,39 | `predict_stage_execution_time` | OWNER: FIX |
| 23 | 1380,1 | 1405,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 24 | 1387,3 | 1412,2 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 25 | 1490,0 | 1515,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 26 | 1493,94 | 1518,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 27 | 1623,0 | 1556,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 28 | 1629,7 | 1561,0 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 29 | 1641,0 | 1568,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 30 | 1643,3 | 1570,3 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 31 | 1664,1 | 1591,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 32 | 1911,1 | 1838,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 33 | 2079,1 | 2006,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 34 | 2303,1 | 2230,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 35 | 2446,1 | 2373,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 36 | 2670,1 | 2597,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |
| 37 | 2894,1 | 2821,1 | `_predict_disaggregated_layer_execution_time` | OWNER: FIX |

### frontier/execution_time_predictor/sklearn_execution_time_predictor.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 33,0 | 34,2 | `<module/class>` | FAMILY: RETAIN/REFACTOR |
| 02 | 36,1 | 38,4 | `<module/class>` | FAMILY: RETAIN/REFACTOR |
| 03 | 72,0 | 78,5 | `<module/class>` | FAMILY: RETAIN/REFACTOR |
| 04 | 133,1 | 143,1 | `<module/class>` | FAMILY: RETAIN/REFACTOR |
| 05 | 370,1 | 380,1 | `_get_attention_family` | FAMILY: RETAIN/REFACTOR |
| 06 | 425,0 | 436,5 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 07 | 540,0 | 556,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 08 | 542,0 | 559,1 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 09 | 559,0 | 577,3 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 10 | 562,5 | 582,4 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 11 | 571,2 | 590,6 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 12 | 578,0 | 602,3 | `__init__` | FAMILY: RETAIN/REFACTOR |
| 13 | 724,74 | 750,13 | `_initialize_file_paths` | PATH: EXTRACT/REFACTOR |
| 14 | 805,30 | 770,15 | `_get_input_files` | PATH: EXTRACT/REFACTOR |
| 15 | 839,0 | 790,2 | `_measurement_family_name` | FAMILY: RETAIN/REFACTOR |
| 16 | 843,0 | 796,16 | `<module/class>` | FAMILY: RETAIN/REFACTOR |
| 17 | 858,0 | 827,1 | `_get_default_measurement_type_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 18 | 866,1 | 835,1 | `_get_default_measurement_type_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 19 | 869,1 | 838,1 | `_get_default_measurement_type_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 20 | 873,2 | 842,2 | `_get_default_measurement_type_for_cluster` | FAMILY: RETAIN/REFACTOR |
| 21 | 878,0 | 848,1 | `_should_enable_measurement_family` | FAMILY: RETAIN/REFACTOR |
| 22 | 881,4 | 851,1 | `_should_enable_measurement_family` | FAMILY: RETAIN/REFACTOR |
| 23 | 887,1 | 854,1 | `_should_enable_measurement_family` | FAMILY: RETAIN/REFACTOR |
| 24 | 892,1 | 859,1 | `_should_enable_measurement_family` | FAMILY: RETAIN/REFACTOR |
| 25 | 914,0 | 882,1 | `_select_measurement_type_for_batch` | FAMILY: RETAIN/REFACTOR |
| 26 | 918,1 | 886,1 | `_select_measurement_type_for_batch` | FAMILY: RETAIN/REFACTOR |
| 27 | 936,1 | 904,1 | `_select_measurement_type_for_batch` | FAMILY: RETAIN/REFACTOR |
| 28 | 943,1 | 911,1 | `_select_measurement_type_for_batch` | FAMILY: RETAIN/REFACTOR |
| 29 | 952,0 | 921,6 | `_activate_measurement_type` | FAMILY: RETAIN/REFACTOR |
| 30 | 1157,5 | 1131,9 | `_require_predictions_for_measurement_type` | FAMILY: RETAIN/REFACTOR |
| 31 | 2908,0 | 2887,2 | `_get_attention_model_names` | FAMILY: RETAIN/REFACTOR |
| 32 | 3498,1 | 3478,1 | `_train_attention_layer_models` | FAMILY: RETAIN/REFACTOR |
| 33 | 4009,1 | 3989,1 | `_predict_for_attention_layer_models` | FAMILY: RETAIN/REFACTOR |
| 34 | 4021,1 | 4001,1 | `_predict_for_attention_layer_models` | FAMILY: RETAIN/REFACTOR |
| 35 | 7366,1 | 7346,2 | `predict_attention_layer_time` | GDN: RETAIN |
| 36 | 7368,0 | 7350,16 | `predict_attention_layer_time` | GDN: RETAIN |
| 37 | 7893,0 | 7891,16 | `predict_stage_execution_time` | DENSE: FIX/REFACTOR |
| 38 | 7905,1 | 7918,1 | `_predict_dense_layer_execution_time` | DENSE: FIX/REFACTOR |
| 39 | 7907,10 | 7920,4 | `_predict_dense_layer_execution_time` | DENSE: FIX/REFACTOR |
| 40 | 7922,3 | 7928,0 | `_predict_dense_layer_execution_time` | DENSE: FIX/REFACTOR |
| 41 | 7941,1 | 7945,2 | `_predict_dense_layer_execution_time` | DENSE: FIX/REFACTOR |
| 42 | 8258,58 | 8263,1 | `_predict_dense_layer_execution_time` | DENSE: FIX/REFACTOR |

### frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py

| ID | Base range | Candidate range | Surrounding owner | Review key / disposition |
| --- | --- | --- | --- | --- |
| 01 | 3,0 | 4,2 | `<module/class>` | OWNER: FIX |
| 02 | 15,1 | 17,1 | `<module/class>` | OWNER: FIX |
| 03 | 46,0 | 49,1 | `<module/class>` | OWNER: FIX |
| 04 | 251,0 | 255,6 | `<module/class>` | OWNER: FIX |
| 05 | 475,0 | 485,1 | `_get_dummy_execution_time` | OWNER: FIX |
| 06 | 565,1 | 575,1 | `_get_dummy_execution_time` | OWNER: FIX |
| 07 | 623,1 | 633,1 | `_get_dummy_execution_time` | OWNER: FIX |
| 08 | 646,1 | 656,1 | `_get_dummy_execution_time` | OWNER: FIX |
| 09 | 652,5 | 662,5 | `_get_dummy_execution_time` | OWNER: FIX |
| 10 | 690,0 | 701,1 | `__init__` | ROUTING: REFACTOR |
| 11 | 695,0 | 707,5 | `__init__` | ROUTING: REFACTOR |
| 12 | 797,29 | 813,6 | `_init_global_routing_allocations` | ROUTING: REFACTOR |
| 13 | 845,2 | 838,5 | `_build_shared_routing_details` | ROUTING: REFACTOR |
| 14 | 864,0 | 861,24 | `_build_shared_routing_details` | ROUTING: REFACTOR |
| 15 | 866,1 | 886,1 | `_build_shared_routing_details` | ROUTING: REFACTOR |
| 16 | 913,0 | 934,5 | `_materialize_layer_ep_workload` | ROUTING: REFACTOR |
| 17 | 917,0 | 943,2 | `_materialize_layer_ep_workload` | ROUTING: REFACTOR |
| 18 | 919,0 | 947,13 | `_materialize_layer_ep_workload` | ROUTING: REFACTOR |
| 19 | 928,2 | 968,2 | `_materialize_layer_ep_workload` | ROUTING: REFACTOR |
| 20 | 936,0 | 977,4 | `_materialize_layer_ep_workload` | ROUTING: REFACTOR |
| 21 | 2155,0 | 2200,48 | `_predict_attention_layer_time_with_query_cache` | CACHE: REFACTOR |
| 22 | 2166,0 | 2259,3 | `_get_execution_time_internal` | OWNER: FIX |
| 23 | 2169,1 | 2264,1 | `_get_execution_time_internal` | OWNER: FIX |
| 24 | 2185,3 | 2280,4 | `_get_execution_time_internal` | OWNER: FIX |
| 25 | 2212,1 | 2308,1 | `_get_execution_time_internal` | CACHE: REFACTOR |
| 26 | 2215,0 | 2312,1 | `_get_execution_time_internal` | CACHE: REFACTOR |
| 27 | 2223,1 | 2320,1 | `_get_execution_time_internal` | OWNER: FIX |
| 28 | 2387,2 | 2484,3 | `_get_execution_time_internal` | OWNER: FIX |
| 29 | 2390,2 | 2488,3 | `_get_execution_time_internal` | OWNER: FIX |
| 30 | 2394,0 | 2494,1 | `_get_execution_time_internal` | OWNER: FIX |
| 31 | 2398,1 | 2498,1 | `_get_execution_time_internal` | OWNER: FIX |
| 32 | 2413,1 | 2513,1 | `_get_execution_time_internal` | OWNER: FIX |
| 33 | 2419,1 | 2519,1 | `_get_execution_time_internal` | OWNER: FIX |
| 34 | 2424,0 | 2525,1 | `_get_execution_time_internal` | OWNER: FIX |
| 35 | 2426,1 | 2527,1 | `_get_execution_time_internal` | OWNER: FIX |
| 36 | 2448,5 | 2549,5 | `_get_execution_time_internal` | OWNER: FIX |
| 37 | 2460,1 | 2561,1 | `_get_execution_time_internal` | OWNER: FIX |
| 38 | 3181,3 | 3282,15 | `predict_stage_execution_time` | OWNER: FIX |
| 39 | 3185,5 | 3298,16 | `_predict_moe_layer_execution_time` | OWNER: FIX |
| 40 | 3243,1 | 3367,1 | `_predict_moe_layer_execution_time` | OWNER: FIX |
| 41 | 3248,0 | 3373,1 | `_predict_moe_layer_execution_time` | OWNER: FIX |
| 42 | 3249,0 | 3375,1 | `_predict_moe_layer_execution_time` | OWNER: FIX |
| 43 | 3316,0 | 3443,3 | `_predict_moe_layer_execution_time` | OWNER: FIX |
| 44 | 3455,69 | 3584,1 | `_predict_moe_layer_execution_time` | OWNER: FIX |

## Verification and limits

New corrective evidence: `test_report_2026-09-16_predictor_owner_once.md` records exact commands, intermediate fixture failures and final **154 passed in6.23s** focused tests plus **6 passed in11.39s** real hybrid CPU Simulator tests. The owner spy observes eight physical layers with stage lookups once; the dense spy observes one mutable snapshot for five identities; two terminal-MTP ranges retain nonzero replay and avoid bounds errors. Existing EP phase/conservation and MLA tests stay intact.

W04 evidence: `w04_cache_review.md` and `w04_performance_review.md` retain numerical cache/EP invariants and the controlled three-pair ablations. W00 representative dummy MoE had zero attention/EP query-cache calls, so its cache arms are negative controls only. The unchanged real hybrid fixture exercised336 attention queries with90 hits per run; outputs were preserved. Those performance samples predate this owner-once fix, so they do not establish final candidate wall time. Mutable snapshot component measurements and final paired timing remain explicitly pending root coordination.

GDN/path/measurement test names above are specific existing regressions inspected for contract coverage; this review did not independently rerun every named suite. Their final branch-wide results belong to root-owned W10/W11 verification. The actual normal Simulator manager supplies required config and GDN artifacts; hypothetical incomplete duck-typed managers are not production interfaces. Manager optional replica-config defaults remain outside this four-module repair and are not used to prove supplied malformed metadata is accepted.

No newly observed unresolved production failure remains in this reviewed corrective scope. No performance budget or acceptance decision is inferred from source cleanup. Native AMD/MI355X execution remains **SKIP: AMD/MI355X hardware unavailable**.
