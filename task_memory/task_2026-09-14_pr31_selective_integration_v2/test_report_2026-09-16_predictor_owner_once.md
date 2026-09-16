## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded W03 construction ownership, terminal replay range, and homogeneous snapshot regression repairs. |
| 2026-09-16 | Added profile-supported homogeneous disaggregation reuse, missing-GDN fail-fast, and final fixture migration evidence. |

# Predictor construction ownership verification

Reviewer/implementer: `/root/w02_acceptance_tests`. Integration and commit owner: root agent. No subagent commit. Requirement: W03/T05–T07 constructs stage-owned values once and preserves independent physical-layer timing/routing; W11 audits every changed predictor hunk. Baseline: `0515589a`; candidate: current shared worktree, including the four predictor modules changed by this bounded step.

## Cause and correction

1. MoE and disaggregation public stage loops predicted every physical layer correctly but also constructed CPU/PP stage fields on every iteration. Stage aggregation charged only the first layer, hiding repeated lookups. Internal layer helpers now receive `include_stage_owned=offset == 0`; subsequent layers retain their own attention, dense/MoE, TP/EP and residual work, while stage-owned fields are zero and their lookups are skipped. Dummy values use the same ownership rule. Disaggregation also queried the current-stage boundary handoff twice within its owner layer; the existing overhead constructor now supplies that value once. A consumer-stage residual may legitimately query the preceding stage's boundary; that is a different input, preserved unchanged.
2. MoE terminal-MTP prediction combined each physical layer offset with the configured full-stage layer count. For an eight-layer normal constructor, replay from offset 1 requested layer 8 and raised `ValueError: layer_id 8 out of range for model with num_layers=8`. The owner now receives the actual public stage count and first physical layer. Only that owner starts terminal replay; the existing suppression guard terminates nested replay normally.
3. Homogeneous dense prediction computed numerical components once but finalized/copied the same mutable source separately for each physical identity. The dense stage method now finalizes that source once before assembling identities. This retains immutable source isolation and distinct physical layer records while sharing the finalized numerical components.
4. Removed unused MoE imports, dense model-manager/model getter fallbacks that admitted incomplete fixture interfaces, and obsolete aggregate `ExecutionTime` documentation/return annotation. The dense regression fixture now calls the complete predictor constructor; numerical quantization hooks are installed after that constructor.

## Execution

Environment: `/usr/bin/python`, Python 3.12.3; no activated conda environment. Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`. CPU checks only. Logs are under `/data/ycfeng/tmp`.

```bash
python -m pytest tests/unit/test_attention_query_cache.py tests/unit/test_mtp_terminal_overshoot_ep_replay.py tests/unit/test_dense_execution_time_layer_scaling.py tests/unit/test_sklearn_disaggregation_execution_time_predictor.py tests/unit/test_typed_ep_predictor_contract.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_predictor_layer_id_semantics.py tests/unit/test_spec_decode_mtp_structural_moe_replay.py tests/unit/test_mla_predictor_runtime_operator_times.py -q --tb=short > /data/ycfeng/tmp/pr33-owner-once-final.log 2>&1
python -m pytest tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q --tb=short > /data/ycfeng/tmp/pr33-owner-once-hybrid-e2e.log 2>&1
git diff --check -- frontier/execution_time_predictor tests/unit/test_attention_query_cache.py tests/unit/test_mtp_terminal_overshoot_ep_replay.py tests/unit/test_dense_execution_time_layer_scaling.py tests/unit/test_sklearn_disaggregation_execution_time_predictor.py
```

## Criteria and observed evidence

| Check | Criterion | Result |
| --- | --- | --- |
| Nine focused predictor suites | All existing arithmetic, EP conservation/admission, MoE physical-layer identity, MLA, MTP structural/replay and cache assertions pass | **PASS: 154 passed in 6.23s** |
| Stage construction spy, MoE and disaggregation | Eight physical layers retain attention 2 ms each; nine CPU/PP ownership lookup methods each called once; schedule fields `[1,0,0,0,0,0,0,0]` ms | **PASS**, normal complete constructors and real Batch/Request |
| Homogeneous dense snapshot spy | Five distinct physical identities 3–7, exactly one mutable-component snapshot, one shared finalized attention payload | **PASS** |
| Real terminal-MTP replay, EP=1 | Full range `(first=0,count=8)` and terminal subrange `(first=5,count=3)` both succeed; exactly one original-batch replay and one suppressed nested replay; terminal value equals independent sum of per-layer block values | **PASS**; no out-of-range layer; nonzero terminal work retained |
| Existing physical EP terminal replay | Physical lanes, routed-token conservation and maximum-by-phase assertions stay intact | **PASS** within 154-test command |
| Real hybrid CPU Simulator suite | Production predictor construction, trained synthetic artifacts, multiple requests, hybrid phases and state lifecycle assertions remain valid | **PASS: 6 passed in 11.39s** |
| Whitespace/conflict check | No diff errors in edited production/test scope | **PASS**, no output |

## Visible intermediate failures

`/data/ycfeng/tmp/pr33-owner-once-initial.log`: 3 failed / 83 passed. All three failures were existing disaggregation numerical hooks rejecting the explicit new private `include_stage_owned` keyword; hooks were migrated to the declared private interface.

`/data/ycfeng/tmp/pr33-owner-once-regression.log`: 2 failed / 89 passed. Both failures were old incomplete quantization stubs installed before the now-complete dense constructor, lacking `configure_from_model_config`. They now replace numerical hooks after normal initialization. No production fallback was added.

## Limits

The deterministic owner/MTP tests establish construction counts and arithmetic, not trained timing accuracy. The hybrid Simulator tests use synthetic trained CPU fixtures and do not establish native NVIDIA/AMD numerical parity. This step makes no wall-time claim: concurrent integration work makes correctness-test durations unsuitable for performance comparison. Controlled mutable-snapshot microbenchmarks remain coordinated with the root agent's final exclusive performance window. Full-branch regression and final commit recording remain root-owned.

## Final W11 profile-supported correction

Root supplied `/data/ycfeng/tmp/pr33-w11-small-dense.profile` for the unchanged W00 small-dense PDD scenario: 36 stage assemblies, 1,440 `as_single_layer` calls and 1,440 mutable-component snapshots; 11,520 copy calls accumulated 0.0846 s within a profiled 0.3961 s run. This located a real remaining path: PDD uses the disaggregation producer, so the homogeneous dense producer's earlier freeze-once repair did not reach it. These supplied profile timings locate work and are not unprofiled acceptance timings.

The disaggregation stage producer now computes and finalizes once when the model is non-MoE, no explicit MoE branch is requested, and the model contains zero GDN layers. It reuses ordered physical identities through the existing stage assembler. All four roles (PREFILL, DECODE, DECODE_ATTN, DECODE_FFN) have a normal-constructor regression comparing against independent per-layer production construction: exact model time, total time, operator map, communication map and physical block values; each optimized stage has exactly one numerical prediction and one component snapshot. Stage-owned payload values follow the existing homogeneous dense contract: the stage projects its first owner once. MoE and hybrid paths retain independent per-layer construction.

The direct attention entry point now resolves its required physical-layer identity first. A GDN layer without loaded artifacts raises a clear `ValueError` before any dense/norm numerical lookup, including dummy direct calls. Two normally constructed fixture tests cover dummy/non-dummy behavior; the real Simulator manager path retains its loaded artifact prediction.

Root separately owns removal of the discarded initial copy in `ExecutionTime.as_single_layer`; that entity edit is outside this subagent's production ownership.

```bash
python -m pytest tests/unit/test_attention_predictor_correctness.py tests/unit/test_colocation_release_review_contracts.py::test_non_dummy_shared_model_manager_registers_profiling_metadata tests/unit/test_comm_operator_families.py -q --tb=short > /data/ycfeng/tmp/pr33-final-five-fixtures.log 2>&1
python -m pytest tests/unit/test_dense_execution_time_layer_scaling.py tests/unit/test_attention_query_cache.py tests/unit/test_sklearn_disaggregation_execution_time_predictor.py tests/unit/test_comm_operator_families.py -q --tb=short > /data/ycfeng/tmp/pr33-disagg-homogeneous-final.log 2>&1
python -m pytest tests/unit/test_dense_execution_time_layer_scaling.py tests/unit/test_attention_query_cache.py tests/unit/test_gdn_runtime_guards.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py tests/unit/test_sklearn_disaggregation_execution_time_predictor.py tests/unit/test_comm_operator_families.py tests/unit/test_attention_predictor_correctness.py -q --tb=short > /data/ycfeng/tmp/pr33-predictor-freeze-final.log 2>&1
```

Observed results: **27 passed in 2.84 s**, **80 passed in 4.01 s**, and final **104 passed in 14.52 s**, respectively. The five failing fixtures from root's full-suite run were migrated: typed model with physical specs and normal dense constructor; explicit fake-manager GDN accessor; PP send/recv only in the first owner's per-layer map. All existing numerical expected values remain unchanged; only the declared once-only PP ownership expectation changed.

No new unresolved production failure was observed. The mutable-snapshot microbenchmark subsequently completed in the exclusive window on `f9099f85`: one-layer snapshot 17.412628 microseconds, eight mutable-layer stage 137.845563 microseconds, eight finalized-layer stage 5.185594 microseconds (five-sample medians). Exact command and raw evidence are recorded in `w04_performance_review.md`. Final paired wall time remains root-owned; these isolated component measurements do not establish a whole-simulator speedup.
