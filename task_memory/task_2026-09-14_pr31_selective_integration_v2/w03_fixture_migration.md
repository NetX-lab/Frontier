## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Migrated six W03 predictor fixture suites to normal construction and explicit layer/stage ownership. |

# W03 fixture migration

Target Component/Phase: W03 timing producer fixtures and W04 initialized routing state.
Reviewer Agent Identity: `/root/w03_fixture_migration`.

## Inspected artifacts and changes

- `tests/unit/test_mixed_layer_decode_ffn_scheduling.py`: predictor fixtures now use the full disaggregation constructor through `CacheFixtureDisaggregationPredictor`, with real checked-in model configurations. Timing assertions use stage public scalar/component views; only `is_moe`, which lacks a public scalar, is explicitly checked on the one layer.
- `tests/unit/test_moe_ep_aggregate_admission.py`: valid typed model configurations replace incomplete model stand-ins on successful paths. Successful timing injections return a real 7 ms `ExecutionTime`; assertions verify the stage preserves numerical timing and dense-family identity. Explicit missing-model-capability fault injection remains on its fail-fast test path.
- `tests/unit/test_moe_predictor_layer_id_semantics.py`: normal constructor initializes all cache and routing ownership state. Models inherit the real `BaseModelConfig` contract. The five-layer test independently asserts IDs `[3, 4, 5, 6, 7]`, attention time 12 ms on each layer, and a fivefold stage total. Raw numerical helper fixtures declare one layer.
- Shared fixture dependency: `tests/unit/predictor_cache_fixtures.py`, owned and supplied by `/root/w02_acceptance_tests`.

## Criteria and execution

Failure targeted: old fixtures bypass constructors, omit model-owned attention identity, return arbitrary sentinels through public timing APIs, or rely on private first-layer delegation from a stage.

Passing conditions: all existing 203 cases pass with preserved numerical and admission expectations; no production sentinel allowance or lazy initialization is added; each multi-layer assertion names the correct scope.

Exact command from the active worktree:

```bash
python -m pytest tests/unit/test_mixed_layer_decode_ffn_scheduling.py tests/unit/test_moe_ep_aggregate_admission.py tests/unit/test_moe_predictor_layer_id_semantics.py -q --tb=short > /data/ycfeng/tmp/pr33-w03-fixture-owned-final.log 2>&1
```

Environment: /usr/bin/python; Python 3.12.3; conda environment not active.

## Observed evidence

- Before migration: **30 failed, 173 passed in 5.83 s**, `/data/ycfeng/tmp/pr33-w03-fixture-owned-initial.log`.
- Intermediate: **16 failed, 187 passed in 4.88 s**. These were migration defects: incomplete numerical fixture constructor arguments, wrong public TP field name, and stale actual-replica IDs after explicit two-replica test reconfiguration. They were repaired in test fixtures without production changes.
- Final: **203 passed in 4.26 s**, `/data/ycfeng/tmp/pr33-w03-fixture-owned-final.log`.
- `git diff --check -- tests/unit/test_mixed_layer_decode_ffn_scheduling.py tests/unit/test_moe_ep_aggregate_admission.py tests/unit/test_moe_predictor_layer_id_semantics.py`: PASS, no output.

No additional production defect was established by this bounded suite. No assertions were deleted to hide numerical differences; no D01 numerical golden was changed here. Direct private-helper tests still isolate numerical dependencies with injected hooks. This evidence does not establish E2E fidelity, native GPU execution, or the full repository unit-test status.

Pending tasks for this delegated scope: none. Newly discovered unresolved issues: none. Parent integration and commit remain with `/root`.

## Parent-authorized MTP/effective-token extension

Additional owned files:

- `tests/unit/test_mtp_terminal_overshoot_ep_replay.py`
- `tests/unit/test_predictor_effective_tokens.py`
- `tests/unit/test_spec_decode_mtp_structural_moe_replay.py`

All predictor fixtures use normal production construction before numerical or role hooks are injected. Actual model configurations now supply the layer identity contract. Routing caches are initialized by production construction, with no test-specific lazy cache repair in production. The terminal replay oracle remains **25 ms**, structural replay remains **29 ms**, and typed physical lane counts/conservation checks remain unchanged.

Initial exact three-file command: **4 failed, 41 passed in 3.51 s**, `/data/ycfeng/tmp/pr33-w03-replay-initial.log`; three failures were missing initialized workload cache and one lacked model-owned layer specs.

Final verification:

```bash
python -m pytest tests/unit/test_mtp_terminal_overshoot_ep_replay.py tests/unit/test_predictor_effective_tokens.py tests/unit/test_spec_decode_mtp_structural_moe_replay.py -q --tb=short > /data/ycfeng/tmp/pr33-w03-replay-second.log 2>&1
```

Observed **45 passed in 3.19 s**. `git diff --check` over the additional three files passed with no output. Environment and validation limits are unchanged. No new production defect was established.

## Final W10 interface integration follow-up

The parent delegated three additional fixture interfaces after its final full CPU run: `test_mixed_layer_decode_ffn_scheduling.py`, `test_pd_decode_moe_layer_accounting.py`, and `test_pdaf_prefill_model_time.py`. Initial15FAIL/146PASS -> final161PASS/3.42s. Exact causes, commands, preserved numerical/lifecycle oracles, and source boundaries are recorded in `test_report_2026-09-16_w10_fixture_integration.md`. No production behavior changed.
