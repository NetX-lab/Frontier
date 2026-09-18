## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Added independent W08 mixed-stage projection tests and PREFILL EP caller review. |

# W08 independent reporting review

Reviewer: `/root/w03_contract_review`. Owned files for this work package are this report and `tests/unit/test_stage_reporting_contract.py`. Initial reporting production changes belong to the root agent. The later explicitly delegated five-file EP transport implementation is documented in `hunk_review_runtime.md`. No commit, GPU command, native profiling, or external operation was performed by this review lane.

## Independent contract and observed results

The tests construct real `MetricsConfig`, `ReplicaConfig`, `ClusterConfig`, `SimulationConfig`, `MetricsStore`, `Request`, `BatchStage`, `LayerAttentionSpec`, `ExecutionTime`, and `StageExecutionTime`. A tiny trace sink records the actual events from `on_replica_stage_schedule`; there is no bypass of the metrics constructor or event projection functions. The numerical oracle is synthetic and intentionally covers multiple families; a valid dataclass model configuration provides the MLA dimensions needed by trace metadata. This is a reporting contract test, not a claim that a shipped model contains this exact mixed topology.

The independent five-layer oracle is:

| Global ID | Family / variant | Nonzero operator | Duration (ms) |
| --- | --- | --- | --- |
| 4 | dense_attention / standard | attn_prefill | 2 |
| 7 | dense_attention / standard | attn_prefill | 3 |
| 9 | latent_mla_attention / mla | attn_mla_prefill | 5 |
| 12 | gated_delta_net / qwen3_5 | gdn_core_prefill | 7 |
| 14 | gated_delta_net / qwen3_5 | gdn_core_prefill | 11 |

The registry supplies mandatory zero-valued attention schema entries only. Expected nonzero operators, identities, durations, cardinalities, grouping, and totals are independently enumerated in the tests.

The first isolated execution covered eight passing cases:

1. Expanded traces emit five records with actual IDs `4,7,9,12,14`, exact family/variant, replica, batch, and duration.
2. Aggregate traces emit three records with durations `5,5,18`, `layer_id=-1`, and global-ID metadata `[4,7]`, `[9]`, `[12,14]`.
3. Requested operation metrics emit all five nonzero observations when utilization and tracing are disabled.
4. Same-family operators with distinct `standard` and `mfa` variants remain separate aggregate records.
5. The dense metrics adapter predicts actual dense IDs `7,9`, preserves routed ID `8`, keeps stage owner schedule overhead `13` despite `999` on replacement predictions, and produces model `38 ms` / total `51 ms`; the input stays at model `10 ms`.
6. The mixed-family component ledger totals `28 ms`; the separate W03 owner fixture retains model `285 ms`, schedule-inclusive `298 ms`, diagnostic `321 ms`, and PP/CPU/draft/terminal values `11/13/17/19 ms` exactly once. Repeated ledger reads agree.
7. With write-metrics disabled and all reporting flags off, no trace context, component ledger, or operation projection is built and no record is inserted.
8. The same disabled-projection guarantee holds with write-metrics enabled but individual reporting flags off.

The disabled-reporting tests inspect event-time work, as required by the hot-path constraint; they do not assert that the constructor allocates no empty metric containers.

## Execution and limitations

Environment: `/usr/bin/python`, Python 3.12.3, local CPU, no conda activation. Final command:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w08-stage-reporting
```

Observed: **8 passed in 2.79 seconds** against root's in-progress W08 production edits; HEAD was `c1e924db` at the following status inspection. `git diff --check -- tests/unit/test_stage_reporting_contract.py` passed.

No RED claim is made: root's production edits preceded completion of valid fixtures. Initial fixture failures were visible and corrected: incomplete structured attention maps required explicit inactive zeros; `mla` was an arbitrary label in the numerical-only W03 fixture and was rebound to registered `latent_mla_attention`; a dense-only trace model lacked MLA dimensions; exact float comparison was changed to `pytest.approx`; the trace kept the caller's integer batch ID. An attempted existing `deepseek-v3` config was rejected by its FP8 serialized-checkpoint validation; this unrelated catalog issue was not changed. The final model fixture uses normal dataclass construction and avoids that unrelated configuration.

These checks do not replace the plan's real request/export E2E, which remains root-owned. No serialized CSV/JSON export or native execution is claimed here.

## PREFILL EP caller evidence and bounded correction

Before root's W08 correction, `handle_prefill_sync_collective()` in `frontier/scheduler/utils/prefill_collective.py` predicted `num_layers=1`, `layer_id=current`, `include_ffn=False` for completion timing and passed that same final-layer-only object into the stage metrics adapter. `prepare_prefill_final_timing()` separately consumed `_prefill_model_execution_components_ms_by_stage`; the complete scheduling scalar did not recover missing typed layer identity or operator payloads. A correct runtime total therefore did not establish correct reporting scope.

Existing phase accumulation is separate: `prefill_collective.py` appends each next attention duration, and `schedule_layer_wave()` in `frontier/scheduler/utils/ep_wave_schedule.py` appends either each dense post-attention duration or EP wave barrier duration. Those scalar scheduling records should retain their timing role. EP lane records own routed work; blindly adding routed MoE into the stage report would duplicate that scope.

Root's current correction follows the bounded reporting route: immediately before final metrics construction, predict attention for the current batch and the complete PP stage using `num_layers=num_layers`, `layer_id=stage_layer_start`, `include_ffn=False`. It leaves the earlier completion-timing calculation intact. `build_prefill_metrics_execution_time()` then walks the real stage IDs, predicts each actual dense FFN layer, preserves routed layers, and keeps the original stage owner. The independent dense-adapter test settles this helper's ID/scope behavior. A real scheduler E2E is still needed to establish full caller integration and routed-lane record cardinality; this lane did not edit schedulers or execute that E2E.

## Follow-up isolation, completion, and transport verification

The root combined E2E execution exposed seven fixture failures after a hybrid Simulator left `IS_MOE=True`; `_store()` attempted a dense normal config while the global still held the earlier model state. The test file now uses the existing `global_vars.reset_global_vars()` operation before and after each case. No production fallback was introduced.

The independent suite now adds three cases: requested ledger completion with utilization disabled (`end=1.25 s`, all pending maps empty, completed total `28 ms`), and conditional capture enabled/disabled for two actual predicted EP lane stages. Enabled capture preserves the exact batch/stage objects and actual lane IDs; disabled capture returns an empty record tuple while preserving the original numerical phase arrays.

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w08-reporting-final --tb=short
```

Observed **11 PASS in 2.96 s**, with the same Python 3.12.3 CPU environment. This includes the source repair to ledger completion independently of utilization. No CPU tests ran during the parent's reserved performance window.

The conditional transport was then checked through existing event/materialization tests:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 python -m pytest tests/unit/test_forward_sync_state.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_replica_identity_contract.py tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w08-ep-transport-final --tb=short
```

Observed **53 PASS / 2 FAIL in 3.75 s**. The two remaining failures occur in existing prefill/decode placeholder-consumption fixtures whose `SimpleNamespace` metrics store has no `ep_wave_reporting_enabled` property. Root owns migrating those fixtures and the actual `MetricsStore.on_ep_wave_schedule` projection. The focused eleven reporting/capture cases pass in this combined execution. These are intermediate integration failures, not a final acceptance pass. Exact source findings, proposed phase timestamps, original/current hunk mappings, and remaining scope concerns are recorded in `hunk_review_runtime.md`.

The subsequent authorized migration of the two placeholder metrics fixtures resolved both failures. Re-running the same five test files with `--basetemp /data/ycfeng/tmp/pr33-w08-ep-transport-fixtures --tb=short` observed **55 PASS in 4.68 s**. Independent comparison of the root phase projection table with the previous phase definitions found preserved legacy/structured gating semantics; details and the floating summation-order limit are in `hunk_review_runtime.md`.
