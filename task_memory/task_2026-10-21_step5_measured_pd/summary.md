# D20-PD CPU consumer handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Parent clarified that READ duration must be strictly positive, matching the consumer and file loader. |
| 2026-10-21 | Archive the verified consumer and remaining supplier/native dependencies. |

## Outcome

Existing PD events, C16 ownership and the Request recompute cursor consume admitted grouped measured transfer coverage without analytical substitution. CPU integration and affected regressions pass201 tests. Independent follow-up closed the two lifecycle findings; this is not native/profile admission or full task qualification.

- Worktree: `/data/ycfeng/Frontier/.worktrees/step5-measured-pd`, branch `feat/step5-measured-pd` from6db2250.
- Shared context fix: f3781ef ports the exact6cc09dc expression;70 affected tests passed before the measured slice.
- Consumer implementation a082c00; lifecycle corrections9625984 freeze admitted retry coverage and unwind failed READ validation while retaining normal P pending KV.
- Edits: transfer events/info/predictor, scheduler PD owners, approved grouped-allocation input extension, Request and dedicated tests. Registry, computation predictor, training, managers, typed loaders and client metrics/comparator are untouched.
- R2 is preserved at b6ba7e5 on `feat/step5-r2-correction`; P1 cb09a27 and636-test evidence are unchanged.

## Shared provider interface

The source cluster's existing `_predictor` supplies:

```python
get_nixl_transfer_coverage(
    *, request, source_replica_config, target_replica_config,
    cache_capacity, target_cost_provider,
) -> NixlTransferCoverage | None

get_nixl_transfer_time_ms(*, coverage, payload) -> float
```

`NixlTransferCoverage` lives in `frontier/entities/kv_cache_transfer_info.py`: original prompt, explicitly admitted computed prefix, exact capacity, both attention-TP sizes/devices, immutable profile identity and `duration_scope="read_to_gpu_ready"`. The supplier owns existing profile/native admission and returns no coverage or raises on missing inputs. Identity is provenance, not authority. No observer verdict or CPU fixture establishes admission.

One existing page resolver sizes legacy and measured payloads. Only grouped measured mode uses the supplied cost; non-grouped GLM retains shared9f58dbe behavior. Preflight, reservation, READ completion, ready admission, first-D computation and release remain in existing owners.

## Verified CPU behavior

Exact coverage fields: `num_prefill_tokens`, `num_computed_tokens`, `cache_capacity`, `source_attention_tp`, `target_attention_tp`, `source_device`, `target_device`, `profile_identity`, `duration_scope`. Prefix must be positive, complete-block aligned and smaller than the prompt. Coverage must match live configs and remain equal during retry/READ. Cost is finite and strictly positive milliseconds through GPU readiness; identity is not native proof.

Actual-main requirements:
- Paired PREFILL/DECODE under `SYS_ARCH="pd-disaggregation"`, NIXL, one serving Replica per role, `vllm_v1`, `vllm_load_balancing`, policy `v0.28.0`.
- All layers Step; PP1, MoE TP1, EP=attention TP*DP; matching P/D attention TP and capacities. Both role providers expose both hooks and use `enable_dummy_mode=False`.
- Grouped memory-planner cache; prefix caching/speculation/MTP disabled. D token budget must fit the full admitted tail; model length must cover the request. Missing tail/mixed computation rows must fail explicitly.
- Tested fixture: h800/h800_dgx, TP8/DP4/EP32, block16/BF16-auto, max_model_len8192, max_tokens_in_batch8192, batch_size_cap1, chunked prefill false, offline/eager. Request4105/9, prefix4080 and3.5ms are CPU fixture inputs, not production calibration.
- H200/TP8/DP2/EP16 requires its own matching admitted inputs. P computation remains whole-prompt; do not infer native P-backoff/T2 proof.

The labelled fixture executes non-dummy constructors, actual transfer events and schedulers: P4105, declared prefix4080, target1020 blocks,103051776 bytes/worker lane, READ3.5ms, eager first-D prefill(query25,context4080), exactly9 outputs and empty final allocation/frontier tables. Timings are fixture inputs, not production data. Missing cost unwinds frontend/pool ownership; ordinary zero-token calls remain rejected.

Commands and RED/GREEN results: `test_report_2026-10-21_measured_pd.md`. Final affected result:201 passed in176.62s. Primary owns the consolidated77-case fidelity matrix.

## Remaining work

1. Primary now owns production transfer inputs/provider and integrated main; computation owner owns shared managers/FFN/bundle. Small measured `frontier.main` is NOT_RUN here. No further provider-source edits without coordination.
2. Real H200 computation/tail/mixed rows and READ-to-GPU-ready calibration are missing; absence must remain a clear failure.
3. Early T2 assessment remains open. P compute is unchanged whole-prompt behavior; no native P-backoff implementation or6144->6128+16 proof is claimed.
4. Native content/readiness/reference admission and all four10% gates remain outstanding. No GPU/auth/platform/cloud/SSH operation, push or consolidation was performed.
