## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded reproduced R01 and the scheduling decision requiring user selection. |

# R01 decision — superseded by approved temporary approximation

## Observed defect

The new real non-dummy Simulator regression fails in GDNBatchFeatures.from_batch with `GDN predictor does not support same-batch prefill+decode mixing`. Capacity is three; requests are (arrival seconds, prefill tokens, decode tokens): (0, 16, 12), (0.002, 33, 3), (0.002, 16, 3). It uses the existing synthetic hybrid profile campaign, production constructor, trainer, loader, scheduler and event path. Synthetic timings establish the control-flow failure, not hardware latency accuracy.

## Historical unselected choices

1. **Recommended: retain vLLM V1 running-first order with phase-pure admission.** In `_schedule_running_requests`, the first successfully scheduled request establishes phase; skip opposite-phase requests before allocation. In `_schedule_monolithic`, do not admit waiting prefill into a decode batch. Use `request.is_prefill_complete`, not token length. Retain slots and KV blocks for deferred requests. Capacity remains configurable and same-phase requests can share a batch. This delays new prefill behind running decode; it intentionally changes formerly failing GDN workload scheduling while preserving ordinary non-GDN behavior.
2. **Restrict GDN to the existing SGLang prefill-first scheduler.** Reject vLLM V1 + GDN during configuration before resource ownership, and use the existing `_schedule_prefill_stage_first` / decode fallback policy. This narrows advertised scheduler support and requires callers to select SGLang; prefill-first can delay decode under sustained prefill traffic. It avoids adding a second phase policy but changes existing accepted deployment configuration.

No option silently relabels mixed batches, sums unrelated kernel estimates, or forces batch size one. The predictor rejection remains a useful invariant check.

## Historical proposed correction scope

Production: two admission boundaries in `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`; no public config field or shared data contract changes. Tests: `tests/integration/test_gdn_phase_admission.py`, extended to verify ownership stability, release/reuse, pure dispatched phases, capacity greater than one and a one-token final prefill chunk. Include running partial-prefill versus running decode, not only new waiting admission.

The scheduler exceeds 2,000 lines. Existing prior cleanup analysis remains applicable: the two checks belong beside existing allocation decisions; moving lifecycle/PP/preemption owners during this defect repair would expand scope. A future split should extract monolithic admission policy after isolating shared allocation/rollback ownership, with exact non-GDN fidelity gates; it is not required for these two checks.

## Historical approval question

User AGENTS.md Approval Gate item 5 requires selection of materially different unresolved designs. The review itself leaves phase-compatible scheduling versus early configuration rejection open. Asked the user to select the two supported outcomes; no production policy was changed while awaiting selection.

1. Record the selected policy.
2. Implement R01, strengthen lifecycle regression, run focused scheduling tests and the required fidelity matrix, then commit.
3. Proceed through R02–R10 in review order; assess C01/C02 and prepare C03 locally before any remote action.

Independent R02 inspection confirms the homogeneous binder remains in `VllmRocmAttentionWrapper.init`; the existing `resolve_runtime_attention_family` can resolve the unique hybrid full-attention KV family without weakening the whole-model binder. No R02 code changes or native execution have occurred.

## Accepted R01-A — temporary prediction approximation

The user challenged changing scheduler semantics and then explicitly authorized treating mixed GDN batches as prefill with a warning and documentation, especially for co-location. The prior two choices were withdrawn and are not implementation requirements. Preserve actual batch/request phases and resource ownership; route the complete GDN workload through prefill estimators using the original feature vector. Emit RuntimeWarning at runtime feature construction, using standard Python repeat filtering. Native mixed profiling/training remain unsupported. Confirmed as an implementable simulator approximation, not native timing equivalence.
