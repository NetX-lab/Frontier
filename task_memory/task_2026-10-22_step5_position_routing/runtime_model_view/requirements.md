# Named-model query implementation requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Bind YC's approval and100-gross four-path CORE implementation. |

## Authority and objective

Decision: **D29-NAMED-MODEL-QUERY**. Complete question/options are in
`../requirements.md`. [Original Request] YC answered:
“批准，类似请求无需申请权限，皆授权”. Implement the approved named-model query.

The existing `_models_view_for_family` accepts an optional keyword-only logical
model name. GEMM point selection requests`moe_grouped_gemm`; other callers remain
unfiltered. Preserve registry-name versus label semantics and the existing
`_frontier_cache_model_name` predicate. No new public configuration or retained
model-view/numerical/proof cache. Current-model native/typed/point/count/context
checks and live family/estimator/row-tuple/identity replacement remain effective.
Only unrelated full-view errors stop aborting this named query. Full initialization
and public unfiltered views retain their behavior. Do not promise skipped view
errors necessarily reappear during context-qualified numerical lookup.

## Scope and acceptance

`../repairs/named_model_query_scope_amendment.json` is authoritative: registry15,
workload5, bindings50, runtime30; source20/tests80/total100. No donor file edits.
Keep114 allocated paths and CORE4404/PRICING3956/native1582/source9942/global10000.

1. Demonstrate a pre-change named-selection failure and then pass selected-set,
   unfiltered order/object/error and runtime-query checks.
2. Cover mixed native/non-native orders, selected native conflict, typed ambiguity/
   conflict/missing contract, DECODE_ATTN, absent/count-model exclusion and the
   approved unrelated-error difference. Reuse current replacement/point failures.
3. Run relevant unit suites and compare baseline/candidate failures. Use readable
   tests and existing fixtures; unallocated test paths may be run/imported, not edited.
4. Rebind the existing208-event wave to committed source without method replacement.
   Keep the protected output oracle unchanged. Validate full values, keys/order,
   JSONL/ledger, snapshots,64 wrong keys and3902B prefix; preserve288 numerical/
   144 fresh proof/64 collective/16 point calls and one binding read.
5. Record actual step/cumulative path/owner/global usage. Both review axes and
   parent verification precede integration. Final union/frozen regression, full
   6144/1024/2400s CLI and native/GPU/E2E remain separate required gates.
