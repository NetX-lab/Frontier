# Independent trainer and consumer scope

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Resolve the approved reachability boundary and withdraw the extra independent-public-consumer implementation prerequisite. |

TaskID: D29-INDEPENDENT-CONSUMER-SCOPE. Read-only investigation by
ses_e999f9819ffe45GDvuxH9fJPKj on frozen8f96408c, followed by direct parent reads
of the decisive code and approved requirements. No source or test edits.

## Decision: D29-TRAINER-CONSUMER-BOUNDARY

D29 requires both independent and shared trainer entrypoints to select the actual
operator/frame/precision/source binding before loading and fitting duration/count
models, preserving their per-binding caches. It requires the two existing runtime
consumers to use the binding registry on the approved Step P/D position path.
It does not require a new independent public-predictor binding implementation.

Root previously treated the absence of an independent public manager/binder as an
additional D29 blocker. That prerequisite is withdrawn after checking the actual
product call graph and scope. The latent issue is not fixed or claimed supported;
it is outside this implementation assignment. Do not add a predictor registry mixin,
new manager, null fallback, new mode or further allocation for it.

## Decisive evidence

- requirements.md16–24 restricts active scope to Step P/D, PP1/TP8/DP2/EP16 without
  SP/MTP. Its independent-reference routing wording refers to routing math, not
  independent predictor construction.
- design.md869–944 specifies both real trainer entrypoints, selected rows/targets,
  pre-loader validation, duration/count fits, metadata and per-binding cache/model
  identities. It does not promise a separate independent public consumer.
- moe_routing_policy.py194–207 admits position routing only under pd-disaggregation
  in PREFILL/DECODE with the approved geometry. The MoE constructor calls this
  validator before parent initialization (sklearn_moe_execution_time_predictor.py600–602).
- config/config.py747–749 classifies pd-disaggregation as disaggregated. In
  simulator.py152–174 that path unconditionally creates the shared model manager
  and supplies it to every predictor. Thus the approved product position path
  cannot reach an independent consumer with model_manager=None.
- docs/general/step5-preview.md43–47 describes monolithic Step as a controlled
  structural/synthetic-cost path, explicitly not trained prediction qualification.

## Retained latent finding and test boundary

Independent `_train_moe_models` uses a name-keyed result dictionary, so its returned
projection can overwrite same-name entries even while separate bindings are fitted
and cached. It does not attach the complete registry markers. A directly constructed
P/D predictor with model_manager=None can bypass the product manager wiring, and an
unqualified monolithic measured Step predictor can reach the existing FFN manager
dereference. Neither is admitted as an approved D29 serving mode.

The investigation retained a read-only scratch repro at
`/data/ycfeng/tmp/d29_indep_consumer_probe.py`; it stubs operator binding like the
existing unit fixture and demonstrates the actual None dereferences and name-keyed
storage, not a complete public run. Proposed source changes and speculative budget
were not adopted. Current CORE230/registry360 repair allowances remain governed by
the original-baseline receipts and effective overlays, not by an earlier constructor
commit's unrelated230-line history.

Continue both actual trainer fit/cache/preflight checks. Qualify the runtime path
through the shared manager used by the actual P/D entrypoint. Registry mixed
legacy/native view P2, CORE review findings and point-diagnostics persistence remain
in-scope work; actual-main, strict77 and native/G3/E2E gates remain unchanged.
