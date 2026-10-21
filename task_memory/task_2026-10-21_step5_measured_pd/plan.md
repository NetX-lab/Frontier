# D20-PD implementation plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define bounded implementation slices and verification seams. |

Dependency graph: `A -> {B, C} -> D -> E`.

| Step | Scope and completion condition |
| --- | --- |
| A | Inspect existing provider, transfer events and C16/Q10 scheduler semantics; record the shared interface needed from the computation owner. |
| B | Implement admitted measured transfer handling in existing start/end events; unsupported coverage fails before ownership changes. |
| C | Implement the admitted D computed frontier/first-step shape inside existing Request/scheduler owners, preserving output accounting and legacy paths. |
| D | Exercise actual event/scheduler lifecycle with labelled CPU fixture costs, missing/wrong coverage and lifecycle negative controls; run affected existing tests. |
| E | Commit verified code and task evidence; run a small actual `frontier.main` case when the computation/bundle interface is available, otherwise identify the exact missing input. |

## Test seams and limits

The user explicitly requested actual `KVCacheTransferStartEvent.handle_event`, `KVCacheTransferEndEvent.handle_event` and scheduler admission/completion. These are the agreed CPU integration seams. Fixture costs represent independent input data, never production timing defaults. Keep the existing allocator, queue and event owners. Work in vertical red/green slices. No new observer, GPU collection or independent transfer engine.

The computation owner implements shared family/profile admission and measured computation. Notify primary before changing any interface outside PD ownership. Parent performs the integrated 77-case comparison after consolidation.
