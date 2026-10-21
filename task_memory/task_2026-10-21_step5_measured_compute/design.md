# Step measured compute design

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record the implemented CPU minimum and remaining normal-initialization/FFN integration. |

## Physical owner and interface

The existing producer already exports clean `time_stats.step5_layer_e2e.*`. It is the sole measured predictor target for this attention family:FULL/piecewise CUDA_EVENT prices native runtime latency;eager KERNEL_ONLY prices correlated kernel-plus-copy device activity. The nine native scoped measurements remain profiling/trace diagnostics,not additional physical charges. The predictor-only root never requests a new native instrumentation scope.

Reuse `AttentionOperatorTimes.op_times` for this root and the existing `AttentionTime.total_time()`/`ExecutionTime` composition. An earlier uncommitted optional-root field was removed once inspection showed the existing structured operator map already carries this owner. Typed `OperatorCost` evidence lives inside the existing structured attention object and survives its normal deepcopy into ExecutionTime. It exposes actual selected modality,duration/count model hashes,exact-row status and explicit fixture provenance. No constructor/scheduler/Request/Batch interface change is needed.

Every charged root requires actual `time_stats.step5_layer_e2e.kernel_count`;sample count or aggregate trace count is not a replacement. Counts train through the same shared `_train_single_model`/atomic pickle path. Missing root/count/physical feature coverage errors are explicit. CPU tests deliberately use23 launches,10 statistical samples and999 aggregate kernels to distinguish the contracts;none are native measurements.

FULL event cost owns its internal device gaps and never gets an additional kernel-gap charge. Eager GPU-active attention needs the existing measured kernel_gap.csv and charges its count once,even when the surrounding step uses event-family FFN terms. Other models retain their prior global-family rule. Input normalization and TP allreduce remain outside the Step native attention root and must be supplied separately by the complete predictor path.

## Trainer/cache locality

`step5.py` validates source geometry/typed ownership and produces the39-column numeric feature vector. The shared family trainer calls its root/count fit through the original cache chain. `step5_cost.py` binds actual batch/source metadata to exactly the same features and routes to the existing on-demand lookup. No parallel trainer,GDN publisher,profiling producer or simulation engine is introduced.

The existing PredictionModelRegistry snapshots its existing artifact files in a content-addressed manifest,after all files exist. Manifest/model provenance,feature/target metadata and artifact hashes are checked on load. Cold/warm and A->B->A reuse are tested without relabelling measurement types or rewriting cache artifacts.

The large sklearn predictor already has a recorded exception in `docs/general/module-size-exceptions.md`. Inspection found no duplicate measured-Step implementation to remove;controlled dummy pricing remains a different supported contract. New fitting/row logic and runtime lookup are in the two small domain modules;the large predictor only delegates and records actual modality-specific count ownership. Moving all legacy family binding is the existing movement-only split,not part of this capability.

## Explicit remaining work and external interfaces

- Normal shared-manager initialization must request the correct attention family independently from generic FFN/CPU operator families;it must invoke bundle publication/load as part of the current chain.
- Complete outside-root input norm and typed dense/shared/routed FFN coverage with true BF16/FP8/exclusion contracts;the first replay deliberately isolates only its measured attention owner.
- Request-free attention-DP dummy forward has no matching ordinary cache-backed root measurement:raise coverage error,never substitute the full root or zero.
- Runtime currently queries configured serving block-table capacity;the producer input allocates only live-length columns. This is an actual physical-feature difference,not harmless interpolation. Keep mismatched native rows unavailable. Positive CPU tests use legitimate near-limit lengths whose producer allocation already equals configured capacity;the4104-context negative case remains.
- Parent-owned producer seam:`frontier/profiling/step5/rows.py:15` must use `family.profiling_ops()` instead of all operators so the predictor-only root does not become instrumentation. The unmodified generic `attention/profiling_mapping.py` shared feature API also needs coordinated Step support;this owner has notified primary and not edited outside its file ownership.

No native/profile admission or four10% gate changes. The prior native strict FAIL remains. CPU evidence supports executable consumption only.
