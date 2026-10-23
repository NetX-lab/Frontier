# Measurement binding retention repair

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Allocate a bounded existing-registry correction from CORE headroom; retain typed/cache identities. |

TaskID: D29-REGISTRY-SEAM. Investigation:
ses_e99d7657effeDfIQ5QCsxfqJJt at COREd96b3279. Root authorizes ordinary in-scope
correctness repair under the existing standing small-scope-adjustment permission.
This is not a new user GPU or numerical-model decision.

## Reproduced cause

Two measured variants of one name/typed contract/precision are stored, but the
registry's (name, identity) key overwrites the first. The tight real Step example
uses FP32 gate contracts from layers3/88 under the FP8/BF16 expert groups: typed
gate identities match, bucket size becomes1 and requesting the first group yields
the second estimator. Archived original red: /data/ycfeng/tmp/d29-core-registry-collision03.log.
The investigator independently reproduced it; implementation must retain an exact
standalone RED/GREEN command and persistent real-pattern regression.

Both estimators are fitted and carry measurement identity/physical rows. Per-binding
cache hashes are already distinct. Thus trainer ordering, failed fitting and cache
collision do not explain the loss. prediction_model_identity.py serialization
must remain unchanged; adding precision/group to the global typed identity would
rewrite unrelated cache identities rather than fix registry retention.

## Required implementation

Extend the existing registry's native measurement retention/query using its actual
measurement marker and existing canonicalization. Actual native binding, source
and selected group metadata distinguish entries; per-point rows/workspaces remain
outside binding identity and remain exact row evidence. Do not invent aliases or
a parallel registry. Preserve no-binding legacy/default behavior and cache hashes.

Select the requested actual binding. Ambiguous native requests must fail clearly,
not silently choose the last insertion. Point recovery and bundle/enumeration paths
must retain every applicable native binding; a legacy projection must not silently
discard variants in an admission/pricing path. Enumerate native variants in bundle
receipts while keeping legacy single-binding labels/content unchanged. Existing
group metadata is evidence already carried by current frames, not a new classifier.

Use actual FP32 gate under FP8/BF16 groups and same-precision expert variants in
tests, plus source/group selection, ambiguity, missing source and legacy controls.
Connect the two CORE consumers to the same registry; keep their post-selection
context validation. No new global serializer or manager implementation.

## Scope and sequencing

Effective overlay: registry_scope_amendment.json. Add only the existing registry
source and one regression test path, increasing union107->109; all owner totals
and total10000 remain unchanged. Allocate300 gross to these paths by withdrawing
300 unused grouped-path allowance from four CORE paths; observed code fits every
revised allowance. The two already-owned caller changes share at most60 additional
gross within their revised caps. Whole repair at most360 gross/four paths.

Use an isolated worktree based on d96b3279 so the already-running seam03 source is
not edited. The registry workstream alone writes these four paths until handoff;
original CORE resumes those paths only after root integration authorization.
Required completion: RED/GREEN, both actual trainer fit/cache paths, the shared
manager binder used by the approved P/D entrypoint, bundle coverage, legacy
regression checks, exact diff and budget, then independent review. This clarifies
the earlier ambiguous "shared/independent trainer/binder" wording; it does not add
an independent public consumer. See independent_consumer_scope.md.
Full actual-main/strict77 and native qualification remain later global gates.
