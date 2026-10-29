# D20 measured compute completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive executable shared Step compute/cache/bundle delivery and verification limits. |

## Outcome

Source commits:CPU minimum `1ca09be`,complete slice `83af0a7` on
`feat/step5-measured-compute`. No merge/push/GPU/auth/platform operation was performed.

The existing shared trainer now fits and consumes Step root/norm/typed FFN and supplied
stage targets,through the original atomic artifact cache and registered bundle publisher.
Root ownership is inclusive once;scoped attention columns are diagnostics. FULL uses
the actual clean-window mean,and eager GPU-active uses actual median;labels and native
timing implementations are unchanged. Kernel counts are explicit targets,not sample counts.
BF16 dense/shared,FP8 routed/exclusions,FP32 router,merge and all92 layer identities are
source-registry bound. Missing measurements remain coverage errors.

Normal shared-manager publication preserves all precision/contract domains and count pairs.
Compatible P/D inputs are combined without overwriting coverage. Kernel-gap and explicit
NIXL files are immutable input bindings;NIXL remains a lazy delegate to primary's loader.

## Deliverables and validation

- `tests/integration/step5_measured_compute_fixture.py`:executable H800/H200 CPU fixture
  generation through the normal manager;no default hardware/scale or native-admission claim.
- `handoff.md`:exact generation/test commands,config fields and remaining production inputs.
- `complete_verification01.json`:selected input/bundle/XML hashes and unchanged baseline-conflict evidence.
- `test_report_2026-10-21_measured_compute.md`:192 new CPU PASS,97 profiling compatibility
  PASS,124 existing regression PASS and1 unchanged legacy assertion failure.
- Public constructor and first/last Stage ownership directly pass. H80032 output
  `/data/ycfeng/tmp/d20_measured_compute_fixture_h80032_03/fixture_manifest.json`
  contains4 bundles x34 original-cache artifacts and explicit CPU-fixture provenance.

## Open/deferred

Final optional independent re-review failed due advisor quota;prior3 findings were fixed
with tests,not declared independently re-reviewed. Primary owns combined PD/transfer/main
and77-case fidelity. Production still requires qualified hardware/runtime-specific timings,
actual per-replay counts and serving-matching physical coverage. Existing native strict
FAIL/four10%gates remain unchanged. The6128+16 tail in CPU fixtures is not serving proof.
Embedding/final norm/LM-head profiling was not expanded. Frozen packets/old profiling branch remain intact.
