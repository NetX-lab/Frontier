# D29 native implementation-design review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record four independent design findings and assign corrections before scope approval. |

## Reviewed phase and evidence

TaskID: `D29-NATIVE-DESIGN-REVIEW`.
Reviewer: native reviewer session `ses_e9e15d18cffemTkCOdK4P941nU`.
Artifact: `native_design.md` at `651415a`, the only file changed since `fa5bae2`.
The reviewer inspected current primary requirements/plan/progress, native
context/model/router and PW/FULL capture paths, and package/adapter/observer
interfaces. This is a design review, not source or runtime acceptance.

## Findings and disposition

| Severity | Finding and evidence | Required correction | Status |
| --- | --- | --- | --- |
| P2 | Client identity/capability propagation omitted: `tests/integration/step5_pd/r2_replay.py:25,42,61` fixes the old adapter; `tests/comparison/dp_placement_pp/pd_proxy.py:203` uses object identity for Step HTTP capture | Derive replay adapter from engine; consume a shared registered HTTP capability. Count both files and associated regression coverage | Assigned to design author |
| P2 | Installed model hash does not prove imported model origin: `r2_runtime.py:8–17,84–87`, `r2_stage.py:14–20`, `native_pd_worker.py:135–137` | Include actual imported Step model origin/hash/root in planned P5/P6 and test wrong origins. Keep existing model pin, minimum 53 total | Assigned to design author |
| P2 | Merely constructing BreakableCUDAGraphWrapper can test one segment: native `compilation/breakable_cudagraph.py:198–227` | T2 must perform a test-only eager break, assert `num_eager_breaks > 0`, and validate post-break routing with updated positions/mask | Assigned to design author |
| P2 | Existing `step5_dispatch_probe.py:454–456` lacks the new positions/mask context | Reuse its config/layer/observer components explicitly in T2 with the required context; do not claim the unchanged entry point validates new routing | Assigned to design author |

The initial 16-file / 716–1143-line estimate is arithmetically consistent but
incomplete. At least the two identified client files must enter the combined
scope; final totals await the corrected design and Frontier/profile design.
The parent resumed author session `ses_e9ea6907effeO1iiR7MJ5b3CIM` for design-only
correction. No implementation is authorized by this review result.

## Supported conclusions and limits

The reviewer found source support for the outer uncompiled Step binding,
modular routing before DP gather, and FULL capture using context mode NONE.
The sparse image-context builder has a concrete purpose: existing
`vllm_burst_driver.py:45–71` deletes the destination and copies the entire tree.
Reuse existing hash/pin functions in the new bounded builder.

The reviewer ran no tests or image/platform operations and did not independently
recompute source/receipt hashes. New CUDA arithmetic, segmented replay, actual
auto-backend execution, G3/state agreement and image identity remain unverified.
Design acceptance is open until the four findings are resolved and checked.
