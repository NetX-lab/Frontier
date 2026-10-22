# D29 native implementation-design review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Close all four design findings after c9e9274; correct the proxy path and record parent budget verification. |
| 2026-10-22 | Record four independent design findings and assign corrections before scope approval. |
| 2026-10-23 | Record four implementation-review repairs, CPU evidence and unverified live boundaries. |

## Implementation review continuation (current)

TaskID: D29-NATIVE-IMPLEMENTATION. Reviewer: ses_e9afdb7c4ffejb3hbkjZCeHFKS.
Inspected phase: native19 implementation at2c4cc9e; root independently verified
418 CPU PASS,19 paths/1017 gross and native source hashes. Review identified
four concrete defects; user explicitly authorized their repair in the same allocation.

| Finding | Remediation and decisive source/test evidence | Disposition |
| --- | --- | --- |
| P1 router capture positional indexing | T2 native_modules.route uses explicit BaseRouter signature and named router_logits. Actual MoERunner._apply_quant_method source calls it entirely by keyword; CPU check captures logits after exactly one native invocation. Native tuple identity and raised exception identity preserved; capture fault remains soft. | RESOLVED_CPU_SOURCE |
| P1 capture output asserted before dummy replay | Pinned BreakableCUDAGraphWrapper.__call__:347–349 separates _capture and _replay; _capture:405–415 stores graphs/output without replay; _replay:433–435 invokes capture.replay. T2 now replays dummy before assertions/replay_baseline_counts. CPU executes pinned PW dispatch/replay and actual T2 post-capture statements starting with stale real buffers. FULL capture remains GPU-only. | RESOLVED_CPU_SEQUENCE; GPU UNVERIFIED |
| P2 instance forward replacement accepted | Checked actual text.forward.__func__ is imported Step5ForCausalLM.forward, __self__ is the actual text object, and source origin matches checked native module. Instance-forward negative disables observer without changing native output/exception. | RESOLVED_CPU_SOURCE |
| P2 backend names not validated | r2_runtime.backend_census reuses Fp8Config.is_layer_skipped inputs and native oracle.backend_to_kernel_cls for exact quant/expert classes. Checks imported kernel/impl/prepare-finalize source objects, modular flags, FP8 fallback component classes and actual apply callables. r2_stage/verify_r2_stage bind12 existing modules through schema_source_hashes; no new native registry/pin/file. Host fixtures now model both FP8 fallback and BF16 modular object relationships. Wrong quant/kernel/monolithic/same-name class/backend/apply/source controls remain soft. | RESOLVED_CPU_SOURCE; actual imported live objects UNVERIFIED |

Before-fix regression:14 FAIL/20 PASS/157 deselected (2.76s),
`/data/ycfeng/tmp/d29-native-repair-red.log`. Final affected check:
485 PASS/36.16s, `/data/ycfeng/tmp/d29-native-repair-affected02.log`.
Commands/environment/limits are in test_report_2026-10-23_native_integration.md.
Four native source files and native_source.patch unchanged. Actual budget1269/1582,
19 paths; repair changes7 existing owned paths. No additional subagent review or
GPU/Docker/platform work was run by this stream.
Repair commit74d5c48; `git diff 2c4cc9e 74d5c48 -- tests` is the scoped code diff,
267 additions/23 deletions. Native19 baseline gross increase252; receipt1269.

Evidence labels corrected: graph script explicitly sets mrv2_serving_qualified
and real_backend_graphs_qualified false. Future healthy script execution means
PASS_THIN_GRAPH_AND_EAGER_MODULES, not MRv2/attention/connector/full-backend graph
or G3 acceptance. No formula/mask/scaling duplication was found by reviewer.

The design review sections below are historical, not current scope blockers.

## Reviewed phase and evidence

TaskID: `D29-NATIVE-DESIGN-REVIEW`.
Reviewer: native reviewer session `ses_e9e15d18cffemTkCOdK4P941nU`.
Artifact: `native_design.md` at `651415a`, the only file changed since `fa5bae2`.
The reviewer inspected current primary requirements/plan/progress, native
context/model/router and PW/FULL capture paths, and package/adapter/observer
interfaces. This is a design review, not source or runtime acceptance.

## Initial findings and assignments (651415a)

| Severity | Finding and evidence | Required correction | Status |
| --- | --- | --- | --- |
| P2 | Client identity/capability propagation omitted: `tests/integration/step5_pd/r2_replay.py:25,42,61` fixes the old adapter; `tests/comparison/dp_placement_pp/pd_proxy.py:203` uses object identity for Step HTTP capture | Derive replay adapter from engine; consume a shared registered HTTP capability. Count both files and associated regression coverage | Assigned to design author |
| P2 | Installed model hash does not prove imported model origin: `r2_runtime.py:8–17,84–87`, `r2_stage.py:14–20`, `native_pd_worker.py:135–137` | Include actual imported Step model origin/hash/root in planned P5/P6 and test wrong origins. Keep existing model pin, minimum 53 total | Assigned to design author |
| P2 | Merely constructing BreakableCUDAGraphWrapper can test one segment: native `compilation/breakable_cudagraph.py:198–227` | T2 must perform a test-only eager break, assert `num_eager_breaks > 0`, and validate post-break routing with updated positions/mask | Assigned to design author |
| P2 | Existing `step5_dispatch_probe.py:454–456` lacks the new positions/mask context | Reuse its config/layer/observer components explicitly in T2 with the required context; do not claim the unchanged entry point validates new routing | Assigned to design author |

The actual proxy file is `tests/comparison/calibration/pd_proxy.py`; the initial
review reported the wrong directory. The corrected design uses the actual path.

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
Those four initial findings required correction before design acceptance.

## Follow-up disposition (c9e9274)

The same reviewer inspected the delta and corresponding interfaces. All four
findings are **resolved at design level**, with no new actionable issue:

- Actual engine adapter propagates through R2 model/proxy/replay, including the
  direct-D reference. Registry `http_capture` replaces object identity.
- Imported model/context/router origin, spec origin, bytes and actual model
  class/forward source are checked under the existing observer soft boundary.
- PW validation requires a genuine eager break, at least two graphs, stable
  outputs and updated post-break routing; it cannot pass as a single segment.
- T2 reuses native dispatch components with explicit positions/mask and separates
  eager module evidence from PW/FULL evidence.

The extra `test_step5_native_reference.py` adjustment is necessary: its direct
`r2_replay.run(args)` call lacks `engine_config`, which production already passes.
The reviewer found no other missing caller in this focused delta.

Revised native-facing estimate: **19 unique files / 1003–1582 added+deleted
lines**. Parent independently parsed the N/T/P rows, verified uniqueness and
the sum, and checked both corrected existing paths. These checks pass; no new
routing tests were executed. The complete Frontier/profile/pricing scope and
user numerical approval remain pending. Code, CUDA replay, image, backend and
G3/native acceptance remain unverified.
