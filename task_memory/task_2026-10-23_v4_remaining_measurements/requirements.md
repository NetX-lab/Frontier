# V4 kernel-only attention costs

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded YC's approval of the recommended kernel-only route and CPU implementation/measurement-design scope. |
| 2026-10-23 | Root allocated the two reproduced native collector launch seams under existing repair-scope delegation; GPU authority is unchanged. |
| 2026-10-23 | YC requested completion/commit/handoff of current work; root allocated bounded repairs for five unique collector review findings. |
| 2026-10-23 | Recorded completed H source/CPU integration64f3c950 and preserved G2 as the next-session scope. |
| 2026-10-23 | YC resumed G2 in the current session; existing CPU implementation/design authority and separate GPU approval remain in force. |
| 2026-10-23 | Recorded the seven-file G2 CPU allocation and independently reviewed failure-isolation repairs within the unchanged750-line budget. |
| 2026-10-23 | Accepted G2 CPU source 05952396 after both re-reviews, final separate-suite verification and explicit matrix binding; new GPU authority remains pending. |

## Original requests and decision

[Original Request — execution order] “routing暂时统一采用balanced distribution方案对齐（frontier和vllm的routing都采用均衡分布的设定）”; “先进行workflow和gpu侧的对齐（包括op gap计算），最后再进行cpu overhead（gap）的计算测量”.

[Original Request — proposed choice] “后续 V4 注意力总耗时采用哪条路线？这次选择确定 CPU 实现和测量设计；新增 GPU 运行会在具体方案就绪后单独申请。” Recommended: “以 KERNEL_ONLY 为算子成本，依据原生依赖处理 GPU 流重叠。先补齐覆盖和归因证据，不直接累加现有 scope，也不把未知差额归为 CPU gap。”

[Original Request — YC response] “批准，继续”.

[Original Request — session close] “请你完整当前正在执行任务的收尾，然后完整提交更新docs和codes，生成一段handoff prompts，我将在其他session中继续执行”. Close the active collector source/CPU findings, verify and consolidate accepted code and records on the main branch, and produce a self-contained continuation prompt. Future driver/runtime/GPU work belongs in the handoff.

[Original Request — resume here] “暂不执行handoff，继续在当前session推进任务”. Continue G2 in this session from committed77668066. This supersedes the scheduling instruction to wait for another session; the completed H acceptance and the requirement for a separate concrete GPU approval remain unchanged.

[Decision D-V4-KERNEL-ONLY-GPU-COST] Adopt the recommended per-operator KERNEL_ONLY route. Implement V4-private input/consumer prerequisites and design measurements from native source and retained evidence. Kernel-to-scope uniqueness is not completeness or wall-clock additivity. Serving attention totals require complete attribution plus explicit GPU-stream/dependency accounting. CUDA_EVENT stays separate diagnostic/equivalence evidence; the unexplained uncompressed difference is not labelled CPU overhead or removed by scaling.

## Scope and acceptance

- Continue parent05/09 under approved D-Q10 measured consumption. Preserve balanced_topk and all layout/source/routing/phase/graph/provenance refusals.
- Reuse the selected attention-file path and private sidecar/admission pattern. No new shared config field or registry mechanism is needed for input bundles. Inspect actual callsites before choosing the integration point.
- Audit the accepted prefill trace and pinned native callgraph first: measured/correlated kernels, missing/ambiguous ownership, within-stream order, cross-stream dependencies and unsupported evidence. Keep all samples and original bytes.
- Produce a bounded design before core edits. Default domain limit: five frontier/ files/500changed lines, one owner per file; non-core audit/tools/tests separately accounted. Adjustments follow YC's existing root scope delegation; new shared interfaces/calibration rules remain separate decisions.
- A private input component may be accepted independently, but isolated TP1/eager/six-layer data never become full serving/TP8/DP2/EP16/graph evidence. Missing evidence yields a typed error, not a cost fallback.
- Verify real accepted inputs/metadata, all94 isolated medians, broken provenance/layout/coverage controls, existing affected tests, independent review and the required default matrix against a fixed source. Non-simulator changes may use explicitly bound existing matrix evidence.
- Measurement preparation targets actual per-role TP8/DP2/EP16/PP1/native V2. Existing DSA/V1 replay, EP16-only or TP8-only runs cannot establish equivalence. Inspect the existing native worker/extension path before proposing shared profiling parameters.
- This decision authorizes no new GPU run, platform change, publication, native/C26 edit, broad CUPTI investigation, CPU-overhead collection or large CPU-overhead suite. Completed prefill/native06 allocations cannot be spent again.

## Current reference

### H launch integration allocation

Root authorizes the delivering H owner to add `tests/comparison/dp_placement_pp/vllm_replay.py` (8 changed lines) and `tests/comparison/calibration/pd_replay.py` (4 changed lines). `collector/launch_seam_receipt.json` at owner commit `9366b6e4` reproduces all three blockers: the new extension is rejected for balanced routing, its worker PYTHONPATH is absent, and its profiler configuration is discarded. Reuse one extension registry across these launch paths and `native_pd_worker.py`; no parallel mechanism or permissive fallback. H's allocation is now five non-core files / 787 changed lines (450 collector, 25 native worker, 300 tests, 8 command builder, 4 P/D environment). Existing per-step local/synchronized dispatch evidence remains required. This is necessary wiring for the approved collector, not new calibration semantics or a GPU run approval.

### H review repair allocation

The preceding787-line allocation is superseded for necessary review corrections only. At d94b821a, independent reviews found one P1 and four unique P2 issues: graph-hook lifecycle, slot-mapping ownership, actual source-visible PDL arguments, atomic step-retention admission and bounded detailed error logging. Root authorizes the same five-file owner to use collector550/test400/native-worker25/command-builder8/P-D-environment4, total987changed lines. Record exact changes and RED/GREEN controls; do not remove controls or compress implementation to fit. Both review reports are retained in `collector_primary/review_2026-10-23_initial.md`. No new shared surface, native/C26 change, driver implementation or GPU authority is granted by this repair allocation.

Native06 is ACCEPTED_NATIVE_REFERENCE_R4: originalb62c155, reader integrationd278659, bound by parent `native06_acceptance_20261023/{primary_acceptance_r4.json,frozen_r4_result.json}`. Isolated consumer integratede89b45d:144unit/94exact/legacy102-72, fixed576d278 strict77. Original R3 FAIL/native05 UNVERIFIED remain historical. GLM remains paused; four-metric calibration is unfinished.

Session-close outcome: the private input component is accepted at732406aa; H sourcef8621a8 is accepted and integrated64f3c950 within880/987lines, with five findings closed and merged verification complete (`collector_primary/primary_acceptance.json`). G2 driver/window implementation and a concrete future GPU request remain in `progress.md` and `handoff_prompt.md`; this outcome grants no additional run authority.

### G2 implementation allocation and current authority

Under the existing CPU implementation/design approval, root owns the actual replay driver, window retention and CPU verification after the two initial design delegation attempts failed. The final allocation is seven existing non-core files/750changed lines (production350,tests400), with current owner-checked change433lines. Paths and lifecycle are in `driver_design.md`; `driver/cold_start_scope.md` records the additional existing test path and its reproduced baseline mismatch. No core/native/C26 interface or calibration rule is changed.

Both initial independent reviews reported three distinct failure-isolation defects. The delivering root owner reproduced and repaired all three; both R1 reviews closed them with no new findings. Source `05952396` is now `ACCEPTED_CPU_PROFILE_DRIVER_COMPONENT`, bound by `driver/primary_acceptance.json`, final separate-suite results and an explicit existing 77-case matrix binding. The first-formal 64-step/32-H800 resource sheet in `proposal.md` is ready for a user decision, not an approved GPU allocation. Existing native06 and prefill runs remain historical completed allocations and cannot be reused as spending authority.
