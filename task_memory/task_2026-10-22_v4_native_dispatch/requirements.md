# V4 native dispatch evidence requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded D-V4-NATIVE-DISPATCH-EVIDENCE approval and bounded execution ownership. |
| 2026-10-22 | Recorded balanced-routing and deferred CPU-overhead steering; dispatch scope unchanged. |

## [Original Request]

Primary asked to extend the existing host observer with actual prefill state and CPU DP synchronization inputs, create a versioned checker within5 code/test files and500 changed lines, then run the same V4 workload on32 non-preemptible H800/codesign GPUs after CPU/review/package acceptance, capped at60minutes/32GPUh. YC answered verbatim:

> 实现＋新运行 (Recommended)

Decision: `D-V4-NATIVE-DISPATCH-EVIDENCE`. Evidence/design basis is `../task_2026-10-15_glm53_dsv4_vllm028_h800/native05_acceptance_20261022/proposal.md` and its RCA. Native05/R2 remain immutable historical inputs, not targets for retroactive admission.

## Acceptance

- Reuse existing native host seams and counter/emission ownership; preserve actual native calls/results/exceptions. Capture complete classifier and DP inputs associated with the successful forward/replay. Never infer missing operands from the observed mode.
- No added GPU reads, fences, collectives or per-step I/O. Auxiliary observation failures are soft-isolated and yield UNVERIFIED evidence; healthy native execution continues.
- CPU controls cover prefill-tail vs ongoing q1, peer-induced mode/padding, wrong/missing/ambiguous evidence and count reconciliation. All computable package predicates, source/image pins and independent review pass before the final single-writer freeze.
- A fresh P16+D16 / TP8DP2EP16PP1 run retains the frozen128-request6144/1024 QPS1 workload,16 warmups and3 repetitions. Verify32 ranks,432 requests,7 boundaries, lifecycle and platform/container terminal state.
- No metric/gate/model/engine change or GLM resumption. Root alone performs GPU/platform/auth/SSH/cloud operations and submission. Full calibration still needs a qualified predictor and comparison.

Maintained code/test changes must stay within the approved bound. Report both the successor delta against frozenR2 and physical additions/copies; do not hide new logic in generated task artifacts. Stop for root scope review if the bound cannot be met.

## [Original Request] Follow-up steering

YC's request relayed by root:

> 1.routing暂时统一采用balanced distribution方案对齐（frontier和vllm的routing都采用均衡分布的设定） 2.校准过程中，先进行workflow和gpu侧的对齐（包括op gap计算），最后再进行cpu overhead（gap）的计算测量，不要在当前阶段就引入cpu overhead的大批分析、测量和suits，推迟到矫正最后。

Dispatch scope is unchanged. Necessary CPU correctness controls remain authorized;
CPU-overhead analysis, measurements and suites are deferred until workflow/GPU
alignment, including op gap, is complete. Root owns balanced-routing binding and
coordinates any shared-collector write. `uniform_random` is not balanced. No
freeze/package/launch before balanced runtime binding is reviewed and incorporated.
Native05 remains historical old-routing evidence. The explicit balanced policy
change supersedes the earlier same-engine constraint only for routing; it grants
this agent no routing implementation or platform ownership.
