# V4 native dispatch evidence requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded D-V4-NATIVE-DISPATCH-EVIDENCE approval and bounded execution ownership. |

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
