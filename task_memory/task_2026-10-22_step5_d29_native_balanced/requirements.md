# D29 native balanced-routing requirements

## Modification History

| Date | Change |
| --- | --- |
|2026-10-22|Recorded focused read-only native-route/proposal scope after D28 delivery.|

## [Original Request] D29-NATIVE-BALANCED

Read-only source and bounded CPU probes only; no implementation, native patch, image/package change, GPU/platform/auth/cloud/SSH or further delegation. Preserve sealed unsubmitted capture03 as not balanced. Resolve actual Step FP8/BF16 serving paths for frozen H80032 (P16/D16), TP8/DP2/EP16/PP1; generic monolithic-may-bypass is insufficient. Reuse retained kernel/inventory evidence and search alternative load_balance/round_robin/force_route flags; EPLB placement is not token balance.

YC's verbatim routing request remains:

> routing暂时统一采用balanced distribution方案对齐（frontier和vllm的routing都采用均衡分布的设定）

YC's calibration ordering remains:

> 先进行workflow和gpu侧的对齐（包括op gap计算），最后再进行cpu overhead（gap）的计算测量，不要在当前阶段就引入cpu overhead的大批分析、测量和suits，推迟到矫正最后。

Parent Frontier facts: mainactual06 explicitly sets `--replica_config_moe_routing_distribution_type balanced`; equal1/E ratios are materialized by Hamilton floor/remainder assigned to ascending expert IDs over one Replica-local EP wave's physical token count*K. E352,K8,EP16. Align actual aggregate scope, not only label; per-DP reset differs from global prefix. Existing CPU DP metadata may provide counts/offset without a new collective/GPU readback.

Deliver a compact evidence-backed report/proposal with actual reachable FP8/BF16 kernels, native top-k interface, deterministic balanced count/distinctness/conservation checks at source6128+dummy1/NONE and padded q1/tail cases, options/costs/files/lines, source/image/package/new-generation consequences, honest operator-path identity and remaining live unknowns. Distinguish existing setting from new native implementation. Native extension/locked reference change requires explicit parent decision; do not implement. Primary continues SWA/identity integration; this child does no CPU overhead model/suite work.

## [Original Request] D29 additional semantic check

Per-forward reset can break the independent G3 reference: position6128 gives6128*8%352=96, so full6144's last16 begin at expert96 whereas split D16 reset begins0. Check P6128+D16 vs fullD6144, prefix chunking, real/dummy padding and q1 continuation. Do not weaken/copy actual top-k lists/relabel the reference. Investigate stable logical token offset availability without new GPU readback/collective and what Frontier would need to represent that policy. Materially different outcomes must remain pending parent/user decision; no implementation or new campaign.
