# W128-PROFILE03-PACKAGE requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded the delegated package-only allocation and explicit stopping boundary. |

## Original request and authority

[Original Request] Prepare a fresh 32-H800 diagnostic package in a new branch/worktree `v4-native-profile03-run` from runtime `1437ace6f4a438c1bb879d904e2bede0c777303c`. No seal, launch, platform operations or further delegation. Read the current PRIMARY requirements, plan, profile03 design, integration acceptance and test report. Root accepted 465 PASS / 1 known-baseline FAIL / 0 SKIP and both components' dual reviews with No findings.

[Decision] D-V4-PROFILE03-PACKAGE is authoritative. Ownership is only the existing `v4_p0_native_20261021/build_candidate.py` PROFILE branch plus this new `profile01/profile03/` area. Builder and new profile_plan.json together must stay within 140 gross changed lines against 1437ace6. Generated immutable package content is measured separately. Prior profile02's historical 152 gross lines remain separate. Do not rewrite old packages/plans/acceptances or change core/native/C26/collector/worker/test/checker sources. Preserve native06 branch behavior and the included d87f4c74 entry repair.

[Original Request] Use generation03, runtime1437ace6, role profiler limits prefill512/decode64, records600000/bytes268435456, existing global profiler64/fixed flags, one first-formal window, unchanged balanced canonical workload. Tag `dsv4-h800-p0-profile-20261024-03`; generation `v4-native-v028-h800-p0-profile-20261024-03`; cloud root `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261024`.

[Original Request] Preserve 4x8 H800/codesign/non-preemptible, 64CPU/512mem_gb per replica; 3600s payload, startup/replay1800s, peer1200s, stop90s; TP8/DP2/PP1/localDP1, RDMA/local NFS/personal cloud. Adapt the current profile acceptance guard to the integration receipt's actual status/source_revision, without spoofing legacy status. Bind all13 Git module bytes and copy the PRIMARY acceptance byte-identically into this area. Commit builder/plan/bindings before generation.

[Original Request] Verify13 runtime identities,87native/20binary pins, manifest closure, four node plans with512/64 and600k, actual staged launcher/bootstrap CPU entries using existing checks and explicitly safe stand-ins. Bind existing77-case comparison/domain identity rather than capturing again; preserve prior sealed hashes. No Docker/GPU/StepMind/network/auth/cloud access or local `/mnt` writes. Root owns actual-image/SDK/launch checks. Keep soft diagnostics, add no auxiliary hard gates, and claim no runtime/cost/reference qualification. Logs/caches belong under `/data/ycfeng/tmp`. Commit only owned files, deliver source+records SHAs/package/hash inventory/CPU receipts/root gates, then STOP.

## Acceptance limits

CPU package acceptance does not settle >=3 useful steps per role/lane, unintended drops, raw cross-DP pairing, stream joins, post-stop order or captured per-op ownership. Captured ownership remains UNKNOWN. The one known baseline failure stays explicit and is not relabeled PASS. Resource authority does not authorize this package owner to operate the platform.
