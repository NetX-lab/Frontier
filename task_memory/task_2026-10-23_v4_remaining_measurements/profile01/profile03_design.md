# Next native collection: coverage, DP identity and shutdown evidence

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | YC approved the bounded callback correction/probe and similar diagnostic work; DP's restored-fixture failure is confirmed identical at baseline. |
| 2026-10-24 | Recorded the source-backed collection changes and CPU lifecycle controls; separated the callback decision from the authorized collection repairs. |

## Purpose and current prerequisites

Collect at least three useful first-formal steps per role/lane, actual cross-DP counter correspondence, functioning stream identities and post-stop evidence. Profile02 remains immutable diagnostic evidence with useful counts23/2/61/61. These additions support native tuple/dependency analysis; they do not establish complete operator costs.

- Stream identity: accepted targeted H800 evidence at `stream_probe/results/primary_acceptance.json`; default/auxiliary CUPTI IDs7/13 were joined to traces through two native profiler cycles.
- DP and coverage/lifecycle repairs are integrated at1437ace6; both components' Standards and Spec reviews report No findings. Root's twelve-suite verification observed465PASS/1unchanged-baselineFAIL/0SKIP ([acceptance](integration_cpu/primary_acceptance.json), [report](integration_cpu/test_report_2026-10-24_integration.md)). The old positive32-rank fixture lacks dp_utils; baseline/candidate32 counter/hook dictionaries match. This limitation remains explicit.
- Direct dummy execution is represented in annotations with `step=null`; no synthetic Worker step or counter offset is introduced. Pair raw counters within run generation, role and ordered DP group members, retaining rank/lane membership.
- Captured per-op ownership remains UNKNOWN. The ordinary capture-profiler probe did not expose source-node ownership. The separately approved callback probe is described below.

## Coverage changes

Use the existing native worker's per-role plan construction to select `role_profiler_max_iterations={prefill:512,decode:64}`. Preserve the per-role/lane trace paths and native profiler wrapper. The observed prefillDP1 window spanned405 Worker steps but retained only its first64;512 covers that observed span, not future scheduling variation. Load-driven routing does not guarantee equal lane populations or an upper bound of64 useful steps.

For the next package, select existing `serving_profile_limits.records=600000` and retain `bytes=268435456`. Profile02 rank0 prefillDP0 retained35417 annotations,9229 synchronization rows and64 steps using19930958 accounted bytes, with zero drops. This corresponds to roughly1944 retained rows and0.87MB per observed useful step, including dummy activity. The600000-record cap relaxes the earlier250000 cap; neither it nor256MiB guarantees an arbitrary future window. Counter fields add bytes, and either cap can bind. Report actual drops and retained coverage independently.

Package changes follow source integration. Reuse the existing builder and explicit plan with a fresh run tag/output directory, regenerated manifest and committed harness hashes. Keep balanced routing, native/C26 bytes and workload files pinned. The concrete package budget and final resource sheet are recorded before generation/submission.

## Shutdown changes supported by the CPU controls

Existing worker source SHA256 `9ec701ec12f7a5da0f442f994a353f98c5ae1d2880b9feca38106ea536fd5cab` was unchanged during two controls. Raw command, script, results and capacity observations are at `/data/ycfeng/tmp/glm53_dsv4/profile03_lifecycle_control_jmg_ngi0/`; selected records are retained in `lifecycle_cpu/`.

1. With stop budget10s, peer budget60s and a head marker available at20s, the existing naturally completed headless fixture stops waiting at10s and records missing evidence. This proves the marker wait uses the short stop budget; it does not reproduce a live GPU shutdown.
2. With all four known drain paths already present, an injected enumeration delay keeps the glob quorum waiting20s. This demonstrates a distinguishable failure mode, not the cause of the real cloud delay.

Selected worker correction: check the known peers' explicit drain paths both for quorum and retained drain membership, and use the existing `peer_wait_seconds` for the headless post-stop-marker wait. Preserve native stop escalation, workload/native errors and the soft missing-marker observation. A finite peer budget still cannot guarantee completion after every delayed quorum/stop path; do not claim a timeout guarantee.

Add narrowly scoped monotonic timing for drain-quorum wait, head-marker wait and native stop to a separate diagnostic result field, preserving existing `shutdown_order` meanings. Timings are comparable only within a process. `stop_server` has three timed signal attempts followed by an unbounded final wait; no finite global stop bound is asserted. No pre-stop marker substitutes for the existing post-stop marker.

CPU acceptance: actual worker path with delayed head marker beyond stop budget but within peer budget; late/missing marker remains soft; stale enumeration with visible explicit paths; incomplete peer set still waits; unchanged native stop/return behavior; per-role profiler overrides preserve input config and trace paths. Existing worker/manifest/window tests must pass before review and integration.

## Separate graph decision

Root matched exact-image Torch `graphs.py` and `_graph_annotations.py` to the inspected local bytes. Their exec IDs are built arithmetically from capture-node IDs; this is not the required independent correspondence witness. Source capture receipt: `/data/ycfeng/tmp/glm53_dsv4/graph_api_source_ebaw9_yr/receipt.json`.

The official CUPTI13.0.85 sample is now retained and matches the publisher's archive hash. It dereferences `CUpti_ResourceData.resourceDescriptor` before reading `CUpti_GraphData`, and queries IDs inside the callback. [Source findings](graph_payload_sources.md) distinguish this usage from actual runtime proof. The old probe already derives source scopes from event handles and DAG edges; no new CREATED/runtime callback path is required to test the decoding correction.

Approved by D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY: a fresh copy of the old diagnostic with at most80gross changed probe lines, payload-indirection/CPU controls first, then at most one1H800/codesign/non-preemptible/600s run. Preserve original packages, callback isolation before Kineto, query order and complete9-node/5-kernel/4-event relationship checks, including the deliberately unscoped kernel. Historical3/3 failures remain recorded. Similar bounded diagnostic work may proceed without repeated approval; each concrete scope/attempt ceiling is recorded first. Native/C26 and production-capture integration are outside this probe.

## Execution order

DP and coverage/shutdown source integration is complete; native profile03 package preparation runs alongside callback package validation. Each follows `fresh package + exact image/SDK checks -> concrete H800 submission -> scoped data audit`. Actual >=3 coverage, raw DP pairing, zero unintended drops, stream joins and correct post-stop evidence are verified from the new native run; callback correspondence is verified independently on the probe. A successful process exit alone is insufficient.
