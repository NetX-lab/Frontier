## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | YC deferred naive protocol modeling as optional; resumed D019 with ideal communication. |
| 2026-09-08 | Recorded a verified out-of-scope collective topology limitation. |

## Collective zero-payload topology limit

Independent review reproduced an off-case traffic-generator error in collective-sime564935:2servers ×4GPUs,EP8,pairwise_steps,channels1,exclude_intra_server=True,tensor_bytes0 reaches division byzero in multi-phase chunk counting at htsim_runner.py773. Positive payload behavior is unchanged. The active1server ×8H200 case filters allnetworkpairs before this code and passed the real runner tests. This remains outside the user's single-node calibration implementation scope. Do not claim zero-payload support for every topology or silently expand the repair. Reproduction/source anchors are recorded in review.md.


## Feature work: traced MoE routing distribution import

User explicitly deferred implementation from the current calibration on2026-09-08. Existing capability audit: analysis/routing_import_capability.md. Scope for a separate feature: validated trace input, batch/layer/token-population identity, reuse existing expert-load materializer, explicit handling of differing batch composition, and fresh groundtruth source/dispatch evidence when needed. Do not implement D009 dual-count logging or this importer as part of D011 uniform calibration.


## Optional vLLM naive communication protocol — D020

Status: explicitly deferred by YC on2026-09-08. Current task retains ideal EP dispatch/combine and straggler synchronization. Source evidence and a reviewed proposal are preserved in analysis/d019-communication.md and analysis/d019-communication-review.md. Future scope: explicit communication-runtime selector; shared physicalDP populations without fake Requests; existing five phases with localrouter/broadcast/globalrouting/expert/DPcombine/TP ordering; native same-node broadcast capability. No implementation of these items is part of current D019 execution. Exact naive/ideal traffic and timing parity is not claimed; this approximation must remain visible in final results. Reopen only on explicit user instruction.
