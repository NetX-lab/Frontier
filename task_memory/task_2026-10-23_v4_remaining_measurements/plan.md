# CPU implementation and measurement preparation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Started approved kernel-only route with parallel evidence audit and integration design. |
| 2026-10-23 | Checkpoint correction: split G into G1 (merge/matrix) and G2 (measurement proposal); added H (native V2 profiling extension) that owns the audit's minimum capture additions. |
| 2026-10-23 | H now owns the two reproduced launch seams, five non-core files / 787 changed lines; E entered independent acceptance in parallel. |
| 2026-10-23 | Session-close scope: finish H review repairs (five files/987), integrate verified CPU work and hand off G2. |
| 2026-10-23 | A–H CPU prerequisites complete; H accepted at64f3c950 and G2 handed off. |
| 2026-10-23 | Resumed G2 locally: source-backed window design, one-owner driver/collector implementation, verification/review, then GPU proposal. |
| 2026-10-23 | Assigned G2 implementation to root after two API-quota failures; six non-core files/750 lines, with independent review still required. |
| 2026-10-23 | G2b repaired at433/750lines across seven non-core files; initial reviews complete and follow-up/final verification remain G2c. |
| 2026-10-23 | G2c complete at accepted source 05952396; G2d proposal is ready for a separate GPU decision. |
| 2026-10-23 | Opened the approved H800 collection phase: bounded package preparation, locked-image/source checks, run and retained-evidence verification. |
| 2026-10-23 | Returned the reproduced profile01 entry-import defect to its package owner; fresh profile02 keeps the two-file/300-line allocation and requires actual fresh-process entry checks. |

## Dependencies and ownership

Current status: A/B/C/D/E/F/G1/H and G2a–d **completed for CPU preparation**. H acceptance is `collector_primary/primary_acceptance.json`; G2 acceptance is `driver/primary_acceptance.json` at source `05952396`. The first H800 collection is approved under D-V4-H800-RJOB-AUTHORITY and is entering packaging. The initial and repair allocations below remain historical.

G2 dependencies: `{G2a native window/API evidence, driver callsite design} -> G2b one-owner implementation and CPU controls -> G2c independent Spec/Standards review and root final-source verification -> G2d concrete resource/window proposal -> YC GPU decision`. G2a/b/c are complete for source `05952396`, with seven existing non-core files and 433/750 changed lines. Both R1 reviews closed all findings; final separate suites and repaired-source matrix binding passed (`driver/primary_acceptance.json`). G2d and the GPU decision are complete. Fresh package/source/image checks, collection and runtime qualification remain pending.

Approved collection dependency: `I1 existing builder + explicit profile plan -> I2 CPU package verification and root locked-image checks -> I3 seal/submit/monitor -> I4 retained trace/inventory/client/terminal verification`. Root may inspect quota and prepare verification in parallel with I1. One package writer owns I1; it must stop before sealing. Initial source scope: the existing `v4_p0_native_20261021/build_candidate.py` plus one task-local profile-plan JSON, at most 300 changed lines; reuse existing launch/seal mechanisms. No changes to accepted runtime source, native/C26, original packages or frozen checkers. The package must be explicitly diagnostic-only and preserve the original five workload files.

I2 found one P1 in profile01: the flattened worker cannot import the staged `vllm_replay` from either the launcher or bootstrap environment. The isolated hook probe preloaded `pd_replay` and did not test this entry. I1 is repairing the generated paths for fresh profile02; I2 must recheck both real entry imports, image/source pins and package integrity before I3. Profile01 remains unsealed/unsubmitted. I4's analysis plan distinguishes per-lane useful steps from unproved cross-DP iteration correspondence; no request-set or graph-id inference supplies missing identity.

`A -> {B kernel/dependency audit, C private-input design, D native measurement seam}; C -> E bounded implementation -> F focused verification/review -> G1 merge and matrix gate; {B, D} -> H native profiling extension -> G2 executable measurement proposal`

Historical execution: E and H ran in parallel with disjoint file ownership.

H allocation: `native_serve_profile.py` 450, `native_pd_worker.py` 25, one unit-test file 300, `dp_placement_pp/vllm_replay.py` 8, and `calibration/pd_replay.py` 4 changed lines. The last two additions repair the reproduced extension-registration, PYTHONPATH and profiler-config forwarding blockers (`collector/launch_seam_receipt.json`, owner commit `9366b6e4`). Root owns driver preparation under G2; H owns implementation and source/CPU validation.

Historical H repair allocation superseded the initial limits: collector550, existing test400, worker25, command builder8, P/D environment4; total987, same owner and five paths. All five findings closed before integration. E/F/G1 are accepted at732406aa. The original session-close schedule handed G2 off after H; YC's later resume request superseded that schedule, and G2 has now completed in this session.

| Step | Owner | Acceptance |
| --- | --- | --- |
| A | Root | Approval recorded; previous freeze/results intact; outstanding reference records committed. |
| B | Audit agent | Actual trace/source census distinguishes proven coverage/order/dependencies from missing evidence; originals unchanged. Own only audit records and its analysis script. |
| C | Root | Reuse private sidecar/admission and existing reader/path selection; design integration and typed coverage boundary before core edits. |
| D | Source explorer | Exact native V2 worker/extension/profile seams for TP8/DP2/EP16, graph attribution limits and minimum changes; no invented shared-TP requirement. |
| E | Owner assigned after B/C | Complete private input/consumer component; named file allocation and budget; no unproved total or guard relaxation. |
| F | Root and independent reviewers | Actual accepted-data checks, scoped negatives, source review; fixes owned by implementer. |
| G1 | Root | Merge verified E source; strict fixed-source77 comparison against the bound pre-change capture. |
| H | One implementation owner | New non-core `tests/comparison/calibration/native_serve_profile.py` worker extension (seam §11 estimate 350–450 lines) plus its existing-pattern unit tests. Covers every audit minimum addition: nested outer forward/layer/attention ranges with request/repeat/phase/warmup identity; metadata/mHC/FFN/embedding/norm/logits ownership; per-call event record/wait handle, generation and stream records; PDL launch attributes retained; per step, the lane's own scheduled requests/tokens plus the DP-synced descriptor and graph mode (pinned `gpu/dp_utils.py:34-93` pads every lane to the maximum token count and forces eager when any lane wants eager). CPU-only verification with stand-ins; no native/C26 edit, no GPU. |
| G2 | Root | After H: exact collection needs, resources, duration and acceptance gates for a later GPU decision. |

Read requirements.md before closure. Do not sum scope medians into serving time before ownership completeness and GPU dependencies are proven. Preserve family/PD/graph/TP8 refusals where evidence is absent. A collection proposal is not run approval.
