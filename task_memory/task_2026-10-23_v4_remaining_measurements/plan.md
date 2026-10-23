# CPU implementation and measurement preparation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Started approved kernel-only route with parallel evidence audit and integration design. |
| 2026-10-23 | Checkpoint correction: split G into G1 (merge/matrix) and G2 (measurement proposal); added H (native V2 profiling extension) that owns the audit's minimum capture additions. |
| 2026-10-23 | H now owns the two reproduced launch seams, five non-core files / 787 changed lines; E entered independent acceptance in parallel. |
| 2026-10-23 | Session-close scope: finish H review repairs (five files/987), integrate verified CPU work and hand off G2. |
| 2026-10-23 | A–H CPU prerequisites complete; H accepted at64f3c950 and G2 handed off. |

## Dependencies and ownership

Current status: A/B/C/D/E/F/G1/H **completed**. H acceptance is `collector_primary/primary_acceptance.json`: five findings closed,880/987lines, final-source and merged separate-suite checks passed, joint automatic94 unchanged, existing77/77matrix explicitly bound. G2 is **pending** for the next session; `progress.md` contains its numbered actions. The initial and repair allocations below remain the historical scope record.

`A -> {B kernel/dependency audit, C private-input design, D native measurement seam}; C -> E bounded implementation -> F focused verification/review -> G1 merge and matrix gate; {B, D} -> H native profiling extension -> G2 executable measurement proposal`

E and H run in parallel; they own disjoint files.

H allocation: `native_serve_profile.py` 450, `native_pd_worker.py` 25, one unit-test file 300, `dp_placement_pp/vllm_replay.py` 8, and `calibration/pd_replay.py` 4 changed lines. The last two additions repair the reproduced extension-registration, PYTHONPATH and profiler-config forwarding blockers (`collector/launch_seam_receipt.json`, owner commit `9366b6e4`). Root owns driver preparation under G2; H owns implementation and source/CPU validation.

Current H repair allocation supersedes the preceding initial limits: collector550, existing test400, worker25, command builder8, P/D environment4; total987, same owner and five paths. Five initial review findings must close through owner fixes, targeted source re-review and root verification before integration. E/F/G1 are accepted at732406aa. Under YC's session-close request, G2 is handed off after H closure rather than started in this session.

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
