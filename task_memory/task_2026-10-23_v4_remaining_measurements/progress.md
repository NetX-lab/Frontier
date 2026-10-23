# Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded approval and began CPU audit/design; previous native reference evidence verified intact. |

- completed: YC approved the recommended KERNEL_ONLY CPU implementation/measurement-design route. New GPU runs require a concrete later proposal.
- completed: Root rehashed R4 freeze dependencies/evidence/originals and original R3 prelaunch evidence; all intact. Prior admission records archived at972f0cb1 (24record files; no source changes).
- completed: Private-input design recorded before edits. Scope:2core files/180lines and3non-core files/350lines; a candidate exporter cannot issue its own PASS admission.
- in-progress E: Input implementation in `/data/ycfeng/Frontier/.worktrees/v4-attention-input` (session ses_e992b3667ffeN5frEhBm9plm59). Its first182-test run predates later edits to the reader and preparation tool; acceptance uses reruns on the final commit.
- completed B: Kernel/dependency audit `audit/test_report_2026-10-23_kernel_coverage.md`, `audit/kernel_coverage_receipt_r2.json`, `audit/native_source_evidence.json`; tool `tests/analysis/v4_kernel_coverage.py` (227 lines). 1815 kernels/132 annotations/366 owned kernels reproduced; 1449 kernels unowned, no outer attention boundary, 85 stream waits without event/stream handles, 93 same-stream overlaps (PDL reachable). Retained trace cannot support a serving total; minimum capture additions listed in the report.
- completed D: `native_profile_seam.md` (two rounds) and `native_profile_checks/`. Full native TP8/DP2/EP16 already exists; no shared TP parameter needed. Actual rank0: P NONE at1request/6144tokens; D FULL+PIECEWISE, max11scheduled requests on the16-token descriptor; captured128 is capacity only. Breakable capture is auto-enabled for V4 (server.log line16), so eager attention segments re-run Python at replay. Profiler start/stop restarts work at API level (CPU control); CUPTI re-init under graphs and capture-event survival stay runtime gates.
- completed: Checkpoint corrections applied to plan.md (G1/G2/H) and proposal.md (outer-excluded CUDA_EVENT sums; J3 premise).
- completed root: `baseline_binding.json` regenerated (SHA256 88f0496e…cbf): pre-change 972f0cb1 binds to the 576d278 fixed77 capture; d278659..972f0cb1 changes 24 task-record paths only. A new candidate capture is still required for E.
- completed root: audit tool rerun reproduces `kernel_coverage_receipt_r2.json` (SHA256 caafcedf…c9e1) except the output path and start/end timestamps; rerun at `/data/ycfeng/tmp/glm53_dsv4/root_kernel_coverage_rerun_20261023.json`. Both receipts (~3.9 MB) stay local and untracked, bound by hash and regenerable from the committed tool; `kernel_coverage_receipt.json` (face6d04…d12a) is the superseded first audit pass.
- pending: Bounded core implementation, focused tests, independent review, merged regression/default-matrix accounting and supplemental measurement proposal.
- limits: No additive serving total, physicalTP8/graph costs or full prediction yet. Uncompressed0.9699/1.5241ms difference remains unclassified; no CPU attribution/scaling.
