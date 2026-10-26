# Review record: H3o-(a) operator-level DSV4 cost proposal

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Round 2 submitted and failed (HTTP 429); revision 2a recorded. |
| 2026-10-26 | Round 1 recorded: Codex findings F001-F008, author positions, own additions, remediation in design revision 2. |

## Round 1

- **Component / phase:** H3o-(a) CPU-only proposal (`design.md`, revision 1 at commit `0ce31851`); design review before YC's decision.
- **Reviewer:** StepCode Codex `gpt-6.1-sol-sys-ptu`, effort `xhigh`, request `9c8b2435-c3f3-468e-8c41-6a504ff54fda`.
- **Request:** `review_request_01.md`. **Report:** `codex-review-9c8b2435-c3f3-468e-8c41-6a504ff54fda/review.md`.
- **Verdict:** `changes_required`; P1 × 5 (F001-F005), P2 × 3 (F006-F008). The reviewer ran 81 existing CPU tests (PASS) and two AST checks; no GPU work.
- **Author spot checks (2026-10-26):** the receipt union 0.977821 ms versus span 1.835579 ms (`audit/kernel_coverage_receipt_r2.json:4652-4655`); `chain_gaps_us` starts each graph chain after its first kernel and reads graph-launch kernels only (`kernel_gap/main.py:90-115`); `measured_unequal_combine_time` matches only `expert_parallel_alltoall_combine` (`sequence_parallel_moe.py:133-146`); `primary_acceptance.json:76-78` has all three admission flags `false`; `native_config.json:42` is `balanced_topk`; the shared expert is in `pre_dispatch` (`deepseek_v4.py:44-47`). All match the report.

| ID | Finding (short) | Position | Remediation in revision 2 |
| --- | --- | --- | --- |
| F001 | Layer kernel union used as latency; departs from approved Option A | Agree. Union is busy time; the pipeline also discards interval coordinates | §3.1-§3.2: per-scope costs and a critical path over the source stream graph; raw records keep intervals (file 13); pilot P1 checks A2/A4. Interval-derived union and span are recorded in P1 and in raw records for audit; the training CSV carries per-scope sums and counts only, since no consumer reads union or span. Layer composite is decision D1(b) |
| F002 | Existing kernel-gap path misses eager segments and seams; EP path bypasses the gap wrapper; count naming; TP | Agree. Addition: one CPU look at native profile04 decode dp1 rank 0 shows the decode-only steps as FULL graph replays plus 5,315 eager pre-forward kernels on a separate stream, so the mode must be established per step class first | §3.3; stage 0 R1 census; files 7, 10, 17, 18; decision D5 for reduced TP1 qualified by kernel identity |
| F003 | Missing endpoint contracts, block zeroing, PDD payload admission, shared-expert phase | Agree. Additions: the V4 attention component also accepts `MONOLITHIC` only (`attention/deepseek_v4.py:101`); pre-forward input and metadata preparation kernels are missing from the inventory | §3.5, §4 once-only ledger, files 3, 4, 8, 15 |
| F004 | Rank-0 MoE acceptance used as serving input | Agree | §3.5 rank classes, G3 / G3h, A7, file 16 |
| F005 | Unequal combine bypassed; EP16 small-payload gap certain | Agree. The design's 384 KB was the EP32 minimum | §2 corrected minima; §3.5 classification-based binding; files 8, 9; G5 required; A8 |
| F006 | CUDA_EVENT scopes do not cover eager host cost | Agree | §3.3-§3.4: existing two-stream path; R1 decides materiality; R2 / R2h conditional; decision D2 (whether a per-forward host table is compatible with D-NO-STEP-COST-SHORTCUT) |
| F007 | Model key and features lose workload distinctions; marginal bounds | Agree | §3.6 features from the workload owner, key with planner role and mode, joint coverage A5, group held-out A6, validation migration before deleting `read_isolated_scope` |
| F008 | Routing inventory does not match `balanced_topk`; gating in the wrong family | Agree | §1 runtime scope, §3.5 gating in the MoE family, no hash or score selection charged |

Remaining verification limits: the FULL-graph observation is one rank of one native run, and its host times include profiler overhead. The GPU caps and day estimates are planning caps until C1 and the pilots.

## Round 2

- **Request:** `review_request_02.md` (revision 2 at commit `886290a1`), request `55d630bb-ce7e-40f2-9d0a-658af17f4fed`, same model and effort.
- **Result:** `failed`. StepCode Codex exited 1 after the model gateway returned `429 Too Many Requests` and the reconnect limit (5/5) was exhausted. No report exists; nothing from the run is treated as a review finding. Logs: `/data/ycfeng/tmp/codex-review/55d630bb-ce7e-40f2-9d0a-658af17f4fed/stderr.log` and `stdout.jsonl`. Following the review-skill rule, it was not resubmitted automatically.
- **Lead taken from its command log and verified independently:** the reviewer had run `census_peek2.py` on a retained H200 native trace. The author confirmed that `dsv4-h200-p2-profile-20261025-01` (32 rank traces) and `dsv4-h200-p2-64-profile-20261025-01` (64 rank traces) exist, and reran the script on decode dp0 rank 0 of both: FULL graph replays (249 and 301 `cudaGraphLaunch`) and eager pre-forward kernels on stream 23 (9,937 and 10,853). Revision 2a uses these traces for the H200 runtime census and removes the 32-GPU job R1h.
