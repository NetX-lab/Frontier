# Review record: H3o-(a) operator-level DSV4 cost proposal

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Round 2 result recorded: verdict `changes_required`, F009-F015 (P1 × 4, P2 × 3), closure of F001-F008, author spot checks and positions, remediation in design revision 2c. |
| 2026-10-26 | Round 2 resubmitted on revision 2b (request `4678dfdc-…`) after YC's approval. |
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

### Round 2, resubmission

- **Approval:** YC approved round 2 (requirements item 31) together with decisions D2-D6 and the routing identity rule (items 29-30).
- **Request:** `review_request_03.md` on revision 2b at HEAD `40299c0a`; request `4678dfdc-a4d3-47a2-8c00-de2e7803a915`, same model and effort. It covers revisions 2, 2a and 2b and adds question 7 on routing identity (§3.7, A13, files 20-23).
- **Reviewer:** StepCode Codex `gpt-6.1-sol-sys-paygo`, effort `xhigh`. **Report:** `codex-review-4678dfdc-a4d3-47a2-8c00-de2e7803a915/review.md`.
- **Verdict:** `changes_required`; new findings P1 × 4 (F009-F012), P2 × 3 (F013-F015); no P0 or P3. The reviewer ran 116 CPU tests (`test_balanced_routing.py`, `test_dp_dummy_forward_pricing.py`, PASS), reran the rule check (3,588 rows, 245,088 forward observations, 0 mismatches), and reproduced each new defect with CPU sentinels and counter reads. No GPU work.
- **Closure of round 1 (design level, not implementation acceptance):** F003, F004, F005, F007 and F008 resolved; F001, F002 and F006 partly resolved, with the remaining parts carried by F009, F010, F013, F014 and F015. Accepted without change: D2 host-only `forward_launch` rows (consistent with ADR 0002 if DP, KV and device waits stay excluded and no residual is formed), no union or span training columns, H200-first order and the GPU rules of §7.

#### Author spot checks (2026-10-26)

| ID | Evidence checked | Result |
| --- | --- | --- |
| F009 | `N/vllm/v1/worker/gpu/model_runner.py:1417-1546`: request-state updates, then `dispatch_cg_and_sync_dp`, then `prepare_inputs`, `prepare_attn` and `model_state.prepare_attn`; `sync_entry.py:110-133` and `dp_dummy_forward.py:130-134` | Confirmed. The V2 runner prepares inputs and attention metadata after the DP all-reduce; the existing deferral sets a lane's arrival from its own stage start, and the launch alignment bounds only the host share |
| F010 | `execution_time.py:1116-1135`: the V4 branch returns the serial sum of `timing_operator_times(self, "attention")` and omits `_kernel_gap_times["attention"]`; consumers `collective_timing.py:29-32`, `replica_stage_schedule_event.py:263,558`, `metrics_store.py:1157-1165` | Confirmed |
| F011 | `N/.../gpu_worker.py:1210-1212` (`_dummy_run(1, uniform_decode=True)`); `model_runner.py:647-701, 1500-1546` (dummy `InputBatch` of the synchronized descriptor, dummy attention metadata, FULL re-stage); `input_batch.py:116-206` (sequence length equals query length, every row marked padding); Frontier `attention/deepseek_v4.py:268-276` rejects any dummy with graph metadata | Confirmed. Native counters, first node of each role, rank 0: decode dummies 486-2,233 FULL forwards; prefill dummies 8,408-9,455 FULL forwards (`evidence/routing/mode_census.txt`) |
| F012 | `N/vllm/v1/worker/utils.py:601-607` (first tail gives uniform `None`); `dp_utils.py:52-95` (synchronized uniform count and max query length, final `dispatch` at 84-90); `cudagraph_utils.py:75-100, 382-410` (FULL descriptors require the uniform count; PIECEWISE accepts any); counter row 428 of the H200 P2-32 decode node: local `[12, 8]`, desired `[1, 2]`, uniform `[0, 1]`, final PIECEWISE `[16, 16]`; captured FULL descriptors 1-128, PIECEWISE sizes up to 256; Frontier `vllm_v1_iteration_policy.py:98-107` gives NONE to a first tail under `full_decode_only` | Confirmed. Reran the reviewer's census: decode desired-mode differences in 347 / 274 / 803 rows (P0 / P2-32 / P2-64), prefill in 14 / 14 / 53 rows (`evidence/routing/review_counter_checks.txt`, copied with its script from the scratch directory; both rerun byte-identical). The design's sentences "the uniform-disagreement branch is not reached" and "no forward with a mode difference" are wrong; `dp_sync_census.py` labelled every non-NONE difference as a FULL width difference |
| F013 | `kernel_gap/main.py:107-115`: one `covered_end` across all streams of a launch | Confirmed; the reducer measures global-union idle time |
| F014 | `N/vllm/models/deepseek_v4/attention.py:515-573` (`enable=hidden_states.shape[0] <= VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`); `envs.py:2016-2017` (default 1024); `utils/multi_stream_utils.py:45-61` (no auxiliary stream under Breakable capture); the producer records the threshold (`dsv4_attention_runner.py:356`) | Confirmed |
| F015 | `dsa_vllm_runner.py:597-604` forces `enforce_eager` for timing; `dsa_plan.py:787-794` rejects graph per-scope timing | Confirmed |

#### Positions and remediation (design revision 2c)

| ID | Finding (short) | Position | Remediation in revision 2c |
| --- | --- | --- | --- |
| F009 | DP barrier and first-layer timing in the wrong order | Agree | §3.7 item 3: the first pre_moe room computes the DP rendezvous from each lane's pre-DP ready time, then places post-DP preparation and first-layer device work after it; the V4 path replaces the launch alignment; the probe's `prepare_inputs_e2e` is split at the DP all-reduce if R2h runs; files 21-22; A13 (d) with the 110 ms sentinel |
| F010 | No composed-duration consumer; V4 getter is a serial sum without gap | Agree | §3.2: composition returns the composed elapsed and per-scope start offsets; the canonical ledger keeps every scope cost; `execution_time.py` V4 branch delegates and charges the gap once; trace offsets from the composition; files 2, 27, 28 |
| F011 | Synchronized V4 dummy keeps the eager one-row no-metadata workload | Agree | §3.7 item 4: V2 dummy workload (descriptor rows and requests, sequence length equal to query length, padding rows, dummy metadata, FULL re-stage), no request metrics or KV credit; dummy coordinates in C1 and G1h; files 3, 4 |
| F012 | Local dispatch differs from native; first tail synchronizes to the wrong mode | Agree | §3.7 items 1-2: native `FULL_AND_PIECEWISE` local classifier as a new value `full_and_piecewise` of the existing `decode_cuda_graph_mode` field (V4 only in H3o), PIECEWISE pads tokens and not requests, PREFILL admitted for this value; the synchronized descriptor follows the source including the uniform and max-query redispatch; census claims corrected; A13 (c) starts from Frontier batches; files 20, 24, 25 |
| F013 | Global idle scalar is not a critical-path gap | Agree; of the two corrections offered, the qualified approximation is taken, since it keeps the `kernel_gap.csv` schema | §3.3: V4 gap measured per stream (idle time after the later of the previous same-stream kernel and its dependency), one value per mode and TP, added per kernel along each path before the maximum; A4 per group; failure stops the route; file 17 |
| F014 | One fixed stream graph per variant | Agree | §3.2: conditional graph selected by rows against the recorded `multi_stream_gemm_token_threshold`, capture kind and the indexer's short-context state; file 2; S1 and P1h cover both sides |
| F015 | No qualified FULL replay route for P1h | Agree | §5 stage 1a: a minimal FULL and PIECEWISE replay route before P1h (scope ownership by kernel signature from the eager hook pass, replay intervals from the same call, exact one-to-one match or the window fails); files 13, 17 split into a bootstrap part |

Open after revision 2c: D1 stays YC's decision. The CPU-probe binding to the V2 runner is checked on CPU before R2h. A further review round needs YC's approval.
