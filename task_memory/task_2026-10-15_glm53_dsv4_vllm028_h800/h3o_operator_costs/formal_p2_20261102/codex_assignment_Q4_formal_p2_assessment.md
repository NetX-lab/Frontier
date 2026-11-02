# Task: Formal P2 assessment against ticket 19 (Q4)

## Goal

Settle the formal ticket 19 verdict for the two H200 priority cases:

- P2-32: P16+D16, TP8/DP2/EP16.
- P2-64: P32+D32, TP8/DP4/EP32.

Workload for both: 128 requests, 6144 prompt / 1024 output tokens.

All P2 input edges have now been decided:

- **Attention:** the independent-pass check was adopted (D-P2-Q1-CHOICES).
- **NIXL timing:** the H200 affine pair 390.39349928254796 Gbit/s / 0.0236521842818856 ms is the formal input (D-P2-Q2-NIXL-TIMING).
- **Payload:** QUALIFIED as source-derived and metric-invariant (D-P2-PAYLOAD-C).

The two-fix replays (`two_fix_20261101/`) still consumed the H800 pair from the B29 inputs (126.2 Gbit/s, 0.0647 ms). Ticket 19's `frontier_reference_run` forbids cross-hardware cost substitution. The lead therefore started two formal fresh-cache CPU replays that override only those two fields with the H200 pair:

- Runner: `formal_p2_20261102/run_formal_replay.py`, committed at `abd1e141`.
- Both started at 2026-11-02T03:09:44Z.
- Command records: `formal_p2_20261102/p2_32_command.json` and `p2_64_command.json`.
- Scratch root: `/data/ycfeng/tmp/glm53_dsv4/formal_p2_20261102/{runs,gates,logs}`.
- Readings: `formal_p2_20261102/readings/`.
- The two-fix runs took 10,242 s (P2-32) and 12,442 s (P2-64). The formal runs should finish around 06:00Z and 06:40Z, gate and readings included.

Deliver these results:

1. **Current input status.** Give one row for each of the 27 Q1 ledger inputs (`input_qualification_20261101/ledger.json`). For each row, give:
   - its current status;
   - the decision and evidence that set it;
   - for any row still OPEN or NEEDS_DECISION, whether ticket 19's final PASS needs it for each case, citing the ledger limitation and the ticket text.

   The Q1 ledger files are a frozen snapshot: do not edit them.

   Re-assess "Formal hardware/operator coverage as a whole" against its own recorded criterion, now that its three missing items (attention, H200 timing, payload) are resolved. Return PASS or FAIL with evidence.

   Do not change any criterion. A row that cannot close from retained evidence stays OPEN, with a proposal and its cost.
2. **Input binding.** For each formal run, match every consumed input to its qualified ledger entry by path and SHA-256:
   - profiling tables;
   - the trace and request population;
   - the resolved config, including the H200 pair;
   - the source commit and the driver.

   A consumed input that is missing from the ledger, has a different hash, or has a fallback or omission is a finding.
3. **Formal versus two-fix.** Compare each formal run with its two-fix run:
   - the forward sequence;
   - KV transfer start and completion as the scheduler sees them;
   - request metrics;
   - gate rows and readings.

   Expected: identical, except fields that carry the NIXL pair or the wire end time. That follows from the Q2 event proof: every possible posting step has a next-step gap of at least 18.724965 / 24.380076 ms, and the price is 0.968 ms. Report any other difference with evidence. Do not patch or rerun anything.

   Show the comparison method is sound first: run it on two-fix against B29, which are known to be bit-identical.
4. **Gate readings.** For each case and each of the three repeats, report from the formal gate and readings:
   - the four warm-reading relative errors (mean client TTFT, mean request TPOT, THPT, mean request E2E), each gated at 0.10 under D-WARM-TARGET;
   - the unchanged all-request readings beside them;
   - the excluded request IDs and their evidence, plus the original and retained window endpoints.

   Pooled values are diagnostics only. A pooled PASS never overrides a repeat FAIL.
5. **Milestone map.** Map every checkbox item of ticket 19's `frontier_reference_run`, `comparison_and_calibration`, `promotion_32_to_64` and `final_acceptance` to:
   - its evidence path;
   - PASS, FAIL, NOT_APPLICABLE or BLOCKED_ON_DECISION, using the ticket's labels.

   Cover non-dummy timing without fallbacks, the feature flags, effective `max_model_len`, `max_num_batched_tokens` and the memory budget, all 128 exact-length completions, and the non-gating p50/p95 diagnostics.
6. **Promotion receipt.**
   - Evaluate the P2-32 → P2-64 30% predicate exactly as the ticket states it, and draft the receipt contents.
   - Establish from retained records when the P2-64 native reference was collected, relative to any receipt. The ticket says the receipt "must exist before formal 64-GPU native submission, not be supplied retrospectively".
   - Return the P2-64 verdict with options for YC, each with evidence and cost. At least: (a) YC waives the launch order for the existing P2-64 reference; (b) issue the P2-32 receipt and run a new formal native P2-64 campaign. For (b), estimate GPU-hours and resource class from the retained records of the original P2-64 native campaign.
7. **GPU-hours.** List every P2 H200 GPU job actually run (native references, profiling, collectives, NIXL, failed attempts) with:
   - job ID;
   - purpose;
   - GPUs and resource class;
   - GPU-hours, with the source of each number.

   Give the total, and mark unknowns as unknown.
8. **Verdict for each case.** State PASS, FAIL or BLOCKED_ON_DECISION, list every blocking reason, and state the limitations that qualify the verdict.

## Approach

- Authorities:
  - ticket `tickets/19-comparison-and-acceptance.md` (milestones and Acceptance);
  - in `requirements.md`: D-P2-INPUT-QUALIFICATION, D-P2-Q1-CHOICES, D-P2-Q2-NIXL-TIMING, D-P2-PAYLOAD-C, D-WARM-TARGET, D-WARM-TWO-FIX, D-WARM-EP-DEFER, decision 46 (measured communication allowed) and D-NO-INTERRUPT;
  - `summary.md` items 1–5 of the P2 input section.
- Evidence:
  - `input_qualification_20261101/{ledger.md,ledger.json,decisions.md}`;
  - `test_report_2026-11-01_q2_h200_nixl_timing.md`;
  - `test_report_2026-11-02_q3_payload_bound.md`;
  - `two_fix_20261101/test_report_2026-11-01_two_fix.md`;
  - `warm_target_20261101/test_report_2026-11-01_warm_reading.md`;
  - `test_report_2026-10-31_ex_situ_input_audit.md`;
  - `test_report_2026-11-01_collective_exact_repeatability.md`;
  - the native reference receipts used by `a10/a10_gate.py`.
- Reuse the existing tools rather than writing new arithmetic: `a10/a10_gate.py`, `tests/analysis/e2e_warm_reading.py`, `two_fix_20261101/compare_b29.py` and the readings layout of `two_fix_20261101/readings/`.
- **Waiting for the replays.** Do items 1, 2 and 5–7 first. Poll the two command records about every 10 minutes until both show a terminal `status` (`FINISHED` or `FAILED`). Then do items 3, 4 and 8 on the formal outputs.
  - If a replay fails, report its log tail and stage, and give the verdict as incomplete.
  - If a replay is still running at 09:00Z, return with everything else done and the comparison ready to run.

## Scope and context

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`, HEAD `abd1e141` or a later records-only commit.
- Untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` are not yours.
- Task records: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`.
- Constraints:
  - Do not edit `frontier/`, `data/profiling/`, thresholds, native references, formal tables, the formal timing input or the Q1 ledger files.
  - Do not start, stop or modify the running replays, their command records, their scratch roots or `formal_p2_20261102/readings/`.
  - Retained evidence and receipts are read-only.
  - No RJob, platform, cloud or GPU action.
  - Do not commit; the lead reviews and commits.
- Writable path: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_20261102/assessment/`, with scratch in `assessment/scratch/`.
- Environment:
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6).
  - Variables: `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
  - Run heavy reads under `timeout`.
  - The two replays use most of the host's CPU memory headroom: keep your own processes small and do not run another replay.

## Expected return

- Report language: English, plain words, actual values.
- `assessment/report.md` covers results 1–8. Put machine-readable results next to it: `input_status.json`, `binding.json`, `formal_vs_two_fix.json`, `gate_readings.csv`, `milestones.json`, `promotion.json`, `gpu_hours.csv`, `verdict.json`.
- For each check, give the command, environment, expected versus actual, and PASS/FAIL.
- List changed and new paths, and any remaining blocker with evidence and a recommended next action.
