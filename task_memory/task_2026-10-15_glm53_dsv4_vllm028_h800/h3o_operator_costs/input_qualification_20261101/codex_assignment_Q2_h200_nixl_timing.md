# Task: Propose the H200 NIXL READ timing input and its validation rule for P2 (Q2)

## Goal

The DSV4-Flash P2-H200 replays (P2-32: P16+D16, TP8/DP2/EP16; P2-64: P32+D32, TP8/DP4/EP32) still price each KV-cache READ with the H800 fit (126.2 Gbit/s, 0.0647 ms). Ticket 19 `frontier_reference_run` forbids cross-hardware inputs, so this input is OPEN in the Q1 ledger. Under D-P2-Q1-CHOICES (timing route B) the lead ran the nixl6144 H200 measurement at the exact P2 payload. Its data are now downloaded and verified.

Use these data to settle the timing edge, or to return one concrete decision for YC:

1. Describe the measured H200 READ behavior: repeatability of each exact case across the three passes, agreement of the equal-byte 6143 and 6144 cases, and how completion depends on requests in flight and active pairs. Report per-trial and full-batch tails. Do not drop slow trials, select passes or fit to native E2E results.
2. Find the READ concurrency the P2 replays actually produce, per scale, from the retained replay outputs: how many READs one decode replica or TP pair has in flight at once, and how many pairs read at the same time. State whether the measured grid covers it.
3. Propose an H200 timing input and its validation rule, chosen without looking at E2E metric outcomes. Candidates include, but are not limited to:
   - a bound: every measured completion in the covered concurrency stays below the earliest next decode-step start inside its transfer window (at least 18.724965 ms for P2-32 and 24.380076 ms for P2-64), so with `nixl_read_completion=engine_step` no event moves when the price changes inside the bound;
   - an exact-point or concurrency-indexed value from these data;
   - a physical affine model refitted on these data (it must have a positive intercept without clamping).
   For each candidate give its definition, the validation rule (proposed limits such as per-case median 5% / p90 10% are not yet adopted), the result on every case and pass, the behavior outside the measured domain, whether it needs a Frontier interface change, and its cost.
4. If the recommended candidate can be expressed with existing Frontier configuration and can move a READ completion across a decode-step start, run the two P2 CPU replays with it and compare them with the B29 / two-fix replays. If it cannot move any completion, prove that from the event windows instead of replaying.

Acceptance: every measured case and pass is reported; the concurrency coverage question has a cited answer; each candidate has a stated rule and its actual result; the recommendation is specific enough for YC or the lead to adopt with one choice.

## Approach

- Measurement: package `h3o_operator_costs/input_qualification_20261101/packages/nixl6144/` (README, stage config, plan). Run evidence and the lead's checks: `h3o_operator_costs/input_qualification_20261101/nixl6144_run/` (`verification.json` 11/11 PASS, `case_summary.csv` with critical-path and per-reader columns, `case_pass_spread.csv`, `paired_6143_6144.csv`, `observation_notes.md`, `platform_terminal.json`) and `test_report_2026-11-01_nixl6144_h200.md`. Raw download: `/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download/` (`C_nixl/nixl_reads.jsonl` holds every trial). Producer and readers: `frontier/profiling/collectives/nixl_kv_read_benchmark.py` (`read_cases`, `summarize_case`).
- Verified facts to start from (re-derive what you rely on): one 6144-token READ takes 0.964 ms (1 pair) / 0.971 ms (8 pairs) at the median, against 2.987 ms from the H800 pair; completion is proportional to bytes; pass spread is at most 0.882%; tails exist mainly at 8 active pairs; the largest single trial is 27.497 ms at 8 pairs x 16 in flight, above the 24.38 ms minimum P2-64 next-step gap.
- Earlier H200 run NIXL02 (no 6144 point; affine fit rejected: intercept −46.68 µs, p90 residual 29.18%): `h3o_operator_costs/b30b/f3_nixl02_20261101/`, `test_report_2026-11-01_f3_h200_nixl.md`. Keep that FAIL on record.
- Event-window evidence: `h3o_operator_costs/two_fix_20261101/nixl_window_check.py` and `.json`, `test_report_2026-11-01_two_fix.md`. Replay driver `h3o_operator_costs/a10/a10_driver.py`, gate `a10_gate.py`, supervised replay example `h3o_operator_costs/two_fix_20261101/run_replay.py`.
- Proposal context: `h3o_operator_costs/input_qualification_20261101/decisions.md` ("H200 NIXL timing"), `ledger.md`.
- Payload bytes stay as committed (`0fb8955d`: selected pages × 1,002,240 B). Whether native request bytes match is a separate open edge for YC; note any dependence of your proposal on it.

## Scope and context

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`, HEAD `58ce307e` or a later records-only commit. Tree clean except the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`, which are not yours.
- Task records: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/{requirements.md,plan.md,progress.md,summary.md}`; tickets `tickets/05-dsv4-flash-family.md`, `09-predictor-bundle.md`, `11-h200-regression-delivery.md`, `19-comparison-and-acceptance.md`. Read D-P2-Q1-CHOICES, D-P2-INPUT-QUALIFICATION, D-WARM-TARGET, D-WARM-TWO-FIX and D-COLLECTIVE-EXACT-REPEATABILITY in requirements.md.
- Constraints:
  - Do not change thresholds, references, folds, formal tables, the formal measured CC input or the formal NIXL input. Return the proposed input instead.
  - Do not edit `frontier/` or `data/profiling/`. If a timing model needs a source or interface change, return it as a proposal with evidence.
  - Do not model host-zero changes, READ lifecycle, offsets or handshakes.
  - Never remove requests, repeats, trials or passes because they are slow. Keep every original FAIL on record.
  - Retained evidence, receipts and the raw download are read-only.
- No RJob, platform or cloud action.
- Writable paths: put evidence, scripts and the report under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/`. Reusable analysis scripts may go under `tests/analysis/`, with focused tests under `tests/unit/` when they carry logic worth guarding. Do not commit; the lead reviews and commits.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Put scratch and replay outputs under `nixl_timing_q2/scratch/` (each replay is about 315 MB; `task_memory/` is ignored by Git, and the lead commits only the small evidence files). `/data/ycfeng/tmp` is outside your writable scope. A full CPU replay takes about 2.7 h (P2-32) or 3.3 h (P2-64) and needs a 64 GiB address-space limit; use `systemd-run --scope -p MemoryMax=30G` or `timeout` for memory-heavy inspections where available.

## Expected return

- Report language: English. Use plain words and give actual values.
- `nixl_timing_q2/report.md` with the measured description, the concurrency coverage answer, every candidate with its rule and result, and the recommendation; machine-readable results next to it.
- For each check: commands, environment, expected versus actual values, PASS/FAIL.
- Changed and new paths, and any remaining blocker with evidence and a recommended next action.
