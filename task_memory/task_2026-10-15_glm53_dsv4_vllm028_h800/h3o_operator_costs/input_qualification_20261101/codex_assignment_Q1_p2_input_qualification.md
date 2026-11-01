# Task: Qualify the formal inputs of the DSV4 P2-H200 replays (Q1)

## Goal

Formal P2 acceptance for DeepSeek-V4-Flash on H200 (P2-32: P16+D16, TP8/DP2/EP16; P2-64: P32+D32, TP8/DP4/EP32) needs two things. The numerical part is already met: under D-WARM-TARGET the warm readings pass all four metrics within 10% in every repeat at both scales (B29 replays, reproduced bit-for-bit by the D-WARM-TWO-FIX replays). What remains open is **input qualification**. The plan, ticket 05/09/11/19 status lines and the two summaries say warm numerical PASS "does not qualify omitted attention workloads, hardware, NIXL/payload or memory".

Settle each open qualification edge with evidence, closing as many as the existing criteria allow:

1. Build one qualification ledger covering every input the P2 replays consume. For each input record its origin, its consumer, the criterion that applies to it (with the requirement, ticket or decision that sets that criterion) and the evidence. Give each input a status: QUALIFIED, OPEN with the exact missing evidence, or NEEDS_DECISION.
2. Close every edge that retained evidence plus CPU work can close under criteria that already exist. CPU replays are allowed when they settle an edge.
3. If an edge can only close with a new GPU measurement, build the measurement package with the existing producers and package builders and verify it on CPU. Give the GPU type and count, preemptible or not, the expected container GPU-hours, the outputs and the acceptance checks. **Do not submit it.**
4. If an edge can only close with a criterion that does not exist yet, return a concrete proposal: evidence, options with their costs, and a recommendation. Examples: accepting an input because P2 metrics provably do not depend on it, promoting the median-of-three attention table, choosing a CSA FULL measurement correction, or a model for the H200 NIXL timing.

Acceptance: the ledger exists and every edge named in the records has a status backed by cited evidence. Each closure carries its commands and actual results. Each GPU package passes its own CPU checks. Each proposal is specific enough for YC to answer with one choice.

## Approach

Known open edges and the evidence that already exists (read the records for detail; verify rather than trust these summaries):

- **Attention coverage.** 1,373 of 1,421 workloads are complete. 48 are omitted (24 partial, 24 fully skipped). The forward classes FULL/dummy, NONE/dummy, NONE/prefill and PIECEWISE/dummy have no complete evaluable workload, so overall attention is NOT_EVALUABLE. NONE/prefill is eager prefill, the main part of TTFT. Find out why each class is unevaluable (missing measured layer windows, missing rows, or something else). Then determine which omitted workloads the P2 replays actually query, and whether retained data can evaluate them. Reports: `test_report_2026-11-01_forward_attention_continuation.md`, `h3o_operator_costs/a6_forward_attention_20261101/`. Promoting the median-of-three table, and fixing the scope-level CSA FULL FAIL (`test_report_2026-11-01_a6_csa_full_cause.md`), both need YC decisions. D-A6-FORWARD-ATTENTION replaced the scope-level attention verdict with forward-level qualification.
- **H200 NIXL timing.** The formal KV transfer input is still the H800 fit (126.2 Gbit/s, 0.0647 ms). That is cross-hardware, and `frontier_reference_run` in ticket 19 forbids cross-hardware substitution. The H200 NIXL02 measurement is complete and its integrity verified, but the affine fit was rejected: the intercept is negative and the eight-reader rows vary widely (`test_report_2026-11-01_f3_h200_nixl.md`, `h3o_operator_costs/b30b/f3_nixl02_20261101/`). The two-fix report also found that with `nixl_read_completion=engine_step`, the decode lane finds each READ at its next step start, and those starts are at least 18.72 ms (P2-32) or 24.38 ms (P2-64) apart inside every transfer window (`h3o_operator_costs/two_fix_20261101/test_report_2026-11-01_two_fix.md`, `nixl_window_check.py`). Determine what this means for qualification. Do not clamp the intercept, select passes, or fit to native E2E results.
- **NIXL payload.** Wire bytes now follow the native packed registration (selected pages × 1,002,240 B; commit `0fb8955d`). Actual native request wire bytes were never observed. Decide whether source evidence plus the observed layout is enough under the existing criteria, or what observation would be needed.
- **Memory.** `num_blocks` 109,934 / 112,914 comes from native startup logs (`kv_admission`, ex-situ audit F4). Frontier's planner gives +5.55% / +6.83%, and the F4 planner replays produced metric artifacts byte-identical to B28. Spec 10.4 reportedly permits case-bound measured capacity. Startup memory identities are recorded in `h3o_operator_costs/qualification_codex_20261031/`.
- **Hardware identity and coverage.** Find what the records require (tickets 05/09/11/19, the prior audit `h3o_operator_costs/qualification_codex_20261031/qualification_report.md`) and check it against the actual run receipts.
- **Already ruled.** The measured collective backend is an allowed formal input (decision 46). P2 uses exact collective points only, and both P2 consumers pass the D-COLLECTIVE-EXACT-REPEATABILITY rule. K = 19 is a recorded F2 finding; the 1-GPU interpolation gives 18.29, and K18 replays were run as diagnostics.

Start with the existing qualification audit (`qualification_codex_20261031/`) and the ex-situ audit (`test_report_2026-10-31_ex_situ_input_audit.md`). Reuse their checks; do not repeat completed work. Prefer exact evidence from the B29 / two-fix run roots over new runs.

## Scope and context

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. HEAD `e6990a21`, branch of this worktree; tree clean except the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`, which are not yours.
- Task records: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/{requirements.md,plan.md,progress.md,summary.md}`, tickets `tickets/05-dsv4-flash-family.md`, `09-predictor-bundle.md`, `11-h200-regression-delivery.md`, `17-dsv4-canonical-case.md`, `18-native-ground-truth-h200.md`, `19-comparison-and-acceptance.md`. The current decision is D-P2-INPUT-QUALIFICATION in requirements.md; read D-WARM-TARGET, D-WARM-TWO-FIX, D-WARM-EP-DEFER, D-A6-FORWARD-ATTENTION, D-COLLECTIVE-EXACT-REPEATABILITY, decision 46 and D-CALIBRATION-PRIORITIES as well.
- Run roots: B29 replays `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/{p2_32,p2_64}_warm_k19b29/`; two-fix replays `/data/ycfeng/tmp/glm53_dsv4/two_fix_20261101/runs/{p2_32,p2_64}/`; native downloads `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/`. The replay driver is `h3o_operator_costs/a10/a10_driver.py`, the gate is `a10_gate.py`, and `h3o_operator_costs/two_fix_20261101/run_replay.py` shows a complete supervised replay.
- Constraints:
  - Do not change thresholds, references, folds, formal tables or the formal measured CC input.
  - Do not edit `frontier/` or published `data/profiling/` tables. If a source change looks necessary, return it as a proposal with evidence.
  - Do not model host-zero changes, READ timing, offsets or handshakes.
  - Never remove requests or repeats because they are slow. Keep every original FAIL on record.
  - Retained evidence and receipts are read-only.
- **No RJob or platform action.** Do not submit, stop or query RJobs and do not change cloud storage. The lead submits packages through the `/ds4-exec` skill. Resource rule for any package: at most 32 non-preemptible H200 GPUs across all personal reservations, queued jobs included; anything beyond that must be wholly preemptible. Packages read and write cloud data only under `/mnt/codesign-exp/ycfeng`.
- Writable paths: the working root. Put evidence, ledgers, scripts and packages under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/`. Reusable analysis scripts may go under `tests/analysis/`, with focused tests under `tests/unit/` when they carry logic worth guarding. Do not commit; the lead reviews and commits.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `TMPDIR=/data/ycfeng/tmp`. Write scratch and replay outputs under `/data/ycfeng/tmp/glm53_dsv4/input_qualification_20261101/`; `/tmp` is memory-backed. Limit memory-heavy inspections with `timeout` and `systemd-run --scope -p MemoryMax=30G` where available. A full CPU replay takes about 2.7 h (P2-32) or 3.3 h (P2-64) and needs a 64 GiB address-space limit.

## Expected return

- Report language: English. Use plain words and give actual values.
- `input_qualification_20261101/ledger.md` (or JSON plus a short markdown), with every input, criterion source, evidence and status as described in Goal.
- For each closure: commands, environment, expected versus actual values, PASS/FAIL.
- For each GPU package: path, CPU check results, GPU type/count/preemptible, expected GPU-hours, outputs and verification plan.
- For each NEEDS_DECISION edge: the proposal with options, costs and your recommendation.
- Changed and new paths, and any remaining blocker with evidence and a recommended next action.
