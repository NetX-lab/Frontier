# Review: mainline drift, completed tickets and reset plan — 2026-10-25

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | First version: drift and loop findings, completed-ticket review, reset plan D31. |

Reviewer: the primary session, at the user's request of 2026-10-25 (see `requirements.md`, D31).
Inspected: this task's `requirements.md`, `plan.md`, `progress.md`, `summary.md`, all 30 tickets,
the D25 native workload records (`../task_2026-10-22_step5_full_workload/`), the D29 sub-task
(`../task_2026-10-22_step5_position_routing/`, including the main09 terminal evidence), the D29
native image records (`../task_2026-10-22_step5_d29_native_balanced/`), the profiling task
records, `git worktree list` and the unmerged branches.

## 1. Has execution drifted from the goal?

The goal is: **Frontier simulates Step-5-Preview, and its TTFT, TPOT, E2E latency and throughput
are calibrated against the native vLLM run to within 10%.**

Yes, execution has drifted a long way. After ten days (2026-10-15 to 2026-10-25):

- **No Frontier-versus-vLLM comparison exists for the frozen workload.** The only completed native
  workload results are the D25 H800 32-GPU and 64-GPU runs of 2026-10-22. They used the original
  router, which Frontier is no longer configured to model (D29 chose `position_cyclic`).
- **Frontier has never finished a run of the workload.** The most advanced attempt, main09 on
  2026-10-25 (job `exp-1025-123506-624501`), simulated **one** request. Both the offline and the
  online mode hit the 2400 s timeout. The offline mode completed 109 of 1024 decode iterations.
- **Frontier has no measured Step-5 operator timing.** main09 used a synthetic CPU fixture,
  `/data/ycfeng/tmp/d29-parent-position-fixture03/` (`fixture_attention_cuda_event.csv` alone is
  5.6 GB of synthetic rows). A real H200 profiler run on 2026-10-21 (`exp-1021-120711-905260`)
  finished natively in two minutes. It was then rejected by the rule that its median must agree
  within 5% with an independent second measurement, and no profile has been admitted since.

### Loops that consumed most of the time

1. **D29 position routing, 2026-10-22 to 2026-10-25.** About 100 commits, a 107-file and
   10,000-line scope ledger, self-imposed limits on how many times a script may be called,
   SHA-256 binding chains, and two independent reviews for scripts used only by this task. The
   work also left 30 worktrees whose commits were never merged. Its output is a single synthetic
   request that does not finish within 2400 s.
2. **The simulator slowness came from the diagnostics that D29 added.** In main09's 109 decode
   iterations, each mode wrote `frontier_ep_wave_lane_ledger.jsonl` (about 22 GB) and
   `op_traces.jsonl` (about 28 GB). That is roughly 460 MB of JSON per decode iteration. A single
   ledger record for one expert-parallel wave is about 145 KB, and about 140 KB of that is the
   `collective_proofs` text. The per-iteration wall time was 14–17 s, and peak memory was 57 GB.
   These outputs prove internal consistency. The E2E comparison does not need them.
3. **Native qualification of vLLM itself** (tickets 04–08, the native parts of 12 and 13, and
   the G1/G2/G3 levels). This work checked kernel correctness, exact KV-cache bytes, observer
   validity and full-rank census data. Ticket 06 alone has about 20 milestone sections. The native
   run is the reference measurement by definition, so Frontier only needs its timings and its
   configuration. Proving that vLLM's kernels are correct does not improve simulation accuracy.
4. **Profiling admission.** Smoke 02 and 03, D18 and D22 all failed or stalled on the same 5%
   independent-median rule. No measured profile reached Frontier.
5. **Process overhead.** The task documents use compressed shorthand, for example
   "9375 source/9389 global" and "native19 1350/1582". The main task directory holds 415 files,
   and there are 101 Step-5-related worktrees. Reconstructing the state took most of this review.

## 2. Review of the completed tickets

| Ticket | Verdict | Serious error or omission |
| --- | --- | --- |
| 01 baseline reuse | Usable | None serious. The dense-initialization time is 2.4% over its frozen budget and stays unresolved. The 77-scenario baseline remains the regression reference for Frontier core changes. |
| 02 native reference lock | Usable but out of date | It locks the base image `sha256:69308454…`. The routing image now required by D29, `hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d958…`, changes four Python files and turns on `position_cyclic` only when `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=position_cyclic` is set. **That image has never run on a GPU.** The first native run must confirm in its log that the strategy was active, otherwise the result does not match the Frontier configuration. |
| 03 text-model inventory | Usable | None. The model has 92 layers: 69 sliding-window (SWA) and 23 sparse (DSA) attention layers, and 88 MoE and 4 dense FFN layers, with 352 experts and top-8 routing. |
| 04 G1 kernel and state qualification | Correct, but not useful for the goal | It checks the correctness of vLLM's kernels, not timing. The 128 zero-scale mismatches do not affect timing. No further work should go here. |
| 06 G2 single-role (simulation scope) | Usable, with one omission | It measured KV-cache capacity on a TP16 single-role startup. The workload topology is TP8 with DP2 or DP4 per role, and H200 has 141 GB per GPU instead of 80 GB. **Frontier's KV-block capacity for H200 must come from the H200 workload run's own log**, meaning `num_gpu_blocks` for each DP lane on P and D. D28's override of 32,000 D-side blocks was an H800 pressure diagnostic and does not apply to the H200 workload. |
| 15, 17, 19 (finish sync), 20 (structural parts) | Code merged, not verified at workload scale | The simulator never completed the workload, so these parts have only been tested on CPU fixtures. |

Further serious findings:

- **The existing H800 reference does not match the current Frontier configuration.** D25 used the
  original router and dummy weights. Frontier with `position_cyclic` cannot be compared against
  it. Either re-collect H800 with the routing image later, or compare H800 only after routing is
  resolved.
- **`tests/integration/step5_pd/priority_cases.json` marks `h200_64` as preemptible.** This
  conflicts with the 2026-10-25 rule that allows only non-preemptible jobs. It must change before
  any H200 64-GPU staging.
- **The communication inputs used in main09 are synthetic for H800**
  (`fixture_collectives/h800/collective_local.json`, scenario `h800_rail`). H200 needs its own
  communication inputs, either a measured H200 table or the analytical backend.

## 3. Recommendations (adopted as D31)

Stop:

- New native qualification work (G1/G2/G3 levels, exact-byte KV checks, observer validation, census
  data). Reopen a piece only when a measured E2E error needs it as an explanation.
- D29's process rules: call quotas, line budgets, hash binding chains and double reviews for
  task-only scripts. One ordinary review per core Frontier change is enough. Acceptance is decided
  by the four E2E metrics.
- Proof ledgers and operator-level traces in E2E runs. They stay available as opt-in diagnostics
  for short targeted runs.

Do, in this order (the dependency graph is in `plan.md`):

1. Fix the documents (this review, D31 in `requirements.md`, the new current plan, the progress
   entry, the summary, and the ticket statuses).
2. Make the Frontier E2E run fast: disable the proof ledger and operator traces for E2E runs, then
   measure the single-request wall time and the full-workload wall time.
3. In parallel with step 2, collect H200 32-GPU native results with the routing image (step_main,
   non-preemptible, 32 GPUs), using the existing D25 stager.
4. In parallel, collect H200 Step-5 operator profiles with the existing `frontier/profiling/step5`
   profiler on one H200 GPU, over the shapes this workload actually reaches. The within-run
   coefficient of variation ≤ 5% stays a hard check. Agreement with an independent median becomes
   a logged diagnostic instead of a blocking rule.
5. Run Frontier for H200 32-GPU with the measured profiles, compare the four metrics, and calibrate
   in the D29 order: workflow first, then GPU operators and kernel gaps, then CPU overhead last.
6. Collect H200 64-GPU native results (64 GPUs, non-preemptible), then run Frontier and compare.
7. Return to H800 when quota allows.

Leave as they are: the unmerged D29 record and candidate branches. They are listed in `summary.md`
as parked. Merging 30 record branches does not advance the goal, and deleting them needs the user's
approval.
