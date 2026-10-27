# Test report: native host-probe runs for the KV side-storage reset host time

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | h200_64 probe result (section 7): all checks PASS, profile accepted for step G; step G host run started. |
| 2026-10-27 | Corrected after codex review 223fee59 (F001–F004): failure classification (3.1), lost-step location and the tightened completeness checks (3.2, 4), withdrawn perturbation bound (5); review fixes and their checks in section 8. |
| 2026-10-27 | h200_32 probe run (`exp-1027-124813-737818`) analyzed: two estimator acceptance rules corrected (`cdcba47b`), DP2 profile written, perturbation bound recorded; h200_64 probe submitted. |

## 1. Scope

Calibration item 3 step 2 (`plan.md`): measure, per role, the host time `h` of a step that
resets the Step-5 KV side storage, from the first reset `.tolist()` return to the forward
launch, without the wait for slower DP engines (`design.md` 4.3). One probe run per DP size:
h200_32 (DP2) for step E, h200_64 (DP4) for step G. The probe runs are diagnostic and are never
a reference (D31.10); the profile holds component host terms only.

## 2. h200_32 probe run

- Stage `/data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_hostprobe03/stage` (clean worktree
  `step5-d31-native-02bd453`, 02bd453a); job `exp-1027-124813-737818`, 4 × 8 H200, `step_main`,
  non-preemptible, creator `i-fengyicheng`; Succeeded after 2866 s (`status_terminal.json`).
- Replay: same workload as `h200_32_sweep03` (420 s length sweep, three repeats of 16 warmup and
  128 formal requests). Repeat means TTFT 913.68 ms, TPOT 53.659 ms (sweep03: 912.24, 53.303).
- Client failures: 3 in the length sweep, 1 in repeat 1 warmup, none in formal phases
  (sweep03: 3 in the length sweep).

## 3. Two acceptance rules corrected (commit `cdcba47b`)

### 3.1 Idle boundaries after a request that never reached a role

- Observation: in hostprobe03 every boundary after repeat 1 warmup was recorded as unverified
  ("Idle/fresh completion deadline", 120 s wait each); in each one both roles were idle and
  decode had one completion fewer than expected (`/data/ycfeng/tmp/d31_diag/probe03/boundaries.txt`).
- Cause: the warmup request failed at the client and never completed on decode. The proxy
  reused a keep-alive connection to decode, sent the request headers, then got an exception
  (aiohttp `connection_reuseconn`, `request_headers_sent`, `request_exception` within 1 ms,
  HTTP 502; proxy line 8405, `proxy.txt`, `failed.txt`). Whether decode received the request
  is unknown; the timing fits a connection the server had already closed (keep-alive 5 s), but
  that is not proven. The replay expected one completion per role for every planned request,
  so the missing decode completion carried into every later boundary.
- Fix: `canonical_replay.phase_completions` counts only the requests the client saw complete
  (`recorded_requests − failed`); `boundary_reached` holds the rule; the estimator re-checks
  each formal-window boundary from its last observation with the same rule, counting from the
  verified length-sweep boundary. Idle gauges are snapshots that update only with engine
  outputs, so this rule alone cannot show that a failed request left no work in a formal
  window; section 8 adds the work-accounting and preemption checks that reject such a window.
- Tests: `test_formal_windows_lie_between_the_idle_boundaries` (reset case passes, the same
  shortfall without a client failure, a busy lane and an unverified sweep fail);
  `test_step5_canonical_native.py` adds a `decode_unreached` fault (all boundaries verified).
  A one-off check with the old rule (`/data/ycfeng/tmp/d31_diag/probe03/oldrule/`) reproduces
  the five deadline failures.

### 3.2 Probe files without an `end` line

- Observation: no probe file of hostprobe03 has an `end` line; the step
  numbers are contiguous; the last written step lies −0.046 to +0.023 s from the final formal
  event (`tails.txt`, `laststeps.txt`).
- Cause: the native P/D launcher stops with "stopping engine client mode=abort timeout=0s" and
  "force killing sub process" in hostprobe03 and in sweep03 alike (`shutdown.txt`), so
  `GPUModelRunner.shutdown` never runs. The file
  is line-buffered, so only the last step and the 1–2 steps whose device events were still
  pending are lost. They lie inside the last formal window (which ends at the idle boundary
  seconds later), after the role's last step with requests.
- First fix (replaced): the estimator required contiguous step numbers starting at 1 per file,
  and equal scheduled tokens per role across the three formal windows on TP rank 0 only. That
  missed a lost tail step on a nonzero TP rank and could not see work that was missing in all
  three windows alike (codex F001). Section 8 replaces it with an every-rank check against the
  client-completed work.
- Result: prefill 784,384 tokens and decode 132,992 tokens in each of the three windows
  (`tokens.txt`: per engine 392,192 / 66,496, equal in every window).

## 4. Estimator run

Command (repo root, d29 runtime env):

```bash
(ulimit -v 31457280; PYTHONPATH=$PWD timeout 900 /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python \
  tests/integration/step5_pd/kv_zeroing_host_time.py --case h200_32 \
  --stage-dir /data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_hostprobe03/stage \
  --clean-summary /data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_sweep03/stage/evidence/replay/canonical_summary.json \
  --job-id exp-1027-124813-737818 --min-samples 100 \
  --output /data/ycfeng/tmp/step5_d31_profiles_20261026/host06/kv_block_zeroing_host_h200_32.json \
  --report /data/ycfeng/tmp/step5_d31_profiles_20261026/host06/kv_block_zeroing_host_h200_32_report.json)
```

Exit 0. Copies: `evidence/d31_h200_32_kv_block_zeroing_host_20261027.json` (profile) and
`evidence/d31_h200_32_kv_block_zeroing_host_report_20261027.json` (report).

| Check | Expectation | Prefill | Decode | Result |
| --- | --- | --- | --- | --- |
| Resetting steps | ≥ 100 | 384 | 14,144 (of 20,915 steps, 67.6%) | PASS |
| Hooks per resetting step | 23 | 23 on all | 23 on all | PASS |
| DP all-reduce join | partial only after the last step with requests | 13,643 joined, 4 trailing partial | 19,936 joined, 0 | PASS (the 4 are dummy steps after prefill's last request; section 8 rule) |
| Window edge clearance | > 0 | 3.19 s | 3.19 s | PASS |
| Window tokens per engine, every TP rank | client-completed work: 128 × 6,128 / 128 × 1,039 | 392,192 × 2 per window | 66,496 × 2 per window | PASS (section 8) |
| Preemptions in formal windows | 0 | 0 | 0 | PASS (section 8) |
| `h` mean (profile value) | — | 37.44 ms | 5.11 ms | — |
| `h` median, p25–p75 | — | 37.02, 35.42–38.54 | 5.09, 4.67–5.32 | — |
| Previous step still running at first hook | decode ≈ 1 | 1.3% | 100% | as expected |
| Device gap of resetting steps vs `h` | close on decode | 96.5 ms median (arrival gaps) | 5.28 vs 5.11 ms mean | PASS (decode, 3%) |
| Device gap of non-resetting decode steps | ≤ 0 (host ahead) | — | median −30.0 ms, p75 +0.09 ms | PASS |
| Trend of `h` over request counts | < 1.0 ms | one count (1) | 0.13 ms over 1–29 | single mean |
| Wait for slower DP engines | split out | 0 | mean 0.58 ms | — |
| Perturbation (client shift vs clean repeat range) | within | TTFT +1.44 ms vs range 8.00 | TPOT +0.357 ms vs range 0.207 | **FAIL (TPOT) → `PERTURBED`** |

Observations:

- Decode: the device idles for the host time on resetting steps (device gap 5.28 ms against
  `h` 5.11 ms), while on non-resetting steps the host runs ahead of the device. This confirms
  the mechanism in `design.md` section 2 with direct timestamps instead of the profiler.
- Prefill: every formal prefill step holds one request of 6,128 tokens. The device was idle
  before 98.7% of them, so the 37 ms is host work (the 23 hooks, input preparation, the DP
  all-reduce and the launch), not a wait for the device. The slowest TP rank launches about
  10 ms after rank 0 (`rank0_host_ms` 27.7 against 37.4); the forward starts with the slowest
  rank.

## 5. Perturbation and decision

- Probe cost inside `h`: the wrapper adds about 4.3 µs per hook call pair on the local CPU
  (`/data/ycfeng/tmp/d31_diag/probe03/overhead/bench.py`, indicative only), about 0.1 ms for 23
  hooks, plus a few stamps around the DP sync and the launch. The probe writes its lines before
  the first hook, which on decode lies inside the device wait.
- TPOT check failed: the probe run's TPOT is 0.357 ms above the clean sweep03 run, against a
  clean repeat range of 0.207 ms. The probe run's own repeats are tight (TPOT 53.62–53.68 ms),
  and only one clean h200_32 run exists (sweep02 was disturbed: TPOT 55.0–58.7 ms), so the
  run-to-run offset between jobs is not known.
- Withdrawn (codex F003): the earlier bound (decode `h` high by at most 0.53 ms, TPOT by at most
  0.36 ms). It assumed the whole shift came from `h` on resetting steps; the probe also stamps
  non-resetting steps and the DP sync, so the shift cannot be assigned to `h` alone.
- Decision: no correction factor. The profile stays labelled `PERTURBED` and is not accepted.
  Step E and step G runs with it are exploratory. Accepting it needs a probe on/off measurement
  within one job (same nodes, alternating phases), which is deferred while the GPU budget runs
  E and G.

## 6. Limits

- The prefill value holds for the 6,128-token prompts of the canonical workload. Its dependence
  on prompt length (block count, input size) is not measured; the probe files also cover the
  length sweep's prefill steps, which could measure it if another workload needs it.
- `h` starts at the first `.tolist()` return. On prefill the host work before the first hook is
  also on the critical path when the device is idle; it is not in the profile.
- The probe source's docstring still says the shutdown writes an `end` line; the staged copies
  were not changed, so the h200_64 stage hashes stay valid.

## 7. h200_64 probe run

- Stage `/data/ycfeng/tmp/step5_d31_workload_20261027/h200_64_hostprobe01/stage` (clean worktree
  `step5-d31-native-02bd453`, 02bd453a; run definition of `h200_64_sweep02` with the 780 s
  sweep). Submitted through Pi `d31-h200-64-hostprobe01-submit-1027f` (prompt
  `/data/ycfeng/tmp/ds4_runs/prompts/d31-h200-64-hostprobe01-submit-1027f.md`); job
  `exp-1027-141317-610373`, 8 × 8 H200, `step_main`, non-preemptible, creator `i-fengyicheng`;
  Succeeded at 07:30 UTC (launcher finished, poll log
  `/data/ycfeng/tmp/ds4_runs/d31-h200-64-hostprobe01-submit-1027f/scratch/poll_hostprobe01.log`).
- Replay: all 7 idle boundaries verified; 14 failed length-sweep requests (soft), none in warmup or
  formal phases. Repeat means TTFT 1,181.52 ms, TPOT 61.21 ms (sweep02: 1,179.88, 61.26). The two
  tracebacks in the server logs are vLLM `EngineDeadError` during API-server shutdown, after the
  replay finished. The stage keeps the earlier replay rule (no counter work accounting); the
  estimator's every-rank token check and preemption check cover the formal windows (section 8).

Estimator command: as in section 4 with `--case h200_64`, the h200_64 stage,
`--clean-summary .../h200_64_sweep02/stage/evidence/replay/canonical_summary.json`,
`--job-id exp-1027-141317-610373`, output `host06/kv_block_zeroing_host_h200_64.json` and
`host06/kv_block_zeroing_host_h200_64_report.json` (log `host06/estimator_h200_64.log`). Exit 0.
Copies: `evidence/d31_h200_64_kv_block_zeroing_host_20261027.json` (profile) and
`evidence/d31_h200_64_kv_block_zeroing_host_report_20261027.json` (report).

| Check | Expectation | Prefill | Decode | Result |
| --- | --- | --- | --- | --- |
| Resetting steps | ≥ 100 | 384 | 17,216 (of 37,879 steps, 45.4%) | PASS |
| Hooks per resetting step | 23 | 23 on all | 23 on all | PASS |
| DP all-reduce join | partial only after the last step with requests | 13,612 joined, 2 trailing partial | 21,131 joined, 0 | PASS |
| Window edge clearance | > 0 | 1.34 s | 1.34 s | PASS |
| Window tokens per engine, every TP rank | client-completed work: 128 × 6,128 / 128 × 1,039 | 196,096 × 4 per window | 33,248 × 4 per window | PASS |
| Preemptions in formal windows | 0 | 0 | 0 | PASS |
| `h` mean (profile value) | — | 36.92 ms | 5.96 ms | — |
| `h` median, p25–p75 | — | 36.29, 35.08–37.56 | 5.88, 5.54–6.21 | — |
| Previous step still running at first hook | — | 94.3% | 100% | see below |
| Device gap of resetting steps vs `h` | close | 36.71 vs 36.29 ms median | 6.09 vs 5.88 ms median | PASS |
| Device gap of non-resetting decode steps | ≤ 0 (host ahead) | — | median −40.1 ms, p75 +0.04 ms | PASS |
| Trend of `h` over request counts | < 1.0 ms | one count (1) | −0.02 ms over 1–17 | single mean |
| Wait for slower DP engines | split out | 0 | mean 0.90 ms | — |
| Perturbation (client shift vs clean repeat range) | within | TTFT +1.64 ms vs range 9.73 | TPOT −0.045 ms vs range 0.494 | PASS |

Comparison with h200_32 (DP2) and observations:

| Value | h200_32 (DP2) | h200_64 (DP4) |
| --- | --- | --- |
| Prefill `h` mean | 37.44 ms | 36.92 ms |
| Decode `h` mean | 5.11 ms | 5.96 ms |
| Decode `h` on TP rank 0 | 4.40 ms | 5.15 ms |
| Decode wait for slower DP engines | 0.58 ms | 0.90 ms |
| Decode resetting share | 67.6% | 45.4% |
| Perturbation | `PERTURBED` (TPOT) | `PASS` |

- The prefill value agrees within 0.5 ms across DP sizes. At DP4 the previous step was still
  running at the first hook in 94% of prefill steps (DP2: 1.3%). On TP rank 0 the previous step
  is a dummy step (the step an idle engine runs to stay in step with busy DP peers) before 384 of
  384 formal resetting steps at DP4 and 383 of 384 at DP2; at DP4 that dummy step is still on the
  device in 360 cases, at DP2 in 5 (`/data/ycfeng/tmp/d31_diag/h64_prev.py`). The device gap still matches
  `h` (36.71 vs 36.29 ms median), so the device waits for the host on these steps in both runs.
- Decode `h` is 0.85 ms higher at DP4, on TP rank 0 as well (+0.75 ms), so the difference is not
  only skew between TP ranks. The cause (EP32 against EP16, the nodes, or batch composition) is
  not analysed; each topology uses its own measured value.
- The DP4 decode share of resetting steps is lower because each engine holds fewer requests
  (at most 32 instead of 64), so fewer requests cross a 16-token block boundary in a step.
- Decision: the h200_64 profile passes every check including the perturbation check, so it is
  accepted for step G as measured. Its TPOT shift (−0.045 ms) also suggests the probe itself adds
  little, which supports, but does not prove, that the h200_32 TPOT shift was job-to-job
  variation; the h200_32 profile stays `PERTURBED` until a probe on/off measurement within one job
  (section 5).
- Step G host run (CPU only) started 08:06 UTC from the clean worktree with options
  `/data/ycfeng/tmp/step5_d31_profiles_20261026/options05/predictor_options_h200_64_host.json`
  (the h200_64 base options plus `kv_block_zeroing_host_profile`), output
  `/data/ycfeng/tmp/d31_e05_run/h200_64_m04_host`; it pairs with `h200_64_m04_base`.

## 8. Codex review 223fee59 and fixes

Review report: `codex-review-223fee59-461a-45f2-9eae-3d6cadf74153/review.md` (task directory).
All four findings were confirmed against the artifacts and fixed.

| ID | Problem | Fix | Check |
| --- | --- | --- | --- |
| F001 | Token equality on TP rank 0 across windows does not prove that every rank's steps are complete; equal loss in all windows passes. | `window_tokens` sums non-dummy tokens per (DP, TP) rank in each window, requires all TP ranks of an engine to agree, and requires the engine sum to equal the work of the client-completed formal requests (`expected_window_tokens`). | `test_window_tokens_match_the_client_completed_work_on_every_rank`, `test_expected_window_tokens_count_the_client_completed_formal_requests`, `test_tp_ranks_of_an_engine_run_the_same_batch` |
| F002 | A failed request may still run on a role; idle gauges are stale snapshots, so a boundary can pass while that work lands in a formal window. | `canonical_replay.phase_work` compares the native prompt-token and inter-token-latency counter deltas of each phase with the work of its client-completed requests; a formal mismatch makes the repetition unverified. The estimator rejects any change of `vllm:num_preemptions_total` in a formal window (`window_preemptions`). | `test_step5_canonical_native.py` fault `sweep_work_in_formal` (formal phase of repeat 1 unverified, extra 1,023 inter-token latencies); `test_a_preemption_in_a_formal_window_rejects_the_run` |
| F003 | The 0.53 ms / 0.36 ms perturbation bounds do not follow from the TPOT shift. | Bounds withdrawn; profile not accepted; E/G exploratory (section 5). | Docs only |
| F004 | The four unmatched prefill all-reduces lie inside the last formal window, not after it. | `engine_steps` accepts a partial all-reduce only after the role's last step with requests; any earlier one raises. The report counts `trailing_partial_all_reduces`. | `test_all_reduces_may_lack_participants_only_after_the_last_step_with_requests` (trailing dummy loss accepted; a lost useful tail on a nonzero TP rank, a lost dummy peer before it, and an interior loss rejected) |

Commands (repo root):

```bash
PYTHONPATH=$PWD:$PWD/tests/unit /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -m pytest \
  tests/unit/test_kv_zeroing_host_time_estimator.py -q \
  --basetemp /data/ycfeng/tmp/d31_pytest/est -p no:cacheprovider -p no:logging        # 14 passed
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH=$PWD:$PWD/tests/unit \
  /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_canonical_native.py -q \
  --basetemp /data/ycfeng/tmp/d31_pytest/native -p no:cacheprovider -p no:logging     # 48 passed
```

- Related step5 test files pass (46 passed, `/data/ycfeng/tmp/d31_diag/probe03/pytest_final.txt`;
  `test_step5_priority_runs.py` 44 passed with `FRONTIER_SIMULATOR_PYTHON` set to the d29 python).
  `test_mla_stage3_online_trace_builder.py` has one failure that imports none of the changed
  modules; deferred.
- Estimator rerun on hostprobe03 with the tightened checks
  (`/data/ycfeng/tmp/d31_diag/probe03/recheck/`): exit 0; every TP rank of each engine holds
  392,192 (prefill) and 66,496 (decode) tokens in all three windows; joins prefill 13,643 plus 4
  trailing partial, decode 19,936 plus 0; no preemptions. The profile is identical apart from the
  report path (prefill 37.4398 ms, decode 5.1105 ms), so the stored copies in `evidence/` are
  unchanged.
