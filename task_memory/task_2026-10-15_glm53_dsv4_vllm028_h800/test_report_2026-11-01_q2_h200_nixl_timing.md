# Test report: Q2 H200 NIXL timing proposal, lead review and adoption

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Created: lead review of Codex Q2 (`8ce5ed55`), independent reruns and refit, source check of the idle wake paths, and the adoption decision D-P2-Q2-NIXL-TIMING. |

Authority: requirements D-P2-INPUT-QUALIFICATION, D-P2-Q1-CHOICES (timing route B) and D-NO-INTERRUPT. Assignment: [Q2](h3o_operator_costs/input_qualification_20261101/codex_assignment_Q2_h200_nixl_timing.md). Codex request `8ce5ed55-1835-4760-891e-826e96823967` (outcome `implemented`), report [implementation.md](h3o_operator_costs/codex-implement-8ce5ed55-1835-4760-891e-826e96823967/implementation.md). Evidence: [nixl_timing_q2/report.md](h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/report.md), `proposed_input.json`, `results.json`, the CSV tables and logs in the same directory. Measurement: [nixl6144 report](test_report_2026-11-01_nixl6144_h200.md).

## Codex result (reviewed)

- Proposed input `AFFINE_MEDIAN_P2_SINGLE_READ`: `analytical_kv_cache_transfer_config_network_bandwidth_gbps = 390.39349928254796`, `analytical_kv_cache_transfer_config_network_latency_ms = 0.0236521842818856`, i.e. 0.9684023419505275 ms per 46,103,040-B READ (H800 pair: 2.987238193 ms). Ordinary least squares on all 90 full-batch case/pass medians against bytes per reader; positive intercept without clamping.
- Replay concurrency: in all four retained replays (B29 and two-fix, P2-32 and P2-64; 128 transfers each), transfer windows never overlap, even across DP lanes. One logical transfer at a time per decode pod, which TP8 maps to eight pair READs: the measured coordinate `t6144_pairs8_inflight1`.
- Event proof: 2,034 possible posting steps; the first later step start on the same lane is at least 18.724965 ms (P2-32) / 24.380076 ms (P2-64) away; the H800, two-fix and proposed prices and every observed n=1 trial stay below it. Zero crossings, so no fresh replay.
- Candidates kept on record: empirical bound (full grid fails 11/90 at P2-32, 4/90 at P2-64), exact concurrency point (90/90 median, per-trial rule fails 7/90), affine (90/90 median, maximum held-out median error 0.919518%; per-trial rule fails 8/90).

## Lead checks

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), cwd the worktree, `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, OpenBLAS/OMP/MKL threads 1, `TMPDIR=<nixl_timing_q2>/scratch`, each script under `timeout 120s`. Rerun script: `/data/ycfeng/tmp/q2_lead_review/rerun.sh` (logs next to it).

| Check | Command / evidence | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Scope | `git status --porcelain`; file list of `nixl_timing_q2/` | only new files under `nixl_timing_q2/`; no `frontier/`, table or retained-evidence edit | only `nixl_timing_q2/` new (plus Codex report dir); untracked `.cortexkit/`, `.pi/`, `remaining_measurements/` pre-existing | PASS |
| Rerun analysis | `rerun.sh`: `analyze.py`, `checks.py`, `verify.py`, `render_report.py` | all exit 0; 12/12 checks | all exit 0; `checks.py` 12 tests OK; lifecycle pytest 9 passed | PASS |
| Reproducibility | SHA-256 of every CSV/JSON/log/report before and after two reruns | data tables, `proposed_input.json`, `report.md` unchanged | unchanged; only `generated_utc`/`started_utc` timestamps and the pytest run time differ | PASS |
| Independent refit | numpy OLS on Pi's `nixl6144_run/case_summary.csv` (not Codex code) | same coefficients and held-out maximum | intercept 0.0236522 ms, 390.39350 Gbit/s, 0.9684024 ms; held-out maximum 0.919528% (difference from Codex ≤5e-8 ms, from the CSV's 6-decimal rounding) | PASS |
| Trace completeness | two-fix `forwards.jsonl.gz`, DECODE rows per lane | contiguous `formation_index`; dummy forwards present | 0 missing indices on 2 / 4 lanes; dummy forwards 170 (P2-32) / 612 (P2-64), all `real=False` and included in the step-start set | PASS |
| Idle wake at wire end (source) | `vllm_dp_busy_loop.py` `next_pass_start`; `replica_schedule_event.py` `_connector_step_events` | wake at a wire end only for a zero-cost dummy group or an idle loop with posted READs | minimum DECODE forward durations: dummy 18.724965 / 24.380076 ms, real ≥19.339 / 25.067 ms; no zero-duration forward | PASS |
| Idle wake at wire end (observed) | two-fix report section 4: B29 (2.570035 ms) vs two-fix (2.987238 ms) | a wake at a wire end would make transfer completion depend on the price | the two replays are bit-identical (forwards, KV transfer times, metrics) | PASS |
| P2 coordinate, per-trial rule | `case_pass_results.csv`, rows `t6144_pairs8_inflight1` | each held-out pass: trial median error ≤5%, p90 ≤10% | passes 0/1/2: median 0.350/0.300/0.208%, p90 2.464/0.639/0.636% | PASS |
| Trial-rule failures | same file, `affine_trial_rule_pass=False` | listed, none removed | 8 rows, all at other coordinates (`t6143_pairs8_inflight1` p0, `t6143_pairs8_inflight4` p1, `t6144_pairs1_inflight1` p0, `t6144_pairs8_inflight4` p0, `t6144_pairs8_inflight16` p1, `t6145_pairs8_inflight1` p1, `t6145_pairs8_inflight8` p1, `t6145_pairs8_inflight16` p1) | recorded |

## Decision

Adopted as D-P2-Q2-NIXL-TIMING (requirements.md): the affine H200 pair above is the formal NIXL timing input for P2-32 and P2-64, qualified only at the measured coordinate, with a physical-fit rule plus the per-pass trial median 5% / p90 10% rule at that coordinate. This is stricter than Codex's median-only rule and passes. No replay: by the event proof and the observed two-price identity, P2 outputs with the new pair equal the B29/two-fix outputs.

## Limits

- Inference, not observation: equality of P2 outputs under the new pair follows from the event proof; no replay with the new pair was run.
- Standalone READ timing; serving compute/collective contention is not reproduced. Tails at the P2 coordinate reach 12.554511 ms, below both step gaps, but finite samples are no future-tail guarantee.
- The pair rests on the source-derived payload of 46,103,040 B per TP pair. The native payload edge stays OPEN for YC.
- The runtime does not enforce the domain: any other bytes, concurrency, pair count, hardware or configuration fails qualification and must not use this pair as a qualified input.
