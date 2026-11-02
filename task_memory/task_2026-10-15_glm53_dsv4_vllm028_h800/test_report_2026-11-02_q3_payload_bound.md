# Test report: Q3 native NIXL payload bound under route C, lead review and acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Created: lead review of Codex Q3 (`bd667bb6`), reproducibility rerun, independent recomputation of the bound and the measured check, native source spot checks, and the acceptance of the payload edge under D-P2-PAYLOAD-C. |

Authority: requirements D-P2-PAYLOAD-C (YC: "选择方案c，继续"), D-P2-Q2-NIXL-TIMING and D-P2-INPUT-QUALIFICATION. Assignment: [Q3](h3o_operator_costs/input_qualification_20261101/codex_assignment_Q3_payload_bound.md). Codex request `bd667bb6-ba3c-4f98-8ab5-f459fee8a1cb` (gpt-6.1-sol-sys-paygo / ultra, outcome `implemented`), report [implementation.md](h3o_operator_costs/codex-implement-bd667bb6-ba3c-4f98-8ab5-f459fee8a1cb/implementation.md). Evidence: [payload_q3/report.md](h3o_operator_costs/input_qualification_20261101/payload_q3/report.md), `results.json`, `per_group_bound.csv`, `model_windows.csv`, `selected_reader_trials.csv`, `selected_pass_summary.csv`, `registrations.json`, `source_excerpts.md`, `proposed_ledger_row.json`, `verification.json` and the logs in the same directory.

## Codex result (reviewed)

- **Bound.** `B_max` = 46 pages × 1,002,240 B = 46,103,040 B per TP rank pair, equal to the source-derived payload Frontier already prices. Per group (compressed, swa_0, swa_1, csa_state_0, hca_state_0): 24, 2, 2, 2, 16 pages. Structural bytes 39,521,664 B stay as recorded.
- **Why it is an upper bound.** All P2 requests have 6144 prompt tokens, prefix caching off.
  - Decode (D) loads the whole prompt as external KV and schedules no forward tokens or lookahead slots during the async load.
  - For a sliding-window group, D pads the out-of-window prefix with null pages and allocates only the in-window pages.
  - The connector drops null pages (`get_unhashed_block_ids_all_groups`), then clips each window group to `ceil(W/b)+1` pages.
  - The worker trims each prefill (P) group to D's page count (`_apply_prefix_caching`, non-Mamba path). It then builds one descriptor per page at the packed 1,002,240 B stride (`_compute_desc_ids`, one region).
  - Counting every allocated page would give 2,525,644,800 B; clipping alone would give 50,112,000 B.
- **Checks.** (1) bound PASS; (2) production price 0.9684023419505275 ms, below all 2,034 possible posting-step gaps (minima 18.724965 / 24.380076 ms), zero crossings, PASS; (3) the bound lies inside the eight-pair grid (maximum 817,827,840 B per reader). At the qualifying coordinate, 46,103,040 B per reader, all 1,440 trials are below both gaps (maximum 12.554511 ms). Two prompt cases tie there, t6143 and t6144 inflight1. PASS. (4) wire bytes reach only `get_transfer_time` plus an optional operator trace, which is disabled in all four retained configurations. PASS within that scope.

## Lead checks

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), cwd the worktree, `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, OpenBLAS/OMP/MKL threads 1, each script under `timeout 120s`. Rerun script: `/data/ycfeng/tmp/q3_lead_review/rerun.sh` (logs next to it).

| Check | Command / evidence | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Scope | `git status --porcelain`; `find . -type f -newer <assignment>`; same on the native tree | only new files under `payload_q3/` and the Codex report directory | only those, plus the lead's own `progress.md` edit and an unrelated old launcher log (see Limits); native tree unchanged | PASS |
| Rerun | `rerun.sh`: `analyze.py`, lifecycle pytest, `verify.py` | all exit 0; four checks PASS | exit 0 / 0 / 0; four PASS; 9 passed; four audits PASS, 104 inputs unchanged | PASS |
| Reproducibility | SHA-256 of every CSV/JSON/MD/log before and after | identical | identical, except `results.json` records the caller's `TMPDIR`; a rerun with Codex's `TMPDIR=payload_q3/scratch` restores the original hash `7117685e...36ce` exactly | PASS |
| Independent bound arithmetic | own closed form from the group block sizes and windows (QUALIFIED layout row of the Q1 ledger) | 46 pages, 46,103,040 B | live pages 24/2/2/2/16, clip limits 24/3/3/3/17, selected 46 pages, 46,103,040 B | PASS |
| Independent measured check | own read of raw `C_nixl/nixl_reads.jsonl` (not Codex code) | smallest eight-pair coordinate ≥ `B_max` and every trial there below 18.724965 ms | 8-pair grid of 10 byte coordinates up to 817,827,840 B; selected 46,103,040 B (t6143 and t6144 inflight1); 1,440 trials; maximum completion 12.554511 ms, or 12.570083 ms with preparation and posting added | PASS |
| Independent price | own formula `0.0236521842818856 + B×8/(390.39349928254796×1e6)` | 0.9684023419505275 ms | 0.9684023419505275 ms | PASS |
| Native source spot checks | `single_type_kv_cache_manager.py:250–323`, `get_num_skipped_tokens`; `kv_cache_manager.py:99–109`; `base_scheduler.py:128–140,240–284`; `base_worker.py:99–130,2412–2446`; `pull_worker.py:225–329`; `scheduler.py:918–921,1008–1043` | the cited rules exist and apply unconditionally on the P2 path | null-padded skipped prefix and in-window allocation; nulls dropped before exchange; window clip; non-Mamba suffix trim to D count; concatenated descriptors over one region; zero new tokens and no lookahead during async load | PASS |
| Hybrid allocator dependence | `config/vllm.py:1717–1789`; Codex report | the bound must not rest on clipping alone | with the hybrid allocator disabled no clipping happens, but null dropping and the suffix trim still give 46 pages | PASS |
| Byte sinks | `rg get_grouped_nixl_wire_bytes|wire_bytes|physical_block_size_bytes frontier` | wire bytes reach only the transfer time in retained configurations | predictor line 148 (`get_transfer_time`) and `op_trace_utils.py:894` (trace only, disabled); other `physical_block_size_bytes` uses are memory planning and cache zeroing, which use the fixed layout, not the selected pages | PASS |

## Decision

Accepted under D-P2-PAYLOAD-C. The ledger edge "Actual native request-selected descriptors/rank wire bytes" is **QUALIFIED** for the retained P2-32 and P2-64 configurations, with provenance `source-derived, metric-invariant` (not observed). The bound equals the payload Frontier already prices, so the formal inputs, the two-fix replays and every P2 output stay as they are. No replay.

The Q1 `ledger.md` / `ledger.json` stay as the frozen Q1 snapshot; the current status of each edge is kept in `summary.md` and tickets 05, 09, 11 and 19.

## Limits

- Native final descriptor IDs, ranges and aliases were never observed. The bound is a source derivation checked by host-only method fixtures, not a serving witness.
- Scope: retained P2-32/P2-64 on H200 only, with homogeneous TP8, 6144-token prompts, prefix caching, compression and operator tracing off, and `engine_step` completion. H800, a tracing-enabled run, other prompt lengths and other TP layouts need their own check.
- Finite tail sample: the 12.554511 ms maximum stays 6.17 ms below the P2-32 gap, but finite samples are no future-tail guarantee (same limit as Q2).
- Unrelated finding during the scope check: `native_track_20261020/g0d_launch/launcher.log` (about 2.88 GB) still grows with proxy-connection errors for the old RJob `exp-1020-203529-472675`. No process in this session's namespace holds it open. It was not touched; it is reported to YC.
