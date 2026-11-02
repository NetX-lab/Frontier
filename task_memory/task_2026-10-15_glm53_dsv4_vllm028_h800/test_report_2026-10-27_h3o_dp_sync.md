# Test report: H3o-(b) entries 20-25 (V4 DP-synchronized descriptor and DP rendezvous)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: checks and results for commit `826539f7`. |

## 1. Change under test

Commit `826539f7a6eed61085473f27ab9950845d9c7363`, 19 files, +930 / −62. It implements entries 20-25 of [h3o_operator_costs/design.md](h3o_operator_costs/design.md) §6, and the pre-DP part of entries 27 and 28:

- `decode_cuda_graph_mode=full_and_piecewise` (V4 only, explicit capture sizes, no speculative decoding), with the native local dispatch: FULL for uniform decodes within the FULL request bound, PIECEWISE with token padding only up to the largest capture size, NONE above it. The PREFILL role builds graph metadata under this mode (entries 24, 25).
- For V4, every lane of a forward, real or dummy, enters the first `pre_moe` room at stage start. The room applies the source synchronization rule (lowest mode, widest token count, uniform count only when all lanes agree, each lane's own request count) to every lane before pricing. Each lane's first layer starts at the later of `R + preparation` and the end of its own pre-DP device work, where `R` is the latest entry. The launch-start alignment stays for other models (entries 20-22).
- `kv_cache_block_zero` is its own `pre_dp` phase before attention, in the execution-time record and in the trace (entries 27 and 28, pre-DP part only; the composed attention elapsed time waits for entry 2).
- Tests (entry 23): new `tests/unit/test_dsv4_dp_sync.py` (27 tests); fixture `tests/fixtures/dsv4_native_dp_sync_rows.json`, built by `h3o_operator_costs/evidence/routing/build_dp_sync_fixture.py`; test stubs in 6 existing test files updated for the new architecture-profile read.

Fixture size: 2080 distinct tuples, 161,629 bytes, from the P2-32, P2-64 and P0 serving counters. The design estimated 685 tuples and about 33 KB. The difference comes from adding the recording lane's DP rank to the tuple: without it, two lanes with different own request counts collapsed into one tuple and the test could not check the per-lane request count.

## 2. Environment

- Python 3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`.
- `PYTHONPATH` = repository root, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`.
- Memory limit `ulimit -v 62914560` for simulations and the unit suite, `31457280` for the short reruns.
- Scripts and logs: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/`.
- Fidelity candidate: clean worktree `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-826539f7` at the commit. Fidelity baseline: `/data/ycfeng/tmp/glm53_dsv4/h4b_item2/fidelity/candidate_item2`.

## 3. Checks

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `pytest tests/unit/test_dsv4_dp_sync.py tests/unit/test_e2e_gate.py -q -p no:cacheprovider` | all pass | 67 passed (27 + 40) in 3.20 s | PASS |
| 2 | `python /data/ycfeng/tmp/glm53_dsv4/h3o_b/mutation_check.py` (two wrong synchronization rules: PIECEWISE pads requests; uniform count ignored) | A13 (a) fails on both, passes on the original rule | both mutants fail the test; the original rule passes | PASS |
| 3 | `run_structural.sh offline` and `run_structural.sh online` (`tests/integration/run_dsv4_structural.py --dp 4 --case <case>`) | rc 0; the counts shared with the accepted C17 reports (`/data/ycfeng/tmp/glm53_dsv4/c17_structural_author_20261019/final_combined_r2_<case>/report.json`) unchanged, except trace rows for the new `pre_dp` phase | rc 0 for both. Requests 4, real EP lanes 4128, idle EP lanes 39904, stage rows 12, EP waves 1376: equal to C17. Trace rows 481,136 → 481,144 (+8), equal to the 8 cache-zero invocations now traced as their own phase. Largest stage conservation error 1.0e-9 ms (C17: 1.0e-9 ms) | PASS |
| 4 | `run_fidelity.sh` (`tests/e2e/refactor_fidelity/run_matrix.py run --jobs 8 --clean-cache --continue-on-failure`, then `compare`) | 77/77 cases identical to the baseline (no V4 case in the matrix) | run rc 0, 77/77 produced artifacts; compare: 77 compared, 77 identical, 0 mismatched, 0 candidate-only failures, predictor caches compared and clean. Compare rc 1 only because of the baseline's provenance line: "measured against a modified working tree at 9b8aaf17e8ee, which is not a commit-specific result" | PASS (numbers); provenance note recorded |
| 5 | `run_unit_final.sh` (`pytest tests/unit -q -p no:cacheprovider -p no:logging --continue-on-collection-errors -rfE`) and `h4b_item2/cmp_ids.py h4b_item2/unit_final.log h3o_b/unit_final.log` | failing test IDs equal to the base | 134 failed, 6160 passed, 208 skipped, 108 errors (base: 134 failed, 6133 passed, 208 skipped, 108 errors). Failing IDs: base 241, new 241, none only in new, none only in base. Passed +27 = the new test file | PASS |
| 6 | `bash examples/architecture/run_dummy_smoke_matrix.sh` (log `smoke.log`) | 12 cases complete | "Completed 12 profiling-independent dummy smoke case(s)", rc 0 | PASS |

## 4. Observations and inferences

Observations:

- The non-V4 fidelity matrix is numerically identical, and the unit failure set did not change.
- The V4 structural runs keep every count shared with C17; the only count change is the 8 new trace rows of the `pre_dp` phase.

Inferences:

- The 241 failing unit IDs were present before this change. They are not caused by it, and they are not waived.
- The V4 timeline changes by design (rendezvous instead of launch-start alignment). The structural harness checks structure and conservation, not latency, so it says nothing about V4 timing accuracy. That is tested later on the operator-level path (gate A12, pilot P1h).

## 5. Limits

- No GPU run; 0 GPUs used.
- Entries 27 and 28 are only partly done: the composed attention elapsed time and its start offsets need entry 2 (`dsv4_composition.py`).
- The fidelity worktree `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-826539f7` is kept; removing it needs YC's approval.
