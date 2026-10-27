# Test report: H3o-(b) entries 2, 27 and 28 (composed DeepSeek-V4 attention phase)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: checks and results for commit `7003cf1d`. |
| 2026-10-27 | §3.1 filled: fidelity 77/77 identical. |

## 1. Change under test

Commit `7003cf1d`, 8 files, +410 / −11. It implements entry 2 of [h3o_operator_costs/design.md](h3o_operator_costs/design.md) §6 and completes entries 27 and 28:

- `frontier/execution_time_predictor/dsv4_composition.py` (new, 173 lines). The conditional stream program of one V4 layer's attention phase, from S1 ([h3o_operator_costs/s1_stream_graph.md](h3o_operator_costs/s1_stream_graph.md)):
  - the outer input projections fork only when the layer's rows are at most the multi-stream GEMM threshold, and run serially above it;
  - `execute_in_parallel` forks in every mode (the CSA fork: indexer on aux0, attention compressor on aux1);
  - `maybe_execute_in_parallel` runs serially under PIECEWISE (the HCA compressor; the nested indexer fork);
  - the short-context indexer runs its compressor and fill on aux0.
  Each scope weighs its cost plus its kernel count times the gap of its segment class (FULL `cuda_graph`; PIECEWISE `cuda_graph_piecewise`, except the `_sparse_indexer_and_attn` scopes, which are an eager break; NONE `eager`). The device-only timeline returns the composed elapsed and each scope's start offset.
- `serial_attention_phase` is the schedule of the two predictors that have no stream program inputs (no rows, threshold, kernel counts or gaps): dummy mode and the diagnostic step-cost path. It places the scopes back to back in phase order, with no gap, so their numbers do not change.
- Entry 27: the V4 branch of `ExecutionTime._get_attention_layer_execution_time` returns the composed elapsed. The gaps are inside it, so they are charged once.
- Entry 28: expanded V4 traces place each attention scope at its composed start offset and then move the cursor by the composed elapsed. Aggregated trace spans sum their layers and have no layer timeline, so they ignore the offsets.
- Tests: `tests/unit/test_dsv4_composition.py` (new, 9 tests). The structural trace test in `test_dsv4_structural_execution.py` now passes `request_ids`, so it really expands the layers, and it checks that the trace events are contiguous. Two existing tests pass the new `advance_cursor` argument.

The trained path does not call `compose_attention_phase` yet. That is entry 3, after entry 1 (rows and threshold) and entry 10 (kernel counts and gap values).

## 2. Environment

- Python 3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`.
- `PYTHONPATH` = repository root (plus `tests/unit` for the targeted tests), `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`.
- Memory limit `ulimit -v 62914560` for simulations and the unit suite, `31457280` for the targeted tests and the comparisons.
- Scripts and logs: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/entry2/`.
- Base for comparisons: the entries 20-25 runs at `826539f7` in `/data/ycfeng/tmp/glm53_dsv4/h3o_b/` ([test_report_2026-10-27_h3o_dp_sync.md](test_report_2026-10-27_h3o_dp_sync.md)).
- Fidelity candidate: clean worktree `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-7003cf1d` at the commit. Fidelity baseline: `/data/ycfeng/tmp/glm53_dsv4/h4b_item2/fidelity/candidate_item2`.

## 3. Checks

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `pytest tests/unit/test_dsv4_composition.py test_dsv4_structural_execution.py test_execution_time_op_times.py test_metrics_stage_execution_time.py test_dsv4_cache_zero_execution.py test_dsv4_dp_sync.py test_e2e_gate.py -q -p no:cacheprovider` | all pass | 159 passed in 4.14 s | PASS |
| 2 | Mutation: delete `advance_cursor(schedule.elapsed_ms)` from `metrics_store.py`, run checks 1's trace tests, restore from `/data/ycfeng/tmp/metrics_store.py.bak` | the trace tests fail on the mutant | `test_real_stage_trace_preserves_physical_scope_and_source_metadata[True]` and `test_trace_places_attention_scopes_at_their_composed_offsets` fail; after the restore, `grep -c "advance_cursor(schedule.elapsed_ms)"` = 1 and check 1 passes | PASS |
| 3 | `entry2/run_structural.sh offline` and `online` (`tests/integration/run_dsv4_structural.py --dp 4 --case <case>`), then `entry2/compare_structural.py` and `entry2/compare_traces.py` (log `compare_traces.log`) | rc 0; dummy-mode V4 numbers equal to the base, because the serial schedule equals the old serial sum | rc 0 for both. `report.json` equal except `run_dir`. Byte-identical: `request_metrics.csv`, `system_metrics.json`, both ledgers, the ledger summary, `op_precision_metadata.csv`, `requests.csv`. `schedule_decisions.jsonl`: 300 lines, equal except the wall-clock `timestamp`. `op_traces.jsonl`: 481,145 lines; every field equal except `ts_start` and the header's wall-clock timestamp; largest `ts_start` difference 1.1e-16 s (floating-point rounding of cursor + offset against repeated cursor addition) | PASS |
| 4 | `entry2/run_fidelity.sh` (`tests/e2e/refactor_fidelity/run_matrix.py run --jobs 8 --clean-cache --continue-on-failure`, then `compare`) | 77/77 cases identical to the baseline (no V4 case in the matrix) | run rc 0, 77/77 cases produced artifacts; compare: 77 identical, 0 mismatched (see §3.1) | PASS |
| 5 | `entry2/run_unit.sh` (`pytest tests/unit -q -p no:cacheprovider -p no:logging --continue-on-collection-errors -rfE`) and `h4b_item2/cmp_ids.py h3o_b/unit_final.log h3o_b/entry2/unit.log` | failing test IDs equal to the base | 134 failed, 6169 passed, 208 skipped, 108 errors in 388 s (base: 134 failed, 6160 passed). Failing IDs: base 241, new 241, none only in new, none only in base. Passed +9 = the new test file | PASS |
| 6 | `PYTHON_BIN=/data/ycfeng/envs/frontier-py310/bin/python bash examples/architecture/run_dummy_smoke_matrix.sh` (log `entry2/smoke.log`) | 12 cases complete | "Completed 12 profiling-independent dummy smoke case(s)", rc 0 | PASS |

Check 6 was first run without `PYTHON_BIN`; the script then used the system `python3`, which has no `plotly`, and the first case failed with `ModuleNotFoundError` (log kept as `entry2/smoke_wrong_python.log`). That was a command error, not a result; the rerun above sets the interpreter as the earlier reports do.

### 3.1 Fidelity

Logs: `entry2/fidelity/run.log`, `entry2/fidelity/compare.log`, report `entry2/fidelity/comparison.json`.

- Run: 77/77 cases produced artifacts, rc 0, no failed case.
- Compare, baseline `candidate_item2` at `9b8aaf17e8ee` against `candidate_7003cf1d` at `7003cf1da4ce`: 77 of 77 cases compared; 77 identical; 0 mismatched; 0 baseline failures; 0 candidate-only failures; 0 cases missing from one side, missing evidence, differing definitions or not compared; predictor cache file names identical.
- The compare exits with rc 1 only because of the baseline's provenance line: "measured against a modified working tree at 9b8aaf17e8ee, which is not a commit-specific result". The candidate side is a clean worktree at the commit. This is the same provenance line as in the entries 20-25 report.

## 4. Observations and inferences

Observations:

- Dummy-mode V4 runs give the same request metrics, ledgers and trace rows as before; only trace start times move, by at most 1.1e-16 s.
- The unit failure set did not change; the new tests add 9 passes.
- The composition unit tests check, with hand-computed values: the fork below and the serial order above the threshold (elapsed 45 against 62), the CSA fork in FULL and PIECEWISE and the serialized nested indexer fork in PIECEWISE (indexer compressor at 21, elapsed 47), the HCA fork in FULL and NONE and its serialization in PIECEWISE (30 against 35), the short-context indexer on aux0, the gap per segment class charged once (the attention scope time and the block time equal the composed elapsed), and the typed errors for incomplete inputs.

Inferences:

- The serial schedule keeps every dummy-mode number, so the structural harness and the smoke matrix cannot see the composition. The composition is only exercised by its unit tests until entry 3 feeds it trained costs, kernel counts and gaps. Its accuracy is tested on the operator-level path (gate A12, pilot P1h).
- The 241 failing unit IDs were present before this change; they are not caused by it, and they are not waived.

## 5. Limits

- No GPU run; 0 GPUs used.
- The trained path does not call the composition yet (entry 3). The workload builder does not expose the indexer's short-context state yet (entries 3 and 4).
- The fidelity worktrees `fidelity-candidate-826539f7` and `fidelity-candidate-7003cf1d` are kept; removing them needs YC's approval.
