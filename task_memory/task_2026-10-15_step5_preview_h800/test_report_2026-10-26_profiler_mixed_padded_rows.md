# Test report: mixed batches and graph-padded rows in the Step attention profiler (D31 step D(b)/(c))

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: unit-test triage against the D(a) run and the 77-scenario fidelity comparison. |

## Change under test

Frontier's measured Step attention model takes mixed batches (decodes plus a prompt tail) and
rows padded to a CUDA Graph capture size as features. Before this change the profiler could
produce neither, so no profile row could cover those Frontier queries.

- `frontier/profiling/step5/inputs.py`: `Step5ProfileInput` accepts phase `mixed` (one-token
  decodes first, then longer queries in ascending order, as MRv2 orders a batch) and a
  `physical_query_rows` field. It defaults to the valid tokens. Only a captured graph of real
  requests may pad. `read_profile_inputs` pads a graph cell to the smallest capture size, and
  also accepts an explicit larger captured width, which is what the DP forward-start agreement
  gives a lane whose own tokens need a smaller graph.
- `frontier/profiling/step5/replay.py`: `choose_graph_mode` accepts `mixed` (PIECEWISE) and
  returns eager above the largest capture size. The new `padded_rows` gives the physical rows,
  using the existing `frontier.config.graph_policy.resolve_capture_size`. That import is local
  to `padded_rows`, because the stdlib-only smoke judge imports `choose_graph_mode`.
- `frontier/profiling/step5/rows.py`: the prefill and decode token counts come from the cell;
  `valid_tokens` and `padded_tokens` come from the workload.
- `frontier/profiling/step5/main.py`: the native request capacity covers the padded FULL
  requests, and the token budget covers the padded rows.
- `frontier/profiling/step5/vllm_wrapper.py`: native MRv2 padding in the attention metadata:
  - FULL padded requests have empty queries, sequence length 0 and zero block-table rows.
  - Every padded token row has position 0 and slot `PAD_SLOT_ID`.
  - FULL metadata spans the padded rows; PIECEWISE and eager use the valid counts.
  - `is_prefilling` covers the real requests only, as in MRv2's `model_states/default.py`.
  - The output check compares only the valid rows and the live CSA summary pages.
- Tests:
  - `tests/unit/test_step5_attention_replay.py`: new
    `test_graph_dispatch_pads_to_the_smallest_capture_size_and_eager_does_not_pad`.
  - `tests/unit/test_step5_attention_profiling.py`: new invalid cases,
    `test_mixed_and_padded_cells_produce_predictor_features` and
    `test_explicit_inputs_take_native_padding_or_a_larger_agreed_capture_width`.
  - `tests/unit/test_step5_boundary_probe.py`: the dry-run cell now also has
    `physical_query_rows`.

## Environment

- Unit tests: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python` (Python 3.10;
  no `torch` and no `aiohttp`), `PYTHONPATH=<worktree>:<worktree>/tests/unit`.
- Fidelity: `/data/ycfeng/envs/frontier-py310/bin/python`, `PYTHONPATH=<worktree>`.
- Baseline worktree: `/data/ycfeng/Frontier/.worktrees/step5-d31-native-e275c83` (commit
  `e275c83a`). Candidate: `986480f7` plus this change.

## Results

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| All Step unit-test files (`/data/ycfeng/tmp/d31_pytest/run_db_all.sh`, same file set as D(a)) | Failed IDs equal the D(a) list `/data/ycfeng/tmp/d31_pytest/all4_failed.txt`, minus the test D(a) renamed | 134 failed, 2594 passed, 205 skipped (`/data/ycfeng/tmp/d31_pytest/db_all.log`); 2 new IDs | see triage |
| After the fixes: `test_step5_attention_judgment.py`, `test_step5_boundary_probe.py`, `test_step5_attention_profiling.py`, `test_step5_attention_replay.py` | Only the known `torch`-import failure remains | 289 passed, 1 skipped, 1 failed: `test_native_wiring_owns_exactly_one_graph_and_reuses_it_for_both_traces`, which is in the D(a) list (`/data/ycfeng/tmp/d31_pytest/dbnew4.log`) | PASS |
| Fidelity matrix `tests/e2e/refactor_fidelity/run_matrix.py` against `e275c83`, both sides run fresh | 77/77 identical | 77/77 identical, 0 failures on either side (`/data/ycfeng/tmp/d31_fidelity_dbc2/comparison.json`, log `/data/ycfeng/tmp/d31_fidelity_dbc2.log`) | PASS |

Triage of the 2 new failures:

1. `test_step5_attention_judgment.py::test_cli_is_stdlib_only_and_never_overwrites_report`:
   `replay.py` imported `frontier.config.graph_policy` at module level. That runs
   `frontier/config/__init__.py`, which imports `numpy`, and the smoke judge must stay stdlib
   only. Fix: the import moved into `padded_rows`. The judge calls only `choose_graph_mode`.
2. `test_step5_boundary_probe.py::test_cpu_dry_run_binds_existing_cell_and_never_imports_native_wrapper`:
   the printed cell gained the new `physical_query_rows` field. The expected dict now has
   `"physical_query_rows": 1`.

The one D(a)-list ID that no longer appears,
`test_step5_measured_ffn_replay.py::test_attention_cache_preserves_proof_and_does_not_invent_layer_coverage`,
was renamed in D(a) (`..._block_coverage`), so it is not a fix.

An earlier fidelity attempt (`/data/ycfeng/tmp/d31_fidelity_dbc`) reused a copy of the D(a)
baseline outputs and reported 77 mismatches. Each case differed only in the absolute
`output_dir` path in the copied `config.json`. The comparison above reran both sides fresh.

## Limits

- The vLLM wrapper changes run only on a GPU with the Step image. CPU tests cover the inputs,
  the rows and the dispatch rules, not the native metadata build. The wrapper checks each cell
  against an unpadded reference forward when it runs, so the D(g) profile job is the first
  native check of the padding.
- The fidelity matrix does not run the profiler; it confirms that the simulator results did not
  change.
- The predictor-cache naming check (`--clean-cache`) was not run, because it deletes
  `<repo>/cache`.
