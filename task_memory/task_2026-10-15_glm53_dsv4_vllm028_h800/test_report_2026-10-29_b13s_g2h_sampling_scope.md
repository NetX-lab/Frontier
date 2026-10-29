# Test report: B13s, the G2h sampling scope fix and the split jobs (CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: the Codex outcome, the lead corrections, the CPU acceptance checks 1-9 and the unowned review. |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B13s_g2h_sampling_scope_and_split_jobs.md](h3o_operator_costs/codex_assignment_B13s_g2h_sampling_scope_and_split_jobs.md). The defects come from [test_report_2026-10-29_g2h_runner_01.md](test_report_2026-10-29_g2h_runner_01.md).
- Codex request `00a8ac9c`, report `h3o_operator_costs/codex-implement-00a8ac9c-e2e3-477a-a749-ede651e9e158/implementation.md`, scratch `/data/ycfeng/tmp/codex-implement/00a8ac9c-e2e3-477a-a749-ede651e9e158/b13s/`.
- Lead review scratch: `/data/ycfeng/tmp/b13s_review/`. Base revision `5896339e` (B15).

## Codex outcome

Codex reported the CPU work complete. The changes:

- **Defect 1, the missing sampling scopes.** `_WorkerState.close_step_range` now ends the active step through the existing `end_step`. `_wrap_execute_model` keeps `state.active` while the V4 graph-replay step range is open. The range opens in `_wrap_step` and closes in `_wrap_sample_tokens`, or in `run_dummy_step` for the native dummy batch. GLM DSA and eager V4 open no range, so their steps still end when `execute_model` returns.
- **Defect 2, the second bootstrap in one container.** `prepare_package.py` builds two jobs, each with one phase and one bootstrap:
  - `G2H_V4_RUNNER_REPLAY`, tag `v4-h200-g2h-runner-20261029-02-replay`, output `.../g2h_runner_02/replay`;
  - `G2H_V4_BLOCK_OPERATORS`, tag `v4-h200-g2h-runner-20261029-02-block`, output `.../g2h_runner_02/block`.

  The path B row check now compares complete rows. The accepted bootstrap, the base package and `package_01` are unchanged.
- **Premise correction (from the pinned source).** Dummy steps do not sample: `_dummy_run` returns at `gpu/model_runner.py:768` without calling `sample_tokens`. The dummy step keeps its existing closure in `run_dummy_step`. No sampling call is added.
- **Tests.** One native-order test runs `execute_model` and then `sample_tokens` through the real recorder, for five step kinds. One test covers the dummy step. Two failure-cleanup tests were also added; the lead removed them (see below).

## Lead corrections (D-NO-INTERRUPT)

- **Removed the new failure-cleanup branches.** Codex added an `except BaseException` branch in `_wrap_execute_model` and another in `_wrap_sample_tokens`. Both closed the range without saving the step. No caller catches a failed execute or sample and keeps profiling: the engine run ends and the bootstrap exits 1. These branches handle a state that cannot occur.
  - `_wrap_sample_tokens` is back to its committed form.
  - The existing B3 contract stays: after a failed execute with no range open, `active` and `matched_dp_peer` are cleared (`test_case_wrapper_clears_peer_after_recorded_and_unrecorded_calls[True]`).
  - The two new failure tests are removed.
- **`close_step_range`** ends the step without a `None` check. A range opens only for an active step, so the active step is always set when the range closes.
- **`test_dummy_worker_closes_range`**: the test stand-in opened a range with no active step, which `_wrap_step` never does. It now opens the range under an active step and checks that the step is recorded once.
- README: the sentence about failed calls now says that a failure ends the engine run.

## Environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM).
- `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `TMPDIR=/data/ycfeng/tmp/b13s_review/tmp`, `ulimit -v 31457280`.

## Checks (assignment acceptance 1-9)

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | The native-order tests fail on the committed runner code (`head_pytest.py` loads the HEAD copies of `dsa_vllm_runner.py` and `dsv4_graph_replay_runner.py`; both are equal to `git show HEAD:` by `cmp`) | They fail | Exit 1: the 5 native-order cases and the dummy case fail. On HEAD the step range closes with no sampling or logits scope, and the dummy step ends at `execute_model` return (`/data/ycfeng/tmp/b13s_review/head_native.log`) | PASS |
| 2 | The same tests and the required suite pass after the fix: `pytest tests/unit/test_dsv4_*.py test_*graph_replay*.py test_*bridge*.py test_profiling_runtime_boundary.py test_*kernel_gap*.py test_dsa_attention_profiling.py test_dsa_query_one_profiling.py test_dsa_extend_profiling.py test_dsa_full_dummy_profiling.py`, with the known kernel_count test deselected | All pass | 1,412 passed, 28 skipped (torch absent, or `FRONTIER_VLLM_BS_ROOT` unset), 1 deselected, in 59.5 s (`suite.log`). Codex's 1,414 included the two removed failure tests | PASS |
| 3 | Lifetime per step kind against the pinned source | The 976 real steps sample after `execute_model` returns; the 13 dummy steps do not sample | Codex's table of pinned call sites (`gpu/model_runner.py:1703` returns None; `v1/engine/core.py:602-603` samples; `_dummy_run` 695-768 has no sampling) was spot-checked by the lead | PASS |
| 4 | Saved G1h-02 and G2h-01 records give the same rows and gaps with the new code (`check_saved.py` into `/data/ycfeng/tmp/b13s_review/saved_after/`, compared by `cmp` with Codex's HEAD outputs in `b13s/before/`) | Byte-identical diagnostics, attention rows, runner rows and ownership | 8 of 8 files byte-identical. G1h-02: 989 cases, 4,945 attention rows, 0 runner rows. G2h-01: 989 cases, 4,945 attention rows, 2,464 runner rows | PASS |
| 5 | Saved G2h-01 gap values | `cuda_graph` 0.4801221185543367, `cuda_graph_piecewise` 0.6106290602187813, `eager` 0.5065229453923222, `eager_preparation` 1.2117857340253784 µs | Equal to every digit (`saved_after.log`); G1h-02 values also unchanged (0.49773414267747085, 0.6249537992538311, 0.5446049595929223, 1.2208671118771937) | PASS |
| 6 | Unowned review on CPU | Each unowned activity assigned to (a) the sampling and logits scopes after the fix, (b) work outside every hook, or (c) the intended output copies | Section below | PASS |
| 7 | GLM DSA records unchanged | CUDA_EVENT and KERNEL_ONLY records, annotations and rows byte-identical between HEAD and the new code | `check_glm.py` (copied to the review scratch, with HEAD `dsa_vllm_runner.py` from `git show`): CUDA_EVENT SHA-256 `3a32296a…`, KERNEL_ONLY `4ea8314d…`, both PASS, equal to Codex's values | PASS |
| 8 | Split-jobs dry run: `prepare_package.py --dry_run` | Two jobs, one bootstrap each, distinct tags and outputs, the 989 case IDs, 394 path B rows, no package built | Exit 0; engines 686 / 296 / 7; path B 33 zero + 164 write per method, 394 rows; import probe PASS; replay job steps `bridge_run`, `bridge_report`; block job step `block_operators`; `package_built: false` (`dry_run.json`). The job schema is the same set of keys as `package_01/jobs/G2H_V4_RUNNER_OPERATORS.json` | PASS |
| 9 | Hygiene | No unused import in the changed Python files; `git diff --check` clean; `package_01` untouched | `[]` for all five files; diff check clean after the lead removed one blank line at the end of `test_dsv4_runner_operators.py`; `git status` of `package_01` empty | PASS |

## Unowned review (G2h-01, CPU evidence from Codex)

Codex read the three ownership summaries and the two permitted traces (`g1-e0-0000-full.json`, `g1-e0-1040-prefill.json`). The counts include kernels, memcpy and memset.

| Case kind | (a) sampling and logits after the fix | (b) outside every hook | (c) output copies | Total |
| --- | ---: | ---: | ---: | ---: |
| FULL | 3,624 | 0 | 672 | 4,296 |
| PIECEWISE | 3,624 | 0 | 672 | 4,296 |
| first tail | 3,184 | 0 | 592 | 3,776 |
| multi-tail | 77 | 0 | 14 | 91 |
| dummy | 0 | 273 | 0 | 273 |
| prefill | 11 | 6 | 2 | 19 |

- (a) includes the sampler kernels, the argmax, the logits-row gathers and the LM-head GEMM `nvjet_sm90_tst_*` [2, 66, 1] with its memset. The traces place the memset under `aten::linear → aten::matmul → aten::mm`, so it belongs to the logits scope. The GPU run must confirm this.
- (b) is the prefill request-state writes (3 `_apply_write_kernel` and 3 HtoD copies, `gpu/states.py:122-124`) and the dummy batch setup (`InputBatch.make_dummy` fills, aranges and cumsum, and the hidden-row gather at `gpu/model_runner.py:767`).
- (c) is the two `Memcpy DtoH (Device → Pinned)` per real step from `AsyncOutput`; they stay unpriced.
- **Lead decision on (b).** It is not wrapped in this round. The prefill writes are 6 activities in one case; the dummy batch setup is 21 activities per dummy step. Both stay visible in the unowned summary. The dummy setup joins the open item on the dummy-step unowned kernels, to be decided with the G2h-02 traces.

## Expected G2h-02 results (for the GPU acceptance)

- Runner rows: preparation 989, block-table write 976, zeroing 499, sampling 976, logits 976 (4,416 in total). Sampling and logits share each real step's index, with logits nested under sampling. Dummy steps have no sampling row.
- The sampling kernels and the LM-head GEMM leave the unowned list; (b) and (c) remain.
- Path B: the native layout, every planned coordinate, 394 rows.
- Estimated time: replay 0.891-0.958 and block 0.117-0.234 GPU-hours, 1.008-1.192 in total. G1h-G4h would then be at about 2.90-3.08 of 7.

## Limits

- All checks run on CPU, with stand-ins for torch and the vLLM runner. The trace ownership after the fix is predicted from the pinned call order; G2h-02 must confirm it.
- The path B code has not yet run on a GPU.
