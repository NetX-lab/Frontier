# Test report: G2h runner job 01 (B13r package, 1 H200)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: G2h-01 terminal state, the path A acceptance against G1h-02, the path A sampling-scope defect and the path B bootstrap failure. |

## Run

| Item | Value |
| --- | --- |
| Package | `h3o_operator_costs/g2h_runner/package_01`, source revision `763d85f0` |
| Job | `G2H_V4_RUNNER_OPERATORS`, tag `v4-h200-g2h-runner-20261028-01`, `exp-1029-113750-923871`, 1 H200, step_main, non-preemptible |
| Submission | Pi `g2h-runner-submit-1029a` (`/data/ycfeng/tmp/ds4_runs/g2h-runner-submit-1029a/`), StepMind Python `RJobBackend` |
| Wall time | Job Create 11:37:51, Job Stop 12:29:20 (UTC+8) = 3,089 s = **0.86 GPU-hours** (`/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_01/launch.log`) |
| Terminal | `RuntimeError: Waited job … ended with status: failed`; cloud status `"error": "RuntimeError: block_operators exited 1"`. The launcher has exited (pid 2178499 is gone). |
| Cloud output | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_01` |
| Local copy | `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_01/download/` (ssh tar, by Pi) |
| GPU-hours | G1h-G4h: about 1.03 before + 0.86 = **about 1.89 of 7** |

## Environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6).
- `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, plus `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`.
- `ulimit -v 31457280`.
- HEAD `bb6d9b59` for the CPU readers.

## Checks

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Path A bridge verdict (`G2H_RUNNER_REPLAY/bridge/bridge_report.json`) | PASS | `PASS []` | PASS |
| 2 | Path A engine logs `run_e{0,1,2}.log` | The 989 cases finish; no error other than the known tokenizer warm-up warning | Each log ends with its last case and contains one `Traceback`, which is vLLM's renderer warm-up warning (`Tokenizer not available when skip_tokenizer_init=True`, logged as WARNING) | PASS |
| 3 | Four gaps (README command, `dsv4_replay_gaps`, output `/data/ycfeng/tmp/g2h_accept/gaps_01/`) against G1h-02 | Sample counts unchanged (the new step scopes must not reclassify work); values are compared, not gated | Counts equal (169,232 / 289,730 / 30,243 / 91,170). `cuda_graph` 0.4977 → 0.4801 µs (−3.54 %), `cuda_graph_piecewise` 0.6250 → 0.6106 (−2.29 %), `eager` 0.5446 → 0.5065 (−6.99 %), `eager_preparation` 1.2209 → 1.2118 (−0.74 %); host-bound 0 in each class | PASS |
| 4 | Attention rows of the 989 cases (`/data/ycfeng/tmp/g2h_accept/rows_compare.py`, `rows_compare2.py`; reference: HEAD rows of G1h-02, `/data/ycfeng/tmp/b13r_accept/after/rows.json`) | Same row identities and kernel counts; medians close | 4,945 of 4,945 identities equal. The identity is every field except `time_stats.*`, `gap_stats.*`, `layer_window_stats.*` and `moe_routing`; `moe_routing` differs only in the package path in `origin`. No `count` or `kernel_count` mismatch. Per-operator median ratio (G2h / G1h-02): 0.976-1.009; p05-p95 within 0.78-1.20 | PASS |
| 5 | Runner rows (`raw/dsv4_raw_e{0,1,2}.runner.csv`) | `dsv4_input_preparation` and `dsv4_sampling` cost rows; `dsv4_block_table_write` validation rows; `kv_cache_block_zero` and `dsv4_compute_logits` diagnostic rows (README) | `dsv4_input_preparation` 989; `dsv4_block_table_write` 976; `kv_cache_block_zero` 499; **`dsv4_sampling` 0 and `dsv4_compute_logits` 0** | **FAIL** (defect 1) |
| 6 | Unowned summary (`raw/dsv4_raw_eN.ownership.json`) | Only the work that the README lists stays unowned | Unowned share: FULL 2.16 %, PIECEWISE 2.18 %, first tail 2.17 %, multi-tail 2.24 %, dummy 3.58 %, prefill 3.07 %. FULL and PIECEWISE each have 4,296 unowned activities under 107 names. They include the sampler kernels (`_gumbel_sample_kernel` [b, 127, 1], the argmax reduce, `_post_update_kernel`, `_get_num_sampled_and_rejected_kernel`), the LM-head GEMM `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` [2, 66, 1], the index and scatter-gather kernels, and 672 `Memcpy DtoH` | **FAIL** (defect 1; the remaining list will be reviewed after the fix) |
| 7 | Path B `G2H_BLOCK_OPERATORS` | 394 rows | Failed after 1.6 s: `ValueError: Installed base mismatch: models/deepseek_v4/nvidia/model.py` (`native_dummy_bootstrap.py:54`; log `download/G2H_BLOCK_OPERATORS/logs/block_operators.log`) | **FAIL** (defect 2) |

## Diagnosis

### Defect 1: the sampling scopes are not recorded

- **Observed.** The traces contain no `frontier_dsa|<step>|-1|dsv4_sampling` and no `…|dsv4_compute_logits` annotation, in either a FULL trace (`traces/g1-e0-0000-full.json`) or the eager prefill trace (`traces/g1-e0-1040-prefill.json`). The same traces do contain the other step scopes (`dsv4_input_preparation`, `dsv4_block_table_write`, `kv_cache_block_zero`) and, in the prefill trace, `frontier_v4_model_forward|685`.
- **Cause (from the source).** `dsa_vllm_runner._wrap_execute_model` (`frontier/profiling/attention/dsa_vllm_runner.py:406-429`) sets `state.active` only for the duration of `execute_model`.
  - The pinned V2 runner returns `None` from `execute_model`. It then runs `sample_tokens` (`vllm/v1/worker/gpu/model_runner.py:1707`), which calls `self.sample` (1749, with `model.compute_logits` at 1340) and `self.postprocess_sampled` (1809).
  - When `active is None`, `_WorkerState.timed_call` (`dsa_vllm_runner.py:272-304`) runs the function without an annotation.
- **Why the CPU tests passed.** `tests/unit/test_dsv4_runner_operators.py` replaces `state.timed_call` with a stand-in that always annotates. It calls `runner.sample()` directly instead of `execute_model` followed by `sample_tokens`.
- **Effect.** No `dsv4_sampling` cost row and no `dsv4_compute_logits` diagnostic row. The other G2h-01 outputs are unaffected: the attention rows, the four gaps (the sampler kernels were already unowned, outside work in G1h-02) and the input-preparation, write and zeroing rows.

### Defect 2: the bootstrap ran twice in one container

- `native_dummy_bootstrap.py` is the accepted bootstrap, copied unchanged from the base package. It requires the installed `model.py` to have the base hash (`c01be6d3…`) before it writes the overlay (`d3556a47…`).
- G2h-01 is the first job that runs the bootstrap twice in one container. Phase 1 had already installed the overlay, so phase 2's base check failed.
- Path B's own code did not run.

## Fix plan (lead decisions, D-NO-INTERRUPT)

- **Codex assignment B13s** (`h3o_operator_costs/codex_assignment_B13s_g2h_sampling_scope_and_split_jobs.md`, request `00a8ac9c`):
  - keep the recorded step active through `sample_tokens`;
  - add a unit test that runs the native call order;
  - review the unowned list on CPU;
  - build `package_02` with two separate jobs (path A replay; path B block operators), each running the accepted bootstrap once.
- The accepted bootstrap and `package_01` are not changed.
- After B13s is accepted:
  - build `package_02` from the commit;
  - run both jobs in parallel on 1 H200 each, estimated 1.1-1.2 GPU-hours in total (G1h-G4h about 3.1 of 7 afterwards);
  - repeat checks 3-6 on the new path A;
  - run the path B acceptance from `g2h_runner/README.md`.

## Limits

- Checks 3 and 4 compare two separate GPU runs. The differences are reported, not gated.
- Check 6 counts activities per case kind from Codex's ownership report. The per-name split into "sampling" and "other" will be confirmed after the fix.
