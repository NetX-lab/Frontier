# Test report: G2h runner jobs 02 and 03 (B13s packages, 1 H200 each)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: the G2h-02 replay acceptance, the package_02 block failure and its fix, and the package_03 block acceptance. |

## Runs

| Item | Replay (G2h-02) | Block, first attempt (G2h-02) | Block, rerun (G2h-03) |
| --- | --- | --- | --- |
| Package | `g2h_runner/package_02`, source `812b4362`, pinned `cc89c796` | same | `g2h_runner/package_03`, source `412237db`, pinned `e92b62ef` |
| Job | `G2H_V4_RUNNER_REPLAY`, tag `v4-h200-g2h-runner-20261029-02-replay`, `exp-1029-135035-301114` | `G2H_V4_BLOCK_OPERATORS`, tag `v4-h200-g2h-runner-20261029-02-block`, `exp-1029-135037-016686` | tag `v4-h200-g2h-runner-20261029-03-block`, `exp-1029-141735-456588` |
| Resources | 1 H200, step_main, non-preemptible, 7,200 s timeout | same | same |
| Submission | Pi `g2h-runner-submit-1029b`, StepMind Python `RJobBackend` | same | Pi `g2h-block-submit-1029d` |
| Wall time (UTC) | 05:50:35 → 06:41:24 = 50.8 min = **0.847 GPU-hours** (phase 3,005.2 s) | 05:50:37 → 05:54:11 = 3.6 min = **about 0.06 GPU-hours** (phase 5.8 s) | 06:17:35 → 06:29:24 = 709 s = **0.197 GPU-hours** (phase 658.7 s) |
| Terminal | `Succeeded` | `Failed`: `RuntimeError: V4 prefill requires the observed V4 V2 runner` | `Succeeded` |
| Local copy (read-only, by Pi) | `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_02/download/replay/` (17,775,835,743 B across 1,014 files, with the block files) | `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_02_block_early/download/` | `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_03/download/block/` (792 files, 183,397,933 B, sizes equal to the cloud listing) |

GPU-hours for G1h-G4h: 1.89 after G2h-01, plus 0.06 + 0.847 + 0.197 = **about 2.99 of 7**.

### The first block failure and its fix

- `profile_dsv4_prefill.py:20-21` requires `VLLM_USE_V2_MODEL_RUNNER=1`. The bridge driver sets it only for its replay children. The block job runs the prefill entry directly, so nothing set it. The G2h-01 block phase stopped earlier and never reached this line.
- Lead fix (committed before package_03): `g2h_runner/prepare_package.py` starts the block job command with `env VLLM_USE_V2_MODEL_RUNNER=1`, as the accepted base package's prefill job does. package_03 keeps the bootstrap, the prefill entry and the `frontier/` snapshot byte-identical to package_02. Only the block job was resubmitted.

## Environment (CPU readers)

- Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6). Host `python3` for the CSV and trace summaries.
- `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `TMPDIR=/data/ycfeng/tmp/g2h_accept/tmp`, `ulimit -v 31457280`.
- HEAD `5bacace4` for the readers. Scripts in `/data/ycfeng/tmp/g2h_accept/`: `runner_scopes_02.py`, `rows_compare.py`, `rows_compare2_02.py`, `rows_compare2_02v01.py`.

## Replay checks (G2h-02)

`RAW=/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_02/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw`

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Bridge verdict (`bridge_report.json`, `logs/bridge_report.log`) | PASS, 3 runs | `PASS: 3 runs, FULL=342, PIECEWISE=645`; e0 342 FULL, 342 PIECEWISE, 2 NONE; exit codes 0 / 0 / 0 (e0 2,038 s, e1 630 s, e2 66 s) | PASS |
| 2 | Engine logs `run_e{0,1,2}.log` | No error apart from the known tokenizer warm-up warning | No `Traceback`, `Error` or `ERROR` line apart from that warning | PASS |
| 3 | Step scopes per engine (`runner_scopes_02.py $RAW`, log `scopes_02.log`) | Sampling and logits on every real step, none on dummy steps; logits inside the sampling span; preparation on every step | e0: 686 steps (673 real, 13 dummy), sampling 673, logits 673, preparation 686. e1: 296 / 296 / 296 / 296. e2: 7 / 7 / 7 / 7. All five checks true in each engine; 0 logits intervals outside the sampling span | PASS |
| 4 | Runner rows (`dsv4_raw_e{0,1,2}.runner.csv`) | Preparation 989, write 976, zeroing 499, sampling 976, logits 976 (4,416 in total) | `dsv4_input_preparation` cost 989, `dsv4_block_table_write` validation 976, `kv_cache_block_zero` diagnostic 499, `dsv4_sampling` cost 976, `dsv4_compute_logits` diagnostic 976; total 4,416 | PASS |
| 5 | Unowned activity (`dsv4_raw_eN.ownership.json`) | The sampler kernels and the LM-head GEMM leave the unowned list; only (b) work outside every hook and (c) the output copies remain | FULL 672, PIECEWISE 672, first tail 592, multi-tail 14: only `Memcpy DtoH (Device -> Pinned)` (c), 0.34 % of each kind's activity. Dummy 273 (3.58 %): the `InputBatch.make_dummy` fills, aranges and cumsum, as predicted for (b). Prefill 8 (1.29 %): 3 `_apply_write_kernel` and 3 HtoD copies (b), and 2 DtoH copies (c). No sampler or LM-head name is unowned. The LM-head GEMM (`nvjet_sm90_tst_*` [2, 66, 1]) and its memset are now under the logits scope | PASS |
| 6 | Four gaps (`dsv4_replay_gaps`, output `/data/ycfeng/tmp/g2h_accept/gaps_02/`) | Sample counts unchanged; values compared, not gated | Counts equal (169,232 / 289,730 / 30,243 / 91,170). `cuda_graph` 0.4791 µs (−0.22 % vs G2h-01, −3.75 % vs G1h-02), `cuda_graph_piecewise` 0.6093 (−0.22 %, −2.50 %), `eager` 0.5371 (+6.03 %, −1.39 %), `eager_preparation` 1.2983 (+7.14 %, +6.34 %) | PASS |
| 7 | Attention rows (`rows_compare2_02.py` against `/data/ycfeng/tmp/b13r_accept/after/rows.json`, G1h-02; `rows_compare2_02v01.py` against G2h-01 `rows_01/rows.json`) | 4,945 identities equal, no count or kernel-count mismatch, medians close | 4,945 of 4,945 identities equal to both references; no `count` or `kernel_count` mismatch. Per-operator median ratio: 0.978-1.010 against G1h-02, 1.000-1.005 against G2h-01; p05-p95 within 0.80-1.19 | PASS |

`rows_compare.py` reports 0 identity matches. Its identity keeps every field whose name lacks "time", "count", "median" and "mean", so `moe_routing` (whose `origin` holds the package path) breaks every match. `rows_compare2` removes `moe_routing`, as in the G2h-01 report.

### Replay runner values (observations)

| Operator | Role | Rows | Median of the per-row medians (ms) | Range (ms) |
| --- | --- | ---: | ---: | --- |
| `dsv4_input_preparation` | cost | 989 | 0.08234 | 0.05648-0.13498 |
| `dsv4_sampling` | cost | 976 | 0.03024 | 0.02141-0.06329 |
| `dsv4_block_table_write` | validation | 976 | 0.00307 | 0.0-0.00822 |
| `kv_cache_block_zero` | diagnostic | 499 | 0.00374 | 0.00231-0.5921 |
| `dsv4_compute_logits` | diagnostic | 976 | 0.24706 | 0.240-0.269 |

- The sampling row excludes the nested logits scope; each kernel belongs to its innermost scope.
- The replay logits scope (0.247 ms) is about 7.3 times the standalone `dsv4_lm_head` row of G2h-01 (0.033-0.034 ms for 1-24 rows). The replay engine runs with `tensor_parallel_size=1` (engine log), so its LM head computes the full vocabulary. The standalone operator measures one TP8 vocabulary shard. Inference: the 7.3 ratio is the 8-way shard plus a fixed launch cost. This is why the LM head is priced from the operator table and the replay logits row stays a diagnostic.
- Some write rows have a 0.0 kernel median: the scope ran but launched no write kernel on those steps. B17 must handle this when it compares the replay writes with path B.

## Block checks (G2h-03)

`D=/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_03/download/block/G2H_BLOCK_OPERATORS`

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 8 | Layout (`layout.json`, and the `layout` column of every row) | The native V4 layout of `expected_rows.json` | 139 segments, 10 chunks per page, zero-kernel block size 1,024, block sizes [256, 64, 64, 4, 8], five KV groups with [62, 22, 21, 42, 20] members, 112,927 blocks, 128 request rows, row widths [28, 112, 112, 1792, 896], 43 layers. One layout value across all rows | PASS |
| 9 | Coordinates against `g2h_runner/expected_rows.json` `path_b_rows` | The 394 planned rows, compared as complete rows (operator, method, n, w_c, r) | Equal sets, 394 rows, no duplicates: zeroing n 1-32 and 2,520; writes w_c 0-32 with r 0-4 (164 per method) | PASS |
| 10 | Values | Finite and positive; 20 repeats after 3 warm-ups | min, max, mean and median of `time_stats` and `kernel_stats` finite and positive in all 394 rows; `repeat_count` 20 and `warmup_count` 3 everywhere | PASS |
| 11 | Kernel count | One kernel per repeat | `kernel_count_per_repeat` is 20 ones in every row | PASS |
| 12 | Trace ownership (spot check: zeroing n = 2,520 and write w_c = 10, r = 2, record_function intervals) | Per repeat: the hold, one index upload, the operator kernel | 20 `spin_kernel` holds, 20 `Memcpy HtoD (Pinned -> Device)` and 20 `_zero_kv_blocks_kernel` or `_apply_write_kernel`; nothing else | PASS |
| 13 | Step log `logs/block_operators.log` | No error | One warning traceback (the tokenizer warm-up) and the NCCL `destroy_process_group` warning at exit; `row_count: 394` printed | PASS |

Block values (record_function kernel medians, observations): zeroing grows linearly from 0.00189 ms (n = 1) to 0.02909 ms (n = 32), and n = 2,520 takes 2.905 ms. Writes take 0.0038-0.0049 ms over the grid. The cuda_event rows add about 0.007 ms of event overhead at small n (0.00909 ms against 0.00189 ms at n = 1); the KERNEL_ONLY rows are the cost rows (design §3.1).

## Limits

- The replay runs one GPU with TP1. Per-rank shapes for the attention scopes come from the plan; the LM head does not, which the logits diagnostic shows. Input preparation, sampling (on the gathered full-vocabulary logits) and the block operators do not depend on the TP width. This is an inference from the vLLM V1 code path, not a TP8 measurement.
- The `eager` and `eager_preparation` gaps differ by 6-7 % between runs with equal sample counts. These are run-to-run differences between nodes. B16 promotes the G1h-02 gaps, as its assignment states.
- The replay write rows (validation) have not yet been compared with the path B rows; B17 does that comparison.
- Pi left a 0-byte `cloud_status_block_1.err` in `g2h_runner_03/`; it stays in place.
