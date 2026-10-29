# Test report: G2h runner job 04 (native sampling contract, 1 H200)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: the G2h-04 replay acceptance and its comparison with G2h-02 and G1h-02. |

## Run

| Item | Value |
| --- | --- |
| Purpose | Replay the G1h case set with the native request contract (`min_tokens == max_tokens`, `ignore_eos`, temperature 0), so that the sampling scope measures the batch-level `needs_logits_processing = True` branch of the pinned V2 sampler |
| Package | `g2h_runner/package_04`, built in the worktree `.worktrees/g2h-runner-04-build` at `18161ea2`; `status.json` `source_revision` `994d293c` (the last source commit, the `min_tokens` fix); `manifest.json` SHA-256 `d885c37c...3171016b7`; job `G2H_RUNNER_REPLAY`, tag `v4-h200-g2h-runner-20261029-04-replay` |
| Job | `exp-1029-154446-382293`, creator `i-fengyicheng`, 1 H200, `step-main-default`, non-preemptible; StepMind Python `RJobBackend` through Pi |
| Image and runtime | `vllm-openai@sha256:2286e853...`; vLLM 0.28.0, torch 2.13.0+cu130, FlashInfer 0.6.16.post3, Python 3.12.3; `nvidia-smi`: `NVIDIA H200, 143771 MiB, 570.124.06` |
| Queue | Pending from 07:44:46Z (`Queue remaining: H200=0`); last Pending poll 08:38:09Z, `Starting` at the 08:43:15Z poll, `Running` at 08:48:24Z |
| Phase | e0 08:48:21 → 09:23:04 (2,083 s), e1 → 09:33:46 (642 s), e2 → 09:34:56 (70 s); package phase 3,076.2 s |
| Terminal | `Succeeded` (finish 09:39:35Z); the launcher logged `FINAL_STATUS succeeded` and stopped the job |
| GPU-hours | 08:43:15Z (first `Starting` poll) → 09:39:35Z = 56.3 min = **0.94 GPU-hours**, an upper bound because the platform reports no start time. This replaces the 0.854 I gave earlier, which started at the first engine start and left out container start-up |
| Local copy (read-only, by Pi) | `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_04/download/replay/`: 1,012 files, 17,846,523,218 B; cloud total 17,846,477,745 B (`remote_du.txt`; the local total adds directory entries); 1,011 copied files listed in `status.json`, 0 trace bytes left on the worker |

GPU-hours for G1h-G4h: 2.99 after G2h-03, plus 0.94 = **about 3.93 of 7**.

## Environment (CPU readers)

- Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6); host `python3` for the JSON and CSV summaries.
- `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `TMPDIR=/data/ycfeng/tmp/g2h_accept/tmp`, `ulimit -v 31457280`.
- Readers run on the B17 working tree over HEAD `1fcc6880`. Scripts in `/data/ycfeng/tmp/g2h_accept/`: `runner_scopes_02.py`, `sampling_04.py`, `rows_gaps_04.py`, `rows_compare2_04v_g1h.py`, `rows_compare2_04v02.py`, `gaps_compare_04.py`, `runner_values_04v02.py`.

`RAW=/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_04/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw`

## Checks

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Bridge verdict (`bridge_report.json`, `logs/bridge_report.log`) | PASS, 3 runs | `PASS: 3 runs, FULL=342, PIECEWISE=645`; exit codes 0 / 0 / 0 | PASS |
| 2 | Engine logs `run_e{0,1,2}.log` | No error apart from the known tokenizer warm-up warning | 3 matching lines per engine, all inside the `base.py:259` tokenizer warning | PASS |
| 3 | Step scopes (`runner_scopes_02.py $RAW`, log `scopes_04.log`) | As in G2h-02 | e0: 686 steps (673 real, 13 dummy), sampling 673, logits 673, preparation 686. e1: 296 / 296 / 296 / 296. e2: 7 / 7 / 7 / 7. All five checks true in each engine; 0 logits intervals outside the sampling span; `FAILURES []` | PASS |
| 4 | Sampling contract (`sampling_04.py`, log `sampling_04.log`) | `_bias_kernel` and the FP32 logits copy in every real step's sampling scope, none on dummy steps | `_bias_kernel` and `direct_copy_kernel_cuda` in 673 / 673, 296 / 296 and 7 / 7 real steps; 0 dummy steps with sampling; 11 kernels in every sampling scope (one of two gather kernels per step) | PASS |
| 5 | Runner rows (`dsv4_raw_e{0,1,2}.runner.csv`) | Preparation 989, write 976, zeroing 499, sampling 976, logits 976 (4,416) | 3,044 + 1,340 + 32 = 4,416 rows with exactly these counts; the same 4,416 (operator, case) keys as G2h-02 | PASS |
| 6 | Unowned activity (`dsv4_raw_eN.ownership.json`, through check 3) | Only the output copies, the dummy-batch setup and the prefill request-state setup | FULL 672, PIECEWISE 672, first tail 592, multi-tail 14: only `Memcpy DtoH (Device -> Pinned)`, as in G2h-02. Dummy 273, equal to G2h-02. Prefill 10 against 8: one more `_apply_write_kernel` and its `Memcpy HtoD` upload (4 and 4 against 3 and 3). No sampler or LM-head name is unowned | PASS |
| 7 | Four gaps (`rows_gaps_04.py`, output `rows_04/gaps.json`; `gaps_compare_04.py`) | Counts equal to G2h-02 except for the sampling kernels the contract adds; values compared, not gated | `cuda_graph` 169,232, `cuda_graph_piecewise` 289,730 and `eager` 30,243 samples, equal. `eager_preparation` 93,120 against 91,170: +672 in FULL and +1,278 in PIECEWISE, which is 2 x 336 and 2 x 639 real graph steps, one gap for each of the two added sampling kernels. Values: `cuda_graph` 0.4933 µs (+2.97 % against G2h-02), `cuda_graph_piecewise` 0.6213 (+1.97 %), `eager` 0.5069 (−5.61 %), `eager_preparation` 1.2063 (−7.09 %) | PASS |
| 8 | Attention rows (`rows_compare2_04v_g1h.py` against G1h-02 `/data/ycfeng/tmp/b13r_accept/after/rows.json`; `rows_compare2_04v02.py` against G2h-02 `rows_02/rows.json`) | 4,945 identities equal, no count or kernel-count mismatch, medians close | 4,945 of 4,945 identities equal to both; `COUNT_MISMATCH {}` in both. Per-operator median ratio 0.990-1.003 against G1h-02 and 0.989-1.022 against G2h-02; p05-p95 within 0.80-1.22 | PASS |
| 9 | Runner values against G2h-02 (`runner_values_04v02.py`; cost = kernel median + memcpy median, joined by operator and case) | Preparation unchanged (the staged sampling writes run in `add_requests`, outside preparation); sampling higher (the FP32 copy and the bias kernel) | Preparation median 0.09367 against 0.09207 ms, per-case ratio median 1.017 (p05 1.003, p95 1.032). Sampling median 0.05057 against 0.03024 ms, ratio median 1.633 (p05 1.121, p95 2.360). Zeroing 1.002; logits 0.968; writes (validation only) 1.187 | PASS |

## Interpretation

- Observation: with the native contract, every real step runs the logits-processing branch, and the sampling scope costs about 1.6 times the G2h-02 scope. Inference: G2h-02 sampling rows measure the `needs_logits_processing = False` branch and G2h-04 rows measure the `True` branch, on the same coordinates (b = 1-24 real requests), so B17b can key sampling by the branch and b without any scaling.
- Observation: preparation and the 4,945 attention rows do not move beyond run-to-run spread, so the request contract affects only the sampling scope among the measured scopes.
- Inference: the extra prefill staged write (check 6) is the logit-bias request state that `min_tokens` adds in `add_requests`. It belongs to the known unpriced `add_requests` copies (about 15-25 µs on steps with new requests).
- The committed `kernel_gap.csv` stays the G1h-02 reduction (B16 decision). The G2h-04 gap values differ from it by run-to-run amounts; the extra `eager_preparation` samples come only from the two added sampling kernels.

## Limits

- One GPU with TP1, as in G2h-02; sampling runs on the gathered full-vocabulary logits, which is the TP8 path's input too (inference from the vLLM V1 code path).
- The GPU-hour figure is an upper bound from the status polls.
- The replay table is not promoted yet; B17b defines the sampling coordinate first.
