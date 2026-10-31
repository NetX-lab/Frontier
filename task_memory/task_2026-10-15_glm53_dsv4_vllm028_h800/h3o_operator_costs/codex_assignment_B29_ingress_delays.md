# Task: model the serving path outside the engines with two delay fields (stage 4 assignment B29, ingress option A, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Created from decision 46 (requirements.md) and A10 report sections 3.6 and 5.4. |

## Goal

In a native vLLM PD deployment, a request passes through processes outside the engines before an engine queue sees it:

- **Ingress:** the client sends the request to the proxy, the proxy forwards it to the prefill API server, and the API server enqueues it for the prefill engine.
- **Handoff:** after the prefill engine produces its output, the prefill API server returns the response to the proxy, and the proxy sends the request to the decode API server, which enqueues it for the decode engine.

Frontier models neither path. A request enters the first role's waiting queue at its client arrival time, and the KV transfer starts at the prefill batch end. On the H200 A10 replays these paths are about 22 ms per request of the TTFT gap (A10 report section 5.4).

YC approved ingress option A, including its configuration schema change (decision 46). Add two delay fields, default 0:

1. **`request_ingress_delay_ms`:** time from the client arrival to the engine waiting queue of the first role. It delays the request arrival event. `Request.arrived_at` stays the client arrival, so TTFT and E2E still start there.
2. **`request_handoff_delay_ms`:** time from the prefill engine output to the decode engine's enqueue. It delays the KV transfer start event.

With both fields at 0, every result must stay byte-identical.

## Decisions (lead, final)

- **D1, field homes.**
  - `request_ingress_delay_ms: float = 0.0` is a top-level field of `SimulationConfig` (`frontier/config/config.py`), beside `simulation_mode`. Flat flag `--request_ingress_delay_ms`. It is a property of the serving front end, not of a role or of the workload.
  - `request_handoff_delay_ms: float = 0.0` is a field of `BaseKVCacheTransferConfig` (`frontier/config/kv_cache_transfer_config.py`). Flat flag `--analytical_kv_cache_transfer_config_request_handoff_delay_ms`. Every connector (`none`, `p2p_nccl`, `nixl`) gets it, because the handoff path does not depend on how the KV moves.
  - Help texts name the path each field covers, in the words of the Goal section. Units are milliseconds; Frontier's event times are seconds, so convert once where the delay is applied.
- **D2, validation.** Both fields must be finite and at least 0, checked with the existing `_require_non_negative` helper or its equivalent at the config's own validation point. A nonzero `request_ingress_delay_ms` in offline batch mode (the paths of `frontier/simulator.py` that add every request at time 0 without an arrival event, near lines 1288-1330 and 370-390) fails at configuration time with a message that says the field applies to request arrival events. Do not apply it silently there and do not ignore it.
- **D3, ingress semantics.** The arrival event time is `request.arrived_at + request_ingress_delay_ms / 1000`. The event handler keeps calling `on_arrival(self.time, ...)`, so the queue entry time, the scheduling delay and the `request_arrival` ground-truth record's `time_s` are the queue-visible time, and its `arrived_at` field stays the client arrival. Apply the delay in one place used by both arrival paths (`simulator.py` lines 404-421, parallel online path, and lines 1332-1343, sequential path); do not copy the arithmetic.
- **D4, handoff semantics.** The KV transfer start event time is the prefill batch end plus `request_handoff_delay_ms / 1000` (`frontier/events/cluster_batch_end_event.py`, the single `KVCacheTransferStartEvent` construction near line 602). The prefill keeps the request's KV blocks during the delay; the decode target is reserved when the start event runs, as today. Check that nothing else between the prefill batch end and the transfer start reads the start time (the transfer ledger, `Request.on_kv_cache_transfer_start`, the PD-AF lifecycle). State in the report what each consumer now records.
- **D5, identity at 0.** With both fields at 0, the event times are the same floats as today (`x + 0.0 == x`), the events are created in the same order, and every output is byte-identical. Do not add a branch on `delay == 0`.
- **D6, no other changes.** No egress delay, no per-role ingress, no change to the KV transfer time model, no change to `nixl_read_completion`. Those are outside option A.
- **D7, driver values.** The H200 values are inputs of the A10 driver, not code defaults. The driver sets them by default, like `DP_MAX_FORWARDS_IN_FLIGHT`, from named constants with a comment that cites the evidence file and line for each term:
  - `request_ingress_delay_ms` = client dispatch to proxy arrival (P2-64r mean, `h3o_operator_costs/a10/remainder/reference_split_stdout.txt`, the line "client dispatch -> proxy arrival n 379 mean 0.6") + proxy arrival to API-server arrival (3.11) + API-server arrival to `add_enqueue` (6.59) (both from `h3o_operator_costs/a10/k19/ingress_split_wave-20261030-01.txt`) = 10.3 ms. The `add_enqueue` to engine-add span (1.63 ms) is the engine-loop wait, which the scheduler models; it is not part of the delay.
  - `request_handoff_delay_ms` = prefill first-token update to prefill response (2.3 ms, line `p_update->prefill_response` of `h3o_operator_costs/a10/remainder/remainder_stdout.txt`) + prefill response to the decode `add_enqueue` (9.1 ms, line `prefill_response->d_add_enqueue` of `h3o_operator_costs/a10/remainder/pairs_stdout.txt`) = 11.4 ms. Both are wave-run means (n 128). The decode `add_enqueue` to decode-add span (10.4 ms) is the decode engine-loop wait, which `nixl_read_completion=engine_step` already contains (A10 report section 3.6, observation 5).
  - Read each term from its file in a small check (`b29/ingress_values.py`, output `b29/ingress_values.txt`) and assert the sums. If a printed value has more digits than the A10 report quotes, use the file's value and report the difference. If a term is missing from its file, stop and report.
  - P2-32 uses the same values: no P2-32 run recorded `add_enqueue`, and the engine build, proxy and API server are the same (A10 report section 5.4). The constant's comment says so.

## Work

### 1. Configuration

Add the two fields (D1, D2). Update `docs/cli/README.md` with one short subsection: what each field covers, that both default to 0, that `arrived_at` stays the client arrival, and that the values are per deployment and must come from measurements of the serving path.

### 2. Simulation

Apply the delays (D3, D4). Keep the change in the existing modules; no new module.

### 3. A10 driver

In `h3o_operator_costs/a10/a10_driver.py`:

- Set both fields by default from named constants (D7). Add `--without-ingress`, which sets both to 0 and so reproduces the B28 configuration.
- Add `--field NAME=VALUE`, repeatable, which sets any flat field for diagnostic variants. Parse it with the existing `h4a4.parse_field` (same JSON-or-string rule as the H4a-4 driver). A field given twice, or a field that the driver also sets through another option, fails with a clear message. The receipt already records the override diff; check that `--field` values appear there.
- `resolved_inputs` reads `cc.summary_csv_path` for every role. Read it only when the role's CC backend config has that field, so a variant with another CC backend type can run. Record the selected CC backend type of each role in the receipt.

### 4. Tests

New or changed tests under `tests/unit/` (name them in the report):

- Defaults are 0; negative, NaN and infinite values are rejected for both fields; a nonzero ingress delay in offline batch mode is rejected (D2).
- Flat-flag round trip: both flags parse and reconstruct into the nested config.
- Ingress: for one request in an online run, the arrival event time is `arrived_at + d`, `arrived_at` is unchanged, the queue entry time is `arrived_at + d`, and `Request.ttft` grows by exactly `d` against `d = 0`.
- Handoff: for one request in a PDD online run with dummy predictors and `kv_connector=none`, the KV transfer start, the transfer end and the decode first token each move by exactly `d + h` against `(0, 0)`, and the prefill end moves by exactly `d`. Pick inputs where the shift is exact; explain in the report why it is.
- Identity: the same small PDD run, and one co-location online run, with both fields explicitly 0 produce byte-identical `request_metrics.csv` and `system_metrics.json` to the run that leaves them unset.
- A10 driver: `--without-ingress` gives 0 for both fields; the default gives the constants; `--field` duplicates and conflicts fail; `resolved_inputs` works for a role whose CC config has no `summary_csv_path` (use the built config of `p2_32` with `--field cc_backend_config_type=astra_sim_analytical`; do not run the simulator).
- No skip or `xfail`, and no weakened assertion. List each changed existing test line with its reason.

### 5. Short-trace replay check

Run the A10 driver on a short trace for both cases, warm caches, without the observer:

- Trace: the first 8 requests of the A10 trace. Use the existing `--trace` option and write the trace into your scratch. Name the source trace and the selection rule in the report.
- Caches: `cp -a` copies of `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_32_cold_g1hs_r2/cache` and `.../p2_64_cold_g1hs_r2/cache` into your scratch run roots (read-only sources; 1,758 files each). Do not write under `stage4/`.
- Runs per case: `--without-ingress`, and the default (ingress values set). Four runs; memory `ulimit -v 62914560`; at most two at a time.
- Report per request: TTFT, the arrival event time minus `arrived_at`, the KV transfer start minus the prefill end, and the decode first token. Expected: the arrival offset is 10.3 ms and the transfer start offset is 11.4 ms for every request; the TTFT change is near 21.7 ms per request, but it can differ because of queueing and decode step alignment. Explain each request whose change differs by more than one decode step.
- The B28 driver baseline of this trace is the `--without-ingress` run itself; also check that its receipt equals the B28 configuration except for the new fields at 0 and the run paths.

## Acceptance

1. The new and changed tests pass. Name and count them.
2. `b29/ingress_values.txt` shows each term with its source line and the two sums.
3. The short-trace replay check gives the expected offsets for every request.
4. **Regression, candidate against base.**
   - Selection: every `tests/unit` module that contains any of these strings: `kv_cache_transfer`, `KVCacheTransfer`, `RequestArrivalEvent`, `request_arrival`, `arrived_at`, `simulation_mode`, `flat_dataclass`, `SimulationConfig`, `cluster_batch_end`, `a10`, `dsv4`, `pd_disaggregation`, `pdd`. Add the list `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b24/regression_targets.json`. Record the selection in `b29/regression_targets.json`.
   - Run the selection on the base worktree and on the working root with the same command.
   - Compare the outcome of each test node. Every difference must be one of the intended test changes; list them. Any failure must also appear on the base side with the same message.
   - Use a scratch working directory: a `data` symlink to the repository's read-only data, absolute test paths and the repository on `PYTHONPATH`, so the tests' default `outputs` and cache writes stay in scratch.
5. The dummy smoke matrix `examples/architecture/run_dummy_smoke_matrix.sh` passes (12 of 12), run from scratch as in B24.
6. `git diff --stat` shows only paths under "Writable paths". Each Python module stays at or below 2,000 lines, or does not grow if it is already above.

Test command template, from the scratch working directory (`<root>` is the working root or the base worktree):

```bash
ulimit -v 31457280
env PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0 \
  TMPDIR=<scratch>/tmp FRONTIER_TMP_ROOT=<scratch>/tmp \
  PYTHONPATH=<root>:<root>/tests/unit:<root>/tests/comparison/calibration:<root>/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages \
  timeout 3600 /data/ycfeng/envs/frontier-py310/bin/python -m pytest <absolute test paths> -q -p no:cacheprovider \
  --basetemp=<scratch>/tmp/<label> --junitxml=<scratch>/<label>.xml
```

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b29-ingress-delays`. It is a Git worktree on branch `b29-ingress-delays`, created from the main commit that adds this file. Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Base worktree (read-only):** `/data/ycfeng/Frontier/.worktrees/b29-base`, detached at the same commit.
- **Task records (read-only)** are in the main tree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, task directory `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`. The same files are in your working root.
  - `requirements.md`: decision 46 (D-COMM-MEASURED-ALLOWED) and decision 43 (the driver-constant pattern of K).
  - `test_report_2026-10-30_a10_operator_path_replay.md`: section 3.6 (observations 3-5) and section 5.4.
  - `h3o_operator_costs/a10/remainder/` and `h3o_operator_costs/a10/k19/`: the evidence files named in D7.
- **Writable paths, in the working root:**
  - `frontier/config/config.py` and `frontier/config/kv_cache_transfer_config.py`;
  - `frontier/simulator.py` and `frontier/events/cluster_batch_end_event.py`;
  - `frontier/events/request_arrival_event.py`, only if D3 needs it;
  - `tests/unit/`: new test modules, and only the existing tests that the intended change affects, each listed in the report;
  - `docs/cli/README.md`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py`;
  - your records under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b29/`.
  - If you need another path, ask in the report; do not change it yourself.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b29/`.

## Restrictions

- The lead commits. Do not run Git commands that change state. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
- No GPU, network, job submission or package installs.
- No deletions or moves outside your scratch.
- Do not write under `/data/ycfeng/tmp/glm53_dsv4/`, and do not run recursive `grep` or `find` over it. Read the two cache sources only at the paths given above.
- No temporary patches, scaling factors, fallbacks or guards for states the code cannot reach. No literal that should come from configuration or driver input.
- Python 3.10. Memory `ulimit -v 31457280` for tests and `ulimit -v 62914560` for the short-trace replays, and a `timeout` on every command.

## Expected return

- Report language: English. Write the report to `h3o_operator_costs/b29/report.md` and return its content.
- The diff summary, with the reason for each changed function and test line.
- `b29/ingress_values.txt`, and the consumer list of D4 with what each now records.
- The short-trace replay table.
- The regression comparison: commands, counts and node-level differences; the smoke matrix result.
- Deviations and their reasons, and limits.
