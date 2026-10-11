# E2E accuracy gate: Frontier against vLLM

This harness gates Frontier's request-level metrics against vLLM ground truth on the 22 claimed
cells of the H200 scenario matrix (`cases/scenario_matrix.json`). The matrix also lists the one
cell it excludes, with the reason. `run_gate.py` is the single entry point.

```bash
export PYTHONPATH=$PWD
python tests/comparison/e2e_gate/run_gate.py cells
python tests/comparison/e2e_gate/run_gate.py gate \
    --cell e2e_c1_dense_coloc/qps2 --rows tests/comparison/e2e_gate/fixtures/c1_qps2/rows
pytest tests/unit/test_e2e_gate.py -q
```

The tests replay each fixture's rows and compare the reading with the cell's entry in
`cases/published_readings.json`.

## Evidence tiers

Each tier needs more than the one before it. A published number is reproducible to the depth of
the tier you can run.

| Tier | Subcommand | Needs | Reproduces |
| --- | --- | --- | --- |
| Gate replay | `gate` | this checkout; normalized rows (the `fixtures/`, or `normalize` output) | the cell's metric decisions and verdict, its entry in `cases/published_readings.json` |
| Row normalization | `normalize` | raw vLLM run directories and Frontier run directories | the normalized rows and their identity gaps (`sample.json`) |
| CPU artifact replay | `cpu-table` | the pinned CPU-probe logs of the calibration bundle | a case's published CPU-overhead CSVs, byte for byte (SHA-256) |
| Trained Frontier replay | `frontier` | the pinned profiling tables, CPU-overhead CSVs, traces and engine files of the bundle | a published Frontier run: same command, inputs and route-delay draw; predictors train into `--cache-dir` when it is empty |
| Native reference collection | not automated here | an H200 node, vLLM-BS at the pinned commit, the run's image | new vLLM ground truth |

Pinned files live under `--artifacts-root`, the calibration directory of the task that published
the readings. Every path in `cases/scenario_matrix.json` and `artifacts.json` is relative to that
root and carries the file's SHA-256. Each subcommand checks the hash before it uses the file. The
bundle is not part of this repository; a path without its hash is not evidence.

Native collection reuses the existing replay harnesses:

- co-location cells: `tests/comparison/dp_placement_pp/run_vllm_worker.sh` and `vllm_replay.py`;
- PDD cells: `tests/comparison/calibration/run_pd_worker.sh`, `pd_replay.py`, `pd_proxy.py` and
  `pd_metrics.py`.

Each ground-truth run in `artifacts.json` records the following:

- the vLLM-BS commit and harness commit;
- the SHA-256 of each harness script;
- the image, node, resources and software versions;
- the input hashes;
- its row files.

## Files

| Path | Content |
| --- | --- |
| `run_gate.py` | entry point; checks pinned hashes, expands recorded commands |
| `gate.py` | evidence admission, metric decisions, gate kinds |
| `normalize.py` | normalized rows from raw runs (reuses `calibration/e2e_metrics_gap.py` and `calibration/pd_metrics.py`) |
| `dp_pp_modes.py` | DP x PP mode labels of each request (decisions T43-MODEGATE-RULE, T43-PAIRGATE) |
| `admission.py` | admission of a CPU-probe run into a CPU-overhead table |
| `cases/scenario_matrix.json` | per case: architecture, model, devices, TP/PP/DP/EP, scheduler limits, graph mode, routing status, KV transfer, measurement families. Per cell: trace records, request counts, formal window, ground-truth runs, recorded Frontier commands, coverage class |
| `artifacts.json` | Frontier and vLLM-BS identities, instrumentation diffs and overlay patches, ground-truth runs, profiling tables (current and k1d/k1e), CPU-overhead tables with their producer recipes, predictor-cache identities |
| `cases/published_readings.json` | the published reading of every claimed cell: verdict, each gated metric or mode pool with its relative error and status (a pool also with its request count), routing, coverage, and for DP x PP cells each run's mode windows, the mode outcome, reference modes and whole-cell errors |
| `fixtures/c1_qps2`, `fixtures/c3_qps2` | normalized rows of a co-location and a PDD cell |
| `fixtures/c6_qps0.75_drain` | normalized rows of a DP x PP cell: three vLLM runs, five ensemble members, a gated in_phase pool and two reference modes without pairs |
| `fixtures/cpu_probe_c1` | a 100-step slice of a C1 CPU-probe run's engine and stage logs and the producer's CSVs for it |

## Metric contract

A cell's rows hold, for each side, the formal requests in the following columns:

- `arrival_s` and `completion_s`, in seconds on that side's own clock;
- `ttft_ms`, `tpot_ms` and `request_e2e_time_ms`;
- prompt and output token counts.

TTFT is request arrival to prefill completion. That is the canonical TTFT of the repository's
`AGENTS.md`.

The five metrics are:

| Metric | Definition |
| --- | --- |
| `ttft_ms` | mean TTFT of the formal requests |
| `tpot_ms` | mean TPOT of the formal requests with more than one output token |
| `request_e2e_time_ms` | mean request E2E time |
| `request_throughput_rps` | formal requests / formal window |
| `token_throughput_tps` | (prompt + output tokens) / formal window |

The formal window runs from the first formal arrival to the last formal completion. Each metric is
decided on its own: it passes when `abs(frontier - vllm) / abs(vllm) <= 0.10`. When the vLLM value
is zero, the relative error is undefined and the metric is `INSUFFICIENT_EVIDENCE`. A cell passes
only when every gated metric passes.

Evidence admission comes before any number is gated:

- **Request identity.** Each side holds every formal request exactly once and no other request.
  `normalize` records missing, duplicate and unclassified ids in `sample.json`. It writes one
  row per request, so the gate re-checks the missing and unclassified ids against the row index
  and takes duplicates from the record.
- **Token counts.** Every Frontier row carries the vLLM row's prompt and output token counts.
- **Clocks and units.** `request_e2e_time_ms` must equal `1000 * (completion_s - arrival_s)` to
  within 1e-3 ms, and `0 <= ttft_ms <= request_e2e_time_ms` must hold.
- **Routing.** The case's MoE routing-alignment status comes from the matrix:
  - `MATCH` or `NOT_APPLICABLE`: the numbers are gated;
  - `UNSET`: the cell is `INSUFFICIENT_EVIDENCE`;
  - `MISMATCH`: the cell is `FAIL`.

If the identity, token or clock check fails, every metric is `INSUFFICIENT_EVIDENCE`.

## Gate kinds

- **`single`** (co-location): one vLLM run against one Frontier run.
- **`pd_s33`** (PDD; decision S33-TTFT (ii)): like `single`, except the vLLM TTFT is reduced by
  the minimum endpoint offset of the formal requests. The offset is the proxy hop to the prefill
  instance plus the prefill API-server time (prefill TTFT less its model execution time).
- **`dp_pp_pairs`** (DP x PP; decision T43-PAIRGATE):
  1. The two DP lanes move between four modes: `in_phase`, `anti_phase`, `split` and
     `double_split`. Each mode has its own per-token period (`dp_pp_modes.py`).
  2. The formal window is cut into windows of `mode_window_s`. Each request takes a TTFT mode
     and a TPOT mode from those windows.
  3. Every ground-truth run is paired with every Frontier ensemble member (route-delay seeds
     1-5).
  4. A request's TTFT enters the pool of mode M when its TTFT mode is M on both sides; the same
     rule applies to TPOT, for requests with more than one output token.
  5. A pool is gated once at least 20 distinct requests reach it.
  6. Reporting:
     - Modes that hold at least 20% of the vLLM windows are the reference modes. More than one
       makes a multi-mode reference.
     - If no pool is gated (`mode_outcome: empty_pair_set`), the cell is
       `INSUFFICIENT_EVIDENCE`. No single numerical comparison exists, and no favorable run may
       stand in for one.
     - Whole-cell means (over the vLLM runs and over the members) are reported, not gated.

## CPU-overhead publication recipe

A case's CPU-overhead tables (`cpu_overheads.csv`, `cpu_overheads_kernel_only.csv`) come from
the repository's producer, `python -m frontier.profiling.cpu_overhead.vllm_cpu_probe`.
`artifacts.json` records, for the current X2 table and for the earlier k1d/k1e tables:

- the producer arguments, including `--engine_idle_edges_ms 0 100 500` and the pipeline-stage
  count;
- the SHA-256 of every input engine and stage log, plus the `dp_placement` logs of DP>1 runs;
- the SHA-256 of the output CSVs.

`run_gate.py cpu-table --case <case> --table current|k1d|k1e` reruns the producer and compares
the output hashes. There is no second publisher.

The k1d and k1e probe logs come from vLLM-BS commits older than eed75f525. They lack the fields
the current producer needs to tell prefill tokens from decode tokens, and the producer rejects
them. To rebuild those tables, check out the producer commit recorded in `artifacts.json` and
pass it with `--producer-tree`; the module's SHA-256 is checked first. For example:

```bash
git worktree add ../frontier-k1e <producer commit>
python tests/comparison/e2e_gate/run_gate.py cpu-table --case e2e_c2p_moe_coloc_ep2_tp1_dp2_pp2 \
    --table k1e --producer-tree ../frontier-k1e --artifacts-root <bundle> --output-dir <new dir>
```

A CPU-probe run enters a table only when it passes both of the following.

1. **The ground-truth run check.** The engine probe log must:
   - hold every engine step, with stamps in execution order;
   - hold every trace request;
   - have stage records that cover every forward.

   Each run's result is recorded in the table's provenance.
2. **The admission rule of its cell kind** (`admission.py`, `run_gate.py admit`), with the ratio
   inside 0.97-1.03:

   | Rule | Cells | Probe measurement | Reference |
   | --- | --- | --- | --- |
   | `client_e2e` (issues.md I-6) | all cells | mean client E2E | the same cell's clean run in the same job |
   | `send_span` (T43-C4PROBE; together with `client_e2e`) | PDD prefill | median summed per-layer KV-send span of isolated prefills | mean of the job's clean runs of the cell |
   | `isolated_prefill` (T43-ISOPREFILL) | DP x PP cells, whose mode is not reproducible | the first formal request's isolated prefill step | median of the job's CPU-probe cohort |

## Limits

- The calibration bundle (raw runs, tables, traces) is pinned by hash. It is not shipped here.
- MoE routing: every MoE case's reference runs with vLLM-BS round-robin routing
  (`moe_uniform_routing` in the engine file), which matches Frontier's `balanced` split. vLLM's
  native router on the dummy weights does not match it. C4's CPU-overhead table was collected
  with the native router and is kept.
- Coverage: a cell classed `outside_profile` holds formal batches outside the profiled ranges or
  grid spacing (`coverage.gaps` names them). Its numbers are not in-domain evidence.
- Mode boundaries: a DP x PP verdict covers only its gated mode pools.
  `reference_modes_without_gated_pairs` names the reference modes that the reading does not cover.
- Native collection depends on the H200 node's host speed. The admission rules bound that
  dependency, but they do not remove it.
