# D29 real-path timeout feedback-loop report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record measured RED, one-variable diagnostic control, and exact commands. |

Task ID: **D29-CLI-TIMEOUT-REPRO**. Diagnosis only; no implementation or CLI gate
change. Source e41abc073500d61f224eda4fca1123e74e40544d, submodule3dac53c.

## Environment and evidence

- Python `/data/ycfeng/envs/frontier-py310/bin/python`, version3.10.6.
- NumPy2.2.6, pandas2.3.3, scikit-learn1.7.2.
- `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
- All probes/output under `/data/ycfeng/tmp/d29_cli_timeout_repro/`.
- Systemd scopes unavailable; outer timeout60s plus30GiB address-space limit.
- Retained bundle:
  `fixture03/cache/bundle_3e46232621a39c1c5f598921435bd41da1a4c1915b424bc53f3eba09a5bb3e37.json`.
- Selected FP8 model82e252d5,16,464 retained physical rows, pickle SHA256
  `725a481f616e307b28753fb05563f7460cfd42c370e2948d2412bdcc2f82e925`;
  existing loader verifies duration and paired count hashes/model metadata.
- Actual source: query6128/context0, physical6128; idle peer has empty queries,
  physical1; gathered6129. Layer3, EP0. No synthetic replacement of actual point.

## Exact executed RED command

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_cli_timeout_repro/runtime_loop.py --output /data/ycfeng/tmp/d29_cli_timeout_repro/execution-final01 --iterations 2 --execution --no-profile > /data/ycfeng/tmp/d29_cli_timeout_repro/execution-final01.log 2>&1
```

Result: exit1, **RED**, two pricing calls3.453060280997306s and
3.376363863237202s;1056 layer descriptions. RED condition is more than one full
88-layer description pass across repeated identical point calls, not a relaxed
2400-second criterion. Result resides in `execution-final01/result.json`.

For replay, select a new scratch output/log name because the script deliberately
rejects an existing output directory. Do not overwrite these retained results.

## Exact executed one-variable diagnostic control

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_cli_timeout_repro/runtime_loop.py --output /data/ycfeng/tmp/d29_cli_timeout_repro/execution-groups-final01 --iterations 2 --execution --reuse-groups --no-profile > /data/ycfeng/tmp/d29_cli_timeout_repro/execution-groups-final01.log 2>&1
```

Result: exit0, **GREEN for the diagnostic call-count condition only**,
0.4041568008251488s and0.3511730800382793s;zero repeated layer descriptions.
The sole variable is in-process reuse of the tuple returned by the actual
resolver for the exact same model object. No production file was changed.

Both payloads (MoE/communication costs, operator proofs, full plan keys) hash to
`d2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952`.
The paired model, source descriptor, graph/context/layout and pricing are not
relabeled, reduced or skipped. Timing excludes verified bundle setup.

## Profile and initialization observations

| Probe | Actual result | Verdict/limit |
| --- | --- | --- |
| `execution01/profile.txt` | Two production calls15.239s cumulative; resolver14.438s,12 calls; layer description14.414s,1056 calls | Dominant sampled runtime hotspot; profiler adds overhead |
| `execution-groups01/profile.txt` | Two calls0.781s cumulative; real backend0.542s,8 calls; JSON decoding0.588s | Secondary residual hotspot, not separately fixed |
| Query-only baseline/control | Approximately1.10–1.12s versus0.085–0.102s without profiler | Same point/context/features/plan-key hashes |
| `loader01/result.json`, `profile.txt` | Original2,114,905,379-byte CSV,15.053s bounded stop;9.302s parse,5.696s row validation,3021 config validations | Full original input supplied; no full admission result |
| `typed01/result.json` | Actual first row validated8times,5.764ms/row profiled | Microprobe only;213.52s full-map estimate is not observed |
| `init01/result.json` | Actual Simulator constructor sampled5.040s; requests/gap/linear reads observed; no simulation run | Bounded initialization, not main07 retry |

## Actual symptom correspondence and limits

Parent receipt reports main07 offline2400.103553s and online2400.096253s,
both timeout, exactly seven PREFILL-only scheduler events, last simulation times
0.0/0.25, and no final request/system/stage/lane/transfer output. The loop uses
the P6128 point that those scheduler records actually reached. It does not claim
that main07 completed a layer/EP wave, reached decode, or matches old D23 state.

The probe invokes actual inherited production methods through the existing
`tests.unit.test_step5_measured_ffn_replay._predictor` construction seam. It then
overrides model/replica/measurement/role with actual parsed CLI facts and uses a
real Cluster/backend. This avoids expensive full Simulator setup; it is not a
complete constructor/workflow test. One lane/one layer does not measure total
runtime across the entire6144/1024 workload. No assertion here passes the original
2400s/full-output/native/accuracy gates.

Parent's unchanged nine-input/e41/3dac evidence is referenced, not repeated:
`/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_parent/main07_timeout_receipt.json`.

## Proposed next allocation, not implemented

Highest-value seam: existing CORE predictor's fixed-model routed-group
selection at `sklearn_moe_execution_time_predictor.py:121–127` and
`moe_routing_workload.py:133–145`, both currently invoke the same
`routed_layer_groups.py:115–121` -> `:59–107` all-layer resolver repeatedly.
The feature path at `moe_routing_workload.py:703–707` also reaches the selection.

Recommendation for parent allocation: reuse validated model-scoped routed-group
facts through these existing CORE seams, retaining actual model/binding identity
and all per-point source/context/cost checks. A generic global cache or new
provenance/registry API is not implied. Initial estimate:25–45 source gross lines
plus35–60 regression gross lines in an existing owned consumer test, subject to
root's effective path ceilings and current ownership. No budget was spent on
source/tests in this diagnosis.

Secondary proposed PRICING seam: repeated artifact read/JSON decode at
`collective_sim_cc_backend.py:810–811`; estimate10–25 source plus20–35 regression
gross lines if separately allocated. Typed initialization at
`profiling_dataframe_loaders.py:723` -> `typed_contracts.py:96/405` needs separate
allocation and full-field validation semantics; this report authorizes no skip
or cache change. Do not combine those repairs without parent direction.
