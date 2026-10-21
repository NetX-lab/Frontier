# CPU timeout diagnosis verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded actual-cache RED loop and discriminating CPU, collective and trace measurements. |

## Environment and authority

- Task `D23-ACTUAL-MAIN/PERF`; session `ses_e9f519775ffe3Cv0bhEzJSKNhH`.
- Worktree `/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf`, branch `feat/step5-mrv2-perf`, base `ec5de154c19ceeae23a3652f511582858170f813`.
- Python `/data/ycfeng/envs/frontier-py310/bin/python`, version 3.10.6, Linux 5.10.25-nvidia-gpu-x86_64/glibc 2.35.
- No production changes, fitting, fixture/cache mutation, full CLI retry, GPU/platform, external action, further delegation, or timeout increase.
- Immutable fixture `/data/ycfeng/tmp/d23_main_fixture02`; actual05 offline typed bundle selected by `/data/ycfeng/tmp/d23_main_actual05_offline/synthetic_read_cost.json` input binding and `max_tokens_in_batch=8192`.
- Existing SHA256-verified source estimator coverage retained: 94392 CUDA_EVENT attention keys and 184 KERNEL_ONLY keys, duration/count artifact pairs. Evidence remains `cpu_fixture`, never native admission.

## Reusable fast RED command

Run from the isolated worktree; use a new output directory on each invocation.

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --iterations 2 --no-profile --output /data/ycfeng/tmp/d23_perf_red_final01
```

- Expected current verdict: exit 1, RED; independently `source_truth=PASS`.
- Actual: loading 1.478718s, measured 0.527437s, 64 lane queries, source truth PASS, RED.
- Exact repeat changing output to `d23_perf_red_final02`: loading 1.514730s, measured 0.542120s, source truth PASS, RED.
- Final current-script repeat `d23_perf_red_final03`: loading1.483458s, measured0.543830s, RED/source PASS. Python stdlib `subprocess`/`perf_counter`/`resource.getrusage(RUSAGE_CHILDREN)` measured complete child command3.798875s and815716KiB maximum RSS. Receipt `/data/ycfeng/tmp/d23_perf_final_elapsed.json`. `/usr/bin/time` was unavailable; no dependency installation attempted.
- Each query preserves TP8/DP2/MoE-TP1/EP16, 352 experts, balanced seed42 routing, aggregate physical input rows2, scalar/routed feature domains, FP8 routed layer3 and BF16 tail layer88.
- Source truth asserts actual duration and count from the stored exact keys, count7, physical layer identity, precision, evidence kind, artifact SHA256 and paired measurement identity. It does not fabricate final/native acceptance.
- Gate: `(2400 - 1004) / 1024 / 16 / 88 = 0.0009682395s` per FFN lane/layer. Giving FFN the entire remaining budget is deliberately conservative. No budget increase.
- Individual non-profiled queries: min 8.03037ms, mean 8.22655ms, max 11.16541ms. Mean metadata 5.83032ms, mean operator queries 2.38895ms.

## Measured call profile

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --output /data/ycfeng/tmp/d23_perf_ffn06
```

RED / source truth PASS. 32 lane queries measured 0.616861s under cProfile. Profile totals are not wall extrapolations.

| Call | Calls | Cumulative seconds | Observation |
| --- | ---: | ---: | --- |
| `describe_step5_layer` | 32 | 0.452344 | 73.3% of measured profile work; same two layers reconstructed16 times each |
| `resolve_module_precision` | 192 | 0.218655 | Revalidates1406 ignored prefixes and repeats prefix-renaming sets |
| `validate_step5_config` | 97 | 0.164470 | Repeated full layer geometry validation |
| `build_operator_manifest` | 32 | 0.132 | Repeated operator/structural requirement construction |
| `resolve_layer_contract` | 228 | 0.104511 | Full MoE layer classification repeated per operator |
| `_get_on_demand_prediction` | 456 | 0.030027 | Actual duration/count queries; not dominant |
| `_serialize_selected_layer_cache_identity` | 164 | 0.005078 | Not a dominant hash/serialization boundary |
| `_get_family_model` | 228 | 0.003172 | Registry lookup not dominant |

Artifacts: `/data/ycfeng/tmp/d23_perf_ffn06/{prediction.prof,profile.txt,receipt.json}`. Profiled cumulative rows overlap; do not add them.

## Hypotheses and discrimination

1. **Repeated fixed configuration work per EP lane/operator.** Prediction: high full-config/precision/contract calls despite exact numeric cache hits. Supported by32 metadata reconstructions/192 precision calls/228 contract calls, and independently measured non-profiled metadata5.83ms/query.
2. **94392-key coverage/hash scan per lookup.** Prediction: lookup/fingerprint dominates or cache-size work repeats in each measured query. Not supported at this seam: exact lookup and registry together are small, no per-query SHA256/fitting; SHA256 validation is setup-only.
3. **Collective subprocess dominates every round.** Prediction: repeated identical real collective queries launch subprocesses or remain slow. Warm-hit alternative ruled out: identical predictions return in0.110–0.172ms. Cold subprocess setup remains a separate real cost, not yet an allocation of the original998/1004s setup.
4. **Proof/metrics serialization dominates CPU.** Prediction: real metadata/persistence boundary costs more than model queries. Not supported for the measured slice: trace metadata and enabled real JSONL flush took40.5ms/228 events, versus266.4ms FFN queries. Full MetricsStore orchestration not measured.

## Real collective replay

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_collective.py --output /data/ycfeng/tmp/d23_perf_cc03
```

Uses actual05 configuration materialization and existing built collective-sim at `/data/ycfeng/Frontier/.worktrees/step5-mrv2-main/frontier/cc_backend/backends/collective-sim`. Only the location resolver is overridden; prediction work/cost is real. New runner outputs are isolated under the new temporary directory. No clone or build.

| Original shape | Cold seconds | Exact hit seconds | Predicted ms both times |
| --- | ---: | ---: | ---: |
| ATTN_TP8 allreduce,8192 bytes | 1.885434 | 0.00010995 | 0.122465 |
| EP16 all-to-all,16384 bytes | 2.545289 | 0.00017243 | 0.0199104 |

Four requests, two misses, two hits. cProfile:4.425s in subprocess polling out of4.431s total. PASS: exact output repeats, no per-hit subprocess. Native admission false. Artifacts `/data/ycfeng/tmp/d23_perf_cc03/{receipt.json,collective.prof,profile.txt,runner/decode/run.out}`. Earlier exploratory `d23_perf_cc02` measured EP allreduce, not actual EP all-to-all; do not use it as the main EP receipt.

## Enabled trace path

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --trace --no-profile --output /data/ycfeng/tmp/d23_perf_trace_wall02
```

FFN RED/source PASS,0.266396s. Actual source-bound metadata and enabled `TraceStore` JSONL persistence:0.040502s,228 events,172466 bytes. Persisted duration/name/layer tuples exactly match the returned query costs. PASS. Simple row-proof JSON serialization took approximately1ms; it is not a substitute for the measured trace path.

Profile variant omits `--no-profile`, output `/data/ycfeng/tmp/d23_perf_trace_profile02`: trace0.113s profile work, flush0.077s, `TraceEvent.to_dict/asdict`0.072s, only2 metadata descriptions for32 lanes because MetricsStore already caches metadata. No metrics were disabled in this replay.

## Observations, inference and limits

- **Observed cause at expensive production seams:** repeated model-fixed metadata/precision/layer-contract construction occurs before exact scalar cost cache hits and repeats for each EP lane. This is enough to explain a large sustained CPU cost.
- **Inference from actual counts:** measured mean FFN query ×16×88 predicts11.583s per complete routed-layer round, of which8.209s is metadata and3.364s operator queries. These are extrapolations, not an observed full decode round or original workload profile.
- **Correction:** standalone unprofiled attention is GREEN at6.59–7.36ms/query under its conservative7.409ms budget. Earlier attention RED was cProfile overhead, not a standalone wall-time failure.
- Initial998/1004s setup is not fully attributed; no new fitting was performed. Original timeout remains2400s. Cold collective misses are real, but original miss census belongs to the parent.
- This replay does not include public stage assembly, scheduler state transitions, full EP MetricsStore scheduling, all layers, prefill, dummy descriptor execution, or final online/offline acceptance. Those remain parent-owned verification after an approved source change.
- Final syntax check: `timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B -c 'import ast,pathlib;files=[pathlib.Path("tests/performance/step5_mrv2_cpu.py"),pathlib.Path("tests/performance/step5_mrv2_collective.py")];[(ast.parse(p.read_text()),print(str(p)+": syntax PASS")) for p in files]'` → both PASS. No unrelated test suite run is needed for diagnosis-only scripts.
