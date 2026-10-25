# H4a P0 diagnostic path

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | H4a-1: recorded the cost slots, sources and substitutions, the profiler-perturbation check, the keying and out-of-range rules, the KV binding, the expected outcome, and the H4a-2 hook design with its file list. |

**Authority.** D-H4A-FIRST-DIAGNOSTIC-COMPARISON, D-V4-STEP-COST-FIRST-PASS, D-H4A-CODE-SCOPE and D-INPUT-PERSISTENCE in [requirements.md](../requirements.md#decision-2026-10-25--d-dsv4-h800-mainline-correction-current-route-authority); steps H4a-1 to H4a-5 in [plan.md](../plan.md). The result is DIAGNOSTIC. It counts toward neither the 30% promotion gate nor the 10% final gate.

**Case.** P0 on H800-32: P16 + D16, each role one Replica with TP8, attention DP2, EP16, PP1. Native limits: `max_model_len` 7168, `max_num_batched_tokens` 8192, `max_num_seqs` 128. P graph mode NONE; D graph mode PIECEWISE+FULL. KV cache fp8. Chunked prefill, prefix caching, MTP and thinking are off. Workload 68f891b: per repetition 16 warmups and 128 formal requests, all 6144 prompt / 1024 output tokens, deterministic arrivals at QPS 1, three repetitions.

**Reference.** native06 R4 (`../native06_acceptance_20261023/`):

| Repeat | TTFT (ms) | TPOT (ms) | E2E (ms) | THPT (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| r1 | 403.305 | 19.0311 | 19872.12 | 6273.33 |
| r2 | 393.216 | 19.0165 | 19847.10 | 6273.13 |
| r3 | 399.946 | 18.9807 | 19817.24 | 6269.77 |

native06 is never an input to the prediction. It is used only for comparison and, in H4a-5, for attribution.

## 1. Inputs and persistence

`copy_inputs.py` copies every consumed input out of `/data/ycfeng/tmp` with `shutil.copy2`, so modification times are kept. An existing copy must match its source in SHA-256 and mtime; the script never overwrites. `inputs/manifest.json` records, per file, the source path, the cloud origin, the size and the SHA-256.

| Copy | Files | Source | Cloud original |
| --- | ---: | --- | --- |
| `inputs/native06/` | 10 | `/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023/` run and analysis receipts | none recorded |
| `inputs/profile02/` | 11 | `.../profile02_results_readonly_7ba56n6z/artifacts`: run files, the two decode TP0 traces, download receipt, cloud inventory | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261023/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261023-02` |
| `inputs/profile04/` | 9 | `.../profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04`: run files, download receipt, cloud inventory | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261024_04` |

- The SHA-256 of each copied `cloud_inventory.json` equals the `inventory_sha256` in its download receipt (profile02 `33ed03ea…`, profile04 `85b0ef23…`).
- The two profile02 decode traces (23.4 MB and 13.2 MB) are kept in this task directory and are not committed to git. Their cloud originals are listed above, which is the persistent home that D-INPUT-PERSISTENCE names for large raw traces.
- profile04 is a cross-check only. Its decode and prefill traces are not H4a inputs and were not copied; their originals are on the cloud volume.
- The step-cost table is read through Frontier's default profiling path, so it lives at `data/profiling/compute/h800/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json`.
- **Limitation.** Git does not keep modification times. `native_client_rows` requires every `run/` file to be newer than `run_manifest.json`, so a fresh clone of the committed copies fails that check. The on-disk copies pass it.
- The native06 node logs (about 31 MB) stay under `/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023/`. They are not an H4a input. They are copied only if H4a-5 attribution needs them.

## 2. Source choice and profiler perturbation

D-V4-STEP-COST-FIRST-PASS names the complete profile04 lanes (D0/D1 for decode, P1 for prefill). The plan asks for a check of how much the profiler changed the measured run. `perturbation_check.py` computes each profiled run's four client metrics with the canonical client tools (`canonical_workload`, `native_client_rows`, `formal_sample`, `metric_values`) and compares them with native06 in the same repeat. Output: `perturbation_check.json`.

| Run / repeat | TTFT | TPOT | E2E | THPT |
| --- | ---: | ---: | ---: | ---: |
| profile02 R2 | +3.29% | +4.37% | +4.35% | −0.64% |
| profile02 R3 | +1.62% | +4.45% | +4.40% | −0.57% |
| profile04 R2 | +13.06% | +12.52% | +12.53% | −1.17% |
| profile04 R3 | +21.53% | +22.70% | +22.67% | −2.34% |
| profile02 R1 (profiled) | +15463% | +20.13% | — | — |
| profile04 R1 (profiled) | +14665% | +30.99% | — | — |

R1 is the profiled repeat in both runs. Its TTFT includes trace writing and is unusable as a request metric.

**Substitution S1: profile02 replaces profile04 as the step-cost source.** In its non-profiled repeats, profile04 is 13-23% slower than native06 on TPOT, while profile02 is 4.4% slower. The profiler window is the same in both: repeat 1, formal phase, `max_iterations` 64, `qualification diagnostic_only` (profile02 `replay_summary.json` `canonical_repetitions.profiling`: start_profile at monotonic 20331575.41, stop at 20331774.10). This substitution departs from the source that D-V4-STEP-COST-FIRST-PASS names, so it is reported to YC with the H4a-2 scope question.

profile04 cross-check of the decode steps (TP0 rank per lane, step-to-step interval, median):

| Lane | Composition (k, m) | profile04 (ms) | Samples | profile02 (ms) |
| --- | --- | ---: | ---: | ---: |
| dp0 | (0, 1) | 20.036 | 13 | 19.299 (pooled, m 1-2) |
| dp0 | (0, 4) | 19.787 | 45 | — |
| dp1 | (0, 1) | 19.732 | 60 | — |
| dp0 | (1, 0) | 33.912 | 1 | 21.101 (pooled) |
| dp1 | (1, 0) | 21.661 | 1 | — |
| dp0 | (1, 3) | 49.459 | 1 | 30.146 (pooled, m 1) |
| dp0 | (2, 1) | 63.844 | 1 | not measured |

Here k is the number of requests in their first step on this role and m the number of decoding requests. In profile04 the decode cost does not change from m=1 to m=4 (20.04 vs 19.79 ms). The k=1 steps vary much more between runs (21.7-33.9 ms at m=0; 30.1 vs 49.5 ms at m≥1). The k=2 step occurred only in profile04 R1, where the profiler had bunched the requests together.

## 3. Cost slots for one P0 request

| Slot | Source | Measured scope | Coverage in H4a | Status |
| --- | --- | --- | --- | --- |
| P queue wait | Frontier's own scheduler simulation | — | — | modeled; with QPS 1 over two P lanes and a P service under 0.3 s, no wait is expected |
| P prefill step (1 request, 6144 tokens) | profile02 proxy P service time: median of `prefill_response_monotonic − proxy_arrival_monotonic` over the 256 formal requests of R2 and R3 | one request per step, 6144 tokens, both P lanes | (k=1, 6144 tokens, m=0) only | **substitution S2** (below) |
| KV transfer | existing analytical grouped NIXL model: 126.2 Gbit/s plus 0.0647 ms, the K6 fit to the K5 GLM NIXL READ eight-pair measurement (`../w5_glm_comm_20261020/report.md`, tickets 06/07, `../w7_glm_comm_model_20261020/design.md`) | GLM KV layout on the same H800 links | V4 grouped KV bytes of one 6144-token request | **substitution S3**: link parameters measured on GLM |
| D first step (k=1, 1 token) | profile02 decode traces, TP0 rank of each D lane | the recomputed last prompt token, shown in vLLM as `execute_context_1(1)` | m=0: 2 samples; m≥1: 1 sample at m=1 | **substitution S4** for m≥2 |
| D decode step (m requests, context 6143-7167) | profile02 decode traces, TP0 rank of each D lane, pooled | m 1-2, contexts near 6144 (the first 64 profiled iterations of R1) | m 1-128, contexts up to 7167 | **substitution S5** |
| client→proxy, D byte→client | not modeled | native06 about 0.5 ms and 0.15 ms per request | — | left out; below 0.2% of TTFT |

Native06 TTFT parts for comparison (proxy timestamps, R1/R2/R3): P service mean 277.75/268.12/278.13 ms (p50 267.11/264.89/266.17); P response to D first byte mean 124.90/124.44/121.14 ms (p50 118.57/119.48/117.97). profile02 R2/R3: P service mean 279.39/284.51 ms; P response to D first byte 126.16/121.20 ms.

**Step-cost definition.** A decode step cost is the interval from the step's `execute_context_k(tokens)_generation_m(tokens)` GPU annotation to the next step's GPU annotation on the same rank. The last step in a trace has no successor and is dropped. With decode CUDA graphs, the interval and the GPU kernel span of the step agree within 0.013 ms (largest difference over all 120 decode samples), so the choice between them does not move the result. The interval is still the recorded definition: it includes any host time between steps, which CPU-overhead modeling would otherwise cover. CPU-overhead modeling stays off (`skip_cpu_overhead_modeling=True`) and stays last.

**Prefill source.**
- The profile04 P1 lane has 4 useful steps per rank (steps 23, 29, 243, 502 of `['prefill', 1, 0]` in the [retained24 binding](../../task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/review_preparation/retained24_binding.json)). All four are (1 request, 6144 tokens, 0 decode) with graph mode NONE, so they cover the 6144-token slot. But they lie in the profiled repeat. Their CPU annotations last 329.8-334.9 ms and their contiguous kernel spans 378.1-445.9 ms, against a native06 P service mean of 268-278 ms. They are not used.
- The profile02 prefill trace (23 steps, about 315 ms per 6144-token step) lies in the same profiled window and is not used either.
- **Substitution S2.** The P step uses the proxy P service time of profile02's non-profiled repeats: 287.354 ms (256 samples, min 245.06, max 345.53). This time also contains the proxy-to-P HTTP hop, P scheduling and the response, not only the forward pass. Because CPU-overhead modeling is off, this is the only source that covers those parts of the P side. It is 3-7% above the native06 P service mean, in line with profile02's own perturbation.

## 4. Keying and out-of-range rules

A table entry is keyed by role and by the step's composition: k = number of requests in their first step on this role, their token sum, and m = number of decoding requests. The table (`dsv4_step_cost.json`, schema `dsv4_step_cost.v1`, qualification `diagnostic_step_cost`, `formal_gate_admission` false) holds:

| Role | k | k tokens | m range | time_ms | Samples | Measured m |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| PREFILL | 1 | 6144 | 0-0 | 287.354 | 256 | 0 |
| DECODE | 0 | 0 | 1-128 | 19.299 | 117 | 1, 2 |
| DECODE | 1 | 1 | 0-0 | 21.101 | 2 | 0 |
| DECODE | 1 | 1 | 1-128 | 30.146 | 1 | 1 |

Rules:
- **S5.** A decode step costs the same for every m in [1, 128] and every context in [6143, 7167]. Measured: m 1-2 at contexts near 6144; profile04 shows no change from m=1 to m=4.
- **S4.** A k=1 step with m≥1 uses the single m=1 sample for every m in [1, 128].
- Any other composition raises a typed error: k≥2, a prefill length other than 6144, a P step with decoding requests, or a role, model, device or topology that differs from the table header. With deterministic arrivals 1 s apart, k≥2 is not expected.
- No other extrapolation rule is used.

**DP lanes.** Each attention-DP lane looks up its own composition. Lanes that run in lockstep are already handled by the existing EP-wave maximum over lanes; a lane with no work runs a dummy forward, which costs 0 in Frontier.

**Where the cost goes.** The step cost is the whole step on that lane. It is emitted as one registered attention-phase operation per layer, worth `time_ms / num_layers`, so the existing per-layer stage accounting sums to the table value and no MoE, communication or other operator cost is added on top.

## 5. KV capacity binding

Native vLLM reports `GPU KV cache size: 133,280 tokens, Maximum concurrency for 7,168 tokens per request: 18.59x` on every engine (native06 `nodes/*/server.log`), and the P0 binding receipt records 54740 grouped blocks on every rank (`../canonical_case_20261021/p0_binding/receipt.json`). Both roles bind these through Frontier's existing `explicit` memory mode: `block_size` 256, `num_blocks_mode explicit`, `num_blocks` 54740. `frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py` treats `num_blocks` as the shared grouped pool and admits V4 only when grouped with block 256 (lines 159-170). H4a-2 checks that this admission gives 18.59x at 7168 tokens. The full non-KV memory reconciliation stays open for formal acceptance (ticket 17).

With about 10 decoding requests per D lane in steady state, KV capacity is not expected to bind.

## 6. Expected outcome

This is an expectation written before the run, not a target. No value is fitted to native06.

- **TTFT.** About 287 ms (P step) + the analytical KV time + 21-30 ms (D first step) ≈ 310-320 ms plus KV time, against about 399 ms native, so roughly −18% to −22%. The likely leading error: native P response to D first byte is about 120 ms, far more than the analytical NIXL time plus one D step.
- **TPOT.** 19.30 ms per step, plus k=1 steps at 30.15 ms on new arrivals. With one arrival per second and both lanes in lockstep, that adds about 0.1-0.2 ms per step: about 19.4-19.5 ms against 19.01 ms native, roughly +2% to +3%.
- **E2E and THPT.** Within a few percent, because E2E is dominated by 1023 decode steps and THPT by the arrival rate.

## 7. H4a-2 hook design

**Binding.** The table is a sidecar named `dsv4_step_cost.json` next to the resolved attention input file (`atten_input_file`). It is bound by its presence, as `frontier/execution_time_predictor/dsv4_attention_input.py` binds `dsv4_attention.json`. No new config field is added.

**Execution.**
- KV capacity and transfer use the existing grouped allocation (`vllm_v1_kv_allocation.py`) and the existing grouped NIXL transfer events and analytical predictor, unchanged.
- The V4 structural prediction entry gains a step-cost branch. When the table is bound, each layer emits one registered attention-phase operation worth `T_lane / num_layers`, where `T_lane` is the table value for the lane's composition.
- The admission gate becomes "structural = dummy mode, or step cost bound". `frontier/attention/families.py` therefore stays unchanged.
- Roles, topologies and compositions without a table entry keep their typed errors.
- When step costs are bound, the predictor does not load or train sklearn models for this model.
- PD-AF stays rejected (`frontier/attention/deepseek_v4.py:38-39` is unchanged).

**Files and estimated changed lines.**

| File | Change | Lines |
| --- | --- | ---: |
| `frontier/config/cluster_role_config.py` | role configs inherit `attention_backend`, `kv_cache_dtype` and `indexer_kv_cache_dtype` from the base replica config (today the PD role configs fail V4 storage checks without them) | +6 |
| new `frontier/execution_time_predictor/dsv4_step_cost_input.py` | table reader: schema, header and topology checks, composition lookup, typed errors | ~80 |
| `frontier/execution_time_predictor/deepseek_v4.py` | replace the `KV_CONNECTOR != "none"` guard (:457-458) and the graph-mode / single-request limits (:638-649) with the step-cost branch; give the step-cost operation its phase owner | ~40 |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | rename `_predict_dsv4_dummy_structural_execution`, which now also serves step costs, and branch on the bound table | ~15 |
| `frontier/scheduler/utils/expert_parallel.py` | follow the rename | 1 |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | skip sklearn model loading when step costs are bound | ~15 |
| `frontier/execution_time_predictor/base_execution_time_predictor.py` | the structural gate (:45) also accepts a bound table | ~5 |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | skip training when step costs are bound (today the measured PD probe fails with `FileNotFoundError` for `linear_op.csv`) | ~10 |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | route V4 PREFILL/DECODE to the step-cost branch; this file has 3050 lines, so `docs/general/module-size-exceptions.md` gets a note | ~15 |
| `frontier/model_architectures.py` (:702) | `runtime_limitations` text for the new contract | small |
| possibly `frontier/attention/deepseek_v4.py` (:101-102) | lift the co-location-only guard for PREFILL/DECODE | ~5 |

Total: 10-11 `frontier/` files, about 200-250 changed lines. D-H4A-CODE-SCOPE allows about 8 files, so this list goes to YC before any core edit.

Tests to update: `tests/unit/test_dsv4_structural_pd.py::test_grouped_transport_does_not_admit_v4_execution` (new contract) and the tests that call the renamed method. When the reader exists, the H4a-3 builder (`tests/comparison/calibration/prepare_dsv4_step_cost.py`) imports `SCHEMA` and `QUALIFICATION` from it, so both are defined in one place.

**Probe failures that H4a-2 must close.** `probes/probe_v4_pd.py` runs a 3-request V4 PDD case at the P0 topology without editing the repository; its logs are `probes/probe_*.log`.
1. `probe_dummy.log`: `DeepSeek-V4 cache requires FLASHMLA_SPARSE_DSV4`. The PD role configs do not inherit `attention_backend`, `kv_cache_dtype` or `indexer_kv_cache_dtype` from the base replica config, and `probe_dummy2.log` shows the CLI has no per-role flags for them.
2. `probe_dummy3.log` (role inheritance patched inside the probe): `V4 physical timing scopes have no phase owner: ['sp_moe_attention_reduce_scatter']`.
3. `probe_measured1.log` (V4 execution enabled inside the probe): `FileNotFoundError` for `linear_op.csv`, because the model manager tries to train sklearn models.

The cross-check numbers in section 2 come from `probes/step_probe_profile02.py`, `probes/step_probe_profile04.py` and `probes/p04_prefill_span.py`, with their outputs next to them. They read the traces under `/data/ycfeng/tmp/glm53_dsv4/`, which are copies of the cloud originals listed in section 1.

**Verification risk.** The table sits at the default h800 V4 profiling path. After H4a-2, any existing non-dummy V4 test on h800 that resolves this directory would bind it. H4a-2 checks those tests.

**Options for YC.**
- **A (recommended).** The full list above, 10-11 files.
- **B.** The same without the rename (10 files). The method keeps a name that says "dummy" while it also serves measured step costs.
- **C.** A separate stage-level predictor class registered through `ExecutionTimePredictorRegistry`. This is a registry change and duplicates the V4 lane, EP-wave and phase accounting that the structural path already has. Not recommended.

## 8. H4a-3 builder

`tests/comparison/calibration/prepare_dsv4_step_cost.py` builds the table from the persisted profile02 copy:

```bash
T=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python \
  tests/comparison/calibration/prepare_dsv4_step_cost.py \
  --run $T/h4a_p0_diagnostic/inputs/profile02 \
  --engine-config $T/v4_p0_native_20261021/candidate_h800_p0_profile_02/case/inputs/engine.json \
  --device h800 \
  --output data/profiling/compute/h800/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json
```

- The builder checks that the download receipt says `HASH_IDENTICAL_DOWNLOAD`, that the inventory digest equals the receipt's, and that each consumed local file matches the cloud inventory.
- It checks that the engine config matches the run manifest's `inputs_sha256["inputs/engine.json"]` (`1cd52029…`), and that EP is on, prefix caching and chunked prefill are off, and PP is 1.
- It reads the profiled repeat from the run's own `replay_summary.json` and takes the prefill samples from the other repeats.
- Two builds gave byte-identical output.
