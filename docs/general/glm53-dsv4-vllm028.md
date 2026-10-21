# GLM-5.3 and DeepSeek-V4-Flash development status

This branch adds source-derived model representation, structural simulation and
native comparison tooling for GLM-5.3 and the original DeepSeek-V4-Flash.
**It does not yet provide a calibrated predictor for either model.** The target
is prioritized calibration on H800 and H200 at 32 and 64 GPUs, starting with
H80032 for both models. The fixed workload below replaces
the earlier SharedGPT formal case; historical artifacts remain unchanged.

This snapshot includes the W7 measured-DSA, sequence-parallel MoE and measured
communication integrations through October 20, 2026 Asia/Hong_Kong, and the
calibration-scope revision of October 21, 2026. Test results below belong to their
stated revisions and scopes; they are not a claim that every test passes at the
branch tip. Step-5-Preview integration is outside this branch.

## Calibration contract (2026-10-21)

The four priority tasks apply separately to each model:

| Task | Hardware / GPUs | RJob class | Formal calibration dependency |
| --- | --- | --- | --- |
| P0, highest | H800 / 32 | Non-preemptible | Own runtime/coverage admission; final four errors ≤10% |
| P1 | H800 / 64 | Non-preemptible | Basic running validation and same-model H80032 four errors ≤30% |
| P2 | H200 / 32 | Non-preemptible | Own admission and personal non-preemptible budget; no H800 accuracy dependency |
| P2 | H200 / 64 | Entire job preemptible | Basic environment validation and same-model H20032 four errors ≤30% |

Both 64-GPU basic validations and preparation may run **before** the 30% gate;
the formal native reference/calibration campaign may not. Promotion uses all
four errors in each admitted repetition, never an average across metrics,
repetitions, models or hardware. The 30% threshold governs entry, not final
acceptance; every completed calibration still requires four independent errors ≤10%.
Ready higher-priority work receives shared resources first. Both H200 tasks are
P2 and do not wait for H800 accuracy results.

Across all sessions, `i-fengyicheng` may reserve at most **32 non-preemptible
H200 GPUs**, including queued jobs. Every additional H200 submission must be
preemptible; H20064 is always wholly preemptible. H20032 non-preemptible cases
wait for the required budget. The old 70/130-GPU limits no longer authorize runs.
Preserve preempted attempts and restart complete clean repetitions rather than
stitching partial samples together. H800 uses `codesign`; H200 uses `step_main`.

The 32-GPU topology is P16/D16, per-role TP8 × DP2 × EP16 × PP1; 64 GPUs use
P32/D32, per-role TP8 × DP4 × EP32 × PP1. Native refusal is a recorded blocker,
not permission to silently downgrade a task. Common settings remain prefill/
decode disaggregation, online serving, thinking/prefix/chunking disabled,
native CUDA Graph and quantization enabled, and dummy weights.
Each case has **128 requests**,
each exactly **6144 input / 1024 output tokens** (`k=1024` interpretation of 6k/1k),
at **QPS 1.0 with fixed one-second arrivals** from 0 through 127 s. No Poisson
arrivals, SharedGPT sampling or tokenizer re-count defines this workload.
Four metrics are gated independently at
`|frontier - vllm| / |vllm| <= 0.10` on their means: client TTFT; TPOT as the
per-request mean of `(E2E - TTFT) / (output_tokens - 1)`; total token throughput;
and per-request end-to-end latency. THPT is formal prompt-plus-output tokens over
the interval **from the first formal submission through the last formal
last-token completion**, not the 127 s arrival span. A complete repetition has
786432 input + 131072 output = **917504 total tokens**. Internal TTFT, ITL, KV-ready time and
percentiles are diagnostics and do not gate.

Each eligible formal model/hardware/size case needs **3 clean repetitions in one
job**, each with its own independent
16-request warmup of the same shape at QPS 1.0 and formal measurement window,
and both roles idle between repetitions (no
running/waiting requests and KV usage back to zero). Every repetition must finish
all 128 requests with exact native token counts; failed, unfinished, truncated or
early-terminated requests cannot be dropped to create a result. Warmup, startup
and between-repetition idle time do not enter formal throughput. Stable template
IDs keep prompt content unchanged across repetition-qualified transport IDs;
warmup completion counts alone do not prove all-lane graph coverage.

The replacement artifact is
[`data/processed_traces/canonical_fixed_6144_1024_req128_qps1/`](../../data/processed_traces/canonical_fixed_6144_1024_req128_qps1/),
using the existing trace, request-ID, disjoint warmup-pool, manifest and checksum
layout, pinned to the exact `68f891b` package. The earlier
[`canonical_sharegpt_req256_qps4/`](../../data/processed_traces/canonical_sharegpt_req256_qps4/)
artifact is retained as deferred history, not a second formal workload. This
document does not assert synchronization with the separate Step-5-Preview worktree.

MTP follows the native default, disabled for **both** models. Both declare
`num_nextn_predict_layers: 1`, but that does not enable serving speculation.
Final canonical command/runtime parity, effective CUDA Graph mode and capture
sizes, quantization and memory budgets remain evidence to collect, not an actual
freeze established by these requested settings.

Without chunking, the serving configuration must satisfy `max_model_len >= 7168`
and `max_num_batched_tokens >= max_model_len`. Hardware-specific profiling must cover the
6144-token unchunked prompt and the 1024-output decode progression through a
7168-token total context, including reached graph/compression/page boundaries.
The formal communication backend remains **measured**; every touched
collective/NIXL shape needs a curve matching hardware, topology and actual lane
widths. Reuse qualified rows first; no cross-hardware scaling or substitution.

The scope is one fixed workload across the four requested hardware/scale tasks,
not a revival of multi-workload/QPS sweeps or the old 36-cell/holdout matrix.
Existing H800/H200 measurements keep their actual scope and provenance. Memory,
graph, compiled/phase and physical-TP qualification are case-specific; earlier
primitive measurements or basic validations do not establish full calibration.

## Model and reference-runtime contract

| Item | Contract |
| --- | --- |
| GLM | Original `zai-org/GLM-5.3`, `GlmMoeDsaForCausalLM`, 78 backbone layers, 3 dense and 75 MoE layers, 256 routed experts, FP8 configuration. See the [checked-in model config](../../data/config/models/zai-org__GLM-5.3.json). Under vLLM sequence-parallel MoE (EP enabled, TP>1 and DP>1, PP1 in Frontier), each rank stores one full copy of the shared expert instead of a TP shard. The TP8/DP2/EP16 per-rank weight estimate is 52,154,161,152 B (48.57 GiB), 0.53% below the 48.83 GiB native model-load memory delta. Sequence-parallel widths and shared-expert TP1 timing are represented; full-model timing remains uncalibrated. |
| V4 | Original `deepseek-ai/DeepSeek-V4-Flash`, `DeepseekV4ForCausalLM`, 43 backbone layers and 256 routed experts. Source representation includes mixed-precision parameter storage, uncompressed/CSA/HCA attention, the indexer, sequence parallelism and mHC. See the [checked-in model config](../../data/config/models/deepseek-ai__DeepSeek-V4-Flash.json). |
| Native reference | vLLM `0.28.0`, source commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Native observations use separately developed, pinned instrumentation; those vLLM changes are not part of this Frontier branch. |
| Recorded GPU environment | H800/SM90; observed package versions include PyTorch `2.13.0+cu130`, Transformers `5.15.1`, Triton `3.7.1`, FlashInfer `0.6.16.post3`, NIXL `1.3.2` and UCX `1.21.0`. CUDA package labels, loaded libraries and selected kernels are distinct evidence. Package versions do not establish build attestation or qualify every backend. |
| Native weights | `--load-format dummy`: original model structure and real kernels execute on GPUs with random weights. No checkpoint is required. This does not establish production-checkpoint output quality or its expert distribution. |
| Simulator timing | Structural drivers use explicitly synthetic durations. The measured GLM DSA path uses the checked-in H800 tables only within their workload coverage. Neither structural runs nor table-point checks establish full-model accuracy or hardware qualification. |

The [dummy-weight ground-truth decision](../adr/0001-dummy-weight-native-ground-truth.md)
defines the separation between native random weights and simulator dummy timing.
Equal seeds alone do not establish P/D parameter or auxiliary-state identity, so
token-content equality is not asserted without that identity evidence.

## What the branch can represent

| Area | Implemented and exercised scope | Current boundary |
| --- | --- | --- |
| GLM model, attention and storage | Typed dense/routed/shared FFN contracts, full/shared DSA index ownership, measured sparse-MLA/indexer lookup and fitting, source-defined parameter/scale storage and KV capacity. Full 78-layer structural co-location runs use the real simulator entry point in offline and online modes. | Measured DSA rejects uncovered workloads. Old GLM router profiling rows do not satisfy the new grouped-top-k contract, so the checked-in tables alone do not yet admit a full-model non-dummy run. Graph settings in the structural driver establish configuration/accounting only. |
| GLM P/D | Sequential, PP1, eager structural PDD with D-side reservation (the prompt's blocks, as vLLM's remote load holds them; the first D step recomputes the last prompt token in place), transfer readiness, P-side holds, release and re-admission through existing scheduler/cache/transfer owners. The integration driver models P32+D32 as 64 logical devices. | Logical devices are not a 64-GPU qualification. The DECODE cluster scheduler picks each request's D lane through `select_kv_transfer_target`; only `round_robin` and its sticky variant implement it, so the v0.28 DP policy does not yet place NIXL D requests. Transfer payload accounting and same-time readiness/release are simulator contracts; measured NIXL lifetime and timing remain separate. |
| V4 cache | Explicit H800 sparse-FlashMLA cache selection, grouped allocation, sliding/compressed cache state, partial retirement and reuse, plus one cache-zero scope from allocation IDs carried into the Batch. | The exercised layout requires block size 256, DCP1 and the supported FP8 cache selections. Physical reservation extent is not measured store traffic or bytes zeroed. |
| V4 execution | Full 43-layer eager PP1 co-location; TP8 with DP2/4/8 and corresponding EP16/32/64 structural cases. Source-rank sequence-parallel geometry, mHC/shared/routed work, compression boundaries, stage/phase accounting and operator traces are represented. | Costs and destination routing distributions are synthetic. V4 measured execution, P/D, CUDA graphs, prefix caching and speculation are not enabled by this structural path. Complete idle-attention tracing and native overlap remain unqualified. |
| Internal DP placement | Existing balancer gains explicit `policy_version="v0.28.0"`, including request ownership and the selected native scoring/tie behavior. `"v0.10.2"` remains the legacy default. | This is the internal online policy reference; it does not settle P/D placement or establish native DP4 behavior. |
| V1 queue admission | Explicit `max_concurrent_batches=2` admits queued work independently of physical pipeline stages. `None` retains existing pipeline capacity/default behavior. | The opt-in is restricted to the exact V1 scheduler, ordinary-cache MONOLITHIC PP1/DP1, without prefix caching, speculation or GDN. Stage execution stays serial; grouped/disaggregated paths and subclasses are rejected. |
| Profiling producers (vLLM 0.28) | GLM DSA sparse-MLA/indexer attention, block-FP8 linear ops (including sequence-parallel shared-expert rows) and block-FP8 FlashInfer CUTLASS routed experts; V4 compressed attention, sequence-parallel shared-expert rows with clamped SwiGLU and checkpoint-MXFP4 MARLIN routed experts; vLLM forward collectives on serving process groups, NIXL KV READ and EP calls under concurrent KV reads. Rows carry runtime identity columns (runtime versions, source and model-config revision, weight source, device, quantization signature, parallel groups, graph mode). GLM H800 tables and measured communication curves are checked in. | GLM DSA and measured communication consumers are integrated. GLM router rows still need the grouped-top-k rerun. V4 non-dummy timing is not implemented and V4 producers have not run on GPU. |
| Native harnesses | Pinned observer bootstrap, dummy initialization checks, multi-node DP serving commands, externally managed P/D replay and role-specific native cache overrides. | Tools support qualification work; successful preparation or submission does not qualify the workload, observer or predictor. |

The implementation reuses the existing architecture/attention/operator tables,
cache pool, scheduler, predictor and metrics owners. Useful review entry points:

- [Model contracts](../../frontier/model_architectures.py),
  [attention families and cache layout selection](../../frontier/attention/families.py),
  [parameter accounting](../../frontier/utils/param_counter.py).
- [V4 attention workloads](../../frontier/attention/deepseek_v4.py),
  [V4 structural execution](../../frontier/execution_time_predictor/deepseek_v4.py),
  [grouped block allocation](../../frontier/kv_cache/kv_cache_block_pool.py).
- [V1 scheduler](../../frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py),
  [KV-arrival handling](../../frontier/scheduler/utils/kv_arrival.py),
  [DP balancer](../../frontier/scheduler/utils/vllm_dp_load_balancer.py).
- [Native bootstrap](../../tests/comparison/calibration/native_observer_bootstrap.py),
  [P/D worker](../../tests/comparison/calibration/native_pd_worker.py),
  [P/D replay](../../tests/comparison/calibration/pd_replay.py).

## Recorded validation by revision

### Checked-in GLM H800 tables

The model-name path substitution resolves to
[`data/profiling/compute/h800/zai-org/GLM-5.3/`](../../data/profiling/compute/h800/zai-org/GLM-5.3/).
Its [SHA256 manifest](../../data/profiling/compute/h800/zai-org/GLM-5.3/SHA256SUMS)
covers `dsa_attention.csv`, `dsa_attention_kernel_only.csv`, both `linear_op`
tables, both `moe` tables and `moe_config.yaml`. All 12 runtime identity columns
are retained. These measurements used native dummy weights, not simulator
dummy timing or production-checkpoint routing.

Only the [DSA reader](../../frontier/execution_time_predictor/dsa_predictor.py)
opens the isolated `dsa_attention` filenames. Dense/MLA readers keep their own
filenames; no generic `attention.csv` is published in this model directory.
The kernel-only DSA table was re-aggregated from the original JSONL using its
original Python 3.12 environment. Its only byte changes are two header names:
`kernel_count.<scope>` becomes `time_stats.<scope>.kernel_count` for
`attn_dsa_indexer` and `attn_dsa_mla`; measurements and identity are unchanged.

The DSA consumer separates native decode/prefill kernel paths, uses summed key
and selected-key work (causal pairs for the prefill path), and charges the
indexer only to full-index owners. CUDA-event pure-decode rows, including the
known host-stall outlier, are excluded; kernel-only measurements supply decode
timing. Both DSA tables contain 3,708 rows: 1,942 original rows and 1,766
verified query-extend/multi-sequence rows from producer revision `0660717`
on H800. Their aggregation keys are disjoint; measured
cells and all runtime identity columns are preserved. The extension uses the
same schema and does not replace the original measurements. Zero-context FULL
dummy attention remains unmeasured. Uncovered and mixed serving workloads fail
rather than use dummy costs or extrapolation.

The MoE tables contain the verified `grouped_topk_sigmoid` rerun from source
`9dee7defd932624f68d55c75833c57f2a925d8ac`. Both tables have 4,059 rows, with
`grouped_topk_bias` assignment, `sigmoid_renorm_scaled` weights, and native
`FLASHINFER_CUTLASS` expert execution. These replace the old
`standard_fused_topk` measurements without relabeling them. A run on these
tables is labeled **structural smoke on H800 tables, not an accuracy result**;
bypassing dataset admission is not a valid smoke.

These are selected completed checks from the development record. A passing
structural test verifies its modelled invariants, not calibrated predictions.
Detailed native run evidence is retained outside the Git repository; the native
results in the next section cannot be independently reproduced from this branch
alone without its separately pinned runtime, instrumentation and inputs.

| Frontier revision | Recorded result and scope |
| --- | --- |
| `9c63e6b`, `b49a4a0`, `cd3d353`, `dc3912c` | Model contracts, GLM cache/attention/storage and role coverage were accepted in successive source/CPU slices. Tests remain in the [GLM model](../../tests/unit/test_glm53_model_config.py), [attention](../../tests/unit/test_glm53_attention_cost_contract.py), [capacity](../../tests/unit/test_glm53_cache_capacity.py) and [parameter](../../tests/unit/test_glm53_parameter_memory.py) modules. |
| `2da11c8` | GLM structural PDD: 1,124 focused tests; offline and online full-model runs each complete 5 requests and 5 transfers. Actual D-lane capacity waiting, completion, transfer-byte accounting and release are checked by the [structural P/D driver](../../tests/integration/run_glm53_structural_pd.py). |
| `dc7f382` | V4 structural execution: 1,230 tests across 62 isolated modules; full-model offline and online runs each complete 4 requests over 32 logical devices with stage/phase conservation. See the [structural driver](../../tests/integration/run_dsv4_structural.py) and [execution tests](../../tests/unit/test_dsv4_structural_execution.py). |
| `88c030f` | V4 cache-zero/DP8 slice: 1,292 tests across 64 isolated modules; full-model DP8 offline and online each complete 8 requests over 64 logical devices. [Cache-zero tests](../../tests/unit/test_dsv4_cache_zero_execution.py) check allocation/Batch/trace ownership; [lifecycle tests](../../tests/unit/test_grouped_kv_cache_scheduler.py) check release and reuse. |
| `563dfff` | Explicit DP policy: 163 focused checks and three DES cases. A clean candidate passed the unchanged 77-case fidelity comparison against clean baseline `bfa82de`; two placement tests failed there. Only `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` also fails on `bfa82de`; the in-flight preemption-key test failed because of branch commit `3e6a421` and passes after the in-flight decode sample correction recorded below. See [balancer tests](../../tests/unit/test_vllm_dp_load_balancer.py). |
| `8d479ef` | Queue admission: independent focused result 135 PASS / 7 FAIL. All seven failure IDs and normalized exception headings reproduce on clean parent `da13911`. Correction (2026-10-20): `da13911` already contains branch commit `3e6a421`, whose settled decode context shortened MONOLITHIC decode rows by one token while the in-flight sample still expected the old row length. The seven decode-preemption failures came from that commit, not from before this branch, and pass after the in-flight decode sample correction. A clean candidate separately passes the unchanged 77-case fidelity comparison, including artifact and cache-filename agreement. See [queue/overlap integration checks](../../tests/integration/test_vllm_v1_pp_prefill_chunk_overlap.py). |
| `9069538` | Native deployment/coordination source: 156 root tests and independent review accepted. This result covers preparation and lifecycle handling, not native workload qualification. See [worker tests](../../tests/unit/test_native_pd_worker.py) and [P/D harness tests](../../tests/unit/test_calibration_pd_harness.py). |
| `9dec8d1`, `85a5c90`, `5ad40e0`, `0a8cc8e`, `7311c17`, `7e49a80` | Review fixes: GLM shared-expert replication under sequence-parallel MoE, in-flight decode sample length after preemption, the NIXL D-side reservation of the prompt's blocks, the D-lane selector on the cluster scheduler base, and the dense-layer residual-add stage metric. Full `tests/unit` has the same failure set as the parent revision, and the 77-case fidelity comparison is unchanged against `bfa82de`. |
| `20b0621`, `dd119a7` | [Layer and KV-cache inventory check](../../tests/comparison/calibration/native_layer_cache_inventory.py) against native inventories: all 78 GLM and 43 V4 layers and all cache groups match. Two capacity differences remain for calibration: vLLM's null block, which cannot be allocated, and the device memory size Frontier assumes. |
| `f01a1d7`–`8f6c3de` | Profiling producers listed above, for example the [DSA producer](../../frontier/profiling/attention/dsa_main.py), [V4 attention producer](../../frontier/profiling/attention/dsv4_attention_main.py), [MXFP4 MARLIN experts](../../frontier/profiling/moe/mxfp4_marlin_experts.py) and [collective benchmarks](../../frontier/profiling/collectives/vllm_collective_benchmark.py). CPU unit tests and dry-run plans cover them; GPU runs are recorded outside the repository. `4780667` makes shared-expert projections follow the model quantization config, which matches vLLM block FP8 for GLM-5.3, DeepSeek-V3 and step-moe; non-dummy predictions for those models change, and the 77-case fidelity comparison is unchanged. |
| `89290df`, `6c3acc0` | Complete observer bootstrap and per-role cache override accepted separately: 63 bootstrap tests with dependency-combination/fault checks; 132 pressure-harness tests with 45 frozen baseline/candidate checks. |

The older C16/C17/C23 comparisons associated with `2da11c8`, `dc7f382` and
`88c030f` matched all 77 scenario artifacts and cache filenames, but the official
comparator returned exit 1 for recorded modified-source provenance. Those results
are **artifact agreement**, not clean-commit or whole-comparator PASS. The later
clean results at `563dfff` and `8d479ef` are separate evidence. Cache filename
agreement does not compare cache binary contents. The existing
[fidelity harness](../../tests/e2e/refactor_fidelity/run_matrix.py) and
[comparator](../../tests/e2e/refactor_fidelity/compare.py) remain the reference.

## Native evidence and unresolved qualification

Reduced native initialization checks pass for both original models. The V4
diagnostic includes the canonical post-load mHC auxiliary derivation, batched
FP8 scale inspection and MARLIN MXFP4 controls; earlier diagnostic failures are
retained. The original V4 selects V2 Model Runner and MARLIN on the observed
H800 path, so V1 internal hooks do not establish V2 internal coverage.

A full original V4 run exercised 43 layers/256 experts on 16 H800 and generated
four tokens for each of four prompts with finite selected log probabilities.
Platform/container success does not close observer qualification: no EngineCore
trace was retained, teardown included forced worker termination, and the original
launcher exit code is unavailable. Its native prefix-cache default does not
enable Frontier V4 prefix caching.

A full original GLM functional P16+D16 run used TP8/DP2/EP16/PP1 per role and
completed eight HTTP requests, 2,400 prompt tokens and 32 output tokens. Native
parents, containers, platform and launcher completed successfully. The final
40-process collection contains actual multi-chunk prefill, 248 FULL replay host
returns across 31 D steps and eight first-D-input state/order chains. These prove
selected host events; they do not prove GPU readiness or token-content equality.

The functional GLM transfer observer remains **PARTIAL / FAIL**: opaque NIXL
handle serialization lost 320 record positions across 16 D workers, including
prepare-return, submit, DONE and release observations. A minimal repair at
external native commit `169309961` is independently accepted for source/CPU and
new inputs (80 tests and 17 actual-body CPU cases). The separate pressure run
passes business and terminal checks: eight client HTTP 200 responses, 16 P/D
upstream HTTP 200 responses, 7,552 prompt tokens and 640 output tokens. All four
native parents and containers exit 0 without restarts; platform completion is
October 19 at 16:07:17 UTC and launcher exit 0 follows at 16:07:18 UTC.

Post-stop review passes 355 scoped checks, including five same-lane
refusal/release/re-admission chains across 422 allocation refusals and 64 complete
NIXL handle prepare/submit/DONE/release chains. All 40 trace files have no sequence
gaps, parser errors or reported drops. This verifies the observed host lifecycle;
all 40 footers still report `run_verified=false`, and missing-parent records and
unknown receive-report source status remain. Full observer qualification stays
open. The new result preserves the historical functional FAIL.

Other open boundaries remain material:

- Observer perturbation fails the frozen 3% criterion: all/long/short paired
  changes are approximately 3.1642%, 3.3022% and 8.3335%. No sample or threshold
  was changed to obtain acceptance.
- Strict native routing-content capture remains off. Source-derived geometry and
  balanced simulator routing do not supply measured routing occupancy/payloads.
- The retained real GLM compute forwards resolve to their EngineCore parents.
  Broader missing-parent counters also include dummy and NIXL observations;
  driving that counter to zero is not a substitute for missing timing evidence.
- The [existing CPU producer](../../frontier/profiling/cpu_overhead/vllm_cpu_probe.py)
  has no adapter for the new raw observer records. Qualified DP/READ intervals,
  absolute GPU endpoints and a cross-node clock relation remain missing. The
  [raw admission path](../../tests/comparison/e2e_gate/admission.py) remains
  insufficient for numeric publication.
- Complete full-model observation, P/D qualification, the canonical 64-H200
  P32+D32/DP4 reference with 128 exact-length completions in each clean repetition,
  trained artifacts and the four-metric calibration gates remain
  open. One placement failure,
  `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`,
  is inherited from `bfa82de` and remains unresolved; its reproduction is not a
  waiver. The seven decode-preemption failures and the second placement failure
  came from branch commit `3e6a421` and are fixed.

## Review and continuation

Start with the model/storage/cache tests and the three explicit structural
drivers: [GLM co-location](../../tests/integration/run_glm53_structural.py),
[GLM P/D](../../tests/integration/run_glm53_structural_pd.py), and
[V4 co-location](../../tests/integration/run_dsv4_structural.py). Each driver
accepts a new `--output-root`; V4 additionally selects `--dp 2`, `4` or `8`.
Run unit modules in fresh Python processes where their global simulator state
requires isolation. The drivers retain configuration and check request, cache,
stage and operator artifacts; an exit code alone is not their acceptance test.

For example, from the repository root with its CPU test environment installed,
set `FRONTIER_TMP_ROOT` to writable disk storage and use new child directories:

```bash
export PYTHONPATH="$PWD"
python -m pytest tests/unit/test_glm53_model_config.py -q
python -m pytest tests/unit/test_dsv4_model_config.py -q
python tests/integration/run_glm53_structural.py \
  --case offline-none --output-root "$FRONTIER_TMP_ROOT/glm53-colocation"
python tests/integration/run_glm53_structural_pd.py \
  --case all --output-root "$FRONTIER_TMP_ROOT/glm53-pd"
python tests/integration/run_dsv4_structural.py \
  --dp 8 --case both --output-root "$FRONTIER_TMP_ROOT/dsv4-colocation"
```

These CPU commands are structural checks. They do not launch native GPU jobs or
produce qualified calibration data.

Continue from the measured GLM primitive and communication tables into the
approved predictor integration: sequence-parallel MoE token widths and
collectives, the measured DSA timing module, and a measured communication
backend. Then continue into CPU normalization, bundle training, cost replay and
coverage paths.
Complete the trained GLM P/D path before claiming a calibrated GLM predictor;
then compare the fixed canonical case against its admitted native repetitions
and continue measured V4 integration. The older broad holdout matrix is deferred.
Native async disposition needs qualified GLM P/D/schema evidence; DP P/D placement
also needs the separate DP4 observation. Neither a structural run nor a source
policy oracle closes those decisions.

Raw task archives, worker logs, machine identities, cloud locations and private
runtime artifacts are intentionally not included in this review document or its
links. Publishing this branch for review does not merge it or publish the
separate native-instrumentation repository.
