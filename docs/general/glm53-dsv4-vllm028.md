# GLM-5.3 and DeepSeek-V4-Flash development status

This branch adds source-derived model representation, structural simulation and
native comparison tooling for GLM-5.3 and the original DeepSeek-V4-Flash.
**It does not yet provide a calibrated predictor for either model.** The next
product milestone is the full trained GLM P/D path, followed by independent
calibration and holdout checks, then measured V4 integration.

This snapshot describes Frontier code through `6c3acc0` and recorded validation
through October 19, 2026 UTC / October 20, 2026 Asia/Hong_Kong. Test results below
belong to their stated revisions and scopes; they are not a claim that every
test passes at the branch tip. Step-5-Preview integration is outside this branch.

## Model and reference-runtime contract

| Item | Contract |
| --- | --- |
| GLM | Original `zai-org/GLM-5.3`, `GlmMoeDsaForCausalLM`, 78 backbone layers, 3 dense and 75 MoE layers, 256 routed experts, FP8 configuration. See the [checked-in model config](../../data/config/models/zai-org__GLM-5.3.json). Under vLLM sequence-parallel MoE (EP enabled, TP>1 and DP>1, PP1 in Frontier), each rank stores one full copy of the shared expert instead of a TP shard. The TP8/DP2/EP16 per-rank weight estimate is 52,154,161,152 B (48.57 GiB), 0.53% below the 48.83 GiB native model-load memory delta. Shared-expert timing keeps the TP-sharded shape; sequence-parallel token sharding and its collectives are not modeled. |
| V4 | Original `deepseek-ai/DeepSeek-V4-Flash`, `DeepseekV4ForCausalLM`, 43 backbone layers and 256 routed experts. Source representation includes mixed-precision parameter storage, uncompressed/CSA/HCA attention, the indexer, sequence parallelism and mHC. See the [checked-in model config](../../data/config/models/deepseek-ai__DeepSeek-V4-Flash.json). |
| Native reference | vLLM `0.28.0`, source commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Native observations use separately developed, pinned instrumentation; those vLLM changes are not part of this Frontier branch. |
| Recorded GPU environment | H800/SM90; observed package versions include PyTorch `2.13.0+cu130`, Transformers `5.15.1`, Triton `3.7.1`, FlashInfer `0.6.16.post3`, NIXL `1.3.2` and UCX `1.21.0`. CUDA package labels, loaded libraries and selected kernels are distinct evidence. Package versions do not establish build attestation or qualify every backend. |
| Native weights | `--load-format dummy`: original model structure and real kernels execute on GPUs with random weights. No checkpoint is required. This does not establish production-checkpoint output quality or its expert distribution. |
| Simulator timing | The new structural paths use explicitly synthetic durations. Their token, allocation and accounting checks do not measure native latency, hardware fit or communication bandwidth. |

The [dummy-weight ground-truth decision](../adr/0001-dummy-weight-native-ground-truth.md)
defines the separation between native random weights and simulator dummy timing.
Equal seeds alone do not establish P/D parameter or auxiliary-state identity, so
token-content equality is not asserted without that identity evidence.

## What the branch can represent

| Area | Implemented and exercised scope | Current boundary |
| --- | --- | --- |
| GLM model, attention and storage | Typed dense/routed/shared FFN contracts, full/shared DSA index ownership, source-defined parameter/scale storage and KV capacity. Full 78-layer co-location runs use the real simulator entry point in offline and online modes. | DSA measured timing remains guarded. Co-location graph settings in the structural driver establish configuration/accounting only. |
| GLM P/D | Sequential, PP1, eager structural PDD with D-side reservation (the prompt's blocks, as vLLM's remote load holds them; the first D step recomputes the last prompt token in place), transfer readiness, P-side holds, release and re-admission through existing scheduler/cache/transfer owners. The integration driver models P32+D32 as 64 logical devices. | Logical devices are not a 64-GPU qualification. Transfer payload accounting and same-time readiness/release are simulator contracts; measured NIXL lifetime and timing remain separate. |
| V4 cache | Explicit H800 sparse-FlashMLA cache selection, grouped allocation, sliding/compressed cache state, partial retirement and reuse, plus one cache-zero scope from allocation IDs carried into the Batch. | The exercised layout requires block size 256, DCP1 and the supported FP8 cache selections. Physical reservation extent is not measured store traffic or bytes zeroed. |
| V4 execution | Full 43-layer eager PP1 co-location; TP8 with DP2/4/8 and corresponding EP16/32/64 structural cases. Source-rank sequence-parallel geometry, mHC/shared/routed work, compression boundaries, stage/phase accounting and operator traces are represented. | Costs and destination routing distributions are synthetic. V4 measured execution, P/D, CUDA graphs, prefix caching and speculation are not enabled by this structural path. Complete idle-attention tracing and native overlap remain unqualified. |
| Internal DP placement | Existing balancer gains explicit `policy_version="v0.28.0"`, including request ownership and the selected native scoring/tie behavior. `"v0.10.2"` remains the legacy default. | This is the internal online policy reference; it does not settle P/D placement or establish native DP4 behavior. |
| V1 queue admission | Explicit `max_concurrent_batches=2` admits queued work independently of physical pipeline stages. `None` retains existing pipeline capacity/default behavior. | The opt-in is restricted to the exact V1 scheduler, ordinary-cache MONOLITHIC PP1/DP1, without prefix caching, speculation or GDN. Stage execution stays serial; grouped/disaggregated paths and subclasses are rejected. |
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
- Complete full-model observation, P/D qualification, 64-H800 P32+D32/DP4
  qualification, trained artifacts and formal calibration/holdout gates remain
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

Continue from qualified native workflow and measurement inputs into the existing
profiling, CPU normalization, bundle training, cost replay and coverage paths.
Complete the trained GLM P/D path before claiming a calibrated GLM predictor;
then run independent calibration/holdout checks and measured V4 integration.
Native async disposition needs qualified GLM P/D/schema evidence; DP P/D placement
also needs the separate DP4 observation. Neither a structural run nor a source
policy oracle closes those decisions.

Raw task archives, worker logs, machine identities, cloud locations and private
runtime artifacts are intentionally not included in this review document or its
links. Publishing this branch for review does not merge it or publish the
separate native-instrumentation repository.
