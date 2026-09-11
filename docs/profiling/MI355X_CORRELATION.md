# MI355X capture and Frontier correlation

The initial validation target is one SGLang replica using TP=8, EP=1, PP=1
across an eight-GPU MI355X node. `frontier.validation` captures actual batch
metadata and replays it through the existing Frontier predictor without asking
a scheduler to reconstruct the runtime's decisions. The existing `sglang`
scheduler remains available for subsequent serving-level experiments.

## Capture

Run inside the pinned SGLang environment with this Frontier checkout on
`PYTHONPATH`. The adapter uses `sglang.benchmark.one_batch`'s checkpoint loader,
allocation, extend and decode helpers; it requires the CPU length metadata and
`ModelRunnerOutput.can_run_graph` contract present in the tested runtime.

Choose an experiment directory outside the source checkout. Every invocation
requires a new output directory. Workers write separate rank ledgers; the
parent validates and merges them only after every worker succeeds.

```bash
export SGLANG_USE_AITER=1
python -m frontier.validation.sglang_capture \
  --model-path /models/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --tp-size 8 --attention-backend aiter --page-size 1 \
  --random-seed 1234 \
  --chunked-prefill-size 16384 --mem-fraction-static 0.9 \
  --disable-radix-cache \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 4 8 12 16 24 32 \
  --batch-size 16 24 32 --input-len 1024 --output-len 32 \
  --capture-warmups 2 --repetitions 10 --trace --trace-steps 3 \
  --output-dir /experiments/mi355x/capture-001
```

The static runner executes each complete prefill in one call. The server's
chunk budget is recorded but does **not** split these offline batches; for
example batch 32 x 1024 is a 32768-token prefill. This is intentional for the
initial operator/batch calibration. Serving/chunked-prefill correlation needs
actual scheduler batches and is not established by this benchmark.

Normal repetitions are timed without Kineto. `--trace` adds a separate pass
with synchronized `frontier.batch:<id>` markers and exports rank 0 by default.
Use `--trace-ranks 0 1 2 3 4 5 6 7` when all-rank kernel diagnostics are needed.
The profiler pass is excluded from timing statistics.

The manifest records model path, runtime versions, source digest, topology,
launch settings and timing semantics. Per-rank files record graph capture
sizes, GPU memory, token capacity and request-pool capacity. Seeded inputs and
finite-logit checks are shared across ranks. Output hashes must agree across
TP ranks. Differences between repeated greedy outputs are recorded explicitly:
they can change expert routing and must be considered when interpreting timing
variation. No prompt or output text is stored in the batch ledger.

Pin SGLang's `--random-seed` whenever separate captures will share runtime
profiles. Its generated default changes between launches and is part of the
runtime identity. A matching seed still does not guarantee that free-running
decode produces the same token trajectory or expert routes; compare the frozen
trajectory digest and routing records rather than assuming reproducibility.

Inspect the resolved server settings in the manifest: the pinned SGLang AITER
path multiplies the requested memory fraction by 0.85 for model context limits
above 8192. A requested 0.9 therefore resolves to 0.765. Static pool capacity
with prefix caching disabled is not the same as the earlier serving request
limit; the runtime pool and memory records are authoritative for this capture.

`forward_gpu_ms` uses GPU events around model forward and excludes sampling.
`step_wall_ms` includes input preparation, forward and sampling through device
synchronization. TP aggregation takes the maximum elapsed duration across
ranks; it does not sum rank times or assume synchronized host clock origins.
These metrics are distinct from serving TTFT and TPOT.

```bash
python -m frontier.validation.report \
  --capture-dir /experiments/mi355x/capture-001 \
  --output /experiments/mi355x/capture-001/timing-summary.json
```

The report excludes the first four decode steps, takes a median within each
repetition and reports the median and coefficient of variation across repeats.
It also compares every profiled batch against unprofiled repetitions of the
**same step and exact shape**, taking the maximum rank duration in both cases.
The default diagnostic threshold is 5% absolute change, including unexpectedly
faster samples. Do not calibrate full-model latency from materially perturbed
profiles or subtract a global correction factor from operator times. Passing
this check alone does not establish unbiased individual kernel measurements.

## Complete decoder trace decomposition

The explicit `sglang_qwen35_rocm_separate_shared_v1` contract supports one-node
TP=8/EP=1 with separate shared experts, unfused AITER TP reductions and one GPU
stream. It partitions every decoder kernel exactly once against the model-owned
hybrid layer schedule. Unknown operators, missing boundaries, extra layers or
multi-stream traces fail admission. Small timestamp overlaps on the identified
stream remain visible as overlap time; kernel sums are not replaced by spans.

```bash
python -m frontier.validation.operator_trace \
  --capture-dir /experiments/mi355x/capture-001 \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --rank 0 \
  --output /experiments/mi355x/decoder-components.json
```

The output retains batch/rank/layer identity, kernel index ranges, component
sums, decoder span/busy time, source hashes and the profile-perturbation check.
Fusions stay intact: QKV projection includes QK norm; attention output projection
includes its gate; MXFP4 expert computation includes activation quantization and
combine; shared gate/sigmoid/multiply/add is a separate fused component.
Sorting is distinct from expert GEMMs. These runtime components are not all
one-to-one matches for Frontier's existing operator families.

This is a **kernel-only diagnostic dataset**, not a native CSV export. Eager
traces must not be relabeled as CUDA-event profiles. Reduction samples include
rank-arrival skew and are not isolated bandwidth measurements. Work outside the
decoder includes preparation, embedding, final norm, logits and sampling; it
is not a measured full-forward residual. Expert-routing vectors are not yet
captured, so these samples do not establish EP or multi-node scaling.

## Separate eager operator events

Add `--operator-event-layers 0 3 --operator-event-repetitions 5` to the capture
command to measure representative GDN and full-attention layers on the real
loaded checkpoint. After the normal baseline and optional trace pass for each
shape, the adapter temporarily wraps selected model call boundaries and runs
separate eager-prefill-only passes **without Kineto**. Existing decode graphs
and model tensors are unchanged, and callables are restored afterwards.

`operator-batches.jsonl` contains the separate timed TP cohorts;
`operator-events-rankN.jsonl` contains nested scope IDs, parent IDs and inclusive
GPU-event milliseconds. Parent and child durations must not be added together.
Scopes include decoder layers, GDN/attention, normalization, shared experts,
router/top-k, routed experts and attention/MoE boundaries containing reductions.
The files are diagnostic inputs pending mapping/coverage checks, not automatic
native profile replacements. Compare these passes against matching baseline
prefills with `summarize_profile_perturbation` before calibration.

The eager observer deliberately does not time operators inside existing decode
graphs. The tested ROCm PyTorch runtime rejects its `external=True` timing events
during HIP capture. GPU-only Kineto also removes the CPU batch markers and does
not eliminate graph-tracing perturbation. The separate HIP-node path below
avoids these two mechanisms; neither observer silently falls back to eager decode.
Shared/cached module objects are attributed through their active decoder scope,
and every selected scope must execute exactly once in each measured prefill.
These selected-layer samples do not establish coverage of all expert routing
distributions or first-layer versus later-layer initialization costs.

## Separate full-decode HIP graph events

Add `--graph-event-layers 0 3 --graph-event-repetitions 5` to recapture the loaded
model's full decode graphs after all baseline, trace and eager-event passes.
The adapter uses public HIP event-record nodes and capture dependencies,
without Kineto or PyTorch external events. Python wrappers run only during
capture and are restored before measurement. Event handles remain alive for
the worker's graph lifetime; no persistent SGLang source or model weights change.

The default scopes are input normalization, the GDN/full-attention mixer,
attention reduction plus MLP normalization, and MLP including TP reduction.
These scopes are disjoint. Nested timers are rejected because interior event
nodes would add overhead inside the parent span. Custom disjoint leaf scopes
can be selected with `--graph-event-components`; they do not automatically
cover work between those call boundaries.

`graph-batches.jsonl` contains separate profiled TP cohorts, while
`graph-events-rankN.jsonl` retains batch/rank/layer/component identity, physical
graph size, capture ID and `HIP_GRAPH_EVENT` milliseconds. These are event spans,
**not** `KERNEL_ONLY` profiles. A logical batch of 20 using a captured graph of
24 must resolve to physical size 24; missing graphs and multiple capture variants
for the same physical size fail admission.

Use `--freeze-decode-inputs` to make subsequent baseline and diagnostic passes
consume the first baseline repetition's token trajectory for each workload.
This is an explicit teacher-forced timing experiment, not free-running serving.
Tensors are prepared outside measured steps; only trajectory hashes are written.
The final generated tokens are not part of the trajectory because no decode
consumes them. The timing quality check requires identical trajectory hashes
as well as matching step and shape. Fixed inputs remove feedback from changed
greedy outputs, but do not guarantee identical activations or expert routing.

```bash
python -m frontier.validation.graph_event_report \
  --capture-dir /experiments/mi355x/capture-graph-001 \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --output /experiments/mi355x/capture-graph-001/graph-scope-summary.json
```

The report requires complete selected-scope coverage on every decode batch and
TP rank, matching measurement families and physical sizes, and a baseline
perturbation check. It preserves rejected samples and keeps ranks/layers
separate. Summing per-component rank maxima is not a measured critical path.
Even when whole-forward latency passes the check, endpoint overhead can bias
small scopes. This report therefore does not admit native CSV export or claim
full-model correlation. In particular, the coarse MLP/reduction scopes must
not be split into invented compute and communication costs or extrapolated to EP.

For a direct diagnostic of the same decoder boundary used by
`SGLangCostContract`, add `--graph-decoder-event`. This inserts only one HIP
event pair: the start is immediately before layer 0 and the end immediately
after the final decoder layer. Embedding and the final residual/norm, logits,
sampling and host work remain outside the span. Layer-level graph scopes are
optional and should normally be omitted from this run so their additional
event nodes do not enter the whole-decoder measurement.

```bash
# Capture with the ordinary static-batch arguments plus:
--freeze-decode-inputs --graph-decoder-event --graph-event-repetitions 5

python -m frontier.validation.graph_event_report \
  --capture-dir /experiments/mi355x/capture-decoder-001 \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --decoder \
  --output /experiments/mi355x/capture-decoder-001/decoder-event-summary.json
```

The report requires exactly one decoder span per decode batch and TP rank,
checks the captured graph and physical padding identity, and preserves the
per-rank durations rather than summing them. It also reports the same-rank
forward-minus-decoder gap and requires the independent whole-forward
perturbation audit. This remains an in-situ diagnostic: it is not exported as
an isolated primitive profile and is never used to fit a correction factor.

## Expert routing and native boundary admission

Add `--routing-all-layers --routing-repetitions 3 --freeze-decode-inputs` to
capture the expert selections from every decoder layer, or select a subset
with `--routing-layers 0 3`. This is limited to the standard top-k output,
separate shared experts, one stream and TP-only execution without expert
placement remapping. An independent routed-expert capturer must be disabled.

After the untouched baseline, the adapter recaptures graphs retaining references
to the router's actual expert-ID and weight tensors. It adds no counter/copy
kernels to graph capture. Necessary device-to-host snapshots occur after each
measured step; CPU validation, hashes, histograms and compression wait until the
whole repetition ends. Retaining tensors can affect graph-pool allocation and
the snapshot work can still disturb subsequent steps, so this pass remains
profiled and must pass the baseline perturbation check before calibration.

`routing-batches.jsonl` is the routing-only timing ledger;
`routing-routing-rankN.jsonl.gz` stores its per-layer expert counts and hashes.
With `--graph-event-layers`, a subsequent pass captures routing and graph-event
timings together, storing `graph-routing-rankN.jsonl.gz`. Both diagnostic passes
use the same fixed input trajectory as the original baseline.

Records retain separate logical and padding expert-count vectors, per-token
assignment/set hashes, weight hashes and positive-weight slot counts. Valid
IDs with zero weight remain counted as selections: weights alone do not prove
that the backend prunes sorting or GEMM work. Padding sentinels are allowed
only with zero weights; live tokens must select distinct valid experts.

```bash
python -m frontier.validation.routing_report \
  --capture-dir /experiments/mi355x/capture-routing-001 \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --native-features-output /experiments/mi355x/routing-native-features.jsonl.gz \
  --output /experiments/mi355x/routing-summary.json
```

The report requires complete selected-layer coverage for every decode batch and
TP rank. It checks graph/model identity, reports TP routing disagreement and
compares repetitions and matching routing-only/timed passes. Per-token hashes
distinguish different assignments even when the aggregate histograms agree.
Features reuse Frontier's existing `EPLaneWorkload` and `MoELoadImbalanceInput`
for observed EP=1 loads. Logical and physical features are exported separately
on rank 0; TP rank counts are never added together. This exports workload
features, not GEMM timings or a fitted latency model, and does not establish EP
scaling. Expert weights/IDs themselves and prompt/output text are not written.

`frontier.validation.runtime_mapping` audits each SGLang boundary against native
operator registries and the active architecture profile:

```bash
python -m frontier.validation.runtime_mapping \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --layer 0 --phase decode \
  --measurement-type HIP_GRAPH_EVENT \
  --components gdn mlp_including_tp_reduce \
  --output /experiments/mi355x/runtime-map.json
```

The map keeps composite/partial scopes, missing gates, fused residuals and
collectives explicit. Later input norms include the preceding layer's FFN
residual add. The runtime reduces shared+routed MoE output once; that duration
must not be assigned to both native reduction fields. The generic attention
post projection lacks output gating, and fused shared gating/addition has no
native operator. Unknown scopes and layer/phase mismatches fail admission.
No simulator costs or default numerics change through this audit, and no
`HIP_GRAPH_EVENT` CSV import is admitted by relabeling the measurement family.

## GDN held-out validation

The first available component predictor is GDN. The following imports batches
16 and 32 into Frontier's existing kernel-only GDN schema and predicts batch
24 using `ProfiledGDNPredictor`. Calibration and validation sets must be
disjoint, including their physical graph shapes after padding.

```bash
python -m frontier.validation.gdn_correlation \
  --capture-dir /experiments/mi355x/capture-001 \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --device mi355x \
  --calibration-sizes 16 32 --validation-sizes 24 \
  --output-dir /experiments/mi355x/gdn-validation-001
```

Every imported batch must contain exactly the model's number of GDN layers.
The importer rejects incomplete profile boundaries and traces containing
multiple GPU identities. Kernel sums, busy-time union, overlap, gaps and
all-reduce totals remain distinct quantities. GDN error compares the predicted
sum across GDN layers with the held-out measured GDN sum. This is rank-0
component validation, not full-model or end-to-end serving validation.
Both sides of this error calculation are profiled kernel sums. A small held-out
error does not demonstrate agreement with unprofiled decode latency when the
underlying trace fails the matching-step perturbation check.

Projection regressors are fitted separately for prefill, decode and mixed
phases. Matching batch-size sweeps use bounded linear interpolation between
measured endpoints; other unseen shapes use the existing random-forest
fallback. Exact measured rows retain their original lookup values. Sparse
grids do not establish accuracy for extrapolated workloads.

## Native full-model replay

Use the standard `frontier.main` configuration flags after `--`. For the
matching topology, set `replica_config_device=mi355x`,
`replica_config_network_device=mi355x_ubb`, attention/MoE TP=8, EP=1, PP=1,
one replica, block size 1 and `decode_cuda_graph_mode=full_decode_only`.
Pass an explicitly calibrated communication backend. `--audit-only` checks
which compute files are missing without training the predictors.

```bash
python -m frontier.validation.replay \
  --ledger /experiments/mi355x/capture-001/batches.jsonl \
  --manifest /experiments/mi355x/capture-001/manifest.json \
  --output-dir /experiments/mi355x/replay-001 \
  --audit-only -- \
  --simulation_mode offline --sys_arch co-location \
  --cluster_config_num_replicas 1 \
  --replica_config_model_name Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --replica_config_device mi355x --replica_config_network_device mi355x_ubb \
  --replica_config_attn_tensor_parallel_size 8 \
  --replica_config_moe_tensor_parallel_size 8 \
  --replica_config_moe_expert_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --replica_scheduler_config_type sglang \
  --sglang_scheduler_config_block_size 1 \
  --decode_cuda_graph_mode full_decode_only \
  --cc_backend_config_type analytical \
  --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
```

The analytical communication default in this **audit-only** example is not a
calibrated MI355X communication model. Select measured AITER/RCCL profiles or
explicitly calibrated backend parameters before numerical full-model replay.
Removing `--audit-only` requires all active predictor profile families. Eager
prefill consumes CUDA-event profiles; full-graph decode consumes kernel-only
profiles. Existing EP=8 MoE smoke data is not compatible with TP=8/EP=1.

Replay rejects dummy mode and incomplete TP cohorts. Captured latency values
are never passed to the predictor. Predictions use the exact hybrid layer
schedule and are compared with maximum per-rank forward GPU time. Full decode
GDN predictions use captured physical lanes, while logical request progress
remains unchanged. The result is per-batch error, not a reconstructed request
arrival/scheduling timeline.

Native `model_time_ms` currently covers decoder blocks. The captured forward
also includes embedding, final normalization and logits processing. Account
for these boundary operations before treating the full-forward comparison
as numerical parity; the replay summary explicitly labels this limitation.

The opt-in runtime-cost workflow below can collect matching SGLang attention,
GDN, linear/shared-expert, TP=8/EP=1 routed-expert and AITER collective evidence.
These private exact-workload tables do not change the generic simulator. Live
scheduling, mixed batches, CPU overhead, request arrivals and memory admission
remain necessary before reporting serving throughput/latency correlation.

Keep generated profiles, raw traces and experiment reports in the external
experiment directory. Reviewable source changes consist of generic adapters,
schemas, tests and runnable recipes.
## Opt-in SGLang runtime cost adapter

`frontier.runtime_cost.sglang.SGLangCostContract` now composes the supported
Qwen hybrid runtime's physical boundaries into native `ExecutionTime` objects.
It is explicitly constructed by the static validation workflow; registering
the `sglang_runtime` operator family does **not** switch the existing model
architecture, sklearn predictor, serving scheduler or profile CSV defaults.
The first contract is full-graph, non-speculative decode on one node with
TP > 1, EP=PP=DP=1 and separate shared experts. Prefill, partial-stage probes,
overlapping streams, alternative fused communication and distributed EP are
not admitted by this adapter.

The physical `sglang_*` operator map is authoritative. Legacy scalar fields
are compatibility aggregates: for example shared activation and the separate
shared-output gate share the legacy activation carrier, while the combined
shared+routed output reduction uses the MoE TP carrier once. The shared TP
reduction carrier and standalone residual-add carriers remain zero. Later
input norms own the previous layer's FFN residual, and post-attention norms
own the current attention residual. The final FFN residual belongs to the
excluded final norm. Aggregation evaluates every configured layer before
forming native per-layer averages; it never multiplies one observed layer's
costs by model depth.

Each cost query carries a model fingerprint (checkpoint, dimensions, schedule,
quantization), stack identity, layer identity, logical size and **every physical
context length**. Routed sorting and quantization/GEMM/combine queries retain
the canonical EP=1 physical expert-count vector including padding. Planning
rejects zero-weight valid selections until a pruning-aware cost
contract is available; counts alone do not prove the backend's executed work.
Query construction has no observed forward-time or output-logit inputs. The capture
bridge verifies all-rank routing agreement before using rank 0's counts.
Padding sequence lengths include the scheduled query token and must be supplied
explicitly from the active backend's graph contract, not guessed from requests.

For a capture's selected decode batch, write a private coverage plan:

```bash
python -m frontier.validation.runtime_cost \
  --capture-dir /path/to/capture \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --device mi355x \
  --batch-id CAPTURED_BATCH_ID --pass-name routing \
  --output /path/outside/repo/runtime-cost-plan.json
```

For padded batches also pass `--padding-context-lens` with one explicit value
per padded lane. The command writes the exact queries, routing audit and missing
profile coverage. No profile file means an empty coverage table, **not** dummy
timing or a prediction. Add `--profile /path/to/costs.json --predict` only when
a complete compatible table exists.

`ExactRuntimeCostTable` exchanges schema-version-1 JSON with `identity` and
`rows`; each row has the `dataclasses.asdict(CostQuery)` value as `query` and
`dataclasses.asdict(CostEstimate)` as `estimate`. Estimates require the matching
query hash, a strictly positive finite millisecond cost, source provenance,
measurement family and evidence basis. Each coverage-plan entry contains a
canonical `query` plus its `query_key`; the hash belongs to the cost estimate
when building a table row. The strict table performs no fitting,
interpolation, layer tying or extrapolation. Missing, duplicate, cross-stack,
wrong-layer, altered-routing or differently padded rows fail. Exact-table
replay is a composition/coverage check, not held-out generalization evidence.

The cost-provider API allows a separately calibrated model to replace the
exact table later. Full-graph compute requires compatible `KERNEL_ONLY` or
explicitly labeled `HIP_GRAPH_REPLAY` isolated-compute evidence; collectives require independently calibrated
collective evidence. Eager `CUDA_EVENT` values cannot price this graph path.
In-situ kernel sums and `HIP_GRAPH_EVENT` scopes require an explicit
`in_situ_diagnostic` basis and `--allow-diagnostic`; HIP graph events are never
relabeled as kernel-only profiles. The adapter does not infer profile quality
from a provenance string or erase profiling rejection flags upstream.

Next collect matching fused compute and communication profiles, fit using
calibration workloads only, and reserve whole physical graph sizes for
validation (including their padded logical batches). Observed routing remains
conditioning information, not proof that an unprofiled repetition chose the
same experts. Native decoder cost still excludes embedding, final residual/norm,
logits and host work; it does not establish full-forward or serving parity.

Runnable contract and unchanged-default regression checks:

```bash
pytest tests/unit/test_sglang_runtime_cost.py tests/unit/test_execution_time_op_times.py -q
pytest tests/integration/test_validation_simulator_matrix.py -q
```

### Isolated fused primitives and runtime custom reduction

`frontier.profiling.runtime.sglang_primitives` runs under eight-rank `torchrun`
in the pinned ROCm/SGLang environment without loading a checkpoint. It profiles
plain Gemma norm, fused residual/Gemma norm, the shared-output gate/add,
strided attention-output sigmoid gating plus local row-parallel projection,
and the **actual SGLang TP custom all-reduce**. RCCL fallback is rejected; a
generic `torch.distributed.all_reduce` profile is not substituted. The norm
path is SGLang's Triton implementation selected with AITER enabled.

Example split (use fresh, external output directories):

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m frontier.profiling.runtime.sglang_primitives \
  --capture-manifest /path/to/capture/manifest.json \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --device mi355x \
  --sizes 16 32 --invocations 32 128 --repetitions 20 \
  --split calibration --trace --output-dir /path/outside/repo/primitive-train

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m frontier.profiling.runtime.sglang_primitives \
  --capture-manifest /path/to/capture/manifest.json \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --device mi355x \
  --sizes 24 --invocations 32 128 --repetitions 20 \
  --split validation --trace --output-dir /path/outside/repo/primitive-holdout

python -m frontier.validation.primitive_calibration \
  --calibration-dir /path/outside/repo/primitive-train \
  --validation-dir /path/outside/repo/primitive-holdout \
  --output /path/outside/repo/primitive-report.json
```

Each graph contains independent invocations and independent parameters. This
creates a streaming working set, not repeated reuse of one cached weight.
Mutable inputs are restored outside timing; first/last outputs are checked
against FP32-math references before capture and after replay. Timings bracket
the graph replay and are divided by the invocation count. They are labeled
`HIP_GRAPH_REPLAY`, **not** pure kernel durations, in-situ graph-scope events,
or eager CUDA-event costs. Rank-0 kernel traces run in a separate profiling
pass after all uninstrumented measurements, using representative two-call
graphs of the same callables. This avoids ROCTracer omissions on large graph
batches, but does **not** prove complete timed-graph invocation coverage.
Missing representative trace coverage rejects the run. Numerical checks use
synthetic BF16 data, not checkpoint outputs.

`PrimitiveCalibration` requires complete aligned rank cohorts, at least 20
repetitions, two calibration physical sizes, and at least two graph lengths.
It aggregates max-rank local duration per repetition and retains each rank's
median; this is not a clock-synchronized multi-device makespan. The longest
graph is selected in advance. Default quality limits are 10% for graph-length
sensitivity and `(p90 - p10) / median` spread. Held-out sizes must not overlap
training and use the same graph lengths, producer, hardware, environment and
runtime identity. A 10% held-out error check does not modify the fitted model.

The provider performs bounded linear interpolation in **physical size** only;
all model dimensions (including explicit attention head width), quantization,
TP topology and runtime settings are fixed by identity. Norm costs are tied
across layers by fused-residual ownership, both TP reduction boundaries map
to the same calibrated primitive, and a padded logical batch uses its physical
graph size. Out-of-range queries, unstable primitives and unsupported scopes
fail instead of receiving a zero or generic fallback. The provider's call-time
gate checks calibration quality; the independent validation report is still
required before making any held-out accuracy claim.

This is a **partial** provider: the default primitive list excludes stateful
GDN/attention scopes and route-sensitive MoE scopes. The explicit GDN,
attention and MoE calibrations below can add those boundaries only for their
pinned workload contracts. Primitive-level holdout accuracy does not establish
whole-model cache/overlap behavior, full-forward latency, serving performance
or multi-node scaling. The CLI writes a report, not a complete runtime-cost
table, and does not ingest observed forward times.

```bash
pytest tests/unit/test_primitive_calibration.py tests/unit/test_sglang_runtime_cost.py -q
```

### BF16 shared-expert, router and GDN projection scopes

The same producer also accepts six additional `--primitives` selections:
`shared_expert_gate_up`, `shared_expert_activation`, `shared_expert_down`,
`moe_router_linear`, `gdn_input_projections`, and `gdn_output_projection`.
These use actual SGLang linear/activation implementations with independently
initialized BF16 weights and FP32-math numerical references. They do not
load a model checkpoint or change any generic profiling CSV defaults.

The shared gate/up is a merged column-parallel projection; its SiLU/multiply
activation and row-parallel down projection have separate cost boundaries.
The router remains replicated across TP ranks. GDN input projection invokes
SGLang's sequential ROCm helper with both QKVZ and B/A merged projections
sharing one hidden input. The GDN output scope applies SGLang's gated RMS norm
and then the local row-parallel linear; it does **not** include recurrent state
work or an all-reduce. All reductions and shared-output gate/add costs remain
separate.
The producer rejects native activation fallback, unsupported alternate-stream
GDN execution, and model configurations quantizing these dense projections.

For the supported model at TP=8, per-rank weight shapes are:

| Scope | Local BF16 weights (output, input) |
|---|---|
| Shared gate/up | (512, 8192) |
| Shared down | (8192, 256) |
| Replicated router | (512, 8192) |
| GDN QKVZ and B/A | (4608, 8192) and (32, 8192) |
| GDN output | (8192, 2048) |

Shapes are derived from the model configuration, not these example values.
The row-level `dense_spec` records input/output widths, merged partitions,
per-rank weight shapes, dtype, the gated-norm input transform and
excluded-reduction ownership. Consumers
validate its structure and require agreement across ranks, calibration sizes,
graph lengths and independent validation. A kernel-method label alone is not
enough to erase a layout change.

Use the earlier two-run recipe with an explicit primitive list to collect only
these new scopes:

```bash
# Append to BOTH calibration and independent validation producer invocations:
--primitives shared_expert_gate_up shared_expert_activation shared_expert_down \
  moe_router_linear gdn_input_projections gdn_output_projection
```

Keep training sizes and held-out physical sizes disjoint; use the same producer
snapshot for both sides of each experiment. No tuning or correction from the
held-out measurements enters `PrimitiveCalibration`. Native cost composition
still fails until **all** decoder scopes have compatible evidence. In
particular, GDN/attention core, routing/top-k/sorting and routed MXFP4 experts
are not supplied by this extension.

```bash
pytest tests/unit/test_sglang_dense_primitives.py tests/unit/test_primitive_calibration.py -q
```

### Stateful SGLang GDN decode core

`gdn_core_decode` profiles the complete native Qwen GDN boundary between the
two input projections and gated RMS norm. It includes QKV/B/A split and
materialization, packed QKV concatenation, BF16 causal-convolution state update,
the FP32-state packed recurrent kernel, and materialization of the strided Z
gate view. The output gated norm remains in `gdn_output_projection`.

Each captured invocation owns independent convolution and recurrent state.
Correctness checks compare output, convolution state, sampled recurrent heads,
and the materialized Z gate against independent FP32 math before capture and
after replay. The `gdn_core_spec` records per-rank heads, projection widths,
state shapes/dtypes, kernel width, padding sentinel and boundary ownership.

GDN padding differs from dense padding: physical projection/transform work
still executes, while padded recurrent lanes carry cache index `-1` and must
not mutate a request state. Supply logical sizes alongside physical sizes:

```bash
# Example calibration endpoints with four padded graph lanes.
--sizes 16 32 --logical-sizes 12 28 \
--primitives gdn_core_decode gdn_output_projection

# Independent interpolation holdout with the same padding ownership.
--sizes 24 --logical-sizes 20 \
--primitives gdn_core_decode gdn_output_projection
```

Calibration requires a constant `physical_size - logical_size`; queries and
held-out rows with different padding ownership are rejected. This is a bounded
decode calibration for one pinned identity, not a GDN prefill model, generic
batch extrapolation, checkpoint/full-forward fit, or serving-speedup claim.

```bash
pytest tests/unit/test_sglang_gdn_primitives.py tests/unit/test_primitive_calibration.py -q
```

### Stateful SGLang attention decode boundaries

Four opt-in primitives cover the native Qwen attention path before the existing
`attention_post` gate/output-projection scope:

- `attn_pre_proj_qknorm`: TP-local `QKVParallelLinear` plus fused Q/K Gemma RMS
  normalization and extraction of the per-head output gate;
- `attn_rope`: SGLang rotary embedding with the model's partial rotary factor;
- `attn_kv_cache_write`: SGLang's page-size-one cache store; and
- `attn_decode`: AITER ragged paged attention with its model-wide partition
  specialization and NHD BF16 cache layout.

The recorded `attention_decode_spec` pins local Q/KV head counts, head and
rotary dimensions, QKV/gate projection shape, cache dtype/layout/page size,
RoPE parameters, maximum position and boundary ownership. AITER's ragged
`kv_indptr` is constructed explicitly as int32; integer promotion would violate
the native ABI and is not accepted as equivalent metadata.

Attention rows require both logical sizes and active context lengths. The
physical context vector is a uniform logical prefix followed by one-token
padding lanes. Calibration, validation and query-time use must agree on the
active context, padding context and padding count:

```bash
# Example calibration endpoints with four graph-padding lanes.
--sizes 16 32 --logical-sizes 12 28 \
--active-context-lengths 1029 1029 \
--primitives attn_pre_proj_qknorm attn_rope attn_kv_cache_write attn_decode

# Independent physical-size holdout.
--sizes 24 --logical-sizes 20 --active-context-lengths 1029 \
--primitives attn_pre_proj_qknorm attn_rope attn_kv_cache_write attn_decode
```

Very small boundaries can need longer independent-invocation graphs so both
graph lengths are in the same streaming working-set regime. Collect those
primitives in separate cohorts when necessary, then combine only rows with
identical producer source, hardware, environment and runtime identity.
Graph-length choice remains subject to the same 10% sensitivity gate; it must
not be chosen from held-out error. These synthetic BF16 profiles are bounded
decoder costs, not checkpoint-output, full-forward, serving or multi-node
validation.

```bash
pytest tests/unit/test_sglang_attention_primitives.py \
  tests/unit/test_primitive_calibration.py -q
```

### Qwen MoE routing and exact-route MXFP4 experts

The replicated router projection uses the dense primitive above. Select
`moe_routing_topk` explicitly to profile the following AITER fused top-k
boundary over the same physical-size calibration and holdout split. Its shape
contract pins 512 experts, top-k 10, BF16 logits, FP32 output weights, int32
expert IDs, unrenormalized softmax scoring and exclusion of the router linear.
The eager correctness oracle checks the selected expert set and each gathered
softmax probability; only the native fused call is captured and timed.

```bash
# Add to the calibration and independent validation commands, with disjoint sizes.
--primitives moe_routing_topk
```

Sorting and expert execution cannot be inferred from physical batch size alone.
Their queries include the complete 512-entry physical expert histogram for an
audited captured layer, including graph-padding routes. Generate the private
query plan with `frontier.validation.runtime_cost`, then profile each exact
route on all eight ranks:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m frontier.profiling.runtime.sglang_moe_routes \
  --capture-manifest /path/to/capture/manifest.json \
  --query-plan /path/outside/repo/runtime-cost-plan.json \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 --device mi355x \
  --primitives moe_sorting moe_experts_quant_gemm_combine \
  --invocations 32 128 --repetitions 20 --trace \
  --output-dir /path/outside/repo/routed-profiles

python -m frontier.validation.routed_primitive_calibration \
  --profile-dir /path/outside/repo/routed-profiles \
  --query-plan /path/outside/repo/runtime-cost-plan.json \
  --primitives moe_sorting moe_experts_quant_gemm_combine \
  --report /path/outside/repo/routed-report.json \
  --table /path/outside/repo/routed-table.json
```

The producer deterministically reconstructs valid, distinct per-token top-k
assignments from each exact histogram and verifies that Opus sorting reproduces
the packed token/slot and expert-block ownership. Sorting is untimed when
profiling `moe_experts_quant_gemm_combine`; that boundary contains both dynamic
MXFP4 activation-quantization stages, both AITER MFMA expert GEMMs, activation
and combine, but excludes sorting and TP reduction.

Expert graphs allocate independent TP-local packed tensors per invocation.
Zero-valued synthetic MXFP4 weights preserve the exact tensor extents, strides,
shuffle marker and memory traffic while providing an exact-zero output oracle.
They do not load checkpoint weights and are not a numerical checkpoint-output
test. Representative traces must contain both expert GEMM stages. Because these
graphs have a large streaming working set, profile and release one layer at a
time and verify available HBM before choosing graph lengths.

`ExactRoutedCalibration` requires complete aligned eight-rank cohorts, two or
more graph lengths, at least 20 repetitions, matching kernel evidence and a
10% graph-length-sensitivity/spread gate. It selects the longest graph length
in advance and returns costs only for the identical query key, layer and route
histogram. There is deliberately no cross-route interpolation, layer tying or
held-forward correction. Merge admitted rows into a private exact table only
after `ExactRuntimeCostTable.audit` reports complete coverage; generated plans,
profiles, tables, traces and performance results stay outside the repository.

For an exact post-prediction decoder check, the source capture must combine
`--graph-decoder-event` with all-layer graph-pass routing. Build and profile the
runtime-cost plan for one selected `graph-batches.jsonl` batch, then produce a
complete prediction with `frontier.validation.runtime_cost --pass-name graph`.
Correlate that prediction only with the decoder event and routes from the same
batch:

```bash
python -m frontier.validation.runtime_cost_correlation \
  --capture-dir /path/to/combined-decoder-and-routing-capture \
  --prediction /path/outside/repo/exact-runtime-cost-prediction.json \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --output /path/outside/repo/exact-decoder-correlation.json
```

The command independently reconstructs the runtime identity and exact query
keys, requires complete fixed-input TP graph coverage, audits all-layer routing
agreement, validates one decoder event on every rank, and enforces both the
selected-batch and capture-wide perturbation gates before calculating error.
The observation remains post-prediction: it is never used to construct queries,
admit profiles, fit or correct costs. The result does not claim full-forward or
serving parity. Because fixed token inputs do not freeze expert routes, other
graph repetitions are different conditioning points and must be planned,
profiled and reported separately rather than pooled into an exact-route error.

```bash
pytest tests/unit/test_sglang_moe_primitives.py \
  tests/unit/test_sglang_runtime_cost.py \
  tests/unit/test_runtime_cost_correlation.py -q
```
