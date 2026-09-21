## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-09 | Disambiguated full-scope 116.210 ms and reduced-communication 86.412–86.444 ms spans from the canonical 78–79 ms batch-only reference. |
| 2026-09-08 | Recomputed same-run communication and forward ledgers; confirmed large post-MoE TP spans and correlated participant-arrival effects before considering new measurements. |
| 2026-09-08 | Applied D020: retain ideal EP, explicitly defer naive protocol modeling; link completed primitive measurements and active heldout sweep. |
| 2026-09-08 | Proved the naive DP/TP reduction decomposition, corrected the actual custom-AR selection claim, audited the 50-us term, and prepared the exact-group microbenchmark. |

# D019 communication RCA and proposed correction

## Current RCA: larger compute predictions and smaller batch span

This section supersedes the earlier response's priority ordering. The user asks why Frontier compute predictions tend to exceed measured vLLM compute scopes while the integrated Frontier forward is about 20 ms shorter. Investigation is limited to existing artifacts and source. No new GPU execution or model correction was performed.

**Evidence:** the previous response omitted an already measured major component: post-MoE TP all-reduce. The compute table is a partial inventory drawn from separate attention/MoE/detail runs, not the complete additive ledger of the 78-79 ms batch-only reference. Its comparison signs cannot determine the sign of the full batch difference.

## Span identity clarification

The previously cited `116.210144043 ms` and `86.412033081–86.443519592 ms` values are both observed CUDA-event spans, but they belong to different diagnostic executions and have different probe scopes. The full-scope RCA execution is batch 4706 with selected operator scopes and profiler tracing enabled; the reduced communication execution is batch 4250 with only embedding, attention TP AR, and post-MoE TP AR scopes and runtime metadata disabled. Their outer intervals therefore cannot be subtracted, added, or treated as two measurements of one forward.

The `116.210144043 ms` value is the true outer span for that full-scope diagnostic execution, and its selected non-overlapping ledger closes internally. The `86.412033081–86.443519592 ms` values are the true outer spans for the four ranks of the reduced communication execution; the selected communication rows are a partial inventory and the remainder is intentionally unpartitioned. Full instrumentation, profiler activity, host submission gaps, participant arrival/wait, and independent batch/runtime variation can change the outer interval. The approximately 30 ms difference is therefore not evidence of one fixed missing operation.

For the current calibration gate, the canonical batch-only references remain `79.307357788 ms` (before) and `78.118782043 ms` (after). Those runs have no selected per-operator probes and are the references used by the integrated CUDA comparison. They are the closest available clean bracket, while the 116 ms and 86 ms observations remain diagnostic evidence. No production/uninstrumented absolute truth has been claimed beyond that bracket. Consequently, neither diagnostic span closes the operator gate or the CUDA gate.

### Actual Frontier accounting

The integrated `query_receipt.json` contains eleven distinct compute model values. Taking each independent model once per layer, multiplying by 48, and adding the actual reviewed communication costs reconstructs the observed boundary:

| Component | 48-layer ms |
| --- | ---: |
| Compute/memory predictions, including repaired GG and sum | 50.177408283954 |
| Attention TP AR | 4.762262366614 |
| Ideal EP dispatch | 2.125341866667 |
| Ideal EP combine | 2.125341866667 |
| Sum | 59.190354383901 |
| Observed DES boundary | 59.190354383901 |

The arithmetic difference is below 1e-10 ms. This rules out an omitted listed prediction or a layer multiplier error in this particular endpoint reconstruction. It does not establish that all physical vLLM work has a model counterpart.

### Large communication events exist in the retained raw data

Source: `first-batch-op-rca/communication/summary.json` and its four raw DP0 `server.ops` files, batch 4250, request `cmpl-pf4096_dc1024:0-0`, 4096 prefill tokens. Independently re-read all 97 records per rank: one embedding AR, 48 attention AR, and 48 post-MoE AR. Sums, unique sequences, and counts match the stored summary.

| TP rank | Attention AR sum ms | Post-MoE TP AR sum ms | Largest single post-MoE AR ms | Outer batch ms | Outer minus post-MoE AR ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.537248015 | 4.792544022 | 0.102399997 | 86.412033081 | 81.619489059 |
| 1 | 5.649951987 | 25.384415895 | 0.777888000 | 86.437950134 | 61.053534240 |
| 2 | 5.708767995 | 28.413279966 | 0.770143986 | 86.414657593 | 58.001377627 |
| 3 | 5.644864000 | 27.393568158 | 0.705856025 | 86.443519592 | 59.049951434 |

The 25-28 ms values are sums of 48 calls on each rank, not one 20-30 ms call. Ranks are parallel and are never summed. This mode does not record dispatch/combine/compute, so the final column is an unpartitioned remainder rather than compute or CPU time.

### Complete same-run scope ledger

Use the existing full-scope run, batch 4706, DP0/TP0. Keep only the outer even-sequence dispatch/combine scopes, exclude nested `add`, and retain attention output projection together with its unselected AR child. Raw `operators/op_trace_normalized.jsonl` reproduces:

| Disjoint accounting bucket | ms |
| --- | ---: |
| Selected compute scopes excluding attention output projection | 41.336256027 |
| Attention output projection including attention TP AR | 17.231616005 |
| MoE dispatch outer scope | 6.981152035 |
| MoE DP combine outer scope | 4.196768016 |
| Post-MoE TP AR | 37.563167766 |
| Embedding TP AR | 0.222335994 |
| Outer interval outside these selected scopes | 8.678848200 |
| Actual outer batch | 116.210144043 |

These buckets exactly reconstruct this run's outer span. The remainder includes unselected work and gaps, including expert sum that was not selected in this earlier scope inventory. It is not measured CPU time. Neither 116.210 ms nor its buckets are substituted for the later 78-79 ms reference. The table demonstrates why a partial compute comparison can coexist with a larger actual forward.

### Participant arrival contributes to the long communication scopes

**Evidence:** in the communication-only run, TP2 minus TP0 post-MoE AR is 23.620735943 ms, but their outer spans differ by only 0.002624512 ms. Thus the difference is almost entirely relocated to the rest of each rank's timeline. A uniform extra transport cost charged to every rank cannot explain this distribution.

The existing kernel run supplies a direct same-operation timeline. All four DP0 batch-4734 traces have 3077 matched kernel launches, no missing correlations, and `baseTimeNanoseconds=1782967788000000000`. Select post-MoE AR ordinal 28 on all four TP ranks; each rank has exactly 48 such kernels. Times below are relative to the earliest device start for that operation, and apply only to this profiler run:

| TP rank | Host launch start ms | Device start ms | Device end ms | Kernel duration ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | -1.228062 | 0.008295 | 1.763213 | 1.754918 |
| 1 | 1.589029 | 1.667748 | 1.764548 | 0.096800 |
| 2 | -1.316407 | 0.000000 | 1.764004 | 1.764004 |
| 3 | 0.684704 | 0.747948 | 1.763439 | 1.015491 |

Early ranks have entered NCCL while TP1 has not submitted that operation; all four finish within about 1.4 us of one another. This establishes late participant submission in the captured run and supports wait-dominated inflation of early-rank NCCL activity. It does not establish the clean-run delay or identify why the host was late.

Whole-trace corroboration: TP0 has 143.099283 ms recorded device activity and 8.461469 ms idle between activities; TP1 has 54.768490 ms activity and 94.431108 ms idle. The respective envelopes are 151.560752 and 149.199599 ms. Of TP1's idle, 74.125946 ms precedes submission of the next recorded device activity. These are profiler-run quantities, not a CPU addition to the 78-79 ms reference.

### Source mapping and what Frontier represents

`vllm/model_executor/layers/fused_moe/layer.py:1852-1861` executes DP combine before post-MoE TP reduction; `:1599-1613` names that TP-group call `expert_parallel_allreduce`. `all2all.py:28-66` implements hidden/router multicast and DP-global reduction plus slicing. The real-lane post-MoE tensor is 16 MiB and takes the TP4 PyNCCL path, rather than an EP8 all-reduce inferred from its label.

Frontier's `ExecutionTime.get_single_layer_moe_combine_time` consumes ideal EP return; its post-combine accessor has only the retired DP-output and residual fields. This first forward has no standalone post-MoE TP4 charge. However, ideal EP return already represents complete expert-result aggregation. Therefore the supported conclusion is an unmodeled physical protocol decomposition and unqualified participant-timing costs, not a missing mathematical result that can be repaired by adding 28 ms to the existing ideal return.

Frontier already synchronizes modeled EP arrivals using `max(...)` (`scheduler/utils/ep_dispatch.py:105`; combine tracing also validates the latest lane arrival). It cannot reproduce unmodeled host submission skew solely from balanced expert-work predictions. Absence of a blanket straggler mechanism is not the root cause.

### Ranked result and remaining uncertainty

1. High-confidence evidence: major communication/wait intervals were absent from the previous compute-focused explanation. They are observed, not hypothetical missing measurements. The apparent sign contradiction is resolved by the complete ledger.
2. High-confidence source finding: the selected vLLM physical MoE return is DP combine plus TP4 reduction; Frontier retains the ideal EP return under D020. Their phase costs and timing boundaries are not interchangeable.
3. Supported mechanism in the profiler run: delayed participant submission shifts elapsed time between a slow rank's gaps and peers' NCCL scopes. A large NCCL event is not equivalent to pure transfer service time.
4. Still unknown: the exact partition of the latest 18.928428-20.117003 ms reference gap into physical-protocol cost, participant waiting, unselected forward intervals, and probe-induced effects. No same-run complete partition exists for the 78-79 ms reference.

The next discussion should focus on the actual dispatch -> expert compute -> DP combine -> TP4 return critical path and its participant arrival boundaries. Clean/diagnostic reconciliation is relevant only to quantify how much of these identified major effects persists in the reference. It is not a prerequisite to noticing the already observed large communication terms. D020 remains unchanged; no protocol implementation or residual constant is authorized by this analysis.

Read-only verification environment: conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13. Root independently recomputed the delegate's raw communication sums/counts/maxima, normalized full-scope ledger, integrated predictor sum, common profiler time base, and layer-28 launch/device timeline. No runtime simulation, GPU measurement, source edit, or data replacement was performed.

**Active status — D020:** YC explicitly retains Frontier's current ideal EP abstraction. The `vllm_naive` selector, broadcast implementation and physical-DP protocol/schema changes are **deferred optional work**, not pending decisions in this task. The historical protocol proposal below is preserved only as research context and grants no implementation authorization.

Fresh eight-rank primitive measurement and validation are complete: `analysis/d019-communication-measurements.md` and its JSON sibling record all64 operation rows and448 samples, correct groups/bytes, real-lane PyNCCL versus dummy custom selection, and a99-us real TP4 primitive median. Verified baseline benchmark commit: `08f16e58`. Production backend parameters remain unchanged; the current50-us-per-step term is contradicted by the actual primitive duration.

The independent8/12/16/24/32-MiB TP4 sweep has now completed: `d019-communication-sweep.md` and JSON record560event samples,40complete size/kernel joins and a16-MiB heldout PASS. With existing bandwidth/efficiency/latency fixed, a scoped AR-field candidate of4.384788772964477us/step replaces the invalid50-us value; direct estimator evaluation preserves idealEP8all-to-all exactly. This is an aggregate fixed-TP4eager floor, not separately measured CPU launch. The older in-context AR comparison remains14.0–16.6%low, explicitly unresolved. Active next step: independent candidate/scope review, root case-config application if accepted, and fresh first-forward validation. No production parameter has been changed by laneC.

## Established results

1. The first real DP0 attention TP allreduce is **PyNCCL**, despite custom-allreduce being enabled. The BF16 `[4096,2048]` tensor is 16 MiB. Pinned `CustomAllreduce` defaults to an 8-MiB maximum and accepts only strictly smaller inputs. Its source rejects the 16-MiB request; `CudaCommunicator.all_reduce` falls through to PyNCCL. Existing first-forward kernel records independently show `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`. The dummy DP1 `[1,2048]` tensor is 4096 bytes and is custom-eligible when its communicator is enabled. A flag is not evidence of which implementation executes.
2. The naive MoE combine is a **DP2 allreduce over global rows, local-row slice, then TP4 allreduce**. Together they aggregate all eight expert ranks. Frontier's current ideal EP8 return phase already represents returning all routed expert contributions to their source; appending a TP4 reduction while retaining that complete ideal phase duplicates the mathematical aggregation. The correct protocol replacement must substitute the entire naive dispatch/combine decomposition.
3. The 17.899440-ms attention-AR prediction contains **14.400000 ms from the configured per-step launch term alone**: 48 layers × 6 ring steps × 50 us. The remaining analytic transfer plus link latency is 3.499443 ms. The observed lower-density first-real-lane AR is 5.537248–5.708768 ms across its four ranks. This proves which model term dominates; it does not prove that setting the term to zero yields a physically calibrated model.

## Pinned sources and exact groups

vLLM source root: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, source reviewed at `8453dd342c6aa2721aaf4b410998aab2f38bc2ec` before lane A's D019 instrumentation changes.

- `vllm/distributed/parallel_state.py:1107–1182`: rank layout `ExternalDP × DP × PP × TP`; TP groups `[0,1,2,3]`, `[4,5,6,7]`; DP groups `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`; EP group `[0,1,2,3,4,5,6,7]`.
- `vllm/distributed/device_communicators/all2all.py:28–66`: actual buffer allocation/copy, four broadcasts per dispatch, global DP allreduce and local slicing.
- `vllm/model_executor/layers/fused_moe/layer.py:1852–1861`: combine first, then `maybe_all_reduce_tensor_model_parallel` when `reduce_results` is true.
- `vllm/distributed/device_communicators/custom_all_reduce.py:55–59,221–234`: default maximum 8 MiB, byte alignment/contiguity checks, strict size threshold. The H200 architecture table does not raise this default: constructor uses `min(table_limit,max_size)` only with symmetric memory enabled.
- `vllm/distributed/device_communicators/cuda_communicator.py:24–32,105–136`: custom AR enabled only for TP-named groups; DP uses PyNCCL. `VLLM_ALLREDUCE_USE_SYMM_MEM` defaults false in `envs.py:1270` and remains false for the frozen case.
- `vllm/distributed/device_communicators/pynccl.py:110–132`: one `ncclAllReduce` API call on the current stream with an output allocation. NCCL's ring communication rounds are not six independent Python launches.

All message sizes below assume the fixed BF16 case; exact runtime scope metadata remains lane A's identity evidence. Global MoE rows are `4096+1=4097`, not 8192. Routing assignments are a separate dimension and are not used to inflate the naive hidden buffer.

| Runtime phase | Group | Tensor or movement | Bytes | Frequency per layer |
| --- | --- | --- | ---: | ---: |
| Attention output AR, real lane | TP `[0,1,2,3]` | `[4096,2048]` BF16 | 16,777,216 | 1 |
| Attention output AR, dummy lane | TP `[4,5,6,7]` | `[1,2048]` BF16 | 4,096 | 1 |
| Dispatch hidden staging | Each rank, local copy | local input into global `[4097,2048]` buffer | DP0 16,777,216; DP1 4,096 | 1 |
| Dispatch hidden broadcasts | Four independent DP2 groups | source0 `[4096,2048]`, then source1 `[1,2048]` | 16,777,216 then 4,096 | 2 |
| Dispatch router staging | Each rank, local copy | local logits into global `[4097,128]` buffer | DP0 1,048,576; DP1 256 | 1 |
| Dispatch router broadcasts | Four independent DP2 groups | source0 `[4096,128]`, then source1 `[1,128]` | 1,048,576 then 256 | 2 |
| Combine DP reduction | Four independent DP2 groups | global partial expert output `[4097,2048]` | 16,781,312 | 1 |
| Combine slice | Each rank | retain its source-DP rows | view; no copy in this source | 1 |
| Post-combine TP reduction, real lane | TP `[0,1,2,3]` | local `[4096,2048]` partial output | 16,777,216 | 1 |
| Post-combine TP reduction, dummy lane | TP `[4,5,6,7]` | local `[1,2048]` partial output | 4,096 | 1 |
| Embedding TP reduction | Same local TP groups | corresponding local hidden tensor | same real/dummy byte sizes | once per forward |

## Why DP combine plus TP reduction is one global expert aggregation

Let `z[d,t](x)` be the weighted sum of contributions from the 16 experts owned by physical rank `(d,t)` for a global token `x`. It is zero when those experts receive no assignment for that token. After multicast, all eight physical ranks can compute their owned expert contributions for every global token.

The DP combine for a fixed TP coordinate produces `u[t](x) = z[0,t](x) + z[1,t](x)`. Each rank then retains rows belonging to its source DP lane. TP reduction produces `y[d](x) = sum_t u[t](x) = sum_t sum_d' z[d',t](x)` for those local rows. Every expert partition contributes exactly once. There is no extra intra-expert TP partition because the configured MoE TP is one.

Frontier's existing routed all-to-all return plus destination expert sum represents this same final sum through routed data exchange. It has different message volumes, network rounds, staging and tensor replication. Therefore:

- Retain the existing ideal return and append TP4: wrong decomposition, would charge two complete aggregation protocols.
- Replace ideal return with DP global AR + local slice + TP local AR: correct naive decomposition. Preserve named parent intervals so DP and TP child costs are not also charged as independent complete EP return phases.
- Interpret vLLM's post-combine TP as Frontier's existing `moe_tensor_parallel_allreduce` with MoE TP=1: wrong group ownership. It is the attention TP group.

The microbenchmark validates this algebra numerically using rank-valued inputs: DP reduction gives each DP pair's rank sum, then TP reduction gives the sum over the eight ranks (36) for every local output element. This checks operation ownership independently of timing.

## Existing Frontier model and numerical decomposition

`frontier/operators/families.py:376–388,550–580` gives both named EP phases the same ideal all-to-all payload: local routed assignments × hidden size × dtype bytes, EP8 group. The first balanced 4096-token case gives 4096 assignments per EP rank, 16-MiB logical outgoing payload, and `(7/8)×16 MiB` physical modeled bytes per rank. Neither the router-logit broadcast nor naive DP/TP hierarchy is represented.

`frontier/cc_backend/backends/collective-sim/python/collective_sim_core/intra_server_model.py:85–106,135–141,190–194` always estimates allreduce with ring bytes and ring steps. Changing the htsim `allreduce_model` string does not change this intra-server estimator. `collective_sim_cc_backend.py` passes the 50-us term from `cc_backend_config.py:288`; the public field explicitly means **per step**.

For the reached TP4 16-MiB shape:

| Component | Formula | One layer ms | 48 layers ms |
| --- | --- | ---: | ---: |
| Transfer | `1.5 × 16,777,216 / (450e9 × 0.8)` | 0.069905067 | 3.3554432 |
| Link latency | `6 × 0.5 us` | 0.003000000 | 0.1440000 |
| Configured launch term | `6 × 50 us` | 0.300000000 | 14.4000000 |
| Current model total | sum above | 0.372905067 | 17.8994432 |
| Zero-launch counterfactual | transfer + link latency | 0.072905067 | 3.4994432 |

The trace's 17.899440 differs by rounded print precision only. This calculation executes the current backend estimator directly; receipt: `analysis/d019-communication-contract.json`. The zero-launch row is a counterfactual, not an applied correction or accuracy result. Actual RING_LL algorithm efficiency, collective submission, allocation and rank arrival waits remain to be measured. Fitting any of those waits into NVLink bandwidth would corrupt the physical model.

## Initial microbenchmark design — subsequently executed and validated

File: `tests/performance/issue26_h200_collective_microbenchmark.py`.

The script initializes the real pinned vLLM parallel groups and `NaiveAll2AllManager`, with the same eager mode, custom-AR setting and DP token counts. It measures eight operations: runtime TP AR, direct PyNCCL TP AR, DP global AR, hidden multicast, router multicast, complete dispatch, complete combine, and combine-plus-post-TP. Multicast uses the actual implementation including its allocation/copy and root-ordered broadcasts. Composition rows overlap and must not be summed; direct primitive rows are controls, not extra runtime work.

Each operation has 20 warmups and seven event-timed blocks of 48 calls. Events are outside the eager block, with one end synchronization and a CPU-group barrier between blocks; no per-call synchronization or per-call event probes. Every rank records its implementation eligibility, groups, exact bytes, versions, NCCL environment, all samples and host submit timestamps. Those timestamps bound submission skew on the same host, not GPU arrival timestamps. Block measurements can include residual host gaps and are not a clean forward or a pure-wire estimate.

Root owns GPU scheduling. Intended command, in the approved H200 runtime, with actual root-managed output directory replacing the placeholder:

```bash
export PYTHONPATH=/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908
export VLLM_ALL2ALL_BACKEND=naive VLLM_FRONTIER_INSTRUMENTATION=0
/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_h200_collective_microbenchmark.py \
  --output "$D019_COMM_OUTPUT"
```

The launcher must retain the approved image, proxy, H200/step_main allocation, original platform NCCL settings and NFS evidence paths. It must disable diagnostic event logging for this primitive run. This initial preparation text predates root-managed GPU execution. The actual first-generation GPU measurements now PASS their identity/correctness checks; see `d019-communication-measurements.md`. Lane C does not launch GPU jobs.

Acceptance: eight rank JSON records; exact TP/DP/EP layouts and bytes; numerical dispatch/combine correctness PASS; positive finite durations; DP0 PyNCCL and dummy-lane actual selection recorded; report all rank distributions without adding parallel ranks; compare the same primitive with lower-density in-context measurements and investigate spread before modifying timing parameters.

## Historical optional naive protocol proposal — deferred by D020

There is **no current config field selecting the MoE communication runtime**. `moe_gating_routing_runtime_path` selects top-k implementation only. `alltoall_model` and `allreduce_model` choose algorithms for an already selected collective, and cannot select the higher-level naive protocol. The scheduler's `vllm_v1` choice is not sufficient because vLLM supports multiple all2all implementations. Inferring naive from model name, EP size, routing distribution or current test case would hard-code an unrelated policy.

Recommended design decision: introduce one declarative MoE communication-runtime selector with existing ideal routed-all-to-all as its default and explicit `vllm_naive` for this case. Keep this separate from `collective_sim` backend selection. A small protocol owner returns named primitive phases and their groups/payloads from the shared forward's source populations; sibling protocols register in one table. Existing named dispatch/combine execution boundaries can remain the consumers. This choice is materially different from globally replacing Frontier's ideal protocol for every user. D020 has resolved the current decision: keep the ideal protocol and defer this optional feature.

Concrete seams and missing capabilities:

| Change | Existing seam | Required scope |
| --- | --- | --- |
| Protocol selector | `ReplicaConfig` near `config.py:1955`; its existing flat-config construction/copy paths | One new typed/validated field; two protocol entries, no inference from labels |
| Protocol construction | `_predict_expert_parallel_phase_operator_times`, `sklearn_moe_execution_time_predictor.py:1577` | Extract bounded protocol logic to a small owner; avoid growing the 3000+ line predictor |
| Source population input | Shared-forward source batches and `LayerEPWorkload` | Must carry each source DP's token count; one local routed-token count cannot recover `[4096,1]` or mixed populations |
| Named collective operators | `operators/families.py`; `CommOperatorSpec` in `operators/spec.py:170` | Existing spec currently accepts AR/AG/A2A/P2P but not broadcast. Add broadcast through the same registry if represented as child ops |
| Broadcast API | `BaseCCBackend.predict_broadcast` already exists; `CollectiveSimCCBackend.predict_broadcast:789` raises `NotImplementedError` | Implement this existing backend API; no new public method is needed |
| Broadcast backend support | `collective_sim_core/schema.py`, `intra_server_model.py`, runner/htsim collective builder | Native broadcast kind/steps/bytes must be supported end-to-end. Runner currently has no broadcast kind. Do not substitute P2P or reinterpret allgather to hide this gap |
| Post-combine TP phase | Existing `ExecutionTime.get_single_layer_moe_post_combine_time:881` and `CommunicationOperatorTimes` | Charge the naive TP child only after replacing ideal combine; do not revive retired DP gather/scatter field names as a shortcut |

The missing broadcast primitive can be implemented independently through the existing backend interface, but its exact algorithm/bytes contract still needs a bounded design across the optional submodule. Only specializing `num_devices==2` to P2P would leave the next supported DP size incorrectly modeled and is not proposed.

Timing correction is separately bounded: after actual PyNCCL primitive measurements establish the role of launch/transfer components, correct the existing launch-accounting contract or its H200 parameter with evidence. Do not change 50 us to a fitted first-batch residual, do not multiply all comm times by a scalar, and do not assume every TP AR is custom. Fresh first-forward and clean-case validation follow the scoped correction.

Active pending tasks: independently review the completed primitive fit and candidate scope; apply only the supported case-local AR correction after review; validate the new full-forward and clean metrics. Primitive heldout and kernel identity checks are complete; the in-context communication gap remains disclosed. Naive protocol/selector/broadcast implementation is deferred and is not an active blocker. Established issues: real-lane PyNCCL selection supersedes earlier custom-AR wording, and the per-step50-us term dominates the current TP overprediction. No numerical correction is claimed.
