## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-08 | Added the bounded source review and decision proposal for 128-expert routing counts. |

# Global MoE Routing Counts Proposal

## Verdict

**REVISE the one-vector proposal before implementation.** One production-file change can record the exact 128-expert distribution for all post-dispatch tokens, but in the observed target batch that population is 4,097 tokens. It must not be labeled as the 4,096-token real-request distribution. A second production-file change carrying the active DP segment bounds is the minimum design that produces both an auditable 4,097-token dispatch vector and the required 4,096-token active-request vector without logging the dummy worker or correlating DP-local clocks.

This is a diagnostic design verdict only. No source, logging contract, or validation code was changed. It does not attribute any TTFT error; the fresh Frontier numerical result was unavailable during this review.

## Frozen scope and observed target

- vLLM checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`
- Revision: `361d941c97fcec52e544f74b7ab91c54192de9c9`; working tree was clean when inspected.
- Runtime configuration: H200, `TP=4`, `DP=2`, `EP=8`, `VLLM_ALL2ALL_BACKEND=naive`, eager mode, BF16 dummy weights, prefix caching and chunked prefill disabled.
- Existing routing run: `runs/h200-diagnostics-03/runtime/routing`.
- First formal DP0 batch on worker `(dp=0,tp=0,pp=0)` has local `batch_id=4537`, `batch_num_tokens=4096`, and one 4,096-token request. Its first routing row has `num_tokens=4097`, `router_topk=8`, `global_num_experts=128`, but only 16 local counts. The row's local total is 4,224, so it is not a global conservation value. The raw files are `server.batch.dp0.tp0.pp0.jsonl` and `server.routing.dp0.tp0.pp0.jsonl` under that run.

## Evidence ledger

| Claim | Label | Direct evidence | Strength / limit |
| --- | --- | --- | --- |
| Naive dispatch constructs the post-dispatch tensors in DP-rank segment order. | Evidence | `vllm/distributed/device_communicators/all2all.py:28-55` allocates to `cu_tokens_across_dp_cpu[-1]`, copies the local tensor into its rank's `[start:end]`, and broadcasts every rank segment; it applies the same operation first to `hidden_states` and then to `router_logits`. | Direct for the selected `naive` backend. |
| Top-k consumes those post-dispatch tensors before the MoE expert kernel. | Evidence | `vllm/model_executor/layers/fused_moe/layer.py:1804-1808` dispatches both tensors; lines 1811-1842 call `quant_method.apply`; lines 470-487 call `FusedMoE.select_experts`; lines 503-529 then pass the returned `topk_ids` and `expert_map` to the expert implementation. | Direct for the selected non-chunked path. |
| The current hook has the exact `topk_ids` returned to the kernel, before logger-side `expert_map` filtering. | Evidence | `layer.py:1507-1513` creates global top-k IDs, line 1578 logs that tensor, and lines 1581/470-529 return and forward it to the expert implementation. `vllm/v1/utils.py:651-652` copies those IDs once; lines 682-699 then apply `expert_map` only while calculating the logged local counts. | Direct. With EPLB enabled the IDs are global physical IDs after the EPLB mapping (`layer.py:1523-1543`); EPLB is not enabled in this case. |
| The persisted 16-count map cannot reconstruct the missing 128-expert vector. | Evidence | `utils.py:682-699` drops IDs mapped to `-1` and bins remaining IDs by local IDs; lines 759-761 serialize only those local bins. The first formal row records `global_num_experts=128`, `num_experts_per_device=16`. | Direct. |
| One-file logging can recover the full 128-axis vector for the 4,097-token dispatch population. | Inference | `utils.py:651-652` already has all global IDs on CPU before the branch. `torch.bincount(flat_ids, minlength=global_num_experts)` is already used when `expert_map is None` at lines 670-680. Applying it unconditionally before the branch preserves all 128 expert IDs. | High confidence; requires a unit check and one fresh routing run. |
| The active worker can also recover its own 4,096-token vector without observing the dummy worker. | Inference | `all2all.py:35-42` defines exact source segments; `vllm/forward_context.py:65-68,114-116` retains their cumulative boundaries; `layer.py` already imports `get_forward_context`. Slicing `topk_ids[start:end, :]` for the active DP rank before binning selects the original active rank's rows. | High confidence for `naive`; the proposed context must be explicitly scoped to that backend/path. |
| All four TP workers have identical global top-k vectors. | Unknown | They have the same shape and complementary local expert maps, but current logs persist disjoint local projections rather than a common global vector. | Require exact vector equality in the new run; do not assume it or sum current local maps as if they came from one routing decision. |
| The extra row in this collective round is semantically a dummy row. | Inference | The active row has 4,097 post-dispatch tokens while its local batch has 4,096. `vllm/v1/engine/core.py:1101-1113` makes an idle DP engine execute a dummy batch, and `vllm/v1/worker/gpu_worker.py:556-557` calls `_dummy_run(1)`. | The current logs have no collective-round ID, so they do not independently join the remote dummy invocation to this active row. The 4,096 active slice does not depend on that join. |

## Exact minimum extension options

### Option A: one production file, dispatch population only

Change only `vllm/v1/utils.py` in production, plus its existing unit-test file:

1. In `FrontierMoeRoutingLogger.log_routing`, immediately after `flat_ids` validation and before the `expert_map` branch, compute a dense 128-entry count vector from the already-copied CPU tensor.
2. Preserve every existing local field. Add clearly scoped fields such as:
   - `dispatch_num_tokens` (4,097 in the target row);
   - `dispatch_total_routed_tokens` (expected `4097 * 8 = 32776`);
   - `dispatch_global_expert_counts`, a dense list whose length equals `global_num_experts` and whose indices are global expert IDs.
3. To control JSON growth, emit the added vector only for rows whose existing batch metadata has `batch_num_prefill_tokens > 0`. The current task is restricted to prefill 4096; decode-wide global vectors add no evidence for this decision.

Estimated scope: **one production file, about 15-25 executable/validation lines**, and `tests/core/test_frontier_moe_routing_logger.py`, about 15-30 test lines. No `layer.py`, `gpu_model_runner.py`, dummy path, environment variable, or batch schema change is required.

Option A proves the 128-expert distribution for the actual 4,097-token kernel input. It does **not** produce a 4,096-only vector. The name must say `dispatch`, and downstream analysis must retain the 4,097 denominator.

### Option B: two production files, dispatch and active-request populations (recommended)

Change `vllm/model_executor/layers/fused_moe/layer.py` and `vllm/v1/utils.py`, plus the existing unit test and task validator:

1. In `FusedMoE.forward_impl`, only when `do_naive_dispatch_combine` is true, obtain `get_forward_context().dp_metadata.cu_tokens_across_dp_cpu`. Derive the current DP rank's `[start,end)` row span using `get_dp_group().rank_in_group`. The buffer order is the order established by `NaiveAll2AllManager.naive_multicast`.
2. Extend `FrontierMoeRoutingContext` with optional diagnostic metadata, preferably one immutable value such as `source_dp_span: Optional[tuple[int, int]]` and an optional `dispatch_dp_num_tokens: Optional[tuple[int, ...]]`. Pass it through `frontier_moe_routing_context` and `log_frontier_moe_routing_from_context`. Keep it `None` for other MoE paths so the logger cannot silently apply naive ordering to DeepEP/PPLX/chunked layouts.
3. In `log_routing`, reshape the already-copied IDs as `[num_tokens, router_topk]`, calculate both dense vectors, and serialize:
   - the Option A `dispatch_*` fields;
   - `source_dp_rank`;
   - `source_num_tokens` (4,096);
   - `source_global_expert_counts`, a dense 128-entry list;
   - `dispatch_dp_num_tokens` (`[4096, 1]` if confirmed in the fresh rerun).
4. Fail explicitly if the span is out of range, its length does not equal the active logger batch's `batch_num_tokens`, either vector length differs from 128, or either conservation equation fails.

Estimated scope: **two production files, about 45-70 lines total** (`layer.py` about 15-25; `utils.py` about 30-45), `tests/core/test_frontier_moe_routing_logger.py` about 35-55 lines, and `tests/e2e/issue26_diagnostic_identity_analysis.py` about 25-40 validation lines. These are estimates, not an implemented diff.

No `gpu_model_runner.py`, `gpu_worker.py`, engine coordinator, or cross-process logging change is needed. Option B is the minimum proposal that closes both meanings of “global”: all 128 expert IDs and the real active request's token population.

## Ordering and identity contract

- `topk_ids[row, slot]` is available with the same row order as the dispatched `hidden_states` and `router_logits`. DP segments are concatenated in DP-rank order. Counts deliberately discard `slot` order because load modeling needs multiplicity, not the rank of an expert within each token's top-k list.
- The proposed vectors do not persist hidden states, router logits, or per-token IDs. Logging those tensors would be much larger and is unnecessary for the expert-load question.
- `batch_id` remains worker-local. Join routing rows only to the batch log with the same `(dp_rank,tp_rank,pp_rank,batch_id)`. Validate the four TP copies inside one active DP lane by that identity. Do not join DP0 batch 4537 to any DP1 batch ID or timestamp.

## Routing-only overhead and validation

The existing routing logger already performs `topk_ids.detach().to("cpu")` once per layer (`utils.py:651`) and flushes each JSON row (`utils.py:764-765`). Both options reuse that CPU tensor, so they add no second top-k D2H transfer. Option A adds one CPU `bincount`; Option B adds a slice and a second CPU `bincount`. Dense 128-entry JSON data increases CPU and I/O cost, which is why the proposal limits the new vector to prefill-containing batches.

The run must remain routing-only. `tests/e2e/issue26_h200_diagnostics_worker.sh:33-46` already separates operator and routing runs, unsets CUDA op logging in routing mode, and enables only `VLLM_FRONTIER_MOE_ROUTING_LOG_PATH`. The routing run is diagnostic and its TTFT/operator times remain excluded. A fresh A/B can quantify added perturbation, but no claim that it is negligible is supported before that run.

Option B can be validated without a dummy log, global batch ID, or cross-worker clock:

- first formal DP0 local batch: `source_num_tokens == batch_num_tokens == 4096`;
- `len(source_global_expert_counts) == len(dispatch_global_expert_counts) == 128`;
- `sum(source_global_expert_counts) == 4096 * 8 == 32768`;
- `sum(dispatch_global_expert_counts) == 4097 * 8 == 32776`;
- elementwise `dispatch - source` is nonnegative and sums to 8 for this observed shape;
- all 48 MoE layers satisfy those equations;
- the four active DP0 TP workers produce identical source and dispatch vectors for each worker-local batch/layer key. Failure of this equality is evidence that TP routing is not replicated and blocks selecting one TP worker as canonical;
- each worker's existing local 16-count map equals the projection of that same worker's dispatch vector through its `expert_map`.

## Rejected alternatives

1. **Log the dummy worker and join it later:** the current dummy path calls `_dummy_run(1)` without `FrontierMoeRoutingLogger.start_batch/activate/finish_batch` (`gpu_model_runner.py:3224-3237` versus real execution at lines 2377-2433). Adding it requires a new synthetic lifecycle plus a collective-round identity to join unrelated DP-local counters. It is broader, duplicates information already present on the active worker, and still cannot safely use wall-clock correlation.
2. **Sum the existing eight local maps:** only the real worker's four TP processes log this round; the dummy side is unlogged. Even when eight files exist across the run, DP-local batch IDs refer to different scheduling histories. Summing them fabricates a collective join.
3. **Treat the 4,097 dispatch vector as the 4,096 request vector:** this violates the observed denominator and can move eight assignments among arbitrary experts. The difference is small in total count but is not known per expert.
4. **Persist full `topk_ids`, hidden states, or router logits:** those payloads are unnecessary to answer the load-vector question and would greatly amplify the already observed routing instrumentation cost.

## Decision for grill-me

- Choose **Option A** only if the immediate question is the actual kernel input's full 128-expert load including the remote row.
- Choose **Option B** if the vector will be compared with Frontier's 4,096-request routing input. This is the recommended boundary because it records both populations, preserves the raw dispatch truth, and avoids any dummy/global-clock contract.
- Keep full vectors on all four active TP workers for the first fresh validation. Restricting to `tp_rank=0` is safe only after exact cross-TP equality is observed; it should not be assumed from the current local projections.

## Residual unknowns

- Exact cross-TP vector equality is unmeasured.
- The incremental wall-time and file-size cost of either extension is unmeasured.
- The current artifacts do not independently identify the remote one-token execution as dummy for a specific collective round; Option B makes that identity unnecessary.
- No fresh Frontier baseline was available here, so neither routing distortion nor this logging gap is established as a material TTFT cause.

## Root decision scope presented to YC

The pending D009 question is deliberately limited to the one-production-file dispatch-global vector. The first target is to qualify the actual kernel population4097 and its complete EP load distribution; this vector is not promised as a4096-only Frontier replay input. The source-DP vector recommended by the independent reviewer would serve that later replay purpose, but requires additional context-interface changes and is not included in the currently pending user question. Root communicated the4096/4097 distinction explicitly before any shared record edit. No implementation or rerun has been authorized yet.

The existing logs do not retain topk_ids, so obtaining the additional counts does require a fresh instrumented routing run. No additional top-k computation is needed inside that run; the extension reads the current inference call's existing values.
