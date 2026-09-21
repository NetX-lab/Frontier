## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Independently reviewed the timing contract, diagnostic perturbation, MoE collective semantics, and raw kernel-activity coverage. Communication supplement remains pending. |
| 2026-09-08 | Validated all 24 selected communication rank/batches and expanded first-kernel coverage to DP0's four TP ranks; retained numerical and full-run limits. |
| 2026-09-08 | Verified all 400 communication-run client completions and incorporated root's successful launcher/service completion evidence. |

# Independent review of first-batch operator RCA

**Verdict: ACCEPT the bounded timing and source-semantic findings and communication collection integrity; numerical attribution remains INCOMPLETE.** Communication collection passes both the full 400-request completion check and the 24 selected rank/batch scope check. The current records do not support sizing a production repair from the fully instrumented operator spans, declaring the third kernel trace complete, or passing a three-aligned-batch operator gate.

Reviewer: `/root/route_instrumentation`, distinct from the operator RCA author `/root/first_batch_op_rca`. The reviewer implemented the host route diagnostic and separately reviewed the first-forward EP shape contract; the reviewer did not implement the operator collector or comparison helpers.

## Inspected evidence

- `timing_contract.md`.
- `kernels/collector_failure_evidence.json` and the three raw Chrome traces it references.
- `operators/summary.json`, `operators/first_batch.json`, `operators/first_batch_rank_totals.json`, normalized scope artifacts, and operator gap summaries.
- `kernels/first_trace_summary.json` and its first raw trace.
- `../../test_report_2026-09-08_first_batch_op_rca.md`.
- Pinned diagnostic source in `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/utils.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/qwen3_moe.py`, and `vllm/distributed/communication_op.py`.
- Clean source in `vllm/v1/engine/processor.py`, `vllm/v1/engine/async_llm.py`, and `vllm/v1/metrics/stats.py`.

## 1. CUDA-event and official TTFT boundaries

**ACCEPT, with an explicit boundary qualification.** Clean `process_inputs` establishes the recorded host arrival timestamp; the output handler constructs `IterationStats` after receiving EngineCore outputs; `first_token_latency` subtracts the former from that iteration timestamp. This includes relevant host/transport/queue/model/sampling work after recorded arrival and before first output receipt. It is neither an HTTP socket-to-client metric nor pure CUDA time. Host work before the recorded arrival and frontend/client work after the sampled output-receipt boundary are outside it.

The batch CUDA start is recorded before `set_forward_context(...)`, while the end is recorded after the model and context exit, before logits/sampling. In this DP case, `set_forward_context` can execute the CPU-group DP token-count collective before `self.model(...)`. Thus describing the interval as a forward stream elapsed interval is correct, but it must not be narrowed to GPU kernels strictly inside the model call. Stream idle caused by late host submission, CPU-side context preparation, dependency waits, and collective synchronization may appear between these CUDA events. Pure preprocessing before the start event and postprocessing after the end event remain outside the span.

The Frontier profiling inputs themselves use CUDA-event timing. Accordingly, `skip_cpu_overhead_modeling` does not imply that all predicted components represent kernel-active time only. A blanket addition of measured host intervals would risk double-counting host-induced gaps already present in profile values.

## 2. Full-scope 116 ms versus batch-only 80 ms

**ACCEPT as a comparability warning, not as a causal decomposition of 35.9 ms.** The artifacts report `116.210144043 ms` for the fully scoped first DP0/TP0 batch and `80.335617065 ms` in the independent batch-only execution, a difference of approximately `35.874527 ms`. These separate executions expose substantial instrumentation/run sensitivity. They do not isolate an exact overhead attributable exclusively to event-record calls, nor provide a correction factor for clean production operators.

Within the scoped run, DP0/TP0's deduplicated selected-scope sum is `107.531295843 ms`, leaving `8.678848200 ms` in that instrumented batch outside the selected spans. This residual is not a CPU-only measurement. Nested `add` and duplicate EP outer/inner scopes must remain excluded from additive totals, as the current RCA states.

The cross-rank distribution directly warns against equating one collective scope with network service time: first-batch `expert_parallel_allreduce` totals are approximately `37.563 / 37.883 / 5.653 / 37.672 ms` on TP0/1/2/3, while whole-batch times remain approximately `116.2–116.4 ms`. Rank TP2 spends more time in several preceding scopes. This supports synchronization/arrival sensitivity; it does not independently allocate every millisecond to rank waiting or network transmission.

**Update:** communication-only measurements are now independently checked in section 5 below. They reduce the observed whole-batch difference relative to the separate batch-only run, while retaining significant rank-dependent collective elapsed times.

## 3. Additional MoE TP all-reduce semantics

**ACCEPT the source finding; cost and modeling repair are not yet established.** Qwen3 initializes `FusedMoE(..., reduce_results=True)`. With the selected naive path, `reduce_output` first calls EP dispatch-manager combine; `all2all.py` performs a DP-group reduction and local slicing. Then `maybe_all_reduce_tensor_model_parallel` calls `tensor_model_parallel_all_reduce`, whose implementation explicitly uses `get_tp_group()`. The scope name is `expert_parallel_allreduce` when EP>1, but its group in this call is TP4.

This is an actual additional collective in the selected vLLM protocol. It is not an independent EP8 all-reduce inferred from the scope label. It also must not be assigned the entire instrumented 37 ms as production network cost.

Frontier's idealized EP dispatch/combine phase may already describe completion of the mathematical expert-output aggregation using a different collective protocol. Therefore a correction must reconcile the modeled collective algorithm and its phases with the naive DP-broadcast/DP-reduce/TP-reduce implementation. Merely appending a TP4 cost to an otherwise ideal EP model is not yet validated by this source observation alone.

## 4. Kernel collector integrity and first-batch scope

**ACCEPT the missing-activity finding.** The reviewer independently loaded each raw Chrome trace, selected CUDA runtime/driver `LaunchKernel` events, and matched their correlation IDs against `kernel` device activities. Results exactly reproduce the collector evidence:

| Raw trace batch / PID | Recorded kernel launches | Recorded device kernels | Launches without a kernel activity |
| --- | ---: | ---: | ---: |
| 4734 / 4302 | 3077 | 3077 | 0 |
| 4735 / 4302 | 3221 | 3221 | 0 |
| 4736 / 4302 | 3221 | 3216 | 5 |

The five missing correlation IDs in batch 4736 are `95991`, `96010`, `96018`, `96038`, and `96052`. These launches exist in the raw activity stream; their correlated device durations are absent. This is incomplete diagnostic device evidence, not proof that the model omitted those operations. Filling the missing durations with zero, extrapolating them from another batch, or declaring the third-batch per-op gate passed would be unsupported.

For batch 4734, all 3,077 launch correlations are unique and matched one-to-one to 3,077 positive-duration kernel activities; the reverse kernel-to-launch set is also complete. The trace additionally contains 144 device memcpy activities, so the summary's 3,221 device-event count is consistent. All these recorded device events belong to device 0 in this per-rank trace.

The first trace has 48 prefill-attention kernels, 96 `fused_moe_kernel` launches, and 48 SiLU activation kernels. Its selected parent scopes include 48 each for the per-layer attention, shuffling, grouped GEMM, and post-MoE all-reduce operations, with 96 expected duplicated dispatch/combine scopes and 96 gating scopes. These checks support coverage of the first recorded rank's 48-layer forward under the instrumentation contract.

**Required wording:** “complete recorded-launch/device correlation coverage for the first profiled rank/batch” is supported. “Complete unperturbed GPU execution,” “all eight ranks complete,” and “all three aligned logical batches complete” are not supported by this criterion. Launches and activities both omitted by a collector cannot be excluded solely by correlation-set equality; the 48-layer scope/kernel checks provide corroboration but not unlimited completeness.

The first trace's device envelope is `151.560751953 ms`, recorded activity union `143.099283203 ms`, and gaps between recorded activities `8.461468750 ms`. These are quantities from another instrumented execution, not replacements for the batch-only 80 ms baseline. NCCL device-kernel durations can include collective waiting, so even a kernel activity sum is not equivalent to useful arithmetic or network transfer service time.

## Report reconciliation and pending evidence

At initial review time, the main test report still said the record-function/kernel supplement was pending. The author subsequently updated that report and supplied expanded DP0 coverage. The first profiled DP0 forward is inspectable across its four TP ranks; later batches still have missing device records, so overall INCOMPLETE remains appropriate. This reviewer does not edit author-owned artifacts.

Pending items are:

1. Runtime costs for the actual collective protocol, with rank waiting distinguished from claims about pure transport cost.
2. Exact global MoE/source shapes and profiling-family alignment retained in the final numerical interpretation; local 4,096-token membership is insufficient on its own.
3. A complete three-logical-batch operator gate if that broader gate is claimed. Current raw third-batch activity coverage fails it.

## Read-only verification record

CPU environment: conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13. A bounded standard-library JSON inspection processed the three named traces sequentially and reported only correlation counts, missing IDs, device identities, and selected kernel/scope counts. Source inspection used `rg` and bounded `sed`. No code, collector, GPU run, or author evidence was changed.

One ancillary inspection attempted `len(first_batch["op_rows"])` and raised `TypeError` because `op_rows` is a count field. This did not affect the completed raw-trace correlation or scope checks; no evidence value was substituted to hide the inspection error.

## 5. Communication supplement: independent update

Source: `communication/summary.json`, backed by `analysis/h200-rca-comm-01/runtime/operators`. The mode manifest selects exactly `tensor_parallel_allreduce`, `attn_post_proj_tp_allreduce`, and `expert_parallel_allreduce`, using CUDA events with default scopes and per-scope aggregation. Runtime metadata remains disabled. This is a reduced-instrumentation diagnostic, not clean E2E evidence.

**PASS, selected collection integrity:** the reviewer independently joined the eight raw `server.ops.dp*.tp*.pp0.jsonl` files to their raw selected batch rows. All 24 selected rank/batches have exactly 97 records: one embedding TP all-reduce, 48 attention-projection TP all-reduces, and 48 post-MoE TP all-reduces. Each scope sequence is present exactly once (`0`, or `0..47`), rank/phase/token fields agree with the batch, and the selected indices are exactly `0,1,2` per worker. Raw sums reproduce the saved summary. The four TP peers have matching request/token vectors for each of the six DP-local selected forwards. Total verified scope rows: 2,328.

This validates the local identities and communication records. It does not convert the six DP-local forwards into three proven globally aligned EP rounds or establish that their later batch memberships match another execution.

The first formal DP0 batch is local batch 4250 on all four TP workers and contains only `cmpl-pf4096_dc1024:0-0` with 4,096 prefill tokens:

| TP rank | Whole batch CUDA-event ms | Attention TP all-reduce, 48-layer sum ms | Post-MoE TP all-reduce, 48-layer sum ms |
| --- | ---: | ---: | ---: |
| 0 | 86.412033081 | 5.537248015 | 4.792544022 |
| 1 | 86.437950134 | 5.649951987 | 25.384415895 |
| 2 | 86.414657593 | 5.708767995 | 28.413279966 |
| 3 | 86.443519592 | 5.644864000 | 27.393568158 |

Relative to the separate batch-only first forward of `80.335617065 ms`, TP0's whole diagnostic batch is `6.076416016 ms` higher (`7.563788%`). That is materially closer than the full-scope run's `116.210144043 ms`, but neither the smaller difference nor a percentage below 10% makes this a clean production or matched-run overhead estimate. Replacing the full-scope data with communication-only data for the same observed scope is appropriate; algebraically mixing unrelated runs to force the TTFT residual to close is not.

The fresh Frontier attention TP prediction is `17.899440 ms` across 48 layers, versus `5.537248–5.708768 ms` measured across the four diagnostic ranks. This supports a substantial directional overprediction for this selected attention collective comparison and justifies reviewing its backend parameters/algorithm. It does not establish a universal multiplicative calibration factor or independently determine clean transport-only time.

The added post-MoE TP all-reduce remains strongly rank-dependent even with only communication scopes enabled. Its minimum `4.792544 ms` cannot be called clean wire time, nor used as a cost to append to Frontier while subtracting the attention discrepancy. Each scope is a stream elapsed interval containing dependency and participant-arrival effects; the fastest rank can still incur waiting and host-induced gaps. Rank variation plus nearly equal full-batch durations is consistent with arrival/wait redistribution, but these scopes alone do not provide a complete causal split of waiting versus transport service.

### Expanded first-kernel coverage

`kernels/all_dp0_collector_coverage.json` extends the prior report to TP0–TP3. The reviewer independently reopened all four first-batch 4734 traces: every rank has exactly 3,077 recorded kernel launches and the same 3,077 correlated device kernels, with no missing correlation. The supported scope is therefore the first profiled DP0 forward across four TP ranks, still not all eight GPU participants or unperturbed production.

The author's expanded table reports that batch 4735 has two missing activities on TP1 and TP2, and batch 4736 has five missing activities on every TP rank. This strengthens the limitation on subsequent kernel comparisons; no whole-three-batch PASS is warranted. The reviewer previously checked all three TP0 traces directly and now checked all first-batch TP traces; later TP1–TP3 failures are recorded from the author's raw-anchored coverage table without rerunning every collector helper.

No source or author evidence was modified. Only this review was updated. The initial lookup used the parent's provisional filename `communication_summary.json`, which did not exist; `rg --files` located the actual `communication/summary.json`, and all numerical checks used that actual file and its raw source paths.

### Communication full-run completion

**PASS:** the reviewer independently read the final communication `client.jsonl`: exactly 400 unique request IDs, comprising 300 warmup and 100 formal requests, all with `completion_tokens_observed=1024`. Root's persisted `analysis/h200-rca-comm-01/batch_validation.json` reports PASS for the 100 formal requests and eight workers, with its local-batch/global-round limitation preserved. Root additionally reported the successful launcher completion marker and systemd `ExecMainStatus=0`, inactive state. The reviewer did not independently rerun platform inspection for that service-exit observation.

The final root-owned run manifest was reread and reports `COMPLETED_DIAGNOSTIC`, consistent with the observed completion artifacts. Full-run execution completion does not close the missing raw kernel activities, aligned-global-batch identity, or numerical causal decomposition discussed above.
