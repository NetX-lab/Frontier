# WARM-ACCOUNTING: two retained-evidence checks

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Classified NIXL wire-size and EP-dispatch gap-count discrepancies; recorded timing limits. |

## 1. Result and authority

Two small **confirmed simulation_logic discrepancies**, not a macro TTFT fix:

1. Structural component bytes are passed into NIXL **wire** pricing even though
   native transfers padded packed blocks: +6,581,376B per TP worker/handle.
2. EP dispatch launches three kernels but its registry count is one:
   86 missing eager gaps per43-layer P-step, +0.046836ms direct device cost.

No current warm witness supports filling the remaining READ or P-step residual
with a host/queue/wire constant. Retained timing structure is informative but
does not identify an additional measured-input correction. Investigation stops
at these classifications. No residual decomposition or FIFO sweep was rerun.

Authority: parent requirements.md:149–165 D-WARM-TARGET;356 D-NO-NEW-MODELING;
422–423 host-zero/four gap classes;489–492 serving-delay exceptions;
plan.md:156–190. This request permits checks only, not fixes, table promotion,
new GPU, full replay, main-record updates or edits to old reports/source/skill.

## 2. NIXL: confirmed pricing-domain mismatch

### 2.1 Bytes and selected blocks

Existing current function explicitly returns **structural component bytes**,
not descriptors/slab padding/TP aggregate:
`frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py:19–25,50–63`.
Its result is nevertheless passed to `get_transfer_time` at130–135. The helper
is correct for its documented meaning; the discrepancy is the wire-price input.
Capacity functions independently retain physical slab geometry:
`frontier/attention/families.py:190–261` and
`frontier/attention/memory.py` physical_block_size_bytes/get_group_size_bytes.

CPU AST extraction of those exact definitions, with recorded model geometry and
valid logical P/D tables at6144 tokens, reproduces every current ledger exactly:

| Group | Selected pages | Component B/page | Component total B | Native packed total B |
| --- | ---: | ---: | ---: | ---: |
| compressed | 24 | 1,002,240 | 24,053,760 | 24,053,760 |
| swa_0 | 2 | 823,680 | 1,647,360 | 2,004,480 |
| swa_1 | 2 | 786,240 | 1,572,480 | 2,004,480 |
| csa_state_0 | 2 | 870,912 | 1,741,824 | 2,004,480 |
| hca_state_0 | 16 | 656,640 | 10,506,240 | 16,035,840 |
| **Total** | **46** | | **39,521,664** | **46,103,040** |

Native source, pinned2cf0a6915ce544dc493a0990f2ea38d81601128a:
`/data/ycfeng/tmp/opencode/vllm-v0.28.0/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:960–1028`
registers one packed region with one block-stride descriptor per physical block.
Native registration logs give stride1,002,240B; e.g. wave64
`artifacts/nodes/gpu-h200-0142.lgcm.sh.istep.fun/server.log:237`.
`pull_worker.py:287–335` selects the blocks and posts READs. Four inspected native
files match supplied expected-source hashes; three additional inspected files
have no individual hash in that worker receipt. The checkout is clean at the
pinned commit; native_checkout_identity.txt and validation.json preserve this
provenance limit. Descriptor telemetry counts vary because physical descriptors
coalesce: **46 selected physical pages is not necessarily46 wire descriptors**.

Retained scheduler stats: wave64 has128 retained request windows:127 contain8
samples and one contains16 samples from aggregate updates, which cannot be
uniquely assigned to that request. Every sample is46,103,040B. Lifecycle32 has384
diagnostic request windows, each with8 equal-byte samples. Native source posts
one corresponding-rank READ per TP worker; eight TP workers do not imply a TP=8
multiplier in per-worker wire pricing. The16-sample window is a join ambiguity,
not evidence of16 native reads for that request.
These all-request windows establish geometry and are not promoted as a new warm
timing reference. No filtering was done: canonical first-contact identity cannot
be transferred to a different instrumented run without its own handshake witness.
Evidence: payload_summary.json,
wave64_handle_stats.json, lifecycle32_handle_stats.json.

### 2.2 Smallest proposal and effect

Preserve structural component semantics/ownership. In the existing grouped NIXL
consumer, derive wire bytes from the already-selected pages times
`capacity.physical_block_size_bytes` and use that derived quantity only for the
existing network price. Do not hard-code46,103,040 or multiply by TP; do not
rewrite the structural helper's documented meaning silently. No new mechanism,
config flag, warmup model or timing offset is necessary. Primary/YC must approve
implementation under this checks-only assignment.

All six current configurations use126.2Gbps and0.0647ms latency. Exact arithmetic:
`8*(46,103,040-39,521,664)/126.2e9*1000 = 0.417202916ms` extra wire time in **both
cases** (14.275% of native bytes were absent from the wire-price input).
This is not a supported estimate of mean TTFT: READ find/report boundaries can
quantize completion by a step. It cannot be offered as the~9ms residual fix.

Independent validation: CPU check unused retained lanes and held-out prompt
boundaries (e.g.256/257 and6143/6145) against existing logical block selection and
native registration geometry; preserve component-byte and ownership invariants.
Cost0GPU. New native transfer-byte evidence can be read from a future approved
run at an unused prompt length; no new GPU run is required to implement the
geometry correction. End-to-end effect requires an authorized fixed-input replay
after correction; none was run here.

### 2.3 READ lifecycle: structural abstraction, timings unresolved

Frontier records a single wire end per request, posts at engine-step start,
finds completed READs **before posting that step's new READs**, and reports at the
owning output: `frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py:39–77`.
The ledger's start is engine enqueue, not actual READ post:
`frontier/events/kv_cache_transfer_start_event.py:228–271`.

Native binds metadata and calls start_load_kv, yields the forward host code, then
calls get_finished: pinned
`vllm/v1/worker/kv_connector_model_runner_mixin.py:85–109`.
Thus find-before-post at an abstract step start is **not the literal native host
order**. Native's host return is not necessarily GPU completion, so simply moving
the poll to device-step end would also be unsupported. Eight workers post/poll
independently; a single Frontier wire completion does not explicitly represent
their spread. These are known abstraction differences, not measured corrections.

Retained stats are aggregated at scheduler update. They contain8 byte/duration
samples but **no handle IDs or per-worker start_load_kv/transfer/check-state/
get_finished timestamps**. The single16-sample wave64 window also exposes the
limits of request attribution from aggregate scheduler statistics. Raw durations
and post-duration samples are retained in the JSON artifacts; their all-request
means are not a warm reference and are not used to choose a correction. Neither
max-minus-mean nor telemetry duration can be equated to pure wire time or priced
as a tail offset; scheduler update is not wire completion.

Exactly needed observation: on a matched warm request, retain request ID, TP rank,
engine-step/forward ID, transfer handle, selected block/descriptor bytes, monotonic
post start/end, every check_xfer_state result/time, get_finished time, model host
return, device-forward start/end, and output/scheduler receive report. Use a common
clock or record clock alignment. This separates worker posting skew, actual
completion, polling and output aggregation. Timing validation needs a newly
approved native worker-instrumented run at the original P/D allocation/load;
a reduced pair may validate bytes but not full-case phase timing. No quota or
GPU-hour budget was checked; **0GPUs used here**. No lifecycle change is proposed
without this evidence and YC approval for any new mechanism.

## 3. P-step: confirmed EP-dispatch gap count, no macro correction

### 3.1 Current accounting and retained native spans

All128 current real P forwards per case/table close arithmetically. Units ms.

| Segment | P2-64 B29 / pass1 / pass2 | P2-32 B29 / pass1 / pass2 |
| --- | ---: | ---: |
| KV zero | 2.910935 / same / same | 2.910935 / same / same |
| Block-table writes | 0.008889 / same / same | 0.008889 / same / same |
| Prepare | 0.236566 / same / same | 0.236566 / same / same |
| Dispatch, composed stream path | 276.596636 /276.156918 /276.496140 |206.657574 /206.217856 /206.557079 |
| Post | 0.101127 /same /same | 0.101127 /same /same |
| Coordination/overlap remainder | 0.012586 /same /same |0.039904 /same /same |
| **P-step total** |**279.866738 /279.427020 /279.766243**|**209.954995 /209.515277 /209.854500**|

Native TP0 launch-correlation joins, **kernel-only** totals, not clean/warm prices:

| Native forward | Segment | Kernel count | Kernel sum | Union | First–last span |
| --- | --- | ---: | ---: | ---: | ---: |
| P2-64 dp0, index64 | prepare |67|0.145823|0.145824|0.366274|
| same |dispatch|2313|276.816072|274.029458|278.648247|
| same |post|11|0.096512|0.096512|0.111553|
| P2-32 dp0, index0 |prepare|67|0.135550|0.135551|3.496871|
| same |dispatch|2313|336.232272|333.487177|337.465946|
| same |post|11|0.563073|0.563073|0.576897|
| P2-32 dp0, index64 |prepare|67|0.135235|0.135234|1.722531|
| same |dispatch|2313|259.068224|257.974135|327.578429|
| same |post|11|0.102655|0.102655|0.332672|

The previous P2-64 forward's post contains the next real forward's pre-DP zero
and block-table work: zero kernel2.912941ms versus Frontier2.910935ms; four write
kernels sum0.023135ms versus priced0.008889ms (a~0.014ms diagnostic price difference,
not proof of a repeatable input defect). Source ownership is already pre_dp in
`frontier/execution_time_predictor/deepseek_v4.py:28–31`.

Runner inputs at6144/NONE/prefill/request1 contain67 kernels and9 copies; retained
native prepare has67 kernels (this reducer did not count native copies).
Native post11 includes one LM-head kernel and one logits-gather, leaving9 sampling
kernels, matching the needs_logits_processing0/b1 table. Evidence: runner_rows.json,
data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/model_runner_replay_kernel_only.csv,
p2_64_segments.json, p2_32_segments.json.

### 3.2 Witnessed count defect

Native dispatch on all three inspected real forwards has:
129 AllGatherV kernels,43 EP Reduce kernels,43 SP ReduceScatter and44 SP AllGather
(43 attention plus final MHC gather). Current executed SP scopes also have43
attention calls. `collectives.json` counts **price queries**, including repeats;
its44 queries/real-forward for one SP price must not be diagnosed as44 executed
attention layers. Scope pricing gives43 after its current eager gap is removed.

Native source dispatch has hidden_states/topk_weights/topk_ids:
pinned `vllm/distributed/device_communicators/all2all.py:119–127`;
`cuda_communicator.py:629–655` loops the tensor list into three NCCL calls.
Retained129/43=3 independently witnesses the executed path.

Current registry `frontier/operators/families.py:802–816` does not specify an EP
kernel count, inheriting1 from `frontier/operators/spec.py:199–200`.
`frontier/execution_time_predictor/dsv4_runtime.py:31–38` deliberately reads that
registry count for communication, not the per-layer count dictionary. EP dispatch
is not an attention scope-gap override. This is a **confirmed simulation_logic
count defect**, not a reason to remeasure or rescale the G5 kernel prices.

Smallest proposal: record dispatch count3 (combine stays1) in the existing COMM
registry entry. No caller special-case, new flag, host model or input-table change.
`gap_verify.py` exercises the actual isolated _operator_gaps with current count1
and a proposed registry3 fixture. At eager0.5446049596us:
`43*(3-1)*0.5446049596/1000 = 0.0468360265ms` direct P-step increase in **both cases**.
This is0.0065%/0.0129% of the prior native P2-64/P2-32 TTFT, not a useful macro
accuracy gain. Fixed downstream boundaries would yield that same small direct
TTFT increase, but no mean TTFT estimate is supported for quantized FIFO/READ
alignment; no replay was run. It is consistent with D-WARM-TARGET and reuses an
existing registry field, so no new modeling mechanism is required. Implementation
still awaits the primary/YC decision because this task is checks-only.

Independent validation: use unused retained dp1 TP0 forwards and an unused decode
graph class to validate the three physical calls; CPU-only,0GPU. After an approved
implementation, verify registry/gap pricing for NONE/PIECEWISE/FULL against those
held-out counts and run affected existing regressions. Do not select a new gap
value from these P-step totals.

### 3.3 Remaining accounting limitations and exact next observation

No additional count/overlap/input defect was witnessed. Current attention graph
uses actual per-scope gap plus fork/join critical path, not a serial scope sum:
`dsv4_composition.py:88–108,158–187`; observer respects that composition:
`h3o_operator_costs/a10/a10_observer.py:117–133`.
Native stream code agrees with those NONE fork/join branches:
pinned `vllm/models/deepseek_v4/attention.py:461–501,895–965` and
`vllm/utils/multi_stream_utils.py:45–61,105–132`.

P2-64 dispatch has2.787ms overlap and4.619ms idle. P2-32 index64 has69.604ms idle,
consistent with its peer-delayed diagnostic profile and unsuitable for pricing.
These intervals do not identify a missing operator. Kernel-sum, union and critical
path are different quantities; profile/host overhead and warm identity also limit
absolute comparisons. Source/state agreement is not proof of correct durations.

Host-zero is explicit: parent requirements422–423,
`frontier/execution_time_predictor/dsv4_measurement.py:16–20`, current config
skip_cpu_overhead_modeling=true. The earlier~7.5ms host estimate is reused as a
hypothesis only, not upgraded to a current measured cost. No host-zero change is
proposed; it needs YC approval and new clean/warm evidence.

Exactly needed if pursuing the remaining macro gap: paired clean and instrumented
warm P forwards at the original P/D geometry, excluding supplied first contacts;
retain per-layer scope and stream IDs, launch/kernel intervals, counts, workload
coordinates and graph mode, runner copies/zero/write ownership, EP/SP dependencies,
DP rendezvous, model host entry/return and device completion. In a future authorized
replay, retain each scope's device cost, gap/count and composed start/end so scope
duration, launch idle and peer waits can be compared independently. Current native
full-model trace and Frontier observer do not both expose that matched per-layer
input/start detail. No supported macro TTFT effect can be quoted before it exists.
CPU inspection of unused lanes costs0GPU; new matched timing requires a separately
approved native instrumented run at original allocation, not a smaller fake proxy.

## 4. Commands, verification and deliverables

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
Python `/data/ycfeng/envs/frontier-py310/bin/python`, version3.10.6;
`PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp`, `ulimit -v31457280`.
Exact reproducible sequence is run.sh; the individual commands actually executed:

```bash
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
OUT=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/warm_accounting_20261101
PY=/data/ycfeng/envs/frontier-py310/bin/python
timeout 60s "$PY" "$OUT/inspect_current.py" > "$OUT/current_stdout.txt"
timeout 60s "$PY" "$OUT/payload.py" > "$OUT/payload_stdout.json"
timeout 60s "$PY" "$OUT/segments.py" p2_64 > "$OUT/p2_64_segments_stdout.txt"
timeout 60s "$PY" "$OUT/segments.py" p2_32 > "$OUT/p2_32_segments_stdout.txt"
timeout 60s "$PY" "$OUT/gap_verify.py" > "$OUT/gap_stdout.json" 2> "$OUT/gap_stderr.txt"
timeout 60s "$PY" "$OUT/validate.py" > "$OUT/validation_stdout.txt" 2> "$OUT/validation_stderr.txt"
```

PASS7/7: supplied source hashes; structural/packed payload arithmetic;
equal-byte native samples with explicit aggregate-window ambiguity;
six current byte prices/rate arithmetic;
selected runner row counts; retained three-forward EP/SP kernel-count witnesses;
actual gap-code fixture. Outputs: validation.json/validation_stdout.txt,
gap_verify.json, payload_summary.json and per-request handle stats,
current.json and both per-case segments JSON, runner_rows.json.

AST extraction loads only inspected pure definitions, not a trained predictor or
GPU runtime. Initial package import and an ad hoc config-key lookup failed and
were corrected in the new inspection only; progress.md preserves those limits.
No persistent simulator/comparison test/source edits, old-report updates, GPU calls,
full replay, tuning or commit. existing_core_changes.txt is empty. Parent task
records have existing primary-session modifications; this agent did not write
them. The new directory is ignored by .gitignore:171; primary archival will need
explicit force-staging of its selected artifacts (ignore_status.txt). A self-check
found no drift; checkpoint.md records
the route correction: stop investigating classified mechanisms and return decisions.

Next decision: approve the two existing-path corrections separately from any new
READ/host modeling. Fixing them is justified by structure/count witnesses, not by
fitting the residual. Parent acceptance and commit remain with the primary.
