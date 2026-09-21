## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Audited current routing import paths and reusable workload materialization. |

# Routing import capability

## Result and scope

Active branch task/issue26-ttft-h200-20260907, HEAD85240ab5. No public or internally wired file import for vLLM routing exists in the active co-location path. No runtime configuration has been changed; equality is a confirmed objective, not a delivered result.

## Inspected boundaries

- frontier/config/config.py:1909 defines moe_routing_trace_path as deferred. Its only production references declare, propagate, and validate the field. _validate_pdaf_trace_replay_contract at5343 returns immediately outside PD-AF, so its help text does not imply a co-location rejection or import.
- sklearn_moe_execution_time_predictor.py:772 _init_global_routing_allocations generates balanced/random/skewed/zipf ratios once per layer. :829 copies these into replica/layer/expert routing_details. No batch/iteration key exists.
- :253 _emit_routing_details_snapshot writes an output log. It is not a trace reader.
- frontier/scheduler/utils/layer_workload.py:14 uses batch.total_num_tokens and static replica/layer lookup to call frontier/moe_ep_workload.py materialize_layer_ep_workload.
- frontier/moe_ep_workload.py:726 uses Hamilton integer allocation, preserving total tokens times topk, then contiguous EP ownership splitting. This existing module should be reused, rather than introducing a second load allocator.

## Concrete next implementation proposal

Use Option B in routing_global_counts_proposal.md: extend only existing diagnostic context/logger production paths in vLLM layer.py and utils.py, plus focused validation. Expected production change45–70 lines, subject to actual diff. Preserve local fields, add128-entry dispatch and source-DP counts and DP segment lengths from existing naive-dispatch metadata; reuse the existing CPU topk copy. Collect prefill-containing batches for this same case in a separate fresh H200 step_main diagnostic. No router policy, backend, model weights or clean metric change.

Acceptance: all48 first-formal-prefill layers, source assignments4096*8=32768; actual dispatch assignments equal observed dispatch token count*8 (do not force4097 on a new run); full expert domain; local counts equal the appropriate dispatch projection; cross-TP equality measured before choosing a canonical vector. Routing overhead is diagnostic and cannot supply clean TTFT.

Next, define a batch-aware input resolver at the existing scheduler/predictor routing boundary. Static per-layer ratios alone cannot provide per-batch equality. First validate the isolated4096prefill identity; do not stretch its vector across other requests. If scheduling changes batch membership, aggregate batch histograms do not support exact repartitioning: request/token identity or an explicitly restricted aligned batch replay is necessary. That subsequent interface design remains open and must be grounded in the fresh records.

## Alternatives eliminated

- Setting moe_routing_trace_path: no consumer exists.
- Selecting matching balanced/random labels or seeds: does not reproduce measured per-expert counts.
- Importing a run-average per-layer ratio: changes the equality requirement to a statistical approximation.
- Dispatch-only4097 counts normalized onto4096 real tokens: changes the population and integer counts.
- Joining existing DP-local batch IDs: no shared collective-round identity; missing worker records remain missing.

## Validation evidence and limits

Direct CPU execution used /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python (conda dev-vidur-v03-hopper-e2e; Python3.13.13), PYTHONDONTWRITEBYTECODE=1, from the active worktree. The generator method was extracted with ast and executed unchanged, avoiding unrelated predictor initialization/training. The real pure workload module was loaded with importlib.util.spec_from_file_location.

PASS: generator with nonexistent trace path returns48 layers of1/128 balanced weights. PASS: synthetic counts256 each, adjusted expert0+91 andexpert1-91, are exactly reconstructed for4096tokens/topk8. Changing the population to4097 produces32776 assignments and a different vector. These synthetic numbers are capability-test fixtures only; they are never calibration inputs. This check does not test a full simulator invocation, arbitrary floating-point ratios, or real vLLM routing equality.
