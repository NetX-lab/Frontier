# Review request: calibration item 3 design (host time of the Step-5 KV block reset)

Report language: English findings; short Chinese summary is optional.

## Scope

- Design under review: `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md`
- Frontier repository (LLM inference discrete-event simulator): `/data/ycfeng/Frontier/.worktrees/step5-preview`
- Native vLLM source that the design describes: `/data/ycfeng/tmp/d29_published_sources01/source/vllm`
- Native staging and probe harness (reference for where a probe run kind would be added): `/data/ycfeng/Frontier/.worktrees/step5-d31-native-320c208/tests/integration/step5_pd/canonical_stage.py`, `/data/ycfeng/Frontier/.worktrees/step5-d31-native-320c208/tests/comparison/calibration/native_pd_worker.py`
- Task context: `requirements.md`, `plan.md`, `test_report_2026-10-26_decode_step_slope.md` in the same task directory.

This is a design review. Do not edit files.

## Background

Native Step-5 decode TPOT is 13-15% higher than Frontier at 20-40 requests per engine. The native device timeline shows about 9 ms GPU idle gaps before some decode forwards. The design attributes them to the host-side sparse-summary KV side-storage reset and proposes a measured, opt-in host term in Frontier.

Project rule D31.10: a table of whole-step times measured in a real run and looked up by batch composition is diagnostic only and must not be used to make Frontier predict. Component host-time terms measured by a probe are allowed.

## Questions

1. Native mechanism. Is the analysis in design.md section 2 correct?
   - Does `KVBlockZeroer.zero_block_ids` -> the Step-5 sparse-summary side-storage reset hooks (`.tolist()` on GPU tensors and Python-list indexing) force a host wait on the previous step's device work under async scheduling with the V2 GPU model runner (`v1/worker/gpu/model_runner.py`)?
   - Does the gloo DP all-reduce in `dispatch_cg_and_sync_dp` propagate the delay to all DP engines of an instance?
   - Are the cited file:line references accurate? Is anything else on the reset path (for example, H2D copies of block ids, the scheduler's `skip_zeroing_group_ids`) misdescribed?
2. Composition in Frontier. Is putting a per-step host term inside `OverheadTime.forward_preparation_time()` (so it is added before `_forward_launch_start_time` in `frontier/events/replica_stage_schedule_event.py`, and taken as a max over DP lanes in `frontier/scheduler/utils/sync_entry.py`) a faithful composition, given Frontier decode uses `max_concurrent_batches=2` and schedules the next step when the previous one ends?
   - Is there double counting or loss through `prepare_decode_final_timing` / `prepare_prefill_final_timing` in `frontier/scheduler/utils/collective_timing.py`?
   - Does the term apply in the correct place for prefill as well as decode, and for the PDD roles that Step-5 uses?
3. Trigger. Is the proposed trigger equivalent to native's?
   - `zero_new_blocks=True` for the Step-5 capacity in `frontier/attention/step5_memory.py`;
   - `batch.cache_block_zeroing` attached in `_create_batch` in `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` (find the exact path);
   - skipping block recording for NIXL read reservations in the vLLM V1 KV allocation module (`_allocate_request`).
   - Does turning on `zero_new_blocks` for Step-5 change any existing timing (for example, does any existing code price `cache_block_zeroing` as a device operator for non-DeepSeek-V4 models)? The design claims no timing change without the profile.
4. Probe and analysis. Is the probe design in section 4 sound?
   - perf_counter stamps plus CUDA events at `StepTimingCollector.forward_start` / `forward_end` (check where these are called in the V2 runner and whether CUDA events there capture the forward's device start/end correctly with CUDA graphs and async output copies);
   - joining DP engines by the `dispatch_cg_and_sync_dp` call index (is the call count aligned across DP engines, including dummy runs?);
   - excess gap = median device gap of resetting steps minus median device gap of non-resetting steps, per role and request-count bucket. Is this an unbiased estimate of the added host time, or is it confounded (for example by resetting steps correlating with scheduling of new requests, prefill/decode mixing, or output-processing stalls)?
   - Does the resulting profile respect D31.10 (component host-time term, not a whole-step table)?

## Expected output

Concrete errors with file:line evidence, severity, and a recommended correction for each. Also list design points you confirmed as correct, with evidence, so the implementer knows what is settled.
