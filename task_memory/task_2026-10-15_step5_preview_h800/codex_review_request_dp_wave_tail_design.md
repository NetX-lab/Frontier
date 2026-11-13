# Codex review request: DP wave tail design (calibration item 4)

Report language: English.

## Scope

Review the modelling design `task_memory/task_2026-10-15_step5_preview_h800/design_dp_wave_tail.md`
in the Frontier LLM inference simulator (repository `/data/ycfeng/Frontier/.worktrees/step5-preview`,
branch `feat/step5-preview`, HEAD `8127b87f`). This is a design review; no code has been written
for it yet. Do not edit files.

Context:
- Requirements and plan: `task_memory/task_2026-10-15_step5_preview_h800/requirements.md`,
  `plan.md` (item 4), evidence report `test_report_2026-10-27_h200_64_ttft_wave_tail.md`.
- Native vLLM source used by the reference runs: `/data/ycfeng/tmp/d29_published_sources01/source/vllm`
  (`v1/engine/core.py`, `config/vllm.py`, `config/parallel.py`, `v1/worker/gpu/model_runner.py`).
- Frontier code involved: `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
  (capacity check 391-427, `_create_batch` 430-470, `on_schedule`/`_run_dp_iterations` 555-640,
  `_end_iteration`, `_issue_dummy_pass`, `on_dummy_forward_end`, `next_pass_start`,
  `hold_batch_output`, in-flight rule 1136-1150), `frontier/scheduler/utils/dp_dummy_forward.py`,
  `frontier/scheduler/utils/forward_start_agreement.py`, `frontier/events/batch_stage_end_event.py`
  (120-185), `frontier/events/replica_schedule_event.py`, `frontier/events/replica_stage_schedule_event.py`,
  `frontier/config/cluster_config.py` (555), `frontier/config/cluster_role_config.py` (261),
  `frontier/config/replica_scheduler_config.py` (98, 334-338),
  `frontier/execution_time_predictor/kv_block_zeroing_host_time.py`,
  `tests/integration/step5_pd/frontier_workload.py` (102).
- Diagnostic evidence: `/data/ycfeng/tmp/d31_diag/wave_frontier/` (`wave_probe.py`,
  `h64host_wave.jsonl`, `probe_summary.txt`, `native_async_flags.txt`, `vllm_batch_queue.txt`).

Project rule D31.10 (user decision): Frontier must not use a "whole-step time table", meaning
measured whole-step times looked up by batch composition. Such tables are diagnostic only.
Device time must come from op profiling; a measured component host time from a probe (like the
item 3 KV block zeroing host time) is allowed.

## Questions

1. Native mechanism (design section 2): is it correct that, with async scheduling and the V2
   model runner, the prefill engine returns from `step_with_batch_queue` before the prefill
   output, so `_has_global_unfinished_reqs` on step 1 sees an unfinished request and the wave
   runs to the next multiple of 16 steps? Is it correct that, while every engine runs dummy
   steps, the host issues them at its own pace (about 5.6 ms) and the device runs them in
   order (about 32 ms each), while a real batch on one engine holds the others' hosts through
   the per-step DP CPU all-reduce in `dispatch_cg_and_sync_dp`?
2. Change 4.1 (PREFILL batch queue of two): is it a faithful model of the native PREFILL engine?
   Check the interaction with the in-flight rule, `hold_batch_output` / `_pop_batch_output`,
   the KV transfer start on the PREFILL role (does it follow the batch end or the output pop in
   Frontier, and which one matches native NIXL handoff timing?), chunked prompts, grouped KV
   allocation and the item 3 KV block zeroing trigger.
3. Change 4.2 (host-paced dummy steps): is "every lane of the DP group has no real batch in
   flight and no unpopped output" the right condition? Does the existing forward-start agreement
   and the per-layer MoE room pairing still pair the k-th dummy forward of every lane when lanes
   queue several dummy forwards on their stage? Does the existing DP-sync logic (`_end_iteration`,
   `consume_dp_sync_release`) still hold when the next pass no longer waits for the forward end?
   Is there any path where a held-back pass could deadlock or double-count host time with the
   item 3 reset host time (`hold_forward_start`)?
4. Profile placement: a new profile `dp_dummy_step_host_time.v1` versus a second term in the
   item 3 profile. Which fits the existing predictor-option plumbing better?
5. Does the design respect D31.10? Is anything in it a whole-step time in disguise?
6. Anything the design misses that would make the DP4 TTFT prediction wrong (for example the
   dummy forward device cost at DP2 creating a false backlog, the dummy forward of the other
   lanes while one lane prefills, or the DECODE role's own DP wave)?

Report concrete errors with file:line evidence and a recommended correction.
