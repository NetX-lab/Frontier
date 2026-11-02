# Task: Check the P2 NIXL payload edge under route C (Q3)

## Goal

The P2 NIXL READ price uses a source-derived payload: 46 selected pages × 1,002,240 B = 46,103,040 B per TP rank pair for a 6144-token request (structural 39,521,664 B). Native vLLM never emitted the final per-request descriptor IDs, so the Q1 ledger keeps the edge "Actual native request-selected descriptors/rank wire bytes" OPEN.

YC chose route C (requirements.md, D-P2-PAYLOAD-C). The payload is accepted for P2 when an upper bound on native bytes still leaves every READ finishing inside the decode-step gap, because with `nixl_read_completion=engine_step` the events and metrics are then the same as with the source-derived payload. Settle the edge against that criterion exactly as recorded:

1. **Upper bound `B_max`.** From the retained native vLLM source, the observed registrations and the retained P2 workload, derive the largest number of bytes one native NIXL READ can move on one TP rank pair for any P2-32 or P2-64 request. A reduction from trimming, null pages, aliases, padding or rank mapping counts only where the retained source applies it unconditionally on the P2 path. Otherwise count every page the request holds in every KV cache group, at the registered stride. Give the bound per group, and compare it with the source-derived 46 pages and the structural bytes.
2. **Model check.** Price one READ of `B_max` with the adopted H200 pair (390.39349928254796 Gbit/s, 0.0236521842818856 ms) through the production `AnalyticalKVCacheTransferPredictor` formula. Compare it with the next-step gap of every one of the 2,034 possible posting steps: minimum 18.724965 ms (P2-32) and 24.380076 ms (P2-64).
3. **Measured check.** `B_max` must be at most the largest bytes per reader measured with eight active pairs in nixl6144. At the eight-pair coordinate with the smallest bytes per reader at or above `B_max`, every trial in every pass must stay below both gaps. Report all trials there; never remove slow ones.
4. **Bytes reach only the transfer time.** Show from the Frontier source which outputs and events depend on the payload bytes. The four compared metrics and the event sequence must depend on them only through the READ transfer time. Bytes used elsewhere, such as memory, scheduling or a reported metric, fail this check.

Each check gets PASS or FAIL against the recorded criterion. If any check fails, or `B_max` lies outside the measured grid, the edge stays OPEN: return the evidence and a proposal with cost (for example, a smaller native witness). Do not change the criterion after seeing a result.

Acceptance: `B_max` comes with cited source lines and a per-group derivation; all four checks have actual values and PASS/FAIL; the ledger row has a proposed new status and wording.

## Approach

- Native source (read-only): `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/`. Start from `distributed/kv_transfer/kv_connector/v1/nixl/`: `base_worker.py:99` flattens group IDs, `pull_worker.py:225` trims and maps ranks, `pull_worker.py:322` calls `make_prepped_xfer`, and `base_worker._register_packed_kv_cache` registers the packed region. Follow the block IDs back to where the scheduler-side connector chooses them, including any clipped-block or sliding-window rule. The line numbers come from the Q1 inspection; check them.
- Registration and layout evidence: 96 native registrations, one region with a 1,002,240 B stride. See `h3o_operator_costs/qualification_codex_20261031/qualification_report.md`, `h3o_operator_costs/b30b/kv_layout_evidence.json`, `h3o_operator_costs/two_fix_20261101/nixl_boundaries.json`, and the ledger rows in `input_qualification_20261101/ledger.md`.
- Frontier payload path: `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py` (`get_grouped_nixl_payload`, `get_grouped_nixl_wire_bytes`, `get_transfer_time`), `frontier/attention/memory.py` (`physical_block_size_bytes`), `frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py`, `frontier/events/kv_cache_transfer_start_event.py`.
- Timing and gaps: `input_qualification_20261101/nixl_timing_q2/` (`proposed_input.json`, `possible_post_windows.csv`, `transfer_windows.csv`, `report.md`) and the nixl6144 data in `input_qualification_20261101/nixl6144_run/` (`case_summary.csv`). Every reader trial is in `/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download/C_nixl/nixl_reads.jsonl` (read-only). `nixl_timing_q2/scratch/reader_trials.csv` holds the same trials in flat form.
- Workload: confirm the P2 request prompt lengths, and any prefix-cache hits, from the retained P2 replay inputs used by `h3o_operator_costs/two_fix_20261101/run_replay.py` and `h3o_operator_costs/a10/a10_driver.py`.
- The payload observer contract `input_qualification_20261101/payload_observation_contract.md` describes route B. It is context only; do not build it.

## Scope and context

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`. HEAD is `7ea5ec97` or a later records-only commit. The tree is clean apart from the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`, which are not yours.
- Task records: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/{requirements.md,plan.md,progress.md,summary.md}`. Read D-P2-PAYLOAD-C, D-P2-Q2-NIXL-TIMING, D-P2-Q1-CHOICES and D-P2-INPUT-QUALIFICATION in requirements.md.
- Constraints:
  - Do not edit `frontier/`, `data/profiling/`, the native vLLM tree, thresholds, references, formal tables or the formal timing input.
  - No replay is needed if all checks pass. If a check shows the bytes can change an event, report that; do not replay.
  - Retained evidence, receipts and the raw download are read-only.
  - No RJob, platform, cloud or GPU action.
- Writable paths: evidence, scripts and the report go under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/payload_q3/`, with scratch in `payload_q3/scratch/`. Do not commit; the lead reviews and commits.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Run heavy reads under `timeout`.

## Expected return

- Report language: English, plain words, actual values.
- `payload_q3/report.md`: the derivation of `B_max` with source citations, the four checks with expected and actual values and PASS/FAIL, the proposed ledger status and wording, and limits. Put machine-readable results next to it.
- For each check: command, environment, expected versus actual, PASS/FAIL.
- Changed and new paths, and any remaining blocker with evidence and a recommended next action.
