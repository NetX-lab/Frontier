# D21-P0-PROFILES requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record parent assignment at bc8ae27 and P0-only implementation limits. |

## [Original Request]

Continue ticket10 in clean `/data/ycfeng/Frontier/.worktrees/step5-p0-profiles` at `bc8ae27`. D22 native18/18 candidates,22/22 output/cache,18/18 prefixes already PASS; no more boundary experiments/retry. Deliver executable native request-free idle-DP producer (`has_attention_metadata=0`) and independent P/D profiles for P0 H80032, through existing native helpers/scopes/staging and unchanged D22 timing. Preserve positive projection/norm/output-zero-fill/FFN work and no attention metadata/history; never substitute ordinary q1, zero durations or H200 timings.

Own `frontier/profiling/step5`, necessary existing linear/MoE adapters/configs and focused tests. Do not edit simulator/predictor, canonical/G3 harness or consumer admission/coverage. Derive measured inputs from D20–D22 requirements, ticket10, measured-compute handoff, actual-main request-free records and locked native source/runtime evidence. P0 is H800/codesign/non-preemptible,32-serving-GPU topology P16/D16,eachTP8/DP2/EP16/PP1; real prompt6144,observed transfer/P-local6128,D-tail16,output1024. Prepared independent rank-local measurements are not32-GPU serving qualification. Real variable decode widths and complete admission must be explicit,not inferred from CPU fixtures. Reuse compatible evidence without relabelling. Linear quality remains unqualified.

Deliver implementation,meaningful CPU checks,exact commands/grid and new sealed parent-only package(s)/budget under this directory. When actual reached shapes are unavailable, provide callable support with bounded prepared coverage and explicit gaps. Preserve all old stages and historical verdicts. No GPU/auth/platform/cloud/SSH/network/delegation. Commit verified scoped changes and records; parent submits.

## Inspected prerequisites

- Parent requirements: `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/requirements.md` D20–D22.
- Local ticket10; `task_memory/task_2026-10-21_step5_measured_compute/handoff.md`; `task_memory/task_2026-10-21_step5_d20_integration/actual_main/{progress,test_report_2026-10-21_actual_main}.md`.
- Source `frontier/attention/step5.py:499–535,726–842`; predictor `step5_cost.py:59–124` read-only.
- Locked local exported vLLM `models/step5/nvidia/model.py:1271–1424,1514–1677`, `sparse_attention.py:377–402`, `v1/worker/gpu_model_runner.py:5805ff`.
- P0 native judgment: actual P6128/D16 and32 H800 identity established; variable reached request arrays,actual graph replay widths and full admission remain absent.
