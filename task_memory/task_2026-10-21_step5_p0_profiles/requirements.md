# D21-P0-PROFILES requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record parent assignment at bc8ae27 and P0-only implementation limits. |

## [Original Request]

Next ticket10 mainline implementation, D21-P0-PROFILES. New clean worktree /data/ycfeng/Frontier/.worktrees/step5-p0-profiles at bc8ae27 contains your D22 + complete measured compute/PD/request-free consumer; switch there. D22 H200 probe has now PASSED18/18 strict candidates,22/22 output/cache,18/18 prefixes (native receipt in task_memory/task_2026-10-21_step5_d22_timing/native_judgment/); no further boundary diagnostic/retry is needed or wanted. P0 H80032 is priority. Goal: deliver executable native producer support and a coherent P0 measurement package feeding the existing Step predictor, starting with the missing request-free idle-DP workload (has_attention_metadata=0) plus required independent P/D rows. Current rows.py emits1 for normal cells; acc3a88 consumer/scopes strictly distinguish native _dummy_run(1): no attention metadata/cache history, but positive projections/norm/zero-fill/FFN work. Do NOT fabricate regular q1 rows, zero duration, change consumer coverage/admission, or relabel H200 times as H800. Own frontier/profiling/step5, narrowly necessary existing linear/MoE producer adapters/configs and focused tests; no simulator/predictor files or canonical/G3 harness. Read current requirements D20-D22, ticket10, measured-compute handoff, actual_main request-free record and runtime features to derive the measured inputs. Use existing native scopes/helpers/job staging and approved D22 cohorts. Existing R1/P0 local evidence gives6144 prompt,6128 transfer,16 D tail,1024 output,TP8DP2EP16; real variable decode/dummy width coverage and complete admission must be explicit, not inferred from synthetic fixtures. Configure P0 H800/codesign nonpreemptible new package(s), preserve old sealed stages. Reuse valid existing measurements where compatible; linear quality remains unqualified. Implement missing producer capability with meaningful CPU checks and provide exact producer commands/grid coverage and parent-only stage/budget. Do not stop at another inventory/design document; where full grid depends on actual native shapes, still deliver the independent callable/producer support and clearly bound prepared coverage. No GPU/auth/platform/cloud/SSH calls or further delegation; primary submits. Commit scoped changes/records under task_memory/task_2026-10-21_step5_p0_profiles. Return compact commits/test evidence, paths, remaining admission gaps and task ID.

## Inspected prerequisites

- Parent requirements: `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/requirements.md` D20–D22.
- Local ticket10; `task_memory/task_2026-10-21_step5_measured_compute/handoff.md`; `task_memory/task_2026-10-21_step5_d20_integration/actual_main/{progress,test_report_2026-10-21_actual_main}.md`.
- Source `frontier/attention/step5.py:499–535,726–842`; predictor `step5_cost.py:59–124` read-only.
- Locked local exported vLLM `models/step5/nvidia/model.py:1271–1424,1514–1677`, `sparse_attention.py:377–402`, `v1/worker/gpu_model_runner.py:5805ff`.
- P0 native judgment: actual P6128/D16 and32 H800 identity established; variable reached request arrays,actual graph replay widths and full admission remain absent.
