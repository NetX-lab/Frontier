# D30 native finish-sync analysis

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Mark approved source/CPU correction complete; retain unmeasured native boundary limits. |
| 2026-10-22 | Preserve inspected source/args evidence and its limits. |

Analysis is complete for the configured algorithm. Approved source/CPU correction
and metadata consumer are implemented and verified; live-native qualification is
pending. Exact implementation and test evidence are in `summary.md` and the test report.
Investigator: `ses_e9e623c4cffe1X4qP7jVRUOnKv`; parent independently read both
conditions. No retained trace directly measures the synchronization boundary.

Native root: `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`.
`config/parallel.py:162` declares `dp_sync_interval=16`;
`v1/engine/core.py:2158` consumes it and2405–2415 increments the counter and
synchronizes on step1 or a multiple of the interval. It resets at wave end.
Pinned hashes: parallel.py
`4b5f3d5938cdb4644367a7c4c33cc657f8834d83a69f8a838ef4168627e12c21`, core.py
`5a0344942c5af99cd85eb68d9eded7f856df40ff0986bda8de6d2ff22f6d357e`.

Capture01 and both D25 engine/config/startup records bind the same image
`a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`.
Native pin checks precede readiness. The canonical command builder/config and
parsed non-default startup args contain no interval override. There is no
DP_SYNC environment hook. This establishes the source/configured value;
collective timestamps and exact end-of-wave trajectories remain unmeasured.

Frontier atfa5bae2: `vllm_v1_engine_replica_scheduler.py:57–59,581–601`
has module constant32 and synchronizes only at its multiples. Preserve this
legacy contract for unaffected models while binding Step to current native.

Existing saved-state inspection finds both P lanes empty of useful requests,
allocations and transfers during two Dq1 intervals, but P dummy wave11 remains.
This is not evidence of a source lease keeping P alive. It also cannot locate
the native first idle synchronization. If all lanes are idle by step1, missing
the first sync can mean31 extra iterations; later16-vs32 boundaries differ by
up to16. Do not assert that every observed P dummy is spurious or that changing
the interval alone closes all workflow/performance gaps.
