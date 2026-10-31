# A6 repeatability probe verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded pre-submission geometry, source, payload and reservation checks. |

## Environment and commands

Workspace: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, HEAD `2b4da71e`; frozen measurement source `a689ed05`. CPU Python `/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6; platform Python `/data/ycfeng/tmp/stepmind-env/bin/python`, 3.10.6. Temporary paths use `/data/ycfeng/tmp`. Shell is bash for proxy initialization, zsh otherwise.

- Geometry/source: `env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/verify_a6_repeatability_20261101.py`. Retained exact script: `h3o_operator_costs/a6_repeatability_20261101/verify_probe.py`. It uses the existing streaming reader with a 30-GiB RLIMIT_AS; systemd scope requires unavailable interactive authentication as recorded in the earlier A6 report. Expected: every clone preserves the original case and native measured geometry, with source/hash equality. Actual: exit0, 11.632083s; 546/546 manifest files, 527 source files, 128/128 clones, 8/8 original raw steps, zero geometry mismatches. PASS. This is not full-grid coverage. The retained geometry JSON omits only the unrelated full-grid `windows` and `classes` catalogues; every observed/predicted step result remains.
- Actual ResourceConfig dry run: `env PYTHONPATH=/data/ycfeng/steptron PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/glm53_dsv4/a6_repeatability_payload_20261101.py`. Exact script retained as `verify_payload.py`. Only `spawn_tasks` is replaced with a capture callback. Actual: exit0, one H200/step_main, non-preemptible,16CPU/128GB, local frozen source, explicit company image, cloud mount,5400s payload timeout. PASS, zero platform operations. Existing Paramiko TripleDES deprecation warnings are retained in the tool output; they do not change the payload.
- Personal reservation query: handbook proxy initialization followed by `timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/glm53_dsv4/a6_repeatability_quota_20261101.py`. Actual exit0; one complete page, six existing personal jobs, all preemptible, zero personal non-preemptible reservations. Quota-group API403 means free capacity UNKNOWN; no platform access or capacity claim is inferred.

Evidence and frozen hypothesis: `h3o_operator_costs/a6_repeatability_20261101/`. `probe_plan.md` records selection, limits, analysis and budget before launch. The package manifest SHA-256 is `c7082cef1d4c6ff32fe8fb106dbef363ced1fbb3c4c747c4eca4007b67e37822`.

## Current result

CPU preparation PASS. RJob `exp-1101-015820-503227` is Running; business/terminal validation is pending. Personal API confirms creator `i-fengyicheng`, one H200, pinned image, local NFS `100.96.128.195:/data/ycfeng/Frontier/.worktrees/g1h-supplement-build` and actual container memory125Gi. Worker `status.json` confirms NVIDIA H200143771MiB/driver570.124.06, Python3.12.3, torch2.13.0+cu130, vLLM0.28.0, FlashInfer0.6.16.post3 and the expected source revision. Evidence: `platform_01.json`, `worker_status_01.json`, `launch_command.json` (still running). A6 qualification and tickets05/09 remain open. Formal measured CC, K19, ingress10.3ms/handoff11.4ms, references, folds and all thresholds are unchanged.
