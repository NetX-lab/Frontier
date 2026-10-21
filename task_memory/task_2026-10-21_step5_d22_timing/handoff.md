# D22-TIMING parent-only GPU handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Seal source-bound full D18 rerun with preemptible H200 resources and unchanged strict bounds. |

## Ready package

- Source: `74df0100ca0f39d6ff134d999909e863c3db6962` on `feat/step5-profiling`.
- Stage: `/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01`.
- Manifest SHA256: `a1dc28ceca71235b000e7f7448464a5ff76d27e76d2d51ee63fe0811908b5312`.
- Preflight/source audit: `evidence/package_preflight01.json`, `evidence/source_stage_audit01.json`:3407 payloads and exact inventory PASS;525 committed source payloads verified. Original stage02 unchanged:3407 hashes/inventory PASS and original manifest `42545748def4f40947037a61ca68541d4f3736cd5e8faaa60a8ec3698be91077`.
- This is the **unminimized D18 probe**, not another boundary experiment:18 attention CUDA_EVENT rows +4 GPU-active KERNEL_ONLY rows, both original linear methods with8 rows each. Same native policy and exact full worker command after run-ID substitution; no shape/grid reduction.
- Resource: **one H200 / step_main / preemptible=True /8 CPU /32GiB**,3600-second workload cap. No non-preemptible H200 allocation is requested. Conservatively retain original20–45-minute planning budget (0.33–0.75 GPU-hours), maximum1 GPU-hour of worker command; platform startup/queue/cleanup are outside that timeout. This is an estimate, not a measured D22 duration.
- Image: `a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`.
- Source mount is the same-path local NFS stage directory; dynamic volume `juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp`. All durable outputs confined to `/mnt/codesign-exp/ycfeng/frontier/step5_profiling/fixed_h200_d22_probe_20261021_01`.

## Parent next actions

1. Before any platform action, read the current StepMind handbook, load personal `i-fengyicheng` auth and check actual quotas and higher-priority P0 needs. D21 personal nonterminal H200 non-preemptible reservations remain <=32 including other sessions; do not change this sealed preemptible package to consume that allowance. No quota/auth/cloud checks were made by the implementation agent.
2. Optionally rerun the frozen offline verifier from the test report with a fresh output suffix. Do not modify or reuse the sealed stage or old output directory.
3. Submit through the existing credential-aware wrapper and **keep its launcher alive** until terminal status:

```bash
bash /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01/submit_with_env.sh
```

The wrapper sources `/data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh`, sets `STEPMIND_BACKEND=rjob`, and executes the existing Python RJob launcher with `--preemptible`. Do not run bare `submit.sh` without credential environment. If a stop is necessary, parent uses the mandated existing `rjob_stop.sh` path, not a new stopping route.

4. Record creator, source mount, worker physical H200 identity/driver/compatibility, terminal status and business outputs. Retain original CSVs, traces and worker logs. Obtain small local copies for judgment; this task did not access the cloud.
5. Create a **new local receipt directory** containing these exact files without changing their contents:
   - `<cloud-output>/attention/summary.json` -> `summary.json`.
   - `<cloud-output>/attention/compute/h200/stepfun-ai/Step-5-Preview/step5_attention.csv` -> `step5_attention.csv`.
   - Same directory's `step5_attention_kernel_only.csv` -> `step5_attention_kernel_only.csv`.
6. Run the new frozen strict judge on that new local directory:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01 \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B \
  /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01/tests/integration/step5_profiling/judge_attention_smoke.py \
  /ABSOLUTE/NEW/LOCAL/RECEIPT_DIRECTORY \
  --expected-event-rows 18 --expected-gpu-active-rows 4 --max-time-cv 0.05 \
  --report /ABSOLUTE/NEW/LOCAL/RECEIPT_DIRECTORY/d22_strict_verdict01.json
```

Replace the two uppercase local-path placeholders with parent-retained paths. The report is exclusive-create. Exit0 is not sufficient: inspect all row/count/functional/prefix/cache/graph/window/normalization/scoped/CUPTI/5% CV and mean-versus-reference checks. Inspect linear outputs under `<cloud-output>/linear/compute/h200/stepfun-ai/Step-5-Preview/` against the existing full linear plan, without inventing independent linear admission from same-pool medians.

## Limits and integration

- D22 changes the layer agreement estimand only. Candidate and reference windows are independently executed and interleaved C/R at matching replay counts; raw boundary singles are labelled diagnostics. No overhead subtraction or threshold relaxation. No quality-based retry; every cell records one attempt. Eager GPU-active and scoped operator protocols are unchanged.
- CPU final873 PASS preserves all851 baseline IDs; real-seam event-overhead RED->GREEN and final source/package identity evidence are recorded. Native D22 timing remains UNVERIFIED until the fresh parent run; historical D18 and H800 Smoke03 FAIL receipts remain FAIL.
- One physical GPU represents isolated TP8-local shapes with process group TP1. This is not full TP8/DP/EP serving, capacity, transfer or batching qualification; no full-grid, canonical PD or predictive E2E acceptance follows.
- Compute branch `83af0a7` was not merged. `rows.py` contains only this task's reference-statistics changes over79c8b45; parent resolves the separately known three-line integration overlap. Source package stays sealed to74df010 even if later integration changes the working branch.
- No nested review/delegation was performed. Parent owns integration/review and actual run/judgment. Stop here; no implementation-agent GPU operation is authorized.
