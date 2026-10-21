# P0 canonical G3/G4 native capture01

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded sealed package, direct personal submission and platform identity verification. |

## Status

**Submitted; platform Starting at the first identity check. Native evidence and G3/G4 judgment are pending.**

- Job: `exp-1021-235955-532667` (created2026-10-21 23:59:55 local time).
- Stage: `/data/ycfeng/tmp/step5_canonical_g3_20261021/capture01/stage`.
- Source: `025d9a9fa638e035b4f9111a2e77ec1a93b70a49`; implementation `1171ab6`.
- Config SHA256: `2f3d858f6913f276dee767fe04c279a0a885e478022e4a60462c241dba9914c3`.
- Cloud output: `/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d21_canonical_g3_01`.
- Scope:163 diagnostic requests/14 waves; three independent direct-D full-local references, eight content-selected requests,128-request pressure attempt, existing boundary/reuse/reset/abort controls and all-rank MRv2 forward census. Clean timing is ineligible.
- Resources:4 replicas×8 H800, codesign, non-preemptible;32CPU/128GB requested per replica;5400-second ceiling/48GPU-hours. Expected roughly20–40 minutes; actual runtime remains unmeasured.

## Verification and submission

- Primary CPU verification:685 affected tests plus69 final source/stage/seam checks passed; see `test_report_2026-10-21_parent_reference.md`.
- Offline stage:49 sealed files,48 tool pins,50 installed native pins,12 observation pins and14 schema pins. Verifier `PASS_OFFLINE_STAGE`; all Python files compile and `bash -n submit.sh` passes.
- Real StepMind Python dry-run passes. Submission uses the existing `submit_native_pd.py`, `launcher_scope.py` and personal StepMind environment, with the RJob Python backend. The provider is custom, so operations remain in the primary rather than the disabled helper provider.
- Pre-submit personal listing used exact selector `kubebrain.brainpp.cn/creator=i-fengyicheng`, pagination exhausted in one page; zero personal nonterminal jobs. codesign GPU quota128; this request32. H200 reservation remains0/32.
- Actual platform identity: creator `i-fengyicheng`, codesign, preemptible=`no`,4×8GPU, both RDMA device limits present, expected locked image digest and local NFS source/target verified.
- Actual NFS: `100.96.128.195:/data/ycfeng/tmp/step5_canonical_g3_20261021/capture01:/data/ycfeng/tmp/step5_canonical_g3_20261021/capture01`.
- Launcher Python PID503604 was alive after submission. `setsid` detached from the tool wrapper; the shell completion notification was not job completion. The existing callback recorded its own SDK exit-stop removal and preserved native submit/wait behavior.

Actual launch command (already executed; do not repeat this generation):

```bash
umask 077
exec setsid bash /data/ycfeng/tmp/step5_canonical_g3_20261021/capture01/stage/submit.sh \
  > /data/ycfeng/tmp/step5_canonical_g3_20261021/capture01/launcher.log 2>&1
```

Retained local evidence, all under `/data/ycfeng/tmp/step5_canonical_g3_20261021/capture01/`:

- `run_definition.json`, `stage_source_manifest.json`.
- `offline_verification.json`, `presubmit_verification.json`, `stepmind_dryrun.json`.
- `quota_presubmit01.json`, `parent_identity01.json`.
- `stage/launcher/{payload,submission_attempt,submitted}.json`, `launcher.log`.

## Native acceptance still required

Inspect worker GPU identities, image/source verification, all request records and native logs, complete resident/component/reference matches, actual READ/GPU-readiness/retain/free and reset/reuse/abort witnesses, actual allocator pressure, all32 rank forward accounting and terminal status. The offered pressure load and10ms abort deadline alone never qualify those controls. Native numerical differences keep existing tolerances/dispositions. Auxiliary diagnostics remain soft: a missing or failed witness does not justify stopping healthy execution.

The MRv2 descriptors will also supply real profile inputs for the corrected producer (`82f4452`). The immutable no-metadata profiling stage01 packages remain withdrawn; no profiling admission is inferred from this submission.
