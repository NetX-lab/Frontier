# Single scope/lifetime GPU diagnostic acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Accepted the completed standalone P1–P6 probe with explicit control/native/cost limits; preserved R1 and all immutable evidence. |

Primary verdict: **ACCEPTED_SINGLE_IMAGE_SCOPE_LIFETIME_WITH_LIMITS**. [Decision](primary_acceptance.json) binds the combined full-source/R2 raw review; this is a standalone diagnostic, not native serving ownership or cost/reference admission. Report date follows the task's Hong Kong label; measurements retain their exact2026-10-24 UTC timestamps.

## Commands and environment

- Unique GPU launcher,2026-10-24T18:12:24.289020Z: `bash /data/ycfeng/tmp/glm53_dsv4/scope_prelaunch_20261025_01/with_personal_env_ready.sh /data/ycfeng/tmp/stepmind-env/bin/python -u -B /data/ycfeng/Frontier/.worktrees/v4-scope-lifetime-package/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime/h800_scope_lifetime_probe_01/launch.py --launch`. [Submission snapshot](../package_primary/root_submission.json), [launcher result](../package_primary/execution/launcher_result.json). Personal local StepMind Python RJobBackend/i-fengyicheng, STEPMIND_BACKEND=rjob; Python3.10.6/brainpp2.8.6.
- GPU command runs one CLI process, observe then unsubscribed control. Locked image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`;1H800/codesign/non-preemptible/16CPU/64mem_gb=64000Mi/600s ceiling. Original absolute NFS source and cloud root remain frozen in [seal](../package_primary/root_seal.json).
- Primary R2 raw audit: `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_scope_lifetime_audit.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime/results/primary_raw_audit_r2.json`. Python3.10.6; [receipt](raw_audit_receipt_r2.json), [stdout](raw_audit_r2.stdout), [stderr](raw_audit_r2.stderr). This prior completed audit was not rerun during closeout.
- Closeout hash/scalar binding: `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/scope_acceptance_20261025_01/verify_retained.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime/results/retained_binding.json`. [Exact source](retained_binding_source.txt), [receipt](retained_binding_receipt.json);79checks PASS/49immutable review artifacts/18frozen inputs and archive copies,0.062230s. Current editable task documents are excluded from immutable hash claims.

## Expected and observed evidence

| Requirement | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Fresh gates before seal/submission |Current storage/project quota/NFS/destination/quota receipts |18inputs/67pins; fresh gates and root seal preceded the unique launch |PASS; snapshots were not reservations |
| Execution identity/terminal |Personal creator, exact image/mount, actual H800, terminal exit0/no restart |`exp-1025-021228-212042`;gpu-h800-0158;GPU-4c9987eb-19e6-793f-0326-c502be48abcd;Succeeded/Completed/exit0/restart0/launcher0;20binding checks |PASS |
| P1 callback delivery |Valid callback-only scalar copies and five complete original tuples |11created/5clones/1exec;all five selected originals are kernel nodes in the directly bound source graph |PASS this probe |
| P2 thread observations |Separate OS/Python IDs |Five rows record both IDs;frame equalities observed separately |PASS observations only |
| P3 freshness |Unique monotonic op window plus exact scope equality |Five exact start/end pairs and five fresh matching scopes |PASS this probe |
| P4 lifetime |DefaultFalse, one post-capture expected-RuntimeError boolean per graph, no source dereference |True boolean and matching rawexec;full frozen-source/R2 review checks restrictions |PASS this probe |
| P5 closure |Full source(originalhandle,fullID) tuple to clone(originalhandle,originalID) |Five unique direct tuples;source/clone namespaces retained |PASS this probe |
| P6 replay |One contained launch, five unique full-ID joins, no missing/duplicate target kernel |Five direct source→clone→replay joins;zero unmatched target kernels |PASS this probe |
| Same-run control |Unsubscribed five-kernel topology and exact output identity |Same kernel names/grid/block and five1024-value vectors(2,2,3,3,2);originalcontrol.verdict=UNVERIFIED retained |PASS topology/output only |
| Raw/download binding |Every output preserved/hash-bound |Fivefiles/224883B, individual hashes exact |PASS |
| Primary/independent verification |R2 findings closed and frozen identities bound |60primary raw checks/0.122845s;73independent scalar checks/17negative rejections;threeR1 omissions CLOSED;79closeout bindings |PASS within declared coverage |

Exact full IDs, callback/window timestamps and all evidence SHA256 values are in [primary_acceptance.json](primary_acceptance.json), [primary_raw_audit_r2.json](primary_raw_audit_r2.json) and [R2 review](review_runtime_r2.json). Final audit152/190lines, SHA256539da5af58b362d0ecdea09af8e9c6c3157f21af7f530fb12c3629ffce658f93;probeSHA256bc638fbdf0111a8ce78383d182d41e19a5799bacc6cc5a657aa79b1a34672685;manifestSHA2569ff1abec5fa2451baa2a285bdf081011eba9a2741448ad39a64b9a27aae4198e.

## Preserved failures and verification limits

[R1](review_runtime_r1.md) and the original audit/result/logs remain preserved. R2 adds explicit observe.errors rejection, exact op start/end pairing and unique successful close-before-Kineto. The initial four-line platform-log read and root-owned0600 cloud-read PermissionError remain in retained operation records; neither counted as GPU acceptance. No package regeneration, source-path change or repeated GPU/CPU-control run occurred.

Archive verification: the initial `git diff --cached --check` returned2 with20 trailing-whitespace observations confined to the two immutable Kineto traces. Both raw traces remain byte-identical to their download hashes. The separate check of code and maintained records excludes only those two files; it must pass before commit. This formatting observation does not change any raw-data or GPU verdict.

The allocation is spent1/1. Six unmatched-created causes remain UNKNOWN. Host driver release is UNRECORDED;traceAPI13000/CUPTI130001 are telemetry, not new driver/library binary qualification. Thread equality does not prove TLS;freshness comes from the monotonic window and exact scope. Control does not establish timing/non-perturbation. This ordinary pointwise probe does not establish native compiled/breakable serving ownership, dependencies, operator/serving cost, clean E2E or reference replacement. Native06R4 remains the reference;P0/R5/R6/R7 are independent unfinished work.
