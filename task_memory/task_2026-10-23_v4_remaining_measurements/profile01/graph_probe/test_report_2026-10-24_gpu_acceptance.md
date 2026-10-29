# Callback graph correspondence: root GPU acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Accepted one H800 graph-correspondence diagnostic after full-data audit, independent review and fresh root reruns. |

Source1aa1e4a5, package6f84a88b/2264f213, merged4add627d. Frozen manifest `ec816706edeb0930756ad7077684a2872b93c7728ae8851264e3e2592409c146`; eight input hashes unchanged. Source/package reviews found no actionable defect. CPU correction/image evidence is in `root_checks/`; final staged-image and actual-SDK entry/constructor checks are in `package_checks/`. SDK submission and local receipt writes were intercepted only in the CPU control; actual platform execution follows below.

## Actual execution

`root_seal.json` and `root_submission.json` bind the exact launch command and authorization. Personal Python RJobBackend submitted exp-1024-134447-148581: H800/codesign/non-preemptible, CPU16/mem_gb64=64000Mi, one replica,600s workload ceiling. Actual NFS source/target is the package on kun-workspace-vgen2 at100.96.128.195. The launcher completed normally; platform/container Succeeded/Completed, exit0/restart0,12container seconds. Image digest2286e853 and worker `nvidia-smi` H800 identity match. Platform-initialization warnings remain in the redacted worker log.

`execution/platform.json`, `execution/worker.log` and `root_run_check.json` retain exact observations. Read-only sudo inventory/tar of the exact personal cloud root retrieved four files/53146bytes; hashes match `execution/download_receipt.json`, with a persistent small raw copy under `results/raw/`. Original cloud bytes and permissions were not changed.

## Data verification

- `results/command.md` records the exact Python3.12.13 commands and136-line independent auditor, archived inside `data_audit.json`. It does not call producer decision functions.
- U1–U4 PASS:9nodes/5kernels/4event boundaries;9unique complete clone callbacks and one graph1→exec2 callback. All five target kernels join through exact full IDs and launch correlation; four belong to two event-delimited scopes and one is unscoped. The separate eager kernel is excluded.
- One target replay, two retained-source snapshots,52diagnostic stages, empty fault/error records; successful unsubscribe precedes Kineto. Six negative fixtures are rejected.
- Independent reviewer `ses_e93abff63ffeaLmGwRc633Vskw` read all raw files and the algorithm: No Findings. It did not rerun or rehash. Root independently executed two fresh processes; all31checks and six negative controls match, byte-identical digest `1a56bb00bd7fa8bde329f2c9e9933e97fc237b2d70232659edc1021281235905` (`results/root_verification.json`).

`results/primary_acceptance.json`: **ACCEPTED_GPU_GRAPH_CORRESPONDENCE_WITH_LIMITS**. Raw native descriptor buffers, worker binary rehash, numerical accuracy and repeated-replay/clone-ID stability were not measured. Old crash cause remains an inference. This tiny probe does not admit general native/V4 serving ownership, operator costs, consumer behavior, reference runs or E2E accuracy.
