# Profile03 root preparation and submission

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Verified frozen inputs, actual image/SDK and submission identity; retained auxiliary checker failures. |

## Source and package

Source99467718/records882b25ae with records-only citation/diff repair438ddfc4 merged as39e23132. Both package reviews closed. Maintained source/config delta68/140gross; generated45-file package remains6983474bytes. Runtime13-module binding to1437ace6, original frozen/checker inputs, actual Git blobs, historical inventory, fixed source diff and package closure pass165root checks (`root_checks/binding/receipt.json`). Initial root check raised `KeyError: 'package_files'`; the inspected schema uses `package_hashes`. Only the checker was corrected.

Actual locked-image five checks PASS (`root_checks/image/receipt.json`):87native sources/23tool sources/20binaries; fresh bootstrap entry and four role/lane hook controls, P512/D64,600000records/268435456bytes. These are CPU-side entry and hook controls with explicit constructor/config stand-ins, not GPU model execution. Docker argv, stdin, Python3.12.3 and complete output are archived there.

Actual StepMind ResourceConfig expression passes in Python3.10.6 (`root_checks/sdk/{input.py,execution.json,receipt.json}`):4replicas×8H800/codesign,CPU64/mem_gb512,non-preemptible/no restart,RDMA defaults,configured hard/default anti-affinity. This constructor check performs no platform operations.

## Seal and launch

Root compares all45owner-package files with merged Git bytes and checks the live package. `python3 -B <package>/seal.py` produces manifest `4e827ca74e26abfba5816b61f6bac095ca1da326957b80d0f5e4d5c0c59423b7`; `python3 -B <package>/launch.py --check-only` returns `INPUT_HASHES_PASS_NO_SUBMISSION`. `root_seal.json` binds24check artifacts, stopped source owner and the45inputs.

Prelaunch personal Python census06:09:06Z: codesign non-preemptible H80083/128, no node with8usable GPUs; this is not a scheduling reservation. Exact personal cloud destination absent, sourcehost kun-workspace-vgen2/NFS100.96.128.195. The final gate rehashes every input/check, requires the census younger than900s and rechecks destination absence.

`root_submission.json` records the exact local `stepmind-env/bin/python -u -B <package>/launch.py` command, authorization, PID and exclusive log `/data/ycfeng/tmp/glm53_dsv4/profile03_launch_20261024.log`. Package-local `write_run_manifest` completed before the actual personal Python RJobBackend submission. One run,3600s per worker; auxiliary diagnostics stay soft.

## Observed platform state and limits

`platform_launch.json`: exp-1024-141341-669621 Starting at06:18:19Z, fourStarting replicas, creator i-fengyicheng, codesign/no. Each requests8GPU/64CPU/500Gi (=512000Mi); both RDMA resource limits and exact current local NFS match. Launcher alive;45input hashes unchanged. Configured hard/default differs from the platform template's preferred podAntiAffinity; actual placement and RDMA transport are not established by either metadata claim.

The focused CPU follow-up (`root_checks/affinity_cpu/receipt.json`) confirms that ResourceConfig iteration retains hard/default and actual brainpp serialization emits `requiredDuringSchedulingIgnoredDuringExecution`. Where that becomes platform-preferred remains unresolved. A single actual-placement snapshot at06:30:36Z (`root_checks/placement/receipt.json`) finds all four replicas STARTING with empty node names/IPs: no host has been assigned. This is not evidence of co-location or a runtime failure. The healthy launcher remains running; no stop/resubmission was performed.

The first inspector failed before querying because its filename `inspect.py` shadowed Python's standard library. Running the same local reader with `-I` isolated imports; the corrected reader uses the documented `get_rjob_infos` fields. `root_checks/platform/` retains the original traceback, corrected script, argv and successful result. No job control or workload change followed the auxiliary error.

Actual GPU identity/startup, business outputs, useful per-lane coverage, raw DP pairing, stream joins and post-stop evidence remain pending. Profile03 remains diagnostic; tiny-probe graph correspondence does not grant native operator ownership or cost/reference/E2E admission. The pre-existing restored-fixture failure remains recorded separately in the integrated465PASS/1FAIL/0SKIP result.
