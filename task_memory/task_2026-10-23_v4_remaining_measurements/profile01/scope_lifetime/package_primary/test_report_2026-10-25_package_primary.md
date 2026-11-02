# Primary package closeout verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Archived completed image/SDK controls and verified the merged unsealed package. |

## Commands and environment

- Merge verification: `python3 -B /data/ycfeng/tmp/glm53_dsv4/handoff_closeout_20261025.py`; local Python 3.10.6. This hashes/parses existing artifacts; it does not execute a new GPU experiment.
- Locked-image command, complete argv and times: [image/receipt.json](image/receipt.json). `docker run -i --rm --pull=never --network=none --cap-drop=ALL --cap-add=DAC_OVERRIDE --memory=6g --cpus=2 --pids-limit=128`; package mounted read-only, Python `-I -B -`, BLAS/OMP/MKL threads 1, CUDA compat library path fixed.
- Executed stdin: [image/phase_stdin.txt](image/phase_stdin.txt). It hash-checks the prepared checker, preserves every original statement in order and adds only stderr phase timing. Witness and phase output: [image/stdout](image/stdout), [image/stderr](image/stderr).
- Actual SDK entry/resources with `spawn_tasks` and receipt writes intercepted: [sdk/receipt.json](sdk/receipt.json), Python 3.10.6. No authentication, platform job or GPU execution occurred.

## Expected and observed

| Check | Passing condition | Actual | Verdict |
| --- | --- | --- | --- |
| Merged package | Every manifest input equals original hash and owner bytes | 18 inputs, 19 files, 184,495 bytes; manifest `9ff1abec...198e` unchanged | PASS |
| Preserved inputs | All recorded original pins unchanged | 67/67 hashes equal | PASS |
| Locked image | Parsed PASS witness, deadline <60s, CUDA false, binary pins equal | Python 3.12.3; 5.324804s; `PASS_CPU_PACKAGE_IMAGE_PINS`; CUDA false | PASS |
| SDK resources | Existing plan and absolute code mount preserved | 1 H800, 16 CPU, 64 mem_gb, non-preemptible; existing cloud mount and original owner package | PASS |
| Actual submission | No intercepted call reaches platform | `CPU_NO_SUBMISSION`, zero new GPU jobs | PASS |
| Prelaunch/runtime gates | Fresh storage/quota/destination, seal and runtime observations | Not performed | NOT RUN |

## Preserved failures and limits

- Initial root image check exceeded the 60s client deadline; retained raw output is `/data/ycfeng/tmp/glm53_dsv4/scope_package_root_image_m16e_5n_/{stdout,stderr}`. Later container completion is not deadline PASS; the slow first startup cause is not established.
- A phase-observation command omitted Docker `-i`, exited 0 with empty output, and its root parser raised JSONDecodeError. [image_without_stdin/receipt.json](image_without_stdin/receipt.json) and empty logs are preserved; this is not successful verification. Restoring stdin produced the valid witness above without changing package or limit.
- The SDK check proves constructor/entry serialization, not actual NFS/RDMA transport or creator identity on a submitted worker.
- Owner 50-case source tests and source acceptance are existing evidence, not newly executed here. CPU, image and SDK PASS do not prove live callback TLS, default-False lifetime, observe/control output identity, general native ownership, non-perturbation, serving time or cost/reference admission.
