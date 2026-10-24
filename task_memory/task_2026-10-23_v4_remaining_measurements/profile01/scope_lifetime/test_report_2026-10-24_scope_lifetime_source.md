# W128-SCOPE-LIFETIME-SOURCE CPU report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded bounded source/CPU delivery and verification limits. |

## Results
- PASS: new module250 gross lines; new tests160. Frozen callbacks/ABI/IDs/stage/target extraction and worktree-local real ServingProfile.range are reused, not copied implementations.
- RED: scratch/red.log, exit1, one actual ctypes callback13 control failed with `Unexpected or empty graph resource callback` before implementation. red_source.txt/red_test.txt preserve the exact initial bytes.
- GREEN: final.log, 48passed/0failures/errors/skips in2.60s;32new controls +16unchanged frozen-payload controls. Earlier green1/green2/green3 logs remain separate.
- GREEN evidence run:48PASS/1.16s;23fixture exports, CUDA uninitialized, peak RSS527296KiB, actual memory limit32212254720bytes. fresh evidence_corrected.log/execution.json retain node/graph/exec IDs, payloads, windows, both thread identities and rejections. CPU fixtures have no GPU admission.
- Auxiliary FAIL preserved: evidence.log finished48test bodies then its recorder raised FileNotFoundError for an assumed memory.max path. Inspected mountinfo proves the host uses the memory controller's memory.limit_in_bytes. Only the fresh recorder changed; original log/exported fixtures and source/test bytes remain intact.

## Commands and environment
Worktree: `/data/ycfeng/Frontier/.worktrees/v4-scope-lifetime-probe`; scratch: `/data/ycfeng/tmp/glm53_dsv4/scope_lifetime_source_hv4239mb`.
Executable: `/data/ycfeng/envs/asyncopd-py312/bin/python -B`; exact version and command archive are in commands.json/hash_receipt.json.
Each CPU process uses `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Final test suffix: `python -B -m pytest tests/unit/test_cupti_scope_lifetime.py tests/unit/test_cupti_graph_resource_payload.py -q -p no:cacheprovider --basetemp=<scratch>/final_temp --junitxml=<scratch>/final.xml`.
The evidence run uses the same pytest arguments through the recorded receipt-only plugin; no controls or source predicates are added by that plugin.
Expected: exactly48PASS, no CUDA initialization, exact direct FullID joins/rejections, source/test within250/160; all immutable source bindings unchanged. Static checks and exact diff/numstat/commands are archived alongside this report.

## Delivery and limits
Source commit: `5e2fe861`; exact source/test SHA256, full diff/numstat/gross budget and immutable closure are in hash_receipt.json/source_changes.diff/source_numstat.txt/budget_receipt.json. Seven inputs and13actual local runtime modules remain equal;45sealed-package/77matrix bindings are preserved without package/GPU-pin remeasurement.
New capability: callback13 live scope/time/thread observations and direct original-handle/source-ID -> clone-FullID -> target-kernel closure at a default graph constructor, without retained source DAG/event nodes. All payloads are copied during valid callbacks.
The program exports trace before P1–P6 analysis, never asserts a positive scope verdict during capture, and retains setup/callback/query/trace errors as UNVERIFIED. Observe closes before Kineto; control has no subscriber and checks topology/output identity only.
CLI: `python probe_scope_lifetime.py --output-dir <absent-directory> --image-digest <approved-digest> [--elements 1024 --warmup 3]`; the existing stage directory and FRONTIER_NATIVE_PROFILE_LIMITS must be supplied. Defaults reuse the unchanged frozen probe domain.
No host CUDA execution, standalone CLI/GPU run, image/SDK qualification, package generation/seal, native integration, dependency mapping, serving/cost admission or new matrix capture occurred.
CPU tests do not measure actual GPU observe/control orchestration, TLS/lifetime, timing or non-perturbation. Primary owns independent dual reviews/root replays and all S3–S5 gates.
Untested alternatives remain runtime-API/activity correlation, clone/exec-only mapping, explicit push/pop independent of TLS and post-capture retained-activity attribution. A future clean negative is limited to the tested live-TLS image/driver/shape, not general impossibility.
