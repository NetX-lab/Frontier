# D28 D24 pressure stage completion archive

## Modification History

| Date | Change |
| --- | --- |
|2026-10-22|Archived committed pressure route and sealed capture03; explicit native balanced-routing blocker.|

## Outcome

Task **D28-D24-PRESSURE-STAGE**: approved D-only32000 wiring is complete and verified offline. Capture03 is sealed **PRESSURE_OFFLINE_READY_ROUTING_BLOCKED**. The later balanced-distribution request has no built-in supported native strategy in the locked source; the unsupported-mechanism report is complete. **Do not submit this packet as balanced-aligned calibration.** Parent owns every live operation and the routing decision.

Code-only commit `6e73330c2d04af382cf65a2c689c5401ca202485`: existing canonical definition/verifier accept the exact D28 role map; golden/fault-injection tests added. Four files,176 insertions/8 deletions, exact `code_diff.patch` and `changed_files.txt` provided for primary review. No recorder/gate/native source/simulator/CPU-stream change, no whole proposal-branch merge. Existing pregrant estimate hook was sufficient.

## Exact packet identity

| Item | Path / SHA256 |
| --- | --- |
|Stage|`/data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage`|
|Imported source commit|`6e73330c2d04af382cf65a2c689c5401ca202485` in owned checkout|
|Final config|`stage/config.json` — `046b6f25c1f993790c135e6740cdf7bed247078f3b99f2713be2bfa97277854f`|
|Seal manifest|`capture03/stage_manifest.json` — `cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d`|
|Source origins|`stage/source_origins.json` — `68c745871f0bb141c4eee5296d65a030fddbcd97bf1c729ab7e7322ea56244f0`|
|Unchanged50 native pins|`capture03/installed_source_pins.txt` — `724d5dfefab52fe9b82624306170d0583966cb94469a97cc9610eda1e7faa2ca`|
|Cloud destination string only|`/mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d24_capture03` — not accessed|
|Expected local mount source/target|`/data/ycfeng/tmp/step5_d24_native_20261022/capture03`|
|Expected creator/image/resources|i-fengyicheng; original locked manifest69308454…;32 non-preemptible H800/codesign P16/D16,5400s/48GPUh|

P argv/expected engine remains golden automatic capacity. D lane0/1 argv includes exactly `--num-gpu-blocks-override 32000`; everything else matches capture02 after path normalization. Base `engine.json` has no override, preventing P mislabel; config/run role map uses the one existing mechanism. Exact four argv/expected engines in `package_static.json`; reference requests execute on the same D pool. Canonical6144/1024,7168/8192/128,graph/runtime/workload/ports/API/pressure/abort/numerical rules unchanged. Capture02/original packet and routing identities remain immutable.

## Observed validation and retained evidence

Task directory `/data/ycfeng/Frontier/.worktrees/step5-d24-stage/task_memory/task_2026-10-22_step5_d28_d24_pressure_stage/` contains requirements/plan/progress,report,build/offline/static/resource/import/staged/parent-evidence receipts,seal copy,review diff/list and native-routing report/probe/receipt.

- **PASS:**711 scoped tests,122.89s, no failures/skips; Python3.12.13/Torch2.9.1+cu128/pytest9.0.2. Exact command and retained failures in test report.
- **PASS:** actual staged48 raw/mean cases,2 startup,2 HTTP,5 locked native method-body pressure cases; paired estimate/free after native retirement, never granted-delta substitute. No added GPU read/sync/formula.
- **PASS:**16 actual imported module origins/21 copied Git files bound to source; isolated imports from both cwd values prevent root/installed-tests shadowing.
- **PASS:**51 tool pins,50 native pins,163 requests/14 waves/8 content/155 readiness-only;25 Python files/bash/expected-node preflight.
- **PASS:** actual ResourceConfig audit-restricted offline construction/sanity check and actual StepMind Python3.10.6 dry-run; both RDMA defaults/fail_policyAny/no restart preserved. No backend/client instantiated.
- **BLOCKED:** built-ins `uniform_random`/`normal_routing` do not provide balanced token assignment. CPU44-token/352-assignment case yields0–4 per expert;6144 tokens yield106–173. Unknown balanced raises ValueError; original model normalized sigmoid scale3 differs from simulation all-one weights. Monolithic path can bypass router. No routing setting applied; effective runtime UNVERIFIED.
- **UNVERIFIED:** all native GPU timing/queueing/preemption/denial/READ/ownership/overlap/state/business/terminal witnesses. Offline pressure capability does not qualify these.

## Parent-only continuation

1. Resolve exact balanced native mechanism and Frontier semantics separately. Existing registration seam would require new strategy/DP/TP/padding/weights/kernel-path design and approval; no unsupported substitution in this packet. Workflow→GPU/op gap→CPU overhead last.
2. For any routing or imported-code integration after this seal, create a fresh generation and rebind real origins/Git blobs. Do not edit capture02/capture03 seals or submit capture03 as balanced calibration.
3. Before an approved next launch, check personal reservations/creator,original image/local NFS/RDMA/P auto,D32000,32 H800/codesign/90min48GPUh; inspect actual loaded sources/argv/pools/native pressure/ref/READ/ownership and business+all terminal states. Validators soft; numerical criteria unchanged.

No GPU/platform/authentication/cloud/SSH/submission/stop, no attempt/submitted marker/job, no further delegation or push. Code and records are committed separately; the final receipt commit and clean status are reported by the handoff.
