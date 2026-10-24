# 05: DeepSeek-V4-Flash operators, profiling, and predictor

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | W128-OCT24-REPAIR-DOCS: collection repairs integrated, graph package sealed/launcher started and profile03 delivered unsealed; full05 remains OPEN. |
| 2026-10-24 | W128-STREAM-ACCEPTANCE-DOCS: limited fresh GPU stream/restart acceptance and CPU counters recorded; diagnostic DP slice active, full05 remains OPEN. |
| 2026-10-24 | W128-OCT24-DATA-DOCS: diagnostic-only profile02 and isolated MoE data accepted; full05 measured-family/predictor scope stays OPEN. |
| 2026-10-23 | W128-BALANCED-DECODE-STAGE-DOCS acceptance: root accepted isolated balanced q1 data after review/fresh reruns; full05 cost/family qualification OPEN, prior audit-pending state historical. |
| 2026-10-23 | W128-BALANCED-DECODE-STAGE-DOCS followup: balanced Succeeded/execution PASS but data audit pending; corrected MoE design/independent packaging active, profile02 still queued and full05 OPEN. |
| 2026-10-23 | W128-BALANCED-DECODE-STAGE-DOCS: accepted balanced-q1 preparation/merge56e9405d and Running1-H800 snapshot13:40:46Z; profile02 remains Pending and full05 remains OPEN. |
| 2026-10-23 | W128-PROFILE02-STAGE-DOCS factual delta: job exp-1023-180821-373637 created10:08:21Z, verified SUBMITTED_PENDING/queuing at10:10:16Z with personal creator/resources/image/NFS/RDMA; full05 remains OPEN. |
| 2026-10-23 | W128-PROFILE02-STAGE-DOCS: profile02 repair/re-reviews and separate final root checks complete, 45 inputs SEALED_READY_DIAGNOSTIC; launch started without evidenced job identity, full05 stays OPEN and profile01 FAIL preserved. |
| 2026-10-23 | W128-H800-AUTHORITY-DOCS: first32-H800 profiling run and standing task-required H800 resources approved; package/runtime/GPU evidence and full05 criteria remain OPEN, no new submission. |
| 2026-10-23 | W128-G2-CLOSURE-DOCS: G2 CPU profile-driver05952396 accepted after three distinct finding closures, separate final suites and bound77; full05 qualification stays OPEN and next collection needs a new GPU decision. |
| 2026-10-23 | W128-H-CLOSURE-DOCS: H CPU collector accepted/merged64f3c950 with five findings closed; G2 profiling/sampling CPU work handed off, full05 measured-family qualification remains OPEN. |
| 2026-10-23 | W128-V4-INPUT-INTEGRATION-RECORDS: private isolated input prerequisite merged732406aa/real94 equality/bound strict77; full05 constructor/family and measured serving costs remain OPEN. |
| 2026-10-23 | W128-V4-NATIVE06-ADMISSION-RECORDS: native06 P0 reference acceptedR4 and D-Q10 integrated; full05 measured-family/predictor qualification remains open. |
| 2026-10-23 | W128-V4-NATIVE06-RUNNING-RECORDS final delta: native06 terminal Succeeded/frozen routing identity FAIL; D-Q10 isolated prerequisite accepted with complete matrix, full05 still open. |
| 2026-10-23 | W128-V4-NATIVE06-RUNNING-RECORDS: current native06 is launched/Running; independent isolated D-Q10 source/root checks complete, fixed576d278 matrix running; full05 remains open. |
| 2026-10-23 | W128-V4-CHECKPOINT-CORRECTIONS-20261023: current b62c155/953, final fidelity and CPU-checked native06; independent D-Q10 contract launched, family/coverage/runtime acceptance still open. |
| 2026-10-23 | W128-V4-MILESTONES-20261023: accepted bounded balanced prefill data, merged native133PASS/actual950 under952; full family/balanced decode/native06 acceptance remains open. |
| 2026-10-23 | W128-V4-BALANCED-STATUS-SYNC: recorded accepted A/C source/CPU/fidelity and independent package/native06 readiness; preserved prior failed-prefill/native05 snapshot and unchanged family acceptance. |
| 2026-10-22 | W128-V4-CURRENT-RECORDS: prefill9d3df65/isolated-decode102c0c9 source merged1687c60/947961d;282mergedPASS/strict77 identical;prefill submitted then failed;native05 business/terminal PASS,graph qualification FAIL. |
| 2026-10-22 | Primary accepted warm-q1 isolated data:12288rows, first-event spike removed, complete trace and exit0 verified; started ongoing-decode reader consumption with full-serving guards closed. |
| 2026-10-22 | Corrected warm-q1 submission Running after aggregate ceiling32->128 only,527 pins retained/business pending;native04 unadmitted,05 unsealed pending versioned full-checker repair. |
| 2026-10-22 | Active DSV4-only scope under D-DSV4-ONLY-GLM-PAUSED; recorded accepted frozen successor/TCP and successor04 owner, while warm-q1 attempt exits1 with cause pending. |
| 2026-10-22 | Updated A/B source acceptance, completed-but-UNVERIFIED P0 native result and warmed exact-q1 CHECK_ONLY_READY package; removed stale active H200-only/cross-model edges. |
| 2026-10-21 | Accepted narrow H800 q1 collection12288rows/1.449GB trace/exit0, retained first-event anomaly; V4 A known-error follow-up and full P0 reference remain in progress. |
| 2026-10-21 | Prioritized V4 H80032 profiling/consumer work and hardware-specific32/64 binding; retained completed H200 measurements within their actual scope. |
| 2026-10-21 | Accepted actual reduced H200 G0-A and started full-model G0-B preparation; H200 profiling and measured-family admission remain pending. |
| 2026-10-21 | Retargeted H200 profiling to ticket 17's fixed 128-request 6144/1024-token QPS-1.0 case, all attention ratios and reached contexts/boundaries through total length 7168; kept native graph/memory and measured V4 integration open. |
| 2026-10-21 | Platform changed to H200 by the canonical calibration contract: attention (all three compression ratios), MoE and linear_op profiling is re-collected on H200 for the touched shapes; H800 tables stay as H800 evidence. |
| 2026-10-20 | Handoff status: W6 DSV4 producers committed (`45361bb`, `4b19e35`, `99f3a69`, `3ea2c3f`); 24-H800 run configs ready, not submitted. |
| 2026-10-20 | G0-B[DSV4] closed; producers are the second profiling wave after GLM; M4 V4 PD work item recorded. |
| 2026-10-19 | Recorded complete C36 paired perturbation FAIL, accepted C38 source and terminal C38/full-V4 workloads pending artifact assessment. |
| 2026-10-19 | Recorded reduced G0-A[DSV4] PASS from C32 plus MARLIN02 controls; full-model, observer and measured-family requirements remain open. |
| 2026-10-19 | Recorded C32 reduced V4 PASS and MARLIN import-path remediation; C35 ordinary-cache scope supplies no V4 agreement or formal qualification. |
| 2026-10-19 | Accepted C29 queue source/CPU and C30 policy oracles, committed C31 bootstrap/inventory and submitted exact original-V4 diagnostic; native/formal outcomes remain separately gated. |
| 2026-10-19 | Accepted C25/C26/C28 source and CPU slices, retained completed one-H800 GLM graph evidence and C27 workflow deltas; native/formal gates remain open. |
| 2026-10-19 | Accepted scoped C23 cache-zero and C24 DP8 CPU evidence; synchronized remaining native inputs, ownership and formal limits. |
| 2026-10-19 | Synchronized C21/C22 local evidence and remaining collector/metric-input/cache-zero/DP8 gates; no native/formal closure. |
| 2026-10-19 | Synchronized accepted C19/C20 commits, final source/CPU evidence, GLM/V4-only ownership and H800-first fallback; all native/formal limits remain explicit. |
| 2026-10-19 | Recorded the actual C20 decode-context defect and minimal existing-frontier correction; boundary acceptance remains pending. |
| 2026-10-19 | YC limited this worktree to GLM/V4; Step contract reuse is external compatibility, not local Step implementation or merge. |
| 2026-10-19 | Accepted C17 dc7f382 after exact26-file identity,1230 focused tests, actual43-layer offline/online conservation, independent APPROVE and77 identical legacy artifacts; began single-owner accepted Step representation integration. |
| 2026-10-19 | Recorded C17 source-request trace-quota correction and retained structural/native acceptance limits. |
| 2026-10-19 | Added explicit source-backed live idle-DP scope for C17; real-request missing-metadata guards and all native/formal gates remain closed. |
| 2026-10-19 | Recorded frozen C16 trace handover and bounded32logical-device C17 first main;64GPU qualification remains open. |
| 2026-10-19 | Corrected the active EP dependency to distinguish allowed synthetic structural execution from native qualification. |
| 2026-10-19 | Recorded unaccepted C17 canonical physical timing, phase and trace consumers; retained native/formal guards. |
| 2026-10-19 | Accepted C15 91d1697 partial source-rank/mHC/shared/endpoints consumer; complete ticket remains PARTIAL/native BLOCKED. |
| 2026-10-19 | Started C15 source-rank SP/mHC/shared execution consumer with one owner; corrected padding geometry versus selected routing-mask evidence. |
| 2026-10-19 | Accepted C14 596451a eager attention workload/component:833isolatedPASS/APPROVE/77exact artifacts-cache; ticket remains PARTIAL/BLOCKED native. |
| 2026-10-19 | Started C14 eager dynamic workload/component integration; full representation and serving acceptance remain open. |
| 2026-10-19 | Final C13 accepted-source status and next dynamic workload/branch/structural gates synchronized. |
| 2026-10-18 | Accepted C13 d3d4229 source collective payloads/actual RS dispatch; next workload/branch/component integration remains required before V4 serving. |
| 2026-10-18 | Accepted C12 7626b3f static source dimensions with607isolatedPASS/181peerPASS/77exact artifacts-cache; C13 source TP-SP payload/RS dispatch under root validation. |
| 2026-10-18 | Started C12 static source tensor dimensions; recorded pinned SP payload, valid/padded row and EP-backend distinctions before execution work. |
| 2026-10-18 | Accepted C10 ba48220 source-only per-layer operator/precision inventory:527isolatedPASS,77identical legacy artifacts and zero clean-cache filename differences; ordered the remaining shape/SP/mHC/routing prerequisites. |
| 2026-10-18 | Accepted source/CPU slices C07 8d4fb3e/C08 92a9da7/C09 4164a04 with exact legacy/cache regression and shared GLM online structural PASS; complete V4 execution/native gates remain open. |
| 2026-10-18 | Recorded C07 parameter/SP-weight ownership and approved C08/C09 grouped-cache implementation; full execution/native/formal gates remain open. |
| 2026-10-18 | Resumed implementation; recorded verified C02 static contracts, remaining execution gaps and frozen regression scope. |
| 2026-10-18 | Documentation review: unblocked original-variant static/structural work; retained native blocker and GLM-first measured integration. |
| 2026-10-15 | Created in the plan phase. |
| 2026-10-15 | Review remediation: `family_predictor_integration` waits for the complete GLM-5.3 path, 09 `full_frontier_path[GLM]`, and for the DeepSeek-V4-Flash stages of decisions 12 and 14 (S1, Q12, Q13). |



## Final execution state — current (2026-10-24)

**Status: PARTIAL measured family; native06 P0 reference acceptedR4, full05 cost/predictor qualification OPEN.** DSV4-only, GLM PAUSED_PENDING_USER_APPROVAL and official mean/30%/10% gates unchanged. Use [Routing milestone — current (2026-10-23)](../progress.md#routing-milestone--current-2026-10-23).

### GPU stream acceptance and active DP slice — current (2026-10-24)

[Fresh stream acceptance](../../task_2026-10-23_v4_remaining_measurements/profile01/stream_probe/results/primary_acceptance.json) is ACCEPTED_GPU_STREAM_IDENTITY_WITH_LIMITS: probe `5f7d0950`, CUPTI repair `0d745706`, 1-H800 `exp-1024-020317-012085` Succeeded/23s/exit0/restart0; independent No findings and root complete 162/162 byte-identical rerun. 8 eager/16 priming joins, 16 stable snapshot rows/28 raw kernels; Torch IDs0/3 versus CUPTI/trace7/13, two complete native-profiler cycles observed. Successful capture/two replays do not establish captured per-op ownership: UNKNOWN, with metadata/version limitations and no serving/operator/dependency/native-reference admission. Old profile02's 192 null maps remain missing.
DPbb50186e and coverage/shutdown20660f4c are integrated1437ace6, each dual-reviewed No findings; [CPU acceptance](../../task_2026-10-23_v4_remaining_measurements/profile01/integration_cpu/primary_acceptance.json) records twelve suites465PASS/1known-baselineFAIL/0SKIP and46 checks PASS. Baseline44/candidate17832-rank fixture failures and all32 counter/hook dictionaries match; FAIL remains FAIL. Raw counters include direct dummy annotations with `step=null`; sampling uses the detached step counter. Actual cross-DP pairing/coverage/shutdown remain UNVERIFIED; frozen schemas and bound77 evidence unchanged.
Graph correction1aa1e4a5:26/80 lines, root87CPU PASS+2expected RED, actual-image Python3.12.3 entry/ABI/bytecode PASS; source/package dual reviews No findings, pack6f84/2264 merged4add627d. [Seal](../../task_2026-10-23_v4_remaining_measurements/profile01/graph_probe/root_seal.json) binds8inputs+manifestec816706…; [receipt](../../task_2026-10-23_v4_remaining_measurements/profile01/graph_probe/root_submission.json) is LAUNCHER_STARTED at2026-10-24T05:44:43Z/PID3103656, not an evidenced job/Running/GPU result. Payload/correspondence/crash cause remain UNKNOWN.
Profile03 delivered9946771/882b25ae in `v4-native-profile03-run`,45files/13runtime1437/87native/20binary pins preserved,68/140 maintained gross; independent review pending, NOT sealed/submitted. D-V4-PROFILE03-PACKAGE selects P512/D64,600000records/268435456bytes; next actual-image/SDK/package gates -> seal/32H800 run -> audit. Fresh graph one1H800/600s uses D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY; old3/3 historical, no blind automatic retry. Require >=3 useful steps per role/lane/actual pairing/stream/post-stop evidence. Native06 R4, Q2a/901ef960, MoE limits, full05/09/19 OPEN, GLM paused/CPU-overhead last and mean/30%/10% unchanged. [Evidence/delivery links](../progress.md#gpu-stream-acceptance-and-active-dp-slice--current-2026-10-24); historical prose/checklists retained.

### Profile02 and local-MoE acceptance — current (2026-10-24)

Profile02 [primary acceptance](../../task_2026-10-23_v4_remaining_measurements/profile01/results/primary_acceptance.json) is ACCEPTED_DIAGNOSTIC_AUDIT_WITH_GAPS: root32-trace non-clock/decompressed-index equality,326328annotations/7600008kernels/useful P0/P1/D0/D1=23/2/61/61;192 null stream maps (`libcupti.so`),6802600 graph kernels' UNKNOWN ownership and UNKNOWN shared cross-DP identity/dependencies remain. ONLY decodeDP1 shutdown error, prefillDP1 true/errors empty; cross-host wall timestamps cannot prove order. Full audit/32indices archived46files/86211102bytes outside immutable run; Git keeps compact receipts, native06 R4 is not replaced.
MoE [primary acceptance](../../task_2026-10-23_v4_remaining_measurements/moe/results/primary_acceptance.json) ACCEPTED_ISOLATED_KERNEL_DATA_WITH_LIMITS after No findings/root exact rerun; source737b6311/docfix941a4450 mergedbe067bb5,204/300lines/image440/20/1003+SDK PASS. Jobexp-1023-235250-540507 Succeeded15:54:35Z,1H800/16CPU/mem_gb128/non-preemptible/1800s,92container seconds/exit0/restart0; first guessed-label false negative preserved, actual codesign/no and normal launcher finish verified.
MoE19outputs/3261347bytes/two four-row tables/12traces/112statistics/delta0/20record_function grouped calls per M/7kernels per call; rank0 KERNEL_ONLY means(ms) M16/32/6152/12288=0.1017937/0.1022997/0.7557434/1.29869045. CUDA_EVENT raw samples/grouped count absent; no other EP ranks/TP8/communication/graph/serving/weights equivalence or native-tuple cost consumption; M6152 conditional dummy-peer only.
Repairs are integrated; next profile03 review/actual-image/SDK/package gates ->32H800 run/audit, alongside graph-probe launcher/platform/data checks; proper tuple admission precedes costs. Q2a ACCEPTED_ISOLATED_DATA/901ef960 remains; full05/09/19 OPEN, GLM paused/CPU-overhead last. [Current evidence](../progress.md#profile02-and-local-moe-acceptance--current-2026-10-24); dated snapshots are history, checklists unchanged.

### Balanced q1 isolated-data acceptance — current (2026-10-23)

[Primary root decision](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/results/primary_acceptance.json): ACCEPTED_ISOLATED_DATA, jobexp-1023-213905-916183/C6143..7166, both6144-row CSVs byte-exact,94 scope series,51200 CUDA_EVENT/45056 KERNEL_ONLY calls and45056 annotations exact. Original samples/methods retained separately.
[Review](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/results/review.md) No findings; root [fresh raw/trace reruns](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/results/root_verification.json) in `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_root_audit_nss5y_ka` returned0/all non-clock fields matched,17.568s/47.700s; final14:35:05Z98305 records/duplicate fields0/eight hashes unchanged. [Data audit](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/results/data_audit.json) admissions=false preserves historical auditor authority; primary_acceptance.json records root's isolated-data decision.
PhysicalTP1/DP1/eager/six-layer/dummy-weight scope only, one sample per context/layer/scope/method;227901/352829 kernels unowned, actual event-handle dependency evidence absent. Consumer/serving/physicalTP8/graph/constructor-family/stationarity/independent expert-histogram admissions remain false; full05/09/19 OPEN.
Historical14:18:37Z profile02 queue/MoE preparation snapshot is superseded by the2026-10-24 acceptance above; Q1 commitb7c0a953 still closes five P0 subitems only, not full17. All failures/checkboxes remain retained.

### Balanced q1 terminal and MoE preparation — historical (2026-10-23)

[Root run check](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/root_run_check.json): jobexp-1023-213905-916183 Succeeded/finish13:48:57Z observed14:18:36Z, container Completed/exit0/restart0/569s; submission→terminal592s = conservative0.16445GPUh upper bound, not verified billing. Actual H800/image digest/530 package and527/529/530 original hashes match. Engine proves max_num_batched_tokens6144/max_num_seqs1/physicalTP1/DP1/V2/eager/balanced_topk;8 outputs/1,601,865,978bytes hash-identical. Independent data/trace/raw-to-CSV audit IN PROGRESS; no data/GPU-cost/serving/operator/TP8/graph admission.
[Corrected MoE design](../../task_2026-10-23_v4_remaining_measurements/moe_grid_design.md) is complete, superseding historical `moe/local_measurement_plan.md`; packaging independently active in `/data/ycfeng/Frontier/.worktrees/v4-moe-local-run`, two non-core files/300-line budget. EP16 rank0/physicalTP1/M16/32/6152/12288, CUDA_EVENT and record_function, all original columns retained;6152 conditional dummy-peer. No source/runtime, rank/time or serving equivalence; no MoE GPU submitted, package/image/SDK/run/data checks pending.
Same receipt:32-H800 profile02 exp-1023-180821-373637 Pending/no replicas at2026-10-23T14:18:37Z, launcherPID3236764 alive; retain queue → scheduling → I4. Q1 commitb7c0a953 retains five P0 closures only, not full17; full05/09/19 OPEN. Native06 R4/private input/H/G2 CPU milestones, standing H800 authority/soft gates, GLM pause and CPU-overhead last unchanged. Running/pending-terminal/MoE-investigation statements below are historical, superseded here; original failed checks remain retained.

### Balanced q1 and queue-time work — 2026-10-23

Accepted preparation source64133239/records9fb4bd7e merged56e9405dad76d30d4ada9d5bdab124f898bf5065, two maintained files/214 of300 lines. [Root seal](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/root_seal.json) records root530-input/512-Git-file PASS, originals527/529/530 unchanged, two full-changed-file reviews No findings and separate root Git/hash verification. Owner131 and256CPU PASS are records, not root reruns; existing77 binding has no fresh capture. Locked-image60native/520tool/20binary and actual bootstrap-wrapper entry PASS before model execution; SDK preemptible=false.

[Platform snapshot](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/platform_launch.json): balanced jobexp-1023-213905-916183 created13:39:05Z, Running at13:40:46Z, launcherPID3647153,1H800/codesign/non-preemptible/16CPU/mem_gb128=125Gi/1800s; profile02 exp-1023-180821-373637 remains Pending, launcherPID3236764 alive. Quota13:35:48Z122/128 is historical, not real-time. [Memory check](../../task_2026-10-23_v4_remaining_measurements/balanced_decode/platform_launch_check.json) retains string-assertion FAIL and exact128000Mi=125Gi PASS without workload change. Earlier submission/quota snapshots below are historical.

Q0 reports: [queue_selection.md](../../task_2026-10-23_v4_remaining_measurements/queue_selection.md). Q1 accepted/committedb7c0a953 closes five P0 subitems only, not17; Q2b exact native MoE population/source plan remains under investigation after provider30008 abort, no accepted grid. Worker nvidia identity, runtime/routing, business outputs and terminal status remain pending; no GPU-cost/serving/operator/TP8/graph admission. Native06 R4, private input/H/G2 CPU milestones, original failures, full05/09/19 OPEN, standing H800 authority, GLM pause and CPU-overhead last are unchanged. [Detailed checkpoint](../progress.md#balanced-q1-and-queued-profile02--2026-10-23).

### G2 profile-driver — current (2026-10-23)

G2 is ACCEPTED_CPU_PROFILE_DRIVER_COMPONENT at frozen source `05952396b0b8e5a794c53e10f9a21213beb5193d`, seven non-core files/433 changed lines within750, zero core edits. [Primary acceptance](../../task_2026-10-23_v4_remaining_measurements/driver/primary_acceptance.json) and both independent R1 [final reviews](../../task_2026-10-23_v4_remaining_measurements/driver/review_final.md) close all three distinct original findings with no new findings; overlapping findings are not summed. The real `pd_replay.py` first-formal both-head start/drain/stop, sampling/`serving_profile_limits` and independent selected-window record/byte retention are implemented, with soft auxiliary errors, preserved native exceptions and unselected compatibility.

Frozen-source [final checks](../../task_2026-10-23_v4_remaining_measurements/driver/final_checks.json), Python3.12.13, separately pass collector32, worker19, streaming39, inventory28, V2 inventory45, R4 reader135, DP75, P-D68, manifest14, cold18, metrics24 and HTTP1, each with zero failures/errors/skips; do not sum suites or prior-H runs. [Final matrix binding](../../task_2026-10-23_v4_remaining_measurements/driver/matrix_binding_final.json) explicitly binds existing4a1b53e6 capture77/77 to05952396, not a fresh capture.

Runtime/GPU qualification and profile output remain UNVERIFIED. Full05 stays OPEN for constructor/family admission, serving totals, TP8/DP2/EP16 cost evidence, graph/stream/kernel coverage and balanced decode; complete09/19 also remain OPEN. Kernel-owner completeness, stream dependencies, actual GPU PDL/GDC, compile/breakable event survival and captured-segment replay ownership remain unproven; no CUPTI restart proof. Native06 retains ACCEPTED_NATIVE_REFERENCE_R4, old R3 FAIL/native05 UNVERIFIED preserved; GLM pause, official mean/30%/10% gates and CPU-overhead last unchanged.

[D-V4-H800-RJOB-AUTHORITY](../../task_2026-10-23_v4_remaining_measurements/requirements.md) approves the current32-H800/codesign/non-preemptible, maximum60-minute/32GPUh, first-formal64worker-step run and task-required H800 RJobs of any necessary scale without repeated resource approval. Each later run records its purpose, resources and evidence and respects actual quota/capacity; calibration predicates, native/C26 source authority, publication, broad CUPTI investigation, CPU-overhead collection and paused GLM scope are unchanged.

Profile02 owner repair `d87f4c74`, records `a90c9dbb`, merged `cc0227e9`, is complete at cumulative152/300 changed lines in two maintained source/config files; both independent re-reviews have no open findings. Initial profile01 fresh entry FAIL (`ModuleNotFoundError: No module named 'vllm_replay'`) remains unchanged, unsealed and unsubmitted. [Review](../../task_2026-10-23_v4_remaining_measurements/profile01/review.md); [delivery](../../task_2026-10-23_v4_remaining_measurements/profile01/delivery.json) is the historical owner-unsealed snapshot.

Final root tests after source/package binding passed separately: actual StepMind-interpreter launcher; real locked-image bootstrap/fresh child with87native/23tool/20binary pins; four isolated hook/role-path probes, each PASS. No actual GPU constructor/model forward/replay was exercised. Root sealed45 inputs, manifest `51e39a42ec863e496f20be8a165230e0dee17c34906ad6bd4fade14bf898ee91`, status SEALED_READY_DIAGNOSTIC. [Root report](../../task_2026-10-23_v4_remaining_measurements/profile01/test_report_2026-10-23_root.md); [readiness](../../task_2026-10-23_v4_remaining_measurements/profile01/profile02_root_readiness.json).

Historical first-submission status: **SUBMITTED_PENDING**, job `exp-1023-180821-373637`, created2026-10-23T10:08:21Z. Snapshot2026-10-23T10:10:16Z is Pending/`RJob is queuing`, `replicas=[]`, not Running or success. Verified creator `i-fengyicheng`, `codesign`, preemptible=no, four replicas each8GPU/64CPU/500Gi (SDK mem_gb=512 converts to500Gi), locked image, exact local NFS `100.96.128.195:<profile02-package>:<profile02-package>` and RDMA1/8; launcher PID3236764 was alive then. [Platform receipt](../../task_2026-10-23_v4_remaining_measurements/profile01/profile02_platform_launch.json); [root submission record](../../task_2026-10-23_v4_remaining_measurements/profile01/profile02_root_submission.json) is the preserved prelaunch-gates snapshot. Next I3 scheduling/business monitoring, then I4 retained-data verification under the [approved single-window plan](../../task_2026-10-23_v4_remaining_measurements/proposal.md). Accepted G2 CPU source05952396 and existing77 binding remain unchanged; native06 stays the clean R4 reference. GPUs/runtime/operator coverage/stream/event/graph/TP8 costs and full05/09/19 remain unfinished. [Parent checkpoint](../progress.md#g2-profile-driver-closure--current-2026-10-23); handoff is informational and earlier package-pending/launch-started/no-job-identity snapshots are historical.

### Historical H-only CPU checkpoint — before G2 (2026-10-23)

The H-only paragraphs below retain the original evidence; their then-pending G2/driver/retention and next-session directions are superseded by the G2 closure above.

**Latest CPU milestone — H completed, G2 pending.** [Primary acceptance](../../task_2026-10-23_v4_remaining_measurements/collector_primary/primary_acceptance.json) records ACCEPTED_CPU_COLLECTOR_COMPONENT, source `f8621a8258ffa445abfb51f5ea73258ad271494e` merged `64f3c9509b09279f321d237e3a95cbe904c6dd22`, five non-core files/880 changed lines within987. Independent Spec/Standards [final re-reviews](../../task_2026-10-23_v4_remaining_measurements/collector_primary/review_2026-10-23_final.md) close all five original findings. Root final-source and merged-source runs separately pass collector28, inventory28, V2 inventory45, R4 reader135, native worker16, DP harness75, PD harness68 and manifest14 with zero failures/errors/skips per suite; no aggregate claim. Merged automatic isolated input94 equals prior merged732406aa; E/F/G1 remain accepted isolated inputs. H's [matrix gate](../../task_2026-10-23_v4_remaining_measurements/collector_primary/merged_matrix_binding.json) binds existing4a1b53e6 capture77/77 explicitly, not a new capture.

H profile output remains UNVERIFIED; GPU stream mapping, actual per-kernel GPU PDL, compile/breakable survival, captured-segment per-replay ownership and GPU CUPTI restart remain UNKNOWN. Driver not implemented; no new GPU run or approval, full05/09/19 or whole calibration acceptance. Native06 R4/old R3 FAIL/native05 UNVERIFIED retain their verdicts. Next session G2 implements real `pd_replay.py` profiling bracket, sampling/`serving_profile_limits` and CPU controls before a concrete GPU request. Collector records since startup; `/start_profile` does not reset retention budget. [Driver design](../../task_2026-10-23_v4_remaining_measurements/driver_design.md) requires window-retention proof, both-head start-before-dispatch/stop-after-drain, unselected compatibility and soft instrumentation/native-exception controls. [Parent closure](../progress.md#cpu-collector-closure--current-2026-10-23); [root handoff](../../task_2026-10-23_v4_remaining_measurements/handoff_prompt.md). GLM paused; CPU-overhead last.

### Retained routing, isolated input and native06 R4 acceptance

A186dfef is accepted/merged292c489; C8c2502c accepted/merged84c806f and finalized39b2bf8 at five files/202 lines (184/184 tests, A/C26PASS, clean strict77/77 identical with451 artifacts). Root `../../task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json` now accepts source39b2bf8/jobexp-1023-023708-916112 as ACCEPTED_ISOLATED_DATA: H800 Succeeded/exit0/restarts0,500s/0.138889GPUh; both6-row CSVs,6144/C0,7168/8192/configured+effective128, physicalTP1/eager/six layers with TP8-local dimensions; raw1engine/6steps/282calls. Python3.12/pandas2.3.3 rebuilds both byte-identically; all8 remote/local hashes match including7,702,607B trace. Durable selected inputs: routing `prefill_results/accepted/`; reports/receipt in `prefill_results/`. Retain anomalies; no independent expert histogram, steady-state, TP8-PD, graph, serving or E2E admission. Historical q1 tables lack balanced identity and cannot qualify balanced decode; ticket09 coverage remains unfinished.

All three B/C P2s closed, ce9ceba root133 merged PASS, e9e79d4 independent APPROVE/source-only and final strict77=77/77/451hashes/no source-cache findings complete (`native_final_fidelity.json`). b62c155 committed existing model-name case input (+2/-1), actual/ceiling23/953=A432+B319+C202, dispatch491; root union `evidence/final_union_b62c155.json`. Identical frontier/tests bytes are not a new capture or independently reviewed correction. Host R3/R2 unchanged; historical950/952 mount evidence, original argparse exit2/model_name FAILED partial13-file06 and native01–05 retained.

Native06 P0 reference ACCEPTED_NATIVE_REFERENCE_R4, reader merged d278659/frozen67f31923/all8/8PASS against unchanged b62c155 run/12.002222GPUh. Client432/432/three rounds/seven boundaries PASS;32ranks/six phases physical graph P NONE/D PIECEWISE+FULL verified, not measured graph/operator costs/physicalTP8 profiling/independent histogram or Frontier prediction/calibration. Final JSON: [routing current root state](../../task_2026-10-22_v4_balanced_routing/progress.md#current-root-state). Old R3 FAIL/native05 UNVERIFIED preserved; full18 and05/09/19 unfinished, no new GPU budget/cap change.

D-Q10 INTEGRATED_ISOLATED_PREREQUISITE e89b45d/source2f87828, merged144PASS/94exact/legacy102positive72negative,3/197/core55 (`../../task_2026-10-23_v4_prefill_contract/integration_receipt.json`). R4 matrix gate reuses actual576d27877/77/451actual hashes/519cache names/777runtime-input bytes, same environment/gitlink/unreached changed unit file, NOT new R4 capture. R4 additional2/223/zeroCore under3/300 separate from historical routing953/dispatch491, no net-union claim. Full05/09 costs/family/predictor remain open; no q1 relabeling/newGPUcollection/full05 PASS.

Isolated private input prerequisite E completed: source4a1b53e6/delivery04a72b08 merged732406aa. Selected attention CSV discovers an externally admitted V4-private sidecar through `read_isolated_scope`; layouts derive from seven non-timing records, not candidate CSV timing. Real candidate02 isolated-only automatic94 medians/layouts equal explicit94, legacy102positive/72typednegative preserved. Root Standards/Spec reviews No findings; root owner194PASS and merged194PASS are separate runs, never summed. [Admission](../../task_2026-10-23_v4_remaining_measurements/input_primary/input_admission.json), [primary report](../../task_2026-10-23_v4_remaining_measurements/input_primary/test_report_2026-10-23_primary.md), [merged binding](../../task_2026-10-23_v4_remaining_measurements/input_primary/merged_matrix_binding.json): fixed4a1b53e6 versus bound576d27877/77 IDENTICAL/451 artifacts rehashed; merged runtime/data/examples/matrix domains byte-equal, only standalone audit differs outside records, no new capture at merge. Full constructor/family admission, serving totals, TP8/DP2/EP16 measured cost, graph equivalence, balanced decode cost and full05/09/19 remain OPEN. Native06 R4 ACCEPTED/old R3 FAIL/native05 UNVERIFIED/GLM pause unchanged. Historical H launch snapshot — superseded by the latest CPU milestone above: H collector launch wiring remains in progress; no new GPU run or CPU-overhead suite.

The prior prefill failure remains failed with no accepted rows. Source startup-bound RCA and capacity fix9b8494c->ef3ac42 are complete; first faulting GPU kernel remains unproven. Native05 stays historical UNVERIFIED. Workflow/GPU/operator-gap alignment precedes CPU-overhead calibration; qualified hardware-matched H3/H4 coverage and18 admission still precede19.

## Historical final execution state — before balanced A/C integration (2026-10-22)

The retained sources, terminal observations and former investigation steps below are historical. Current next actions are recorded above; no old measurement is relabeled balanced.

**Status: PARTIAL measured family; source slices accepted, full serving/native
qualification OPEN.** DSV4-only and GLM PAUSED_PENDING_USER_APPROVAL remain;
official mean and30%/10% gates unchanged.

- Exact-prefill source9d3df65 merged1687c60; isolated-decode R1 source102c0c9
  plus records-only archive7e4b553 merged947961d. P2 independently supplied
  acceptance provenance is accepted, not pending. Root merged four-file suite
  282PASS in94.46s, zero failures/errors/skips; clean merged strict77 executes/
  compares77/77 IDENTICAL, zero findings. Serving/family constructor, PD,
  graph/compiled and first-tail guards remain closed. Existing warmed-q1 isolated
  data remains accepted only within its physicalTP1/eager/six-layer scope;
  event excursions and20–25%median changes remain unexplained.
- Prefill software/package PASS before submission:57standalone+57merged,
  61legacy plans/errors byte-identical, source strict77, image51source/517tool/
  20binary and package527/517/510/51. Geometry6144/C0,7168/8192,planner128/
  effective1,2prefill warmups+3measured repeats per method is unchanged.
  Job exp-1022-024054-801702 was submitted; source/native/mount identity passed.
  Launcher exit1 reports failed at02:49:55 local. Root investigates terminal/
  business cause; no accepted rows or resubmission. Executable package stays
  `/data/ycfeng/Frontier/.worktrees/v4-h800-prefill/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill`;
  root records are not a package copy. Report:
  `task_memory/task_2026-10-22_v4_h800_prefill/test_report_2026-10-22_primary.md`.
- Native05 exp-1022-015758-885725 completed Succeeded4/4,container exit0/
  restarts0,12.144444containerGPUh;432/432 and7boundaries client-business PASS.
  Original-case frozen checker7PASS/rank_runtime_graphFAIL; reference UNVERIFIED.
  `../native05_acceptance_20261022/graph_rca.md`: invalid shape-only dispatch
  assumption plus omitted DP synchronization; independent historical state/DP
  operands/per-step joins missing. Original manifest mtime is correct; no
  copied-mtime workaround or native/graph admission follows. Preserve frozen
  R2,manifest,candidate04/old5efd/9116 and historical native03 failures.

Next root-owned work: prefill actual failure/business RCA ‖ native05 proof
resolution; qualified hardware-matched H3/H4 coverage and18 admission precede19.
Merged report: `task_memory/task_2026-10-22_v4_isolated_decode/test_report_2026-10-22_merged.md`.

## Historical execution state — before final source/terminal receipts (2026-10-22)

DSV4 warm-q1 exp-1022-003156-993689 Succeeded,exit0/restarts0,actual H800,664seconds/
0.184444GPU-hours. Both6144-row tables and all1024 contexts are primary-verified,
raw/summary/CSV byte reconstruction PASS;1.451738230GB completed trace hash verified.
The original first CSA event2838.837891ms is now2.175072ms. Other event peaks and
20–25% median differences remain unresolved; no steady-state/compiled/physicalTP8/
first-tail/full-serving acceptance. Primary report:
task_memory/task_2026-10-21_v4_q1_warm_remeasure/test_report_2026-10-22_primary_measurement.md.
Ongoing isolated-decode reader consumption is assigned to W128-V4-ISOLATED-DECODE-CONSUMER;
constructor/PD/graph/first-tail/serving guards stay closed. First preflight failed BEFORE_SUBMISSION because aggregate32 rejected
Step32+requested1;descriptor ONLY aggregate_H800_ceiling32->128 with527 pins unchanged.
Log:/data/ycfeng/tmp/glm53_dsv4/v4_q1_warm_submit02_20261022.log;root
task_memory/task_2026-10-21_v4_q1_warm_remeasure/observed/identity02.json.
Native04 SEALED_NOT_ADMITTED_NOT_SUBMITTED:full checker staged PYTHONPATH/checkout
mismatch and missing67th breakable_cudagraph.py native pin. Same V4 owner prepares
candidate05 UNSEALED with67pins;all6 CPU joint predicates pass for provisionald06f,
but independent actual-import shadow finding requires correction before freeze.
VERSIONED_CHECKER_FIX ses_ea0af5ae9ffe20NcnuumIBrR5v repairs that binding.
Old5efd/9116/candidate04 immutable;no04 READY_TO_LAUNCH. Final NEW freeze/
seal waits for ALL_COMPUTABLE_FULL_CHECKER_PREDICATES_PASS and all no-more-writes.
GLM06 handoff complete PAUSED_USER_APPROVAL_REQUIRED,no fresh06/tests/build/seal/
children. Isolated measurement is accepted; native admission and DSV4-only/mean/30%/10% rules are unchanged.

## DSV4 priority-case coverage contract — earlier execution checkpoint

D-DSV4-ONLY-GLM-PAUSED makes DSV4 the sole current correction/testing/calibration
and acceptance target. GLM all tiers/tests/new packages/proposal follow-up are PAUSED_PENDING_USER_APPROVAL
pending new explicit user approval;they do not block DSV4. Reuse accepted shared
artifacts without new GLM work. Bind32 GPUs to per-role TP8/DP2/EP16
and64 to TP8/DP4/EP32. Reuse hardware-matched evidence, preserve typed missing
scope/phase/compiled/TP errors, and do not promote H200 measurements into H800
costs. Tickets10/11 control priority, resource class and64-GPU formal promotion.

Latest H800 supplementary result:exp-1021-182611-145615 Succeeded,exit0;
6144rows per CUDA_EVENT/KERNEL_ONLY table,covering cached6143..7166×three
variants×two tile settings. Actual scope is six layers,physicalTP1/eager with
TP8dimensions. One1.449GB completed trace is retained. First event CSA row
2838.84ms remains raw and steady-state timing is unverified;KERNEL_ONLY is
separate. See `../test_report_2026-10-21_v4_h800_decode_primary.md`.
Consumer A/B fixes are accepted at936f913:756 PASS/8 SKIP and merged strict77/77
identical. Qualified full-serving/compiled/physical-TP/first-tail costs remain
open. V4 native candidate03 exp-1021-220603-144957 is finished:384/384 formal,
431/432 all;rep1 first idle invalid,rep2 one warmup502,rep3 client-clean. Global
reference stays UNVERIFIED / REFERENCE_UNSTABLE, not native/formal acceptance.

Warmed exact-q1 sibling package `v4-q1-warm-remeasure` dc86fad/source d2878ed
retains its earlier CHECK_ONLY_READY preparation and seal
7929afaef9deced67543262ad9a3eec398048b68914fd0322b00c97416cbe360 binds two
warmups before the same6143..7166 measured window. Latest launch attempt exits1
with AssertionError in `/data/ycfeng/tmp/glm53_dsv4/v4_q1_warm_20261022_launcher.log`;
primary diagnoses before business cause is established. No submitted/Running/
success/new timings claim. Legacy glm producer-helper names still denote DSV4.
Cold-start R1 APPROVE/primary241PASS21SKIP merged7c1b888,TCP merged30db587,
combined288PASS25SKIP. All writers stopped;exact accepted frozen checker
5efd00ea6544d64440c535d06ad415b6a4d1a25b96ce8863cf4944568f37ec36 is in root
`task_memory/task_2026-10-21_native_cold_start/FREEZE.json`. Native admission
PENDING;recheck exact pins before/after seal. Old9116ab7 immutable,no POST retry.
Active V4 successor04 owner `ses_ea0bffa17ffes1B4qUTxbDHVTT` continues packaging;
no source acceptance substitutes for native/qualified full-serving evidence.

### Retained H200 preparation snapshot (historical)

H200 reduced G0-A is now accepted: exact MARLIN positives/negative controls and7/7 original native checks, with primary verified platform/container/launcher success. See [actual result and limits](../test_report_2026-10-21_h200_dsv4_g0a.md). Full-model TP8/DP4/EP32 G0-B is being prepared; native graph/cache/non-KV memory and H3 remain unverified. The observed reduced `cache_config.block_size=4` and attention backend's SWA kernel block size256 describe different quantities and must not be substituted for one another when deriving the full-model configuration.

Ticket 17 shares the replacement fixed workload: 128 requests, 6144 input /
1024 output tokens each, QPS 1.0 with one-second arrivals. H200 profiles follow
the canonical TP8/DP4/EP32/PP1 per-role layout, 6144-token unchunked prefill and
decode contexts through a 7168-token total, including compression ratios 0/4/128,
window/page boundaries, mHC and hash/score MoE work. They depend on the matching
H200 runtime/backend qualification and replacement shape contract, not a
SharedGPT fetch or tokenizer re-count. No-chunk serving requires
`max_model_len >= 7168` and `max_num_batched_tokens >= max_model_len`.
Actual graph mode/capture sizes, quantization and memory budgets remain native
evidence under ticket 18; source-default MTP is off. H800 qualification/tables do
not transfer to H200, and GLM-first measured integration remains unchanged.

## Historical mainline recovery — 2026-10-20

G0-B[DSV4] closed on 2026-10-20 on existing evidence (`exp-1019-172330-858927`). `primitive_profiles` producers for V4 compressed attention (FlashMLA sparse/SWA, compressor, indexer) and MXFP4 MARLIN MoE are the second profiling wave, after the GLM producer changes land, so shared producer files keep one owner. V4 already represents SP shared-expert replication (parameter bytes −0.74% against native).

- **M4 (gap):** the V4 structural path rejects PD (`frontier/execution_time_predictor/deepseek_v4.py:447`, `frontier/attention/deepseek_v4.py:35`). V4 PD work is required for the 64-H800 PD objective and follows `09.full_frontier_path[GLM]`; see the new checklist item below.

### Status at the 2026-10-20 handoff

- **W6 producers committed (no GPU run yet):** compressed attention `45361bb`, SP shared-expert linear-op rows with clamped SwiGLU `4b19e35`, checkpoint-MXFP4 routed experts through vLLM MARLIN `99f3a69`, allowlist fix `3ea2c3f`. Records: [w6_dsv4_producers_20261020/summary.md](../w6_dsv4_producers_20261020/summary.md), [test report](../test_report_2026-10-20_dsv4_profiling_producers.md).
- **Run configs** at `w6_dsv4_producers_20261020/runs/3ea2c3fe72424c4c330ab439024c3a9de4e90ad5/`: attention 6 engines / 129 cases / 10895 steps (5 h timeout), linear-op 1 SP plan / 323 steps, MoE 4059 steps; 3 jobs x 8 H800. GLM launch output stays byte-identical to W4.
- **Risks:** MARLIN launch and timings are unverified until the GPU run; DSV4 non-dummy consumers do not exist yet (`deepseek_v4.py:439` raises `NotImplementedError`), so the rows are produced but not read. M4 (V4 PD) is open.

## Historical C32/MARLIN02 reduced G0-A evidence

Full original V4 `exp-1019-172330-858927` now has actual 16-rank/43-layer/256-expert inventory and four successful four-token responses, with platform/both containers successful. The native logs retain forced EngineCore shutdown and worker SIGTERM; only 16 worker traces persisted, with no EngineCore trace and 736 missing-parent rows. Strict routing remains OFF. Available structure/initialization/generation evidence passes; complete observer/G0-B remains OPEN. [Current actual evidence and boundaries](../test_report_2026-10-19_c36_c38_full_v4.md).

C32 `16495c2` corrects the existing inspector's batched FP8 scale geometry and selected MARLIN/V4 auxiliary checks, with 43 independent CPU tests. Original reduced V4 `exp-1019-161725-258030` then passes all seven native driver checks on H800 and completes successfully at platform/container level. The actual variant remains original DeepSeek-V4-Flash, four layers/16 experts/TP1, dummy weights, V2 runner and selected MARLIN MXFP4. [Native result](../c32_c34_native_20261019/v4_native_run/result.json); [inspector scope](../c32_v4_inventory_20261019/test_report_2026-10-19_v4_inventory.md).

G0-A[DSV4] now passes in its original reduced diagnostic scope. MARLIN02 `exp-1019-165425-915548` completes the selected operator control with two exact positives and six rejected controls across eight rows at fixed `atol=0.04`/`rtol=0`; platform Succeeded, container exit 0/restart 0 and launcher exit 0. Independent artifact readback confirms rows, frozen inputs and terminal identities. Together with C32's imports/dummy initialization, selected inventory, generation and shutdown, this meets the existing G0-A criteria. [Disposition, exact commands, resources and errors](../v4_marlin_control_20261019/test_report_2026-10-19_v4_marlin_control.md).

MARLIN01's import-path FAIL and historical C31 FAIL remain immutable. G0-B is the next native stage; full-model inventory, physical cache contents, V2 internal observer, graph/PD, trained predictor and formal acceptance remain OPEN. GLM C34 queue observations and authorized C35 MONOLITHIC PP1/DP1 ordinary-cache work do not establish V4 agreement or lift grouped single-flight. Existing V4 guards and GLM-first trained integration remain.

## Historical C29–C31 checkpoint — superseded above

C26 is deployed through the exact seven-file reviewed subset; C29 is excluded. Frontier587ec80 bootstrap and005d5cf inspector retain original runtime/model locks, seven checks and fail-closed unknown formats. Root15inventory tests and independent22bootstrap tests pass. C31 job `exp-1019-152410-496857` is terminal Failed/container1/restarts0 because scale-role and auxiliary inventory checks remain invalid. Native execution itself completes: prompts16/384 each produce4tokens, logprobs are finite, no driver exception and shutdown returns. All136registered tensors are finite;3/3expected hash tables and the first-layer `[24,4096]` FP32 mHC broadcast pass. The twelve invalid scale rows comprise eight unverified MARLIN MXFP4 rows and four batched `wo_a` rows rejected by the inspector's two-dimensional shape assumption. Complete G0-A remains OPEN. Actual runner is V2; the unchanged native prefix-cache default is True, which does not extend Frontier V4 prefix support. Both trace files persist; existing collector retains two files with no errors and qualification false. Worker host evidence remains incomplete with20missing parents. [Verification](../test_report_2026-10-19_native_queue_v4_g0a.md).

## Historical C25–C28 checkpoint

C26 is committed at native ab0a553f75435ebffcd9738368daa26d388baa28. The original NVIDIA model adds the canonical generic post-load hook and reuses its existing mHC broadcast finalizer only when missing; checkpoint loading still refreshes unconditionally. Four root source regressions and a real CPU Torch Parameter/reduction/reload/pointer oracle pass. This is a source-supported repair, not V4 G0-A GPU validation; that milestone stays OPEN. C25's reduced GLM graph diagnostic did not deploy C26 and supplies no V4 graph evidence. C23/C24 structural support remains accepted; grouped second-inflight, zero-only, complete idle attention, native/cache-content, graphs/PD, trained and formal gates remain unchanged. C27 selected async capacity2 requires a single scheduler/cache owner and verified write/reuse settlement before lifting any grouped guard. [Root verification](../test_report_2026-10-19_native_inputs_dummy_init.md); [source delta report](../c25_native_progress_20261019/c27_audit/report.md); [dependency and owner plan](../plan.md).

## Historical C23/C24 checkpoint

C23/C24 are accepted at `88c030fdfee25adf49fbf7718a7166cc09e37052` for source/CPU structural scope only: 1,292 isolated checks, source-backed actual 64-logical-device offline/online, allocation-to-zero execution and unchanged77artifact/cache equality. Ticket05 remains PARTIAL; graph/PD/native/trained/formal acceptance remains OPEN. Actual representative zero counters, source/command identities and limitations are recorded in the [verification report](../test_report_2026-10-19_dsv4_cache_zero_dp8.md). The earlier C21/C22 missing-consumer and DP8 guard descriptions below are historical checkpoints, superseded by this result.

The cache-zero assignment stream preserves canonical group order, preempted assignments and repeated occurrences; immutable Batch metadata is queried without mutation. Normal profiling/trained targets remain disabled. Layer 0 is a step/worker timing anchor; unique slab extent is not native stores/H2D traffic. DP8 admits existing TP8/EP64/PP1/H800/MONOLITHIC/SP semantics with 4 local experts and 5,495,132,636 B/device source storage, native-unverified; DP16 stays rejected. No native zero-only output representation or complete idle-attention trace is claimed. The existing original-V4 dummy auxiliary-state blocker and routingOFF policy still block qualifying native work; avoid repeating closed startup/CUTLASS/census investigations.


## Historical C22 scheduler checkpoint — superseded by C23/C24 above

C22 is accepted at `b5120e6b39f44a79245b04de71e0ed95d8bb34a6` for the existing scheduler test module only: genuine V4 constructors, real Request/Batch callbacks, live pages79→27→27→23→0, exact null prefixes/retained IDs/reference counts, reserved null, no active-batch retirement, completion and second-cycle reuse. Root205affected checks and independent73checks pass; deliberate wrong-frontier and missing-reclaim mutations fail. Configured decision records expose new/zero-required IDs79/0/0/0 twice. This is a lane0 scheduler fixture with two real constructors; it does not validate native content, DP-wave/DES or C20's original full-model trajectories. All479nominated production/config/driver files are unchanged. [Report](../test_report_2026-10-19_v4_cache_lifecycle.md).

The audit identifies an unimplemented operator/workflow scope: required cache-block zeroing currently has allocation identities/debug events but no execution-cost consumer. This differs from attention-output zeroing. Approved Option B already requires its explicit accounting before complete structural execution. Continue through existing batch/operator/predictor/stage/trace interfaces, once per step/physical worker from actual new IDs and canonical physical stride; do not multiply by43layers, sum aliases or invent measured duration. Dummy timing remains explicitly synthetic. A core refactor has not been shown necessary.

DP8/64logical stays rejected by both driver and parameter-memory guards. EP64 source/SP fixtures do not qualify actual constructors/memory planning/metrics/serving. Next establish independent source/storage and real constructor oracles before any range extension. These V4 gaps do not gate GLM native/schema work. Full representation/structural acceptance, native initialization, graph/PD, trained and formal milestones remain open.


**What to build:** Frontier simulates a DeepSeek-V4-Flash Replica with H800-measured, trained timing for every backbone variant. That covers uncompressed, CSA, and HCA attention; the compressor with complete and partial state; the sparse indexer; sliding-window plus compressed attention; mHC; the low-rank output projection; hash-routed MoE in layers 0–2; and score-routed MoE with shared experts and the actual quantized kernels. Compression-boundary costs are modelled. It follows the pattern GLM-5.3 established in ticket 04. (Spec v2 T05, §3.2–§3.4, §4.)

**Blocked by:**
- `model_representation`: 03 `static_contract[DSV4]` and `operator_inventory[DSV4]`; independent of native initialization and GLM's trained bundle.
- `structural_execution`: `model_representation`.
- `primitive_profiles`: `structural_execution`, 01 `G0-B[DSV4]`, 03 `layer_inventory[DSV4]` and `cache_inventory[DSV4]`.
- `family_predictor_integration`: qualified hardware-specific `primitive_profiles`,
  applicable `12[DSV4]` and `14[DSV4]`; no GLM full-path/calibration predecessor
  under D-CALIBRATION-PRIORITIES. Reuse accepted shared mechanisms.

### Historical structural disposition — superseded by current priority status

**Status:** PARTIAL representation; reduced G0-A[DSV4] is accepted from C32 initialization/inventory/generation and MARLIN02 operator controls, while G0-B/C/D and measured work remain OPEN. C02 provides the original43layer schedule. Accepted C07 `8d4fb3e` adds mixed FP8/MXFP4/BF16/FP32 constructor bytes and replicated SP shared weights; C08 `92a9da7` adds the explicit typed cache selectors and five-group source layout; C09 `4164a04` adds the approved shared-pool manager/V1 scheduler semantics. Accepted C10 `ba48220` adds canonical per-layer source operator scopes and operation-specific precision metadata, with527isolatedPASS. Each C07-C10 final clean-cache comparison has77 exact artifacts and zero cache filename differences, with frozen modified-snapshot provenance preserved. The full GLM online structural regression also passes after the shared scheduler changes. C12 `7626b3f` static constructor dimensions and C13 `d3d4229` source TP-SP payload/RS dispatch are also accepted (607/738isolatedPASS; separate77exact artifact/cache comparisons). C14 `596451a` adds the selected eager attention source component with833isolatedPASS/independent APPROVE/77exact artifacts-cache. Accepted C15 `91d1697` adds private source-rank SP/mHC/shared/endpoints consumption, with877isolated tests/independent APPROVE/77exact artifacts-cache. C17 now supplies the controlled full43-layer dummy stage/trace/serving and synthetic routed EP path; native routed assignments, physical cache boundary/content validation and measured family integration remain open; C20 below adds source-logical boundary coverage. GLM-first trained integration and the original variant remain unchanged.

The preceding structural-history details do not override the current status:
scoped G0-A/G0-B and A/B source are accepted; P0 native collection is complete
but UNVERIFIED; complete measured/native/formal family coverage remains PARTIAL.
Original dummy auxiliary initialization was subsequently repaired and exercised;
it is not today's native startup blocker. Historical failures remain unchanged.

## Accepted C17 controlled structural execution

C17 is accepted at `dc7f382700218280980b923ba7fe9128091aaf1a` for the controlled CPU dummy structural slice. All26 staged/committed blobs match the final R3 manifest. Root62modules passed1230tests in26.33s; independent final-source review returned APPROVE with no unresolved material findings. Actual full43-layer offline and online each completed4/4requests,12real stages,481,136operator trace rows and1,376EP waves over32logical devices. Maximum stage conservation error is1.00000097e-9ms and maximum phase error6.94e-18ms. The unchanged77-case clean-cache matrix completed in432.822s:77identical artifacts, zero missing/mismatched cases and zero cache filename differences. Official comparator exit1 remains solely the recorded modified-working-tree provenance at91d1697; this is not a clean-commit capture or full-comparator PASS.

R2 executed37-file combined source includes exact final C16 plus final C17 production/driver bytes. R3 adds only three corrected preexisting tests. Root proved485tracked production/config files and both integration drivers byte-identical before reusing the R2 main/matrix evidence; no R3 simulator or matrix rerun is claimed. Initial1227PASS/3FAIL and earlier main_red_06 remain diagnostic history. Source identity is resolved before trace expansion quota; idle sources consume no real-request quota, while synthetic destination IDs remain separately labeled and native assignments UNKNOWN. Existing idle attention emits no request/stage metrics, so complete physical idle-attention trace coverage is not claimed. Complete compression/window boundaries, DP8/64logical, graph/PD, trained/native and full ticket05 acceptance remain open. [Final report](../test_report_2026-10-19_dsv4_structural_execution.md), [root execution/comparison/identity evidence](../c17_v4_structural_execution_20261019/final_acceptance/), [independent final review](../c17_v4_structural_execution_20261019/independent_review/final_r3/review_report.md).

The selected idle-DP no-metadata branch is admitted only for actual DummyForwardBatch under eager NONE, no-MTP/no-DBO/no-ubatching/compiler-pass-SP-off conditions. Model-internal SP still pads FFN rows. Keep source projections, query padding and output zeroing; do not fabricate cache/index/topk/sparse work. Real Batch missing metadata and unsupported trained/profiling paths remain rejected. C16 transfer tracing is preserved byte-for-byte. Overall ticket05 remains PARTIAL; native/measured work stays BLOCKED on the original missing dummy auxiliary state.

## Representation before measurement

C20 source-logical boundary coverage is DONE at `3e6a421401a97ce11b7232facf3fa2028e56938f`; ticket05 remains PARTIAL. Both actual full43-layer offline128/127/3 and online2051/2050/3 runs complete2/2requests and8real stages, with pinned HCA store/partial, SWA and CSA fill/score/topk/logical-write checks. A real decode snapshot bug is corrected through existing `_get_settled_kv_progress`, with causal RED, PP1/PP2/recompute/role/spec regression,1254rootPASS and independent APPROVE. Preserved77-case artifacts/cache names match; official comparator exit1 remains modified-source provenance only. Defaults17/16/2 and pool256 remain; the long CSA input explicitly uses1024blocks to meet846-block canonical demand. Physical cache retirement/content, complete idle-attention trace, configuration/64logical coverage, graphs/PD/native/trained/formal gates remain open. [Final C20 report](../test_report_2026-10-19_dsv4_boundaries.md).

### model_representation
- [x] Register the original pinned model profile through existing modules; resolve only the 43 backbone entries from the compression-ratio configuration and exclude next-token/MTP entries.
- [ ] Represent uncompressed/CSA/HCA attention, full/partial compressor state, sliding window plus compressed cache, sparse indexer, mHC, low-rank output and routed/shared FFN through the shared contracts. Preserve precision/shape/parallelism metadata and unsupported-path errors.
- [ ] Reuse GLM's shared ownership and Step's contract requirements. Add only missing operator/attention variants; no second layer registry, profiling pipeline or memory framework.

Static evidence: `../ticket03_dsv4_static_contract.md` and `../test_report_2026-10-18_model_contracts.md`. Whole-model operator manifests reject mixed compression; use per-layer specs. C07 expresses replicated shared weights for the source-selected SP topology through existing per-layer contracts. It does not partition tokens or implement SP allgather/reduce-scatter ordering. C08/C09 cache behavior remains structural/source evidence, with native cache qualification and full serving still open. Reports: `../test_report_2026-10-18_dsv4_parameter_storage.md`, `../test_report_2026-10-18_dsv4_grouped_cache_layout.md`, `../test_report_2026-10-18_dsv4_grouped_cache_integration.md`.

C10 extends `build_operator_manifest(..., global_layer_id=...)` and the existing QuantizationManager. The attention variants contain8/20/11 reachable scopes for uncompressed/CSA/HCA. Fused Q/KV preparation has one owner; compressor/input/state work remains distinct from conditional compressed writes and alternate short-context index fill. The mHC/endpoints view has once-only boundary owners and excludes generic norm/residual/embedding duplicates. All new source targets remain false, shared MoE families are not globally modified, and the generic linear profiling producer explicitly rejects this model. Source metadata records routed MXFP4 packed storage, FP8 base/shared weights and mixed BF16/FP32 state; it does not claim selected kernels. Evidence: `../test_report_2026-10-18_dsv4_operator_precision_manifest.md` and `../c10_operator_precision_20261018/`. C12/C13/C14 now supply static dimensions, TP communication payloads and an eager attention workload/component consumer. C15/C17 subsequently completed controlled SP source-rank, mHC/shared/synthetic routed FFN, EP backend/endpoints and phase/trace/stage integration. The whole representation checkbox remains open for physical compression/window cache-state coverage and configuration limits; C20 adds actual source-logical boundary coverage and local GLM/V4 plus existing shared-path regression; eventual three-model same-HEAD integration belongs to the external integration owner. Native routing and measured support remain separately gated.

### structural_execution
- [ ] Execute every configured backbone layer in Frontier's labeled structural dummy timing mode, including compression-boundary/window-rollover and hash/score layer ownership checks. Reconcile contributions and cache state; do not substitute balanced routing for unknown native routing.
- [ ] State which serving paths are represented and which remain blocked by 07/08 or a concrete core limitation. A structural result proves neither native dummy initialization nor calibrated prediction.
- [ ] Demonstrate GLM/V4 and existing shared-path regression here; document reusable contract compatibility for the separate Step/integration owner. Step implementation, its fixtures requiring Step production code and eventual three-model integration are owned outside this worktree. Do not import its unreviewed production changes.

### Next representation and execution dependencies

The source inventory is a list of reachable scopes, not their execution order or a timing sum. Keep the following work attached to this ticket and the existing shared owners; these are not new registries or an approved generic runtime redesign.

| Work | Prerequisite and owner | Acceptance gate |
| --- | --- | --- |
| Workload shapes and branch selection (C14/C17 controlled eager path and C20 logical boundaries DONE; physical-state/configuration coverage OPEN) | Accepted per-layer manifest/precision; one operator/workload owner | Derive projection, real/padded attention-head, compressor and indexer shapes from configuration and request/chunk positions. Separate boundary, off-boundary, short-context and capture paths; mutually exclusive branches are not charged together. TP8 has8 real local attention heads versus64 padded heads in the pinned FlashMLA source path. |
| SP token and communication ownership (controlled C13/C15/C17 slice DONE; native validation OPEN) | Workload shapes; one owner of existing batch/EP workload and communication bindings | For the source-selected PP1/EP/TP>1/DP>1 topology, padded local token count is ceil(T/TP). Local mHC/router/shared/routed work and full-token attention have distinct ownership. Reconcile per-layer allgather, truncation and reduce-scatter, plus one final allgather of mHC-expanded hidden states. Preserve padded versus valid tokens and omit ordinary wo_b/shared allreduce on this SP path. This does not assume the same topology is selected by every backend. |
| mHC and endpoint execution (controlled C15/C17 slice DONE; native validation OPEN) | Per-layer scopes plus SP ownership; same execution owner | Preserve embedding/first pre,42 cross-layer post/pre owners,43 attention-post/FFN-pre owners and final post/head/norm/LM-head once each. Compose through existing stage/timing/trace interfaces; the manifest's family order alone is not a native launch schedule. |
| Hash/score FFN workload (C17 explicit synthetic structural path DONE; native occupancy OPEN) | Per-layer routing contract and valid input/auxiliary state; existing routing-workload owner | First3 layers use hash and remaining40 use score routing. CPU fixtures establish structural identities only. No fabricated native occupancy or silent balanced substitution; missing native auxiliary state keeps native qualification blocked. |
| Controlled full-layer serving (C17/C20 paths DONE; complete operator coverage including cache-zero, full-model physical state and64logical OPEN) | All preceding represented components and accepted grouped-cache initial scope | Execute all43 backbone layers through Frontier's main workflow; reconcile request/stage/operator/communication/cache ownership at compression and window boundaries. Keep unsupported PD/MTP/prefix paths rejected. Native timing and calibration remain separate. |

The pinned SP source is `vllm/models/deepseek_v4/nvidia/model.py` at `_use_sequence_parallel`, `DeepseekV4DecoderLayer.forward`, and `DeepseekV4Model.forward`, together with `vllm/models/common/ops/sequence_parallel.py`. Source audits may run concurrently with GLM ticket07 payload/lifetime work because the files and acceptance gates differ. A demonstrated need to change a core execution interface goes through the existing material-refactor decision rule; no such redesign is assumed by this table.

Accepted C12 `7626b3f` is limited to static, unpartitioned constructor tensor dimensions in the existing QuantizationManager metadata and MetricsStore CSV. CSA/HCA compressor shapes remain variant-specific. Dynamic token counts, rank-local TP shapes, conditional compression writes and actual backend selection are excluded from that global metadata owner. This accepted source/CPU slice has607isolatedPASS, independent APPROVE/181PASS and77exact legacy artifacts/zero cache filename differences. It is not a completed workload representation. Evidence: `../test_report_2026-10-18_dsv4_static_tensor_shapes.md` and `../c12_static_shapes_20261018/`. C13 `d3d4229` is accepted at the source payload/dispatch seam with one owner,738root-isolatedPASS, independent APPROVE100PASS and77exact legacy artifacts/zero cache filename differences. Evidence: `../test_report_2026-10-18_dsv4_sp_payloads.md` and `../c13_sp_payload_20261018/`.

Accepted C13 extends the existing `CommOperatorSpec -> _predict_comm_operator_with_context -> CC backend` consumer with source attention-input allgather, attention-output reduce-scatter and final mHC-expanded allgather scopes. The reduce-scatter alias and predictor dispatch now reach the existing CC backend. At C13 all three source scopes retained target flags=false/no execution-time attribute and normal V4 serving remained rejected; C17 subsequently admits only the controlled dummy path through canonical consumers. For source input rows T and TP size N, local padded rows L=ceil(T/N), total padded rows P=L*N. BF16 input payloads are respectively `L*H*2`, `P*H*2`, and `L*hc_mult*H*2` bytes; weight quantization does not reduce these activations. At T=17/N=8/H=4096/hc_mult=4 the independent source oracles are24,576/196,608/98,304bytes. All three use ATTN_TP. These payload/dispatch checks alone cannot establish timing totals, trace ordering or once-only endpoint ownership; those require the later existing layer/stage accounting slice. No generic DAG or core refactor is established as necessary.

Accepted C14 derives eager attention query/context intervals from Batch scheduling snapshots and the bound layer variant. The read-only outline under `../c13_sp_payload_20261018/next_slice_preparation/` records source oracles: T17/TP8 attention projections use17rows after SP gather/truncation, while communication carries24; main real8/padded64 heads; CSA2051/2052 fill versus scores/topk; HCA127/128 compression boundary. C14 uses the existing predictor component consumer and kept full-family admission closed at that step. C17 supplies controlled full-layer/mHC/FFN/endpoint ownership; trained/profiling/native admission stays closed. The outline is not a finalized new API/file/schema requirement. Explicit eager metadata state, source topology and unsupported graph/warmup/spec/prefix/PD guards must be reviewable. A component timing sum cannot qualify native auxiliary-stream stage latency.

Keep SP padding distinct from routed valid rows. For DP inputs17+10 and TP8, source geometry gives27valid rows and40physical compute rows. The earlier162-assignment arithmetic assumes padding is actually masked; it is not an unconditional native score-router result. The pinned score-router CUDA sqrtsoftplus fast path has no `is_padding` input, while the hash path can apply the mask. Keep actual routed assignments UNKNOWN until the selected route is evidenced. Destination EP assignment rows are not SP local hidden-state rows. Also, pinned `vllm/config/parallel.py` defaults `all2all_backend="allgather_reducescatter"`; `fused_moe/runner/moe_runner.py` dispatch/combine may use it on DP>1/SP paths. Frontier's generic assignment-row ALLTOALL contract does not qualify that native path. For C17 labeled dummy structural execution, select the source allgather/reduce-scatter contract and explicit synthetic destination counts while keeping actual native routing and masking UNKNOWN. Native workload evidence must establish the selected backend and padding behavior before claiming native-compatible or qualified EP dispatch/combine. This is separate from the three TP-SP collectives; measured/native V4 execution admission remains closed.

### C15 accepted source/CPU component — 2026-10-19

Accepted commit `91d169757e11dfa54aae439643508504117946e8` connects source-rank SP geometry, mHC/shared FFN and explicit selected-logits endpoints to a private partial component consumer. It reuses the SP-aware layer contract, existing MoETime, C14 attention and C13 communication once. Root isolated37modules:877 tests passed in12.90s. Independent final-byte review: APPROVE, zero material findings,44focused tests plus48rank queries,43-layer ownership/87collective checks and duration sensitivity. The unchanged77-case clean-cache matrix completed in419.00s; all artifacts across77cases and cache filenames match. Official comparator exit1 is solely frozen modified-source provenance at parent596451a, retained rather than called a clean-commit capture or full comparator PASS. Only the3frozen C15 files were committed; parallel C16 changes and inherited A2 were excluded.

T17/TP8 uses3physical rows per rank, geometric valid counts(3,3,3,3,3,2,0,0), and17attention rows. Zero-valid ranks retain local mHC/shared work. Actual routed assignments remain UNKNOWN because native score/hash masking is branch-dependent. Existing input masks are unsupported in this component. Embedding uses an inclusive synthetic T-row wrapper; LM-head requires explicit selected rows and leaves physical vocab/logits allocation unspecified. Routed EP, shared/routed overlap, separate embedding/logits communication, full phase/stage/trace publication and normal V4 serving/native/formal gates remain open. The component is explicitly PARTIAL and constructs no full ExecutionTime or StageExecutionTime.

Evidence: `../c15_source_execution_20261019/` and `../test_report_2026-10-19_dsv4_source_execution.md`. Archived C14 preparation remains immutable; the narrow routing-mask correction is in `../c15_source_execution_20261019/routing_mask_correction.json`. H800-first, insufficient-quota H200 fallback, aggregate<=64 task GPUs and non-preemptible constraints remain current. No new GPU/native/profiling/calibration campaign or checkpoint download occurred.

### C14 accepted source/CPU component — 2026-10-19

Accepted C14 `596451a` adds eager request/layer attention workload and the existing synthetic component consumer. Root isolated36modules:833PASS in12.91s; independent APPROVE/0unresolved with54focusedPASS and12additional source oracles. The unchanged77-case clean-cache matrix completed in447.49s with77identical artifacts and zero cache filename differences. Official comparator exit1 is solely frozen modified-source provenance, retained rather than called a clean-commit capture or full comparator PASS. This is an eager source/CPU attention component, not full V4 serving or native timing. SP rank-local token execution, mHC/shared/routed FFN, selected-logits endpoints, phase/trace ownership and full43-layer integration remain open. Normal serving/profiling/E2E family guards remain closed. Exact records: `../test_report_2026-10-19_dsv4_attention_workload.md`, `../c14_attention_workload_20261019/`.

Remaining execution order after C15: connect the accepted local source components to existing canonical timing/phase/trace owners; represent hash/score routing and the selected EP backend without assuming padding masking or balanced assignments; then complete43-layer structural serving. Reuse C15 source-rank/mHC/shared/endpoints, C14 attention and C13 communication once rather than repeating their inventories. The read-only21-source readiness and root disposition are retained in `../c14_attention_workload_20261019/next_execution_preparation/`. At T17/TP8 each rank computes3physical rows, total24padded, with valid rows `(3,3,3,3,3,2,0,0)`; attention retains17rows. A valid-zero rank still performs local mHC/shared work. Geometric valid rows do not establish actual routed assignments: the pinned score-router CUDA fast path has no padding-mask input, whereas the hash path can apply the mask. C15 leaves routed assignments UNKNOWN until the selected routing/backend path is evidenced. Source evidence and primary correction: `../c15_source_execution_20261019/source_review/` and `../c15_source_execution_20261019/routing_mask_correction.json`. Reuse existing stage/layer/Batch/CC owners; no generic DAG/core redesign is demonstrated. Explicit selected logits rows and the actual EP backend remain separate required inputs; a component query is not full stage publication. No material core refactor is demonstrated; if a required capability cannot fit existing interfaces incrementally, provide the exact limitation and smallest alternatives before seeking approval.

## Milestones

### pd_support (per M4, model-specific qualified prerequisites)
- [ ] Define and implement V4 structural PD on the existing transfer owners: grouped-cache transfer payload, compressed/sliding state handoff and D-side reservation; explicit errors for unsupported states.

### primitive_profiles
- [ ] Complete matching hardware/scale coverage for17 and10/11, H80032 first,
  then eligible64/H200 cases:6144 unchunked prompt,1024-output decode through7168,
  ratios0/4/128,hash/score routes,4/128-token/KV-page/window and graph boundaries.
  Reuse matching rows; warmed-q1 preparation is not a measurement. Chunk-boundary
  tests remain diagnostic; chunked serving is not the formal workload.
- [ ] Profile mHC (pre, post, head, Sinkhorn or its fused form) as substantive backbone work.
- [ ] Coordinate DSV4 GPU allocation with06 and the active10/11 owners so
  measurements never share a GPU/NIC;paused04/GLM work is not a dependency.

### family_predictor_integration
- [ ] Bind per-layer variants from the configuration. The compression-ratio array's non-backbone entries never execute.
- [ ] Complete the DSV4 operator/importer/trainer/predictor chain using accepted
  per-layer timing,MoE workload and backend adapters;no new04/GLM work needed.
- [ ] Hash-layer routing uses the input and verified auxiliary state; the gate load is never silently replaced by balanced.
- [ ] Run compression-boundary and window-rollover tests, held-out shapes, identity failures, and the fidelity comparison.

## Acceptance

- [ ] Every backbone variant is covered, MTP is excluded, and hash auxiliary state is valid.
- [ ] No scope is charged twice, and no gating load is silently balanced.
- [ ] Preserve prior GLM path/fidelity evidence without new GLM simulation/tests/
  calibration while paused. Retain C07/C08/C09/C10/C12/C13/C14's77 exact legacy
  artifacts/cache and C14's833PASS/C09's historical GLM regression. Verify only
  affected active DSV4 requirements;paired GLM completion is not a current gate.
