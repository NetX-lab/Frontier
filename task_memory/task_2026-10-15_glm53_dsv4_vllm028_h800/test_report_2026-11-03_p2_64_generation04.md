# P2-64 generation04 verification and execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | YC approved one fourth submission under D-P2-64-FOURTH; package-external authorization bound to the frozen manifest. |
| 2026-11-03 | Prepared and independently verified a fourth-generation candidate after the lead-caused interruption; no GPU submission authorized or performed. |

## Current execution state

D-P2-64-FOURTH approves exactly one fourth submission. The approved scope is unchanged:64H200=8x8,all-preemptible,step_main,3600s,64container GPUh maximum. `h200_mainline/p2_64_formal_retry04/approval_receipt.json` binds the user reply `continue` to the reviewed manifest. The package-external wrapper authorization was corrected from the inherited third-job text; `bash -n` passes. No package byte changed.

Submission is assigned to Pi `p64-formal4-submit-1103e`, session `5de845f5-6c54-42d1-aa42-2cb11b6cda13`, custom/deepseek-v4.1-flash-sys at thinking high. This turn ends normally after job/spec verification; an independent label performs monitoring. No fifth submission or automatic retry. Exact job/spec verification has passed; worker/terminal/metric verification is pending.

## Historical offline result and boundary

PREPARED_UNSUBMITTED_PENDING_USER_APPROVAL. Package `h200_mainline/p2_64_formal_retry04/package_01`, manifest SHA256 `d75b0277a3c08592359002d256ac8115844f50fa051efe422815cc5b2d19f081`. The existing three-job allowance is exhausted. This report accepts local package preparation only; neither check-only nor the old promotion receipt authorizes a fourth GPU submission.

The only source change is a nine-line entry in the unchanged `h200_mainline/build_package.py` CASES registry. Compared with retry03, only run_tag and purpose differ. New run_tag `dsv4-h200-p2-64-formal-20261103-04`; purpose explicitly records pending user approval. Nativea06a72a4326cba02234df6cfbdc21b78d3a15d26, harnessf33306d40443cfc0c44842d6ee044f6703c736e7,3x128formal6144/1024requests atQPS1,16pair warmups per repeat and promotion digestd505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596 remain unchanged. Proposed resources remain8x8H200,all-preemptible,step_main,3600s,maximum64GPUh.

## Verification

Cwd `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp`. Builder/check-only interpreter `/data/ycfeng/envs/openmopd-py312/bin/python`3.12.13; independent audit `python3`3.10.6. No conda activation or dependency installation.

```bash
/data/ycfeng/envs/openmopd-py312/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/build_package.py p2_64_formal_retry04 --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/glm53_dsv4/h200/audit_formal04_package.py
bash -n /data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry04.sh
```

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Builder/reconstruction/preflight/offline launch | Existing checks pass | All PASS; seal47files | PASS |
| Independent seal comparison | Every old/new input matches its seal |47/47each;41byte-identical,6identity/provenance files changed | PASS |
| Exact allowed changes | Only package/run/cloud identities,purpose,build time and associated digest | Longest-first exact substitutions reproduce every changed byte | PASS |
| Independent launch.py --check-only | No submission; explicit input-hash marker |exit0,INPUT_HASHES_PASS_NO_SUBMISSION | PASS |
| Frozen sources/workload | Native/harness clean and same commits;all engine/trace/Frontier bytes unchanged | Both clean;all expected bytes preserved | PASS |
| Fresh output/wrapper | No launcher markers or case/runs;wrapper syntax valid | All absent;bash-n exit0;wrapper never executed | PASS |

The six changed sealed files are config.json,case/inputs/native_config.json,overlay.json,source_lock.json,case/manifest.yaml,evidence_readiness.json. Manifest.json is outside its own47-file seal and changes accordingly. Package03 remains at its original digest. Inherited whitespace in frozen native sources is preserved; authored-file whitespace is checked separately.

Pi labelp64-formal4-prepare-1103d/sessionc5995fa2-4a23-46c8-9980-04467e8252e0 finished normally/exit0/settled. Lead inspected the full registry diff and independently reran only the package acceptance needed for this new candidate; R5 tests and Frontier replays were not repeated. Evidence: `h200_mainline/p2_64_formal_retry04/{package_01_build.json,lead_package_acceptance.json,executor_build.log,compare_to_retry03.json}`.

Remaining: fourth-job resource decision, actual native execution, terminal/copy/coverage/R5 and every repeat's four metrics. The historical24.3GPUh estimate has not been validated for this serial pair-warmup generation;64GPUh is the proposed ceiling. `post_run_command.json` pins the existing completed Frontier replay and remains planned/not-run.

Record-format check: the first JSON parse found that executor compare_to_retry03.json held plain command stdout. The original remains unchanged in executor scratch; the durable file now stores that exact stdout and source digest in a valid JSON object. No package or seal byte changed. The affected JSON/record checks were rerun before staging.

Final record validation:38JSONfiles parse,84scoped paths staged,and all authored-file whitespace checks pass. Full staged diff-check reports only the three inherited trailing-space lines228/271/274 in the sealed native_config_parallel.py, whose bytes are identical to generation03 and the native source. No frozen source was normalized. Both Pi turns ended normally with exit0/settled and the required model/thinking identity.

## Submission acceptance — 2026-11-03

Job `exp-1103-131952-259161`, UID `9d02faf7-621e-4368-ae73-4110312c6ae2`, created05:19:52Z. Pi `p64-formal4-submit-1103e` ended normally with exit0 and `[settled] aborted=false`. Lead independently inspected the query scripts and selected platform receipts, checked exact name/UID/creator, requested8x8H200,preemptible=yes,step_main,global queue,restartable=false,timeout3600 and both RDMA limits. Image matches the sealed config; NFS `100.96.128.195:<package>:<package>` matches the resolved submission host, and required codesign-exp mount is present. The manifest still matches approval_receipt.json; prelaunch run_manifest names D-P2-64-FOURTH. Every acceptance predicate PASS.

Direct local `ps -p 81539 -o pid=,ppid=,sid=,lstart=,stat=,args=` and `/proc/81539/cmdline` confirm the launcher remains alive after the Pi parent exited. No launcher_exit.json exists. No stop/control action was issued. Evidence: `h200_mainline/p2_64_formal_retry04/lead_submission_acceptance.json` and selected `generation04_*` receipts/scripts. Independent monitoring runs under Pi `p64-formal4-monitor-1103f`, session `7bdd2659-fd5e-4a5f-9c58-e109958279a6`.

At05:20:58Z the job remained Pending with no replica objects or start clocks. The preceding05:19:35Z capacity snapshot reported47usable GPUs and5eight-GPU nodes; this is a snapshot, not a placement reservation or a global scheduling verdict. Native worker/source/API/pair/formal acceptance remains NOT RUN. Exact GPU usage remains unknown before terminal clocks. Prepared but unexecuted post-terminal copy wrapper: `/data/ycfeng/tmp/glm53_dsv4/h200/copy_p2_64_formal_retry04.sh`, derived only by replacing generation identities in the existing wrapper; bash-nPASS.

## Queue checkpoint — 2026-11-03T05:26:00.691550+00:00

Independent monitor has 6 successful exact-job snapshots, all with the approved UID/creator, no replica objects and no read errors. Latest phase Pending; platform queue reason is Insufficient GPU quota. The current source snapshot is retained in `h200_mainline/p2_64_formal_retry04/generation04_queue_checkpoint.json`. Launcher81539 remains alive; monitor labelp64-formal4-monitor-1103f is active and has a20-minute unscheduled checkpoint. Resource placement blocks native execution; this is not native failure or numerical acceptance. No stop or new submission.
