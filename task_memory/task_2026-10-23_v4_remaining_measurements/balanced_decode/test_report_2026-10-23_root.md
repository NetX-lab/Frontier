# Root balanced decode prelaunch and runtime checks

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archived post-integration input/image/SDK checks, independent seal and the initial running-worker evidence. |
| 2026-10-23 | Verified terminal, constructor, immutable inputs and hash-identical output download; data audit remains separate. |

## Scope and environments

Root worktree: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, merged source56e9405d. Immutable package remains at `/data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run/task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode/h800_decode_balanced_02`. Runtime source97803f07; native2cf0a691 plus unchanged accepted C26. Local Python3.10.6; StepMind virtualenv Python3.10.6; locked-image Python3.12.3. No conda environment was changed.

## Commands and observed results

| Check / command | Expectation | Actual | Verdict |
| --- | --- | --- | --- |
| `git diff --numstat 97803f07..64133239`; full diff retained as `source.diff` | Exactly the two allocated maintained files, ≤300 gross lines |39+175 additions, no deletions;214lines | PASS |
| Python SHA256 checks and `git archive 97803f07 frontier data/config` | Exact package/source/original identities |530inputs;512Git files; originals527/529 and candidate01's530 unchanged | PASS |
| `git diff --exit-code 4a1b53e6 HEAD -- frontier data examples tests/e2e/refactor_fidelity tests/integration/canonical_case` plus comparison SHA256 | Existing77 comparison remains applicable | Exit0; comparison7a8728ec…d6 unchanged | PASS, existing binding |
| `/data/ycfeng/tmp/glm53_dsv4/check_balanced_decode_image.py` generates the pinned verifier input; Docker argv/stdin retained below | Real image bootstrap and fresh wrapper entry; exact60/520/20 pins | Two explicit PASS witnesses, CUDA uninitialized, package unchanged | PASS CPU entry |
| StepMind Python runs `/data/ycfeng/tmp/glm53_dsv4/check_balanced_decode_launcher.py <package>` | Actual ResourceConfig before submission |1GPU/16CPU/mem_gb128/replica1/H800/codesign, preemptibleFalse, correct source mount | PASS CPU entry |
| Personal Python RJobClient quota/census; SSH exact destination stat | Fresh personal output directory; complete personal census; existing32-GPU launcher alive |13:35:48Z122/128H800, only profile02 active/Pending; destination absent | PASS snapshot |
| Existing `bash <package>/launch/launch_glm_producers.sh <package>/jobs/Q1_WARM_6143_7166_BALANCED.json` | Authorized one-GPU submission with live launcher |PID3647153; exp-1023-213905-916183 created13:39:05Z | PASS submission |
| Personal Python exact-job snapshot | Creator, requested resources, non-preemptible placement, local NFS | Running13:40:46Z; i-fengyicheng; H800/codesign; GPU1/CPU16/memory125Gi; NFS100.96.128.195 and owned worktree | PASS after auxiliary check correction |
| SSH read of the exact personal `status.json` | Allocated H800 and healthy measurement startup |13:44:39Z: one H800,81559MiB, driver570.124.06, CUDA available; measurement running | PASS startup only |

Owner131profiler/routing and256related CPU tests were independently inspected as supplied evidence, not rerun by root. No post-run numerical acceptance is claimed.

Exact Docker/SDK argv, declared environment, timestamps, return codes, stdout and script hashes:

- `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_image_cf8j40z6/{receipt.json,stdin.py,stdout.log,stderr.log}`. Docker uses the locked image2286e853…6635, `--pull=never --network=none --memory=6g --cpus=2 --pids-limit=128`, read-only package mount, `VLLM_USE_V2_MODEL_RUNNER=1`, and `python3 -I -B - <package>`.
- `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_sdk_8oci6za_/{receipt.json,stdout.log,stderr.log}`. `PYTHONPATH=/data/ycfeng/steptron`, `STEPMIND_BACKEND=rjob`, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`; native `spawn_tasks` is intercepted before its body.
- `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_quota_bw4wwwq8/{quota,census}.json`; `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_destination_4tbnt9l0/receipt.json`.
- `root_submission.json` retains exact launch argv, directory and PID; log `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_launch_20261023.log`.
- `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_worker_2d2su6cq/receipt.json` retains the exact SSH command and worker runtime. Python3.12.3, Torch2.13.0+cu130, vLLM0.28.0, FlashInfer0.6.16.post3, pandas2.3.3.

## Auxiliary failure and correction

The initial snapshot verifier returned1 on `AssertionError: expected memory '128000Mi', actual '125Gi'`. The original snapshot is retained. `kubernetes.utils.quantity.parse_quantity` proved both equal134217728000bytes; creator, resources, non-preemptible label and mount then passed (`platform_launch_check.json`). No resource, code or workload change occurred, and neither launcher was stopped.

## Pending

Independent validation of all6144rows/method,2048measured steps/96256calls, per-row routing/native origins, raw-to-CSV statistics and trace coverage is in progress. No data admission yet. Corrected local-MoE population design is in `../moe_grid_design.md`; package preparation proceeds independently.

## Root post-run checks

Receipt: `root_run_check.json`. Expected and actual:

- Personal exact-job and replica snapshot, via StepMind Python `RJobClient.api.get_namespaced_custom_object`, `.list([name])` and `._get_replica_kapis(replica.process_id)`: PASS. At14:18:36Z, Succeeded/Completed/exit0/restart0, image2286e853…6635, personal creator. Raw selected receipt `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_terminal_gonqtfhz/receipt.json`; no environment/API credential payload retained. The same read-only command confirms profile02 Pending/no replicas and its live launcher at14:18:37Z.
- `/data/ycfeng/tmp/stepmind-env/bin/python -B /data/ycfeng/steptron/tests/integration/inspect_stepmind_personal_job.py exp-1023-213905-916183`, under the handbook proxy and timeout60s: PASS. Redacted log `/data/ycfeng/tmp/exp-1023-213905-916183.log`; producer and aggregate exit0. Launcher log records success before its normal exit-handler stop call; final platform phase remains Succeeded.
- Cloud SHA256 inventory through `ssh -o BatchMode=yes -o ConnectTimeout=15 shai-ycfeng 'python3 -'`, scoped to the exact personal root from `root_run_check.json`; `scp -B -r shai-ycfeng:<root> <fresh artifacts directory>`; local streaming SHA256 comparison: PASS8/8,1,601,865,978bytes. Inventory/download receipts in `/data/ycfeng/tmp/glm53_dsv4/balanced_decode02_results_6zkg5vz4/`. First attempt used `<root>/.` and failed with `unexpected filename: .`; fresh corrected attempt retained every original byte.
- Local Python3.10.6 SHA256 loop over sealed manifest `files`: PASS530/530; original manifests527/529/530 PASS. Exact manifest and file hashes are retained in the referenced immutable packages.
- Original worker log: PASS nativeV2, physicalTP1/DP1, eager, balanced_topk, LLM budget6144 and resolved scheduler budget6144, max_num_seqs1. Original status: PASS1×NVIDIAH800,81559MiB, driver570.124.06; Python3.12.3/Torch2.13.0+cu130/vLLM0.28.0. This resolves the preparation-time native-budget UNKNOWN without editing the frozen manifest.
- Resource duration:569s container,547s measurement/aggregate phase,592s submission→terminal; all below1800s.592/3600=0.1644444444GPUh is a conservative span upper bound, not a platform billing measurement.

The tokenizer/chat-template warning with `skip_tokenizer_init=True` is retained in the original log; the token-ID measurement completed. No warning suppression, source change, sample removal, gate relaxation or table publication was performed.
