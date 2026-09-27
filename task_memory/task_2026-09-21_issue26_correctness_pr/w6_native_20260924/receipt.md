# W6 native rerun receipt, 2026-09-24

| Field | Value |
| --- | --- |
| Decision | User, 2026-09-24: "A7:尝试重新提交" (requirements.md, interview answers for items 1-7) |
| Submission | `kun-workspace-vgen2`, StepMind Python `RJobBackend` (`STEPMIND_BACKEND=rjob`), personal auth `i-fengyicheng`; launcher `submit_w6_parity.py`, unchanged since `exp-0922-202645-561899` |
| Job | `exp-0924-114241-429126`, created 2026-09-24T03:42:41Z, `Succeeded` at 03:45:49Z, launcher stopped the job record at 03:45:51Z |
| Creator / group | `i-fengyicheng` / `codesign`, tag `H800`, 1 GPU / 8 CPU / 64000Mi |
| Image | `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2` |
| NFS mount | `100.96.128.195:/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr` at the same path |
| Source | worktree HEAD `363a1dd`, no modified tracked file at submission; FP8 profiler code last changed in `f236c17`, the integration test in `f236c17` |
| Worker | `gpu-h800-0203.host.platform.shaipower.com`, `NVIDIA H800`, Python 3.12.11, torch 2.8.0+cu128, vLLM 0.10.2, `VLLM_API_VERSION=0.10.x`, `FP8_AVAILABLE=True`, pytest 9.1.1 and pandas 3.0.6 from the internal mirror (runbook section 10) |
| Command | `python3 -m pytest -q -rA -p no:cacheprovider --no-header tests/integration/test_moe_fused_expert_numerical_parity.py` |
| Result | **8 passed in 14.53 s**, `W6:PARITY_EXIT=0`, `WORKER DONE status=0` |
| Log | `worker_log.txt` in this directory, fetched with `logs_replica` about 2.5 minutes after completion; no credential text |

The first submission attempt of the day, before the user's retry instruction, was
refused locally by the Claude Code permission classifier and never reached the
platform. The retry's outer wrapper (`run_w6_with_logs.sh`) exited 1 after the job
was created: it sources an environment script that sets `set -euo pipefail`, so its
first poll for the job name, a `grep` that had not yet matched, ended the wrapper.
The submitting Python process, which owns the job lifetime, kept running to
completion, so the job was unaffected; only the wrapper's log poller never started.
