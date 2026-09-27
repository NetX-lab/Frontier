# W6 native FP8 mode check receipt, 2026-09-27

| Field | Value |
| --- | --- |
| Decision | Open-PR review D12, approved 2026-09-26: one H800 `codesign` job for web finding W-P35-01 (`b102652`) |
| Why a new test shape | The native FP8 check ran block mode only, so it could not reach the per-tensor and per-token dispatch `b102652` changed. `1bf5b09` runs it in block, per-tensor and per-token FP8 (width 192 outside block mode) and asserts each GEMM input's scale shape. |
| Submission | `kun-workspace-vgen2`, StepMind Python `RJobBackend` (`STEPMIND_BACKEND=rjob`), personal auth `i-fengyicheng`; the W6 launcher plus source hashes and a cloud-volume log copy |
| Job | `exp-0927-102404-042641`, created 2026-09-27T02:24:04Z, `Succeeded` at 02:27:29Z, launcher stopped the job record at 02:27:29Z |
| Creator / group | `i-fengyicheng` / `codesign`, tag `H800`, 1 GPU / 8 CPU / 64000Mi |
| Image | `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2` |
| NFS mount | `100.96.128.195:/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr` at the same path; cloud volume `juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp` |
| Source | worktree HEAD `1bf5b09`, no modified tracked file at submission; worker sha256 of the test (`5459e1f1...`) and of `moe_vllm_kernel.py` (`d5dec728...`) equal the local files |
| Worker | `gpu-h800-0076.host.platform.shaipower.com`, `NVIDIA H800`, Python 3.12.11, torch 2.8.0+cu128, vLLM 0.10.2, `VLLM_API_VERSION=0.10.x`, `FP8_AVAILABLE=True`, pytest 9.1.1 and pandas 3.0.6 from the internal mirror |
| Command | `python3 -m pytest -q -rA -p no:cacheprovider --no-header tests/integration/test_moe_fused_expert_numerical_parity.py` |
| Result | **10 passed in 15.63 s**, `W6:PARITY_EXIT=0`, `WORKER DONE status=0`. The seven zero-tolerance comparisons pass as before; `test_fp8_path_runs_on_the_gated_activation[block]`, `[per_tensor]` and `[per_token]` pass, so vLLM's `moe_kernel_quantize_input` returns scales of shape `(M, K/128)`, `(1,)` and `(M, 1)` for the two GEMM inputs and the fused kernel accepts them. |
| Log | `worker_log.txt` in this directory, fetched with `logs_replica` about 1 minute after completion; no credential text. Full copy on the cloud volume: `/mnt/codesign-exp/ycfeng/frontier/w6_fp8_modes/2026-09-27T0227250000/worker.log` |
| Limit | Structural: FP8 numerics are still not compared against a reference. |
