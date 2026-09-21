## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-07 | Recorded source-anchored recovery findings and pending decisions. |

# Context Recovery

## Provenance

- New worktree: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
- Branch: task/issue26-ttft-h200-20260907
- Base: main d71ad80b0800880808a0857fd30477e6d96592c6
- Historical worktree: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-calibration-20260904 at 0727a1ca. Its source and configuration records are recovery references only. No old CSV, measured latency, trace, or trained cache has been copied into the new task.
- Main includes PR #30 and commit 8f39181b (online DP lanes bound to shared stage forward groups). AGENTS.md specifies one serving Replica with two DP request owners and one EP8 domain.

## Recovered single case

| Setting | Recovered value / current status |
| --- | --- |
| Model | Qwen3-30B-A3B-Instruct-2507; data/config/models/qwen3-a3b-30b-moe.json |
| Architecture | Qwen3MoeForCausalLM, 48 layers, hidden=2048, experts=128, top-k=8 |
| Precision / weights | bfloat16 / dummy; no real weight download |
| Workload | pf4096_dc1024, 4096 input tokens, 1024 output tokens, 100 formal requests, Poisson QPS=2 |
| Parallelism | one 8-GPU pod, ATTN_TP4, ATTN_DP2, MoE_TP1, MoE_EP8, PP1 |
| Runtime | online co-location, vllm_v1, sequential Frontier event execution |
| Scheduler | no chunked prefill; max_num_batched_tokens=16384; max_model_len=16384 |
| CUDA Graph | disabled; enforce-eager on vLLM |
| Attention | FLASHINFER was explicitly selected in the final old clean rerun |
| Warmup | old client schedules 3 x 100 warmup requests before 100 formal arrivals; no drain barrier between warmup and formal scheduling |
| Communication | UNRESOLVED: requirements/manifest say htsim; actual runner says analytical |
| Hardware | H200 step_main required; effective platform selector and access remain unverified |
| Acceptance | canonical request_arrival_to_prefill_completion mean TTFT, absolute and relative error, <=10% gate |

## Image and runtime

Authoritative environment recipe: /data/ycfeng/stepfun-env-handbook/vllm-bs-0.10.2-frontier-env.md.

- Historical image: hub.i.basemind.com/vllm-0.10.2/frontier-env:latest
- Historical verified digest: sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc
- Python: /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python, version 3.10.16
- Torch: 2.8.0+cu128; FlashInfer: 0.3.0; toolkit: /usr/local/cuda-12.8
- Compiled package: /local/ycfeng/anaconda3/envs/vidur_te/lib/python3.10/site-packages/vllm
- VLLM_FRONTIER_COMPILED_PACKAGE selects that package while PYTHONPATH selects vLLM-BS source.
- Current source inspection: /data/ycfeng/tmp/vLLM-BS is clean, branch feature/frontier-comparison-instrumentation, commit c169f48fa0ce16455f64101bc4366b23b1652a43, parent ea95f571e20937c7c908c6d59ddd1cd6bf9268f1. The overlay restores the four-argument MoE topk_softmax ABI. Remote tip has not been refreshed in this task.
- Old dummy model config directory: /data/ycfeng/tmp/issue26-qwen3-dummy. Contents need explicit new-run verification.
- Old server recipe enables VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1; selects CUDA 12.8; preloads the worker libcuda; bypasses proxy for localhost health checks; mounts /data:/data; disables sshd.
- All runtime versions and mounted library compatibility above are historical observations, not an H200 preflight PASS.

## Issues to resolve before numeric comparison

1. Communication config conflict: old requirements/manifest specify htsim; tests/e2e/issue26_frontier_matrix.sh selects analytical. Pending user question recommends collective_sim/htsim. The new worktree submodule is uninitialized.
2. GPU device omission: old runner never sets replica_config_device; current ReplicaConfig.device defaults to a100. Set H200 explicitly and verify effective capacity; data/config/device/h200.json already exists.
3. Arrival mismatch: old Frontier uses seed 42; old vLLM client uses random.Random(20260904) and consumes warmup arrivals first. Matching QPS is not identical dispatch. Use fresh workload/arrival evidence and an existing trace generator where applicable.
4. TTFT semantics: vLLM v1/engine/output_processor.py::_build_frontier_metrics exports first_token_latency as ttft. AGENTS.md canonical TTFT is queue-visible arrival -> prefill completion. Inspect a minimal clean producer capable of exporting those endpoints before claiming equivalence; an instrumented run cannot silently replace clean timing.
5. Warmup state: the old client creates warmup and formal tasks before one asyncio.gather; pending warmup requests can overlap the formal window. Determine the explicit warm-start comparison procedure from new-run semantics.
6. Missing profile coverage: old simulation terminated on absent attn_decode_in_mixed. Regenerate all active attention phases and MoE/linear inputs under current main on H200. Do not merge old rows into new inputs.
7. Routing: old Frontier uses balanced routing, whereas dummy-weight vLLM executes its actual router. Record observed routing identity/load evidence and analyze any distortion before accepting numeric parity.
8. Platform: inherited-proxy brainctl reads return EOF; removing proxy for the read reaches an API Forbidden response for quotagroup reads in default and shai-core. This proves neither quota existence nor job-use permission. No GPU allocation has been attempted. Local nvidia-smi -L produces no devices. No step_main alias or access recipe was found in inspected handbook and shell/SSH configuration.

## Pending decision

D001: Which communication backend defines the new formal case? Resolved: YC explicitly selected collective_sim/htsim and authorized initialization, build, and topology checks. Recommendation: collective_sim/htsim, preserving original intent. Alternative: explicitly authorize analytical, preserving the old runner's actual backend. Backend identity is frozen; H200 intra-server parameters remain pending physical evidence.

## H200 topology reference and collection boundary

On 2026-09-07, NVIDIA's current H200 specification page was read directly: https://www.nvidia.com/en-us/data-center/h200/ . It lists 141 GB GPU memory and 900 GB/s NVLink interconnect, distinguishes H200 SXM HGX systems with four/eight GPUs from H200 NVL systems with two/four-way NVLink bridges, and labels advertised Tensor Core throughput as using sparsity. A platform label of H200 with eight allocated GPUs does not by itself prove an eight-GPU NVSwitch domain. The physical worker topo -m and nvlink -s output remains required before freezing the intra-server topology. This specification reference is not a timing dataset and does not replace fresh H200 measurements.

The step_main access question is resolved operationally: proxy-free rlaunch with --charged-group=step_main selected H200 nodes and accepted the probe job. The earlier quota-GET Forbidden did not imply lack of job-use permission.
