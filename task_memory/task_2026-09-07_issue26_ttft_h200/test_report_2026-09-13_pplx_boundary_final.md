## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the repaired PPLX boundary replay and selected-DP lane analysis. |

# H200 PPLX boundary replay — final artifact

## Outcome

The PPLX launch path and boundary analyzer issue is resolved for the collected
run. The launch now supplies the PPLX Python/runtime overlays, and the analyzer
no longer rejects a valid run solely because vLLM's internal DP load balancer
placed the first formal request on DP1. The measured PPLX first-formal boundary
is:

| Metric | Value |
| --- | ---: |
| Selected lane | DP1 |
| TP0 | 541.981445313 ms |
| TP1 | 542.336853027 ms |
| TP2 | 542.339172363 ms |
| TP3 | 542.371643066 ms |
| Median | 542.338012695 ms |
| P90 | 542.361901855 ms |
| Rank max | 542.371643066 ms |
| Rank spread | 0.390197754 ms |

This is a boundary-only CUDA-event envelope for the fused PPLX dispatch/combine
protocol. It is not interchangeable with the native clean 78--79 ms span, and
it is not a Frontier correction input.

## Execution

- RJob: `yc26-h200-pplx-boundary-20260913-02`
- Cluster: H200, `step_main`, GPU tag `h200`, one 8-GPU worker
- Capacity: `num_gpu_blocks_override=310809`
- Source checkout: `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt`
- Source commit: `cf1ef5de9c0c45aedec4cf9d22c8eb56caf8bf0b`
- Worker: `tests/e2e/issue26_h200_pplx_boundary_worker.sh`
- Persistent output: `analysis/pplx-boundary-20260913-02/`
- Analyzer: `tests/e2e/issue26_pplx_boundary_analysis.py`
- Repaired analyzer commit: `ad0f6f0d22c5b9fdc97c5955cf49b128a1f578d5`

The replay used the exact frozen case: Qwen3-30B-A3B dummy weights, TP4/DP2/PP1/EP8, BF16, FLASHINFER, eager execution, uniform routing, prefix caching OFF, chunked prefill OFF, and 4096-prefill/1024-output. The PPLX runtime overlay was supplied through:

```text
ISSUE26_OPTIONAL_PYTHONPATH=/data/ycfeng/tmp/issue26-pplx-runtime-20260912-03
ISSUE26_EXTRA_LD_LIBRARY_PATH=/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nccl/lib:/data/ycfeng/tmp/issue26-h200-all2all-deps-preload-20260912-01/python/nvidia/nvshmem/lib
```

The replay executed ten complete drained 100-request warmups and 100 formal
requests, producing 1100 client rows. The first formal server identity was
`cmpl-pf4096_dc1024:0-0`.

## Analyzer repair

The old analyzer required `dp_rank == 0`. In this run vLLM assigned the first
formal request to DP1, leaving the four DP0 boundary files empty while DP1
contained one valid row for each TP rank. The repaired analyzer:

1. validates request identity and the one-request 4096-prefill/zero-decode predicates;
2. derives the selected DP lane from validated non-empty rows;
3. requires exactly one selected DP lane;
4. requires complete TP0--TP3 coverage.

The resulting artifact is
`analysis/pplx-boundary-repaired-20260913-02.json`; its Markdown explanation
is `analysis/pplx-boundary-repaired-20260913-02.md`.

## Gate evidence

- client rows: 1100;
- formal requests: 100;
- ten warmup phases: complete and drained;
- prompt/completion identity: 4096/1024;
- first formal batch: size 1, 4096 prefill tokens, 0 decode tokens;
- selected lane: one complete DP1 lane;
- TP coverage: TP0--TP3 complete;
- analyzer execution: `PASS`;
- Python compile check: `PASS`.

The DP1 boundary files contain many ordinary rows from later requests, but only
the validated first-formal row per TP rank contributes to the reported
statistics. DP0 files are empty for this request and are retained as evidence
of the lane-selection behavior.

## Limits and RCA status

The measured 542 ms envelope is approximately 6.8--6.9 times the accepted
native clean reference. The source log reports that the PPLX MoE tuning file
`E=16,N=768,device_name=NVIDIA_H200.json` is missing and a default MoE config
is used. Native/naive also emits the same warning while remaining near 83 ms,
so the warning alone cannot explain the PPLX slowdown. The current evidence
supports a PPLX fused dispatch/combine protocol or tuning/runtime path as the
primary diagnostic target, but it does not isolate the individual dispatch,
combine, expert-GEMM, first-use or JIT costs.

The CUDA-event envelope may include queued device work and excludes host
wall-clock gaps. No per-operator instrumentation was enabled, and no PPLX
timing value is fed into Frontier predictor, communication, operator accounting,
CPU overhead or clean/diagnostic reconciliation.

