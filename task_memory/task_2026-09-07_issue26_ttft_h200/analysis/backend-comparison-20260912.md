## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-12 | Added the H200 warmup-10 alternative all2all comparison and bounded semantic conclusions. |

# H200 all2all backend comparison

## Comparison table

| Backend | Capability / gate | DP0 median (ms) | P90 (ms) | Rank max (ms) | Rank spread (ms) | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| native/naive | PASS; 10 warmups, 1100 rows, identity PASS | 83.551777 | 83.591407 | 83.607101 | 0.060890 | Current reduced batch-only reference |
| DeepEP high-throughput | PASS; 10 warmups, 1100 rows, identity PASS | 86.486446 | 86.506606 | 86.509506 | 0.302368 | Prefill-oriented fused path; +2.934669 ms / +3.51% vs native |
| DeepEP low-latency | PASS; 10 warmups, 1100 rows, identity PASS | 719.293091 | 719.308789 | 719.308899 | 0.111511 | Decode-oriented path under prefill; 8.61x native |
| PPLX | Capability unresolved; no formal timing | unavailable | unavailable | unavailable | unavailable | Build/link failure; do not call runtime unsupported |

## What can be concluded

The native/naive and DeepEP HT runs are both normal-scale H200 reduced
batch-only measurements. DeepEP HT changes the communication implementation and
the fused reduction semantics, but does not reduce the measured span in this
case. It is therefore a useful semantic control, not a validated replacement
for Frontier's ideal communication value.

DeepEP LL passed its independent eight-rank capability probe and its full
formal replay, so the 719.293091 ms result is valid evidence for that backend.
It is not a failed run and it is not a hidden 20 ms collective missing from the
native reference. Source inspection shows why it is a different workload
mapping: `VLLM_MOE_DP_CHUNK_SIZE=256`, 4096-token prefill chunking, synchronous
low-latency dispatch/combine calls, and a decode-oriented backend contract.

The first formal identity is the same across the table:

```text
cmpl-pf4096_dc1024:0-0
batch_size=1
request_num_tokens=[4096]
batch_num_prefill_tokens=4096
batch_num_decode_tokens=0
DP0 TP0--TP3
batch_dp_token_counts=[4096,1]
```

The source and raw artifacts are documented in
`test_report_2026-09-12_h200_all2all_deepep_ll.md`. All measurements are H200
`step_main + h200 + num_gpu_blocks_override=310809`; no H800 value enters this
comparison.

## What remains open

The alternative backend experiment does not identify the complete Frontier
clean gap. In particular, the integrated Frontier result remains
59.190354384 ms versus the separate clean vLLM references
78.118782043--79.307357788 ms, and the compute overestimate changes the implied
non-compute remainder to approximately 28.668--29.857 ms. The experiment does
not separate queue time, host scheduling, collective wait, and pure kernel time
inside the reduced vLLM batch event.

Consequently:

- native/naive remains the canonical vLLM reference for this frozen prefill;
- DeepEP HT is a semantic alternative with a measured +3.51% span;
- DeepEP LL is a capability-valid but prefill-inappropriate semantic variant;
- PPLX needs a separately repaired build/capability path before timing;
- no backend result authorizes a Frontier production correction or
  clean/diagnostic span reconciliation;
- CUDA attribution and op-gap gates remain open.

