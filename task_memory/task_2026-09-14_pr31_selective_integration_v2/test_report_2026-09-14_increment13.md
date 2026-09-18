## Modification History

| Date       | Summary of Changes               |
| ---------- | -------------------------------- |
| 2026-09-14 | Recorded Increment 13 CPU contract and source-boundary verification. |

# Increment 13 Test Report — Experimental SGLang Integration

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: `3.12.3`
- Command: `python -m pytest tests/unit/test_sglang_experimental_increment13.py tests/unit/test_moe_mxfp4_increment10.py tests/unit/test_collectives_increment11.py -q -p no:cacheprovider`
- Result: `18 passed`
- Command: `python -m compileall -q frontier/profiling/experimental/sglang tests/unit/test_sglang_experimental_increment13.py`
- Result: `PASS`
- Command: `git diff --check`
- Result: `PASS`

## Criteria

- Importing `frontier.profiling.experimental.sglang` remains lightweight and does not import SGLang, AITER, or CUDA runtime modules.
- Dense, packed GDN, decode-attention, MXFP4 routed-MoE shape contracts derive from `ModelConfig` and fail fast for invalid TP/layouts.
- Routed replay uses `frontier.moe_ep_workload.generate_moe_routing_ratios`, validates expert-domain/count/top-k conservation, and supports explicit JSON counts.
- Replay plans preserve launch-time `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` as provenance and validate rank mapping without restoring arbitrary environment data.
- Trace importer accepts `.trace.json` and `.trace.json.gz`, requires known SGLang/AITER anchors, rejects incomplete layer passes, and writes experimental summary artifacts.
- Experimental sources contain no discarded runtime-cost/capture-manifest imports and no standard `DEVICE_EVENT` producer.

## Evidence

- PASS: 7 Increment 13 tests cover package import, shape contracts, invalid TP/count failures, shared routing equivalence, explicit route JSON, visibility planning, gzip trace parsing, anchor/layer-count failure, and source-boundary scans.
- PASS: trace row contains `measurement_source=sglang_kineto_trace` and `evidence_kind=in_situ_kernel_sum`; summary files are `gdn-trace-summary.csv` and `gdn-trace-summary.json` under the caller-selected experimental output directory.
- PASS: `dense.py`, `gdn.py`, `attention.py`, `moe.py`, and graph/routed builders defer SGLang/AITER/CUDA imports until builder/replay invocation.
- PASS: routed assignment reconstruction preserves exact total `physical_size * top_k` and distinct expert IDs per token; infeasible histograms raise `ValueError`.
- PASS: combined regression with Increment 10 and Increment 11 tests reports `18 passed`.
- SKIP: AMD/MI355X hardware unavailable. No HIP graph capture, AITER fused kernel, SGLang runtime, RCCL, GPU timing, benchmark parity, or groundtruth parity was executed.

## Limits

The CPU tests establish schema, planning, import, deterministic routing, and trace control-flow behavior. They do not establish numerical correctness of native SGLang/AITER kernels or any MI355X performance value.
