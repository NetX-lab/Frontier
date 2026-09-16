## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded real profiler-boundary and replay source fixes. |

# W07 event and replay sub-step

Environment: /usr/bin/python 3.12.3, no conda; pytest 9.1.1. This is CPU boundary/orchestration evidence only.

```bash
PYTHONPATH=$PWD python -m pytest tests/unit/test_moe_fused_event_contract.py tests/unit/test_moe_mxfp4_increment10.py tests/unit/test_sglang_graph_replay_orchestration.py tests/unit/test_sglang_routed_sorting.py tests/unit/test_sglang_experimental_increment13.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w07-combined > /data/ycfeng/tmp/pr33-w07-combined.log 2>&1
```

Observed combined result: 48 PASS in 3.28 seconds. Criteria: real fused MoE entrypoint accepts DEVICE_EVENT with correct platform guards, validates iterations before allocation, produces finite shared population statistics; routed replay checks actual outputs and exposes the common representative trace callback; sorting conserves expert loads across multiple blocks.

RED evidence preserved in w07_moe_review.md and w07_sglang_review.md: fused event 15 FAIL/1 PASS; sorting 2 FAIL/3 PASS at loads (33,31) and (64,33,0); routed orchestration 4 FAIL/8 PASS. Singleton std is now 0. Samples [1,3] produce mean=2, median=2, std=1. Integer metadata [97,64] against [96,64] fails exactly; floating native tensor tolerances remain unchanged.

Root inspected all four production diffs. The experimental profile_routed_graph return contract is now (row, trace_replay), matching the existing common profile_graph function; no valid repository caller required migration. External research callers must unpack this tuple. This removes duplicate capture/replay/reset logic and unsupported correctness claims. Native AMD/MI355X/AITER/HIP graph, native quantized layouts and vLLM-version timing parity remain NOT RUN/SKIP. Remaining W07 hunk findings continue separately.
