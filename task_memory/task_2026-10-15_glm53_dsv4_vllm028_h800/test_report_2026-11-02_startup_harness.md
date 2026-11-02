# P2-64 startup source binding: frozen harness verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-02 | Verified the source allowlist for the approved native startup correction. |

## Scope and result

D-P2-64-STARTUP-RECOVERY authorizes native commit `a06a72a4326cba02234df6cfbdc21b78d3a15d26` for generation 02. The frozen harness is based on `4861e680`. The only executable change adds that exact source SHA to `vllm_replay.runtime_version`; the original source SHA and pinned image remain accepted. The serving workload, warmup and metrics are unchanged.

## Verification

Working directory: `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`.
Python: `/data/ycfeng/envs/openmopd-py312/bin/python`, 3.12.13; no conda activation.

```bash
PYTHONPATH=/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 120s /data/ycfeng/envs/openmopd-py312/bin/python -m pytest tests/unit/test_native_pair_warmup.py tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py tests/unit/test_native_groundtruth_check.py -q -p no:cacheprovider
```

Expected: every applicable check passes. Observed: **120 passed, 21 skipped in 52.92s**. Log: `/data/ycfeng/tmp/glm53_dsv4/h200/p2_dp_recovery_harness_tests.log`. Skipped native fixtures do not supply new GPU evidence. A direct call to `runtime_version` with the real package engine admitted each of the two exact source SHAs and rejected `f` repeated 40 times with `ValueError`: **SOURCE_ALLOWLIST_PASS**. AST parsing passed for the source and builder.

This is source-binding and CPU harness verification only. Generation 02 still requires a sealed package, pinned-image check and complete new native output before reference acceptance. The old frozen harness remains unchanged.
