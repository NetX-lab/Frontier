## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified the documented native attention backend selection. |

# R08 — CPU contract PASS

Environment: working directory `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`; `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, uv environment, no conda activation.

```bash
PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_rocm_documented_commands.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r08-green
```

Criteria: execute the literal MI355X attention recipe with `--dry-run` while ATTENTION_BACKEND=NO_OP; emitted producer argv must select VLLM_ROCM. Document prefill-only scope and separate decode collection.

Evidence: combined R08/R09 red campaign produced 4 FAIL, 1.35 s; the R08 assertion observed NO_OP. Corrected R08: **1 PASS**, 1.31 s. Logs: `/data/ycfeng/tmp/pr33-r08-r09-red.log`, `/data/ycfeng/tmp/pr33-r08-green.log`.

Native attention causal-reference and timing acceptance is defined by the R02 native cases but has not run on hardware. Dry-run success does not establish kernel execution.
