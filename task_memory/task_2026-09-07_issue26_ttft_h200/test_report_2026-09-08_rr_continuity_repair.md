## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified the approved RR stream-continuity correction. |

# RR continuity repair

Authorization: D016, repairs/d016_human_review.json. Change marker: D016_RR_CONTINUITY.

## Execution

Working directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907.
Conda environment dev-vidur-v03-hopper-e2e, Python3.13.13.

```bash
export TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_cluster_scheduler_dp_lanes.py -q -p no:cacheprovider -k round_robin_preserves --tb=short
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_cluster_scheduler_dp_lanes.py -q -p no:cacheprovider --tb=short
git diff --check
```

## Criteria

The same logical arrival sequence must retain its RR Replica/DP owners across singleton, irregular chunks, empty calls and a single burst, including nonzero prior progress. Preserve the existing grouped return order and cumulative counter. Existing LOR/random/sticky/DP identity checks must still pass.

## Evidence

- Before repair:10 failed,2 passed,8 deselected. Representative failure: singleton owner at index1 was (7,0), expected (7,1). Raw red output: /data/ycfeng/tmp/issue26-rr-red.log.
- After repair:20 passed in2.12s. All12 new cases and8 existing cases passed.
- Correction uses the existing cumulative request ordinal for both Replica and DP, carrying the selected DP through Replica grouping. No config or new state was added.

This is functional scheduler validation, not a TTFT calibration result. No fresh latency or relative-error claim is made. Fresh current-case validation follows the separately approved vLLM snapshot-policy substep; the user's single-case scope supersedes a generic broad scenario-matrix recommendation.
