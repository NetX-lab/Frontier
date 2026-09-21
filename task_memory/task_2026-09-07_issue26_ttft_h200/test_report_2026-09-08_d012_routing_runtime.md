## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded approved D012 implementation and focused verification. |

# D012 routing runtime verification

Execution directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907.
Conda environment: dev-vidur-v03-hopper-e2e; Python /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python, version 3.13.13; sklearn 1.9.0.

## Criteria and evidence

PASS: 54 config/resolver/predictor/CLI checks in 5.20s plus 20 shared-registry/cache checks in 2.40s (74 relevant checks). Existing distribution defaults and allocation remain unchanged; explicit uniform runtime survives CLI construction and replica copying. Distinct runtimes remain distinct in deduplication, registration, projection, and disk-cache loading; equal resolved runtimes still share. Both routing contexts and measurement/precision identities are covered. Fresh selected dataset admits 387 rows per context: analysis/d012_profile_selection.json. These are correctness checks, not numerical TTFT parity.

Exact parent command:
```bash
TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_predictor_layer_id_semantics.py tests/unit/test_pdaf_deferred_trace_contract.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/issue26-d012-config-predictor-02
```

Exact shared-manager command and synthetic-test evidence are recorded in analysis/cpu-master-runtime-01/d012-implementation.md. Synthetic target values 10/110ms are sentinels only.

## Recovered verification failures

The initial parent combined run had 51 passed/3 failed: an in-process CLI test initialized IS_MOE=True and affected following dense fixtures (`RuntimeError: IS_MOE already initialized to True, cannot change to False`). Moved the actual CLI check into a subprocess; the production guard was preserved. Final 54 checks passed with 115 argparse deprecation warnings.

The shared-manager initial fixture lacked required prefill_hot metadata and was corrected using the canonical helper. A broad test name selector also included an unrelated GPU wrapper that requires absent torch; the final explicit selector covers the modified manager and existing registry/cache cases. No production fallback or environment change was introduced. Both initial failures and exact results are retained in the owned implementation report.

## Limit

The full CPU numerical run and official-server TTFT comparison remain pending. This verification does not establish cross-runtime batch population or missing host-work equivalence.
