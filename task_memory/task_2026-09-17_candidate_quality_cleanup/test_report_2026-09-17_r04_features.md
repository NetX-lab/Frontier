## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified ragged producer-to-CSV-to-trainer feature identity. |

# R04 — PASS

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`. Python `/data/ycfeng/tmp/quality-review-env/bin/python` 3.12.3, uv environment, no conda activation. Prefix: `PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```bash
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_profile_feature_roundtrip.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r04-red
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_profile_feature_roundtrip.py tests/unit/test_gdn_artifact_boundary.py tests/unit/test_gdn_training_predictor_increment8.py tests/unit/test_gdn_profile_samples.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r04-green
```

Criteria: actual native producer row construction (native execution/timing boundaries doubled), production pandas normalization and CSV serialization, real GDNTrainer fitting and fresh loading must preserve the equivalent real Batch feature key. Cases: (2,4), (1,7,16), uniform (4,4), plus decode rows. Missing dispersion with explicit lengths must be derived; conflicting dispersion and unresolvable ragged rows must fail.

Evidence: before **3 FAIL**, 6.43 s, `KeyError: query_len_cv`. After **98 PASS**, 16.96 s. (2,4) dispersion equals 1/3; imported and real-batch exact keys match; each key exists in the trained exact lookup and fresh prediction returns the synthetic 0.22 ms core target. The producer now emits dispersion and stateful count. Runtime and imported-row dispersion share one calculation; mixed training rows remain rejected.

Logs: `/data/ycfeng/tmp/pr33-r04-red.log`, `/data/ycfeng/tmp/pr33-r04-green.log`. Native sample correctness is not established by these CPU doubles. Comparison to synthetic expected values is exact (absolute/relative error zero); no measured hardware latency comparison occurred.
