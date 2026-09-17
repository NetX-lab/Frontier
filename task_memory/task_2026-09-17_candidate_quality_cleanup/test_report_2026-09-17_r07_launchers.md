## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified all three shell launchers against canonical output paths. |

# R07 — PASS

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`; Python `/data/ycfeng/tmp/quality-review-env/bin/python` 3.12.3, uv environment, no conda activation. Prefix: `PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```bash
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_profiling_launcher_output.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r07-red
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_profiling_launcher_output.py tests/unit/test_examples_profiling_contracts.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r07-green
```

Executed scripts: `examples/profiling/profile_linear_op.sh`, `examples/profiling/profile_attention_chunked_prefill.sh`, `examples/profiling/profile_moe.sh` under the working directory above. Tests supply an executable stand-in that writes only the selected producer output, then execute the real shell postflight.

Criteria: cuda_event/cuda, device_event, kernel_only/record_function resolve the canonical filenames for every launcher; a stale unsuffixed file cannot satisfy DEVICE_EVENT postflight. Resolve/display/check one path using build_profile_method_output_path; no shell suffix table.

Evidence: **12 FAIL / 6 PASS** before, 1.65 s. After **33 PASS**, 11.91 s, including shell validation and argument checks. ROCm output taxonomy corrected to suffixed files; dedicated gdn.csv remains unchanged. No real native producer was run.

Logs: `/data/ycfeng/tmp/pr33-r07-red.log`, `/data/ycfeng/tmp/pr33-r07-green.log`.
