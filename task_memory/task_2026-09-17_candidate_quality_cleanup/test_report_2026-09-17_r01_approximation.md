## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified explicitly authorized mixed-GDN prefill approximation and unchanged scheduler behavior. |

# R01-A temporary approximation — PASS

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
Python: `/data/ycfeng/tmp/quality-review-env/bin/python`, 3.12.3, existing uv environment, no conda activation. CPU synthetic profiles; no native GPU execution. Production during all checks was reviewed HEAD `d43ae93240444bd4eff9bd99f296d2e751370514` plus the two-file R01 change in `frontier/attention/gdn/features.py` and `frontier/model_architectures.py`. No other production edits overlapped these checks. Later R02–R04 failing regression preparation was not imported by these executions.

Use this prefix for commands below:

```bash
export PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH
export PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl
```

```bash
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_training_predictor_increment8.py -k mixed -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r01-approx-red
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_gdn_training_predictor_increment8.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py tests/unit/test_gdn_profiler_cpu_increment7.py tests/integration/test_gdn_phase_admission.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r01-green
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/integration/test_gdn_phase_admission.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r01-green-v2
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-r01-nondummy
/data/ycfeng/tmp/quality-review-env/bin/python tests/integration/run_scheduler_refactor_fidelity.py --baseline ../quality-baseline-c288a19f --candidate . --output /data/ycfeng/tmp/pr33-r01-fidelity --workers 2
```

The integration scripts' absolute paths are the working directory above plus their listed relative paths. Raw stdout/stderr logs are `/data/ycfeng/tmp/pr33-r01-{approx-red,green,green-v2,nondummy,fidelity}.log`.

## Criteria and evidence

- New pre-fix warning/estimator regression: **2 FAIL**, 7.77 s, original mixed-batch ValueError instead of warning.
- Initial focused correction: **27 PASS / 1 FAIL**, 30.57 s. GDN approximation passes; real mixed dispatch exposed missing `attn_decode_in_mixed` data in the inherited synthetic fixture. Added valid mixed full-attention rows to this new regression only; no production fallback or suppressed guard.
- Corrected integration: **1 PASS**, 11.22 s. Four completed requests; 14 dispatched batches; two mixed batches; one batch includes a one-token prefill chunk. The second dispatched batch remains request B's 31 prefill tokens plus request A's one decode token. Stable slot object identities, released-slot reuse, empty final allocation map, zero allocated KV blocks and no active slots all pass.
- Warning text is observed in the real Simulator output: `GDN mixed batch uses a temporary prefill approximation ... co-location mode; native mixed-batch timing is unvalidated.` Repeated calls may emit repeated warnings as library warning filters change; no once-per-run guarantee is made.
- Both `(decode=1, prefill=1)` and `(decode=1, prefill=4)` select prefill estimators, preserve all physical feature values and original batch phase counts, predict `gdn_core_prefill=0.22 ms`, `gdn_core_decode=0`, and total GDN operator time `0.66 ms`. These equal the independent constant synthetic targets (absolute error 0, relative error 0); they are not hardware latency comparisons.
- Existing reporting-enabled non-dummy acceptance: **8 PASS**, 79.30 s.
- Existing fidelity matrix: **58 PASS / 0 FAIL** against frozen pre-cleanup candidate `c288a19f59bec09529ee18d782fa57218da2c781`. Every artifact comparison uses the existing comparator and tolerance. This is preservation of the prior accepted candidate behavior, not exact parity against main or hardware. The matrix uses dummy timing; the new integration independently exercises trained GDN prediction.

Selected batch observations and all 58 case verdicts: `r01_approximation_evidence.json`.

## Limitations

Mixed GDN timing is explicitly approximate and may lie outside prefill profiling coverage. Native producer/training mixed rows remain rejected. Full-attention layers still require their existing mixed profiles. GDN core timing appears under prefill in operator reporting. No scheduler or resource ownership policy changed. R02–R10 remain separate review work.
