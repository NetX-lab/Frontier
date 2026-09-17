## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded source-stable broader profiling CPU regression and pre-existing failure classification. |

# P3 integrated regression

Source fixed at `4dd63482`; worktree clean throughout. Python `/data/ycfeng/tmp/quality-review-env/bin/python`, version 3.12.3, uv venv, no conda or native GPU execution.
Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

```bash
mapfile -t frontier_p3_tests < <(rg --files tests/unit | rg '/test_([^/]*profil[^/]*|gdn[^/]*|sglang[^/]*|collectives[^/]*|moe_mxfp4[^/]*|mla_model_config_contracts|model_architecture_registry|device_event[^/]*)\.py$' | sort)
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest -q -p no:cacheprovider "${frontier_p3_tests[@]}" --basetemp=/data/ycfeng/tmp/quality-p3-integrated
```

Criteria: preserve configuration, schema, precision, timer, native-adapter admission, artifact and experimental replay contracts; no new candidate-only failure or skip. Actual selected file inventory: `/data/ycfeng/tmp/quality-p3-integrated-tests.txt` (45 modules).

Observed: **721 PASS, 5 SKIP, 2 FAIL**, 31.02 s; `/data/ycfeng/tmp/quality-p3-integrated.log`.

Both failures are the exact previously observed frozen-candidate/pinned-main nodes:

- `tests/unit/test_mha_stage2_profile_modeling.py::test_stage2_cli_writes_reproducible_artifacts`
- `tests/unit/test_mqa_stage2_profile_modeling.py::test_mqa_stage2_cli_writes_reproducible_artifacts`

Current direct reproduction:

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/tmp/quality-review-env/bin/python tests/analysis/mha_phi3/build_mha_stage2_profile_model.py --cuda-op-log /data/ycfeng/tmp/quality-p3-integrated/test_stage2_cli_writes_reprodu0/cuda_ops.jsonl --output-dir /data/ycfeng/tmp/quality-p3-mha-rca
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/tmp/quality-review-env/bin/python tests/analysis/mqa_falcon/build_mqa_stage2_profile_model.py --cuda-op-log /data/ycfeng/tmp/quality-p3-integrated/test_mqa_stage2_cli_writes_rep0/cuda_ops.jsonl --output-dir /data/ycfeng/tmp/quality-p3-mqa-rca
```

Both exit with `flashinfer-python is not installed in the active Python env`. These scripts have no diff against pinned main. Reference failure evidence: `/data/ycfeng/tmp/quality-pre-unit-path.log`, `/data/ycfeng/tmp/quality-main-failures.log`. The initial guessed MHA directory was wrong; the actual Phi-3 script was then read from its test and used above. No production workaround or changed expected output.

Conclusion: no new failure observed in this phase. The suite is not wholly green, and CPU stand-ins/skips do not prove native ROCm/CUDA kernel behavior. Focused exact before/after comparisons are recorded in the corresponding P3 sub-step reports.
