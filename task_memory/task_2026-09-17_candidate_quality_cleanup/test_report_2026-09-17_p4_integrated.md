## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded the complete P4 regression, test-isolation failure and verified correction. |

# P4 integrated verification

Production source: `5cb8794f`; final fixture correction: `ff2b6cdf`. No further production edits. Environment: Python 3.12.3, `/data/ycfeng/tmp/quality-review-env/bin/python`, uv venv with same-interpreter system packages, no conda activation. CPU execution only. Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

Criteria: model/config construction, measurement families/path precedence, supplied-model identity, predictor cache atomicity, saved/loaded artifacts, training, runtime GDN guards and synthetic constructor behavior retain their established assertions. No new failure/skip or weakened production guard.

```bash
mapfile -t frontier_p4_tests < <(rg --files tests/unit | rg '/test_([^/]*train[^/]*|[^/]*model_manager[^/]*|measurement[^/]*|config_owned_contracts|gdn[^/]*|pdaf_config_contract|pd_transfer_types_and_configs|mla_model_config_contracts|quantization_manager_registry|on_demand_prediction_contract|predictor_cache_atomicity|device_timer_contract|model_architecture_registry|typed_ep_predictor_contract)\.py$' | sort)
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest -q -p no:cacheprovider "${frontier_p4_tests[@]}" --basetemp=/data/ycfeng/tmp/quality-p4-integrated-final
```

Selected inventory: 28 test modules, `/data/ycfeng/tmp/quality-p4-integrated-tests.txt`. Final observed result: **545 PASS**, 56.12 s, no failures or skips; `/data/ycfeng/tmp/quality-p4-integrated-final.log`.

Initial same selection used `--basetemp=/data/ycfeng/tmp/quality-p4-integrated`: **542 PASS / 3 FAIL**, 51.00 s. All three GDN SimulationConfig cases followed new dense config tests that left process-global IS_MOE=False. Minimal config-then-GDN reproduction produced 1 PASS / 1 FAIL. Reused the existing test-only reset function in the new config test module, keeping the production inconsistency guard intact. Both complete modules then passed 36 tests, and the complete P4 rerun above closed the original failure. Full reproduction details and original logs are retained in `test_report_2026-09-17_p4_config_contracts.md`.

The 545-test result is CPU contract/regression evidence. Exact artifact/path/default-output comparisons are in the P4 substep reports; it does not establish native-device profiling or production-data fidelity. Final full-unit, non-dummy artifact, fidelity and paired wall-clock evidence is P5.
