## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified manager family selection and explicit replica input. |

# P4 — Manager contracts

## M1: canonical family selection

Invariant: supported dense/MLA/hybrid classification and architecture/device/graph-dependent measurement families remain identical. Dataset-only `model_config=None` still returns non-MLA. Production always supplies replica_config to measurement selection; four test-only omissions now supply their configured replica.

Environment: `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, uv venv, no conda. Baseline working directory `../quality-baseline-c288a19f`; after working directory `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_measurement_family_selector.py tests/unit/test_measurement_path_precedence.py tests/unit/test_device_timer_contract.py tests/unit/test_shared_prediction_model_manager_eager_attention_mla.py tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py tests/unit/test_hybrid_runtime_family.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py tests/unit/test_profiling_governance_minimal_red.py
```

Observed: **127 PASS before**, 17.42 s; **127 PASS after**, 17.61 s. Logs `/data/ycfeng/tmp/quality-p4-manager-before.log`, `/data/ycfeng/tmp/quality-p4-manager-binding.log`.

Additional exact comparison extracts the two methods from `git show c288a19f:frontier/execution_time_predictor/shared_prediction_model_manager.py` and runs them against current methods on the same inputs: three architectures x graph on/off x none/piecewise decode graph x a100/mi355x x all six ClusterType values, plus None/dense/hybrid/structural-MLA models. The initial probe stopped at the expected unsupported TRANS ValueError and did not establish its prematurely recorded PASS claim. The corrected probe compares both values and ValueError class/text: **PASS: 120 exact measurement selections, 24 exact TRANS rejections and four family classifications**; `/data/ycfeng/tmp/quality-p4-manager-binding-exact-v2.log`. Original failed probe remains at `/data/ycfeng/tmp/quality-p4-manager-binding-exact.log`. No production/expected-output change was made for that harness error.

These checks preserve family choice, not native kernel performance. Full unit and final non-dummy/fidelity gates follow all P4 edits.

## M2: constructor-owned registries

Removed None/lazy dictionary reconstruction from the four family registry selectors. Constructor initializes all twelve selected dictionaries before dummy/normal branching. The remaining getattr has a validated declarative attribute name and no default; it is selection, not missing-state recovery. Family names, typed/legacy identity domains and precision dictionaries are unchanged.

Updated only three affected fixture owners to construct a real manager with empty clusters and a cache-dir-bearing metrics double; fake models are inserted after construction. Tests no longer duplicate a subset of manager initialization or tolerate missing registry fields.

Ran the M1 command above plus `tests/unit/test_on_demand_prediction_contract.py tests/unit/test_typed_ep_predictor_contract.py`: **181 PASS**, 19.77 s; `/data/ycfeng/tmp/quality-p4-manager-registries.log`. Existing 127-test M1 run is the pre-change control. Assertions preserve cache loading/training avoidance, typed identities and measurement/precision lookup outputs.

Direct frozen-method/current-method comparison on a normally constructed manager verifies all four selectors x three families return the identical dictionary object both before and after mutation; manager attribute inventory is unchanged. **12 exact identity checks PASS**, no dynamic attributes; `/data/ycfeng/tmp/quality-p4-manager-registry-exact.log`. No new cache, dictionary schema or public model-view behavior.
