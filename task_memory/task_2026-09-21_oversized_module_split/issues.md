# Oversized Module Split — Issues and Resolutions

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created with the issues found during the `config.py` split. |
| 2026-09-21 | Added I5 to I8, the four regressions the scheduler split introduced. |

## I1 — `ClusterConfig` is constructed at runtime by a method that moved out

| Field | Record |
| --- | --- |
| Found by | `tests/unit/test_pdaf_config_contract.py` and `tests/unit/test_simulator_transfer_predictor_lifecycle.py`, 6 failures |
| Symptom | `NameError: name 'ClusterConfig' is not defined` at `frontier/config/cluster_role_config.py:135` |
| Cause | `get_cluster_configs_for_disaggregation` builds one `ClusterConfig` per role. When that method moved into the `ClusterRoleConfigBuilder` mixin, the name was imported only under `TYPE_CHECKING`, because the survey had recorded it as an annotation. It is a real constructor call, and a module-level import would close a cycle: `cluster_config` imports the mixin. |
| Fix | A lazy import of `ClusterConfig` inside the method, matching the existing `_get_cc_backend_configs` pattern in the same package. |
| Prevention | Any name a moved method *constructs* or *calls* must be imported at runtime, not under `TYPE_CHECKING`. The static free-name analysis cannot distinguish the two; only executing the code does. |

This is the reason the split is gated by the unit suites in addition to the fidelity matrix: no fidelity case had reached this path before the unit run, because the disaggregated example wrappers exercise it but the failure surfaced first in the faster unit selection.

## I2 — Long-prefill co-location case exceeded the model context

| Field | Record |
| --- | --- |
| Found by | The first fidelity baseline capture |
| Symptom | `Sequential simulation ended with non-empty scheduler state`, exit code 1, at simulated time 7.296 ms |
| Cause | The case asked for 4096 prefill tokens plus 16 decode tokens on Llama-2-7b, whose context is 4096. No request can ever be admitted, so the queue never drains. |
| Resolution | Case parameters corrected to 3584 prefill tokens. This is a property of the configuration, not a simulator defect, so nothing in `frontier/` changed. |

## I3 — MoE fidelity cases violated the shared-domain invariant

| Field | Record |
| --- | --- |
| Found by | The first fidelity baseline capture |
| Symptom | `ValueError: Frontier shared attention/MoE parallel domain requires attn_tp*attn_dp == moe_tp*moe_ep`, and the wrapper's own stricter guard `ATTN_TP == MOE_TP * MOE_EP` |
| Cause | Two invalid case topologies. |
| Resolution | The EP1 case now uses `ATTN_TP=1`. The MoE attention-DP case was replaced by an EP4 topology: the MoE wrapper's guard ignores `attn_dp`, so it cannot express `attn_dp > 1`, and the dense matrix already covers DP lanes. |

## I4 — Predictor cache names differed for reasons unrelated to the refactor

| Field | Record |
| --- | --- |
| Found by | The cache-file-name check in the fidelity comparator |
| Symptom | 138 of 426 cache entries had different hashes between the two checkouts while every simulated output was identical |
| Cause | `ExecutionTimePredictionModelManager._get_hash_relevant_config` includes the profiling input file paths in the model hash, and the two `examples/profiling/smoke_simulator_*_csv.sh` wrappers default their data base to an absolute path under their own repository root. |
| Resolution | Both cases now pass `DATA_DIR_BASE=data/profiling`, which resolves identically from either checkout. After recapture the difference is zero, so the check can now detect a real training-identity change. |

## I5 — A moved method referenced a module-level logger object

| Field | Record |
| --- | --- |
| Found by | The fidelity matrix, 66 of 67 cases |
| Symptom | `NameError: name '_frontier_vllm_v1_sched_decision_logger' is not defined`, raised from `vllm_v1_iteration_policy.py` inside `_emit_schedule_decision_event` |
| Cause | The decision-log guard tests the module-level logger object directly, not only the logging function. The extraction imported the function into the new module but not the object, which stayed behind in `vllm_v1_decision_log.py`. |
| Fix | `schedule_decision_logging_enabled()` added to `vllm_v1_decision_log.py`; the guard calls it. No module outside `vllm_v1_decision_log.py` now names the logger object. |

## I6 — `deque` missing from the main scheduler module

| Field | Record |
| --- | --- |
| Found by | `tests/unit/test_simulator_transfer_predictor_lifecycle.py` |
| Symptom | `NameError: name 'deque' is not defined` in `__init__` |
| Cause | The rebuilt import header dropped a name the retained code still uses. |
| Fix | Import restored. |

## I7 — `validate_gdn_runtime_support` missing from the main scheduler module

| Field | Record |
| --- | --- |
| Found by | `tests/unit/test_gdn_scheduler_slots.py` and five other GDN suites, 16 failures |
| Symptom | `NameError` in `__init__` |
| Cause | The guard is used both by the retained constructor and by the extracted allocation methods. The extraction moved the import instead of duplicating it. |
| Fix | Import restored in both modules. |
| Prevention | A static check now parses each split module and reports every name that is loaded but neither imported, defined locally, nor a builtin. It found this one. Its only remaining hit is `BaseCCBackendConfig` in `frontier/config/cluster_config.py`, which is a string annotation the flat CLI generator resolves through its own lazy-import special case, exactly as the pre-split `config.py` did. |

## I8 — A test patched the decision logger on the module that no longer calls it

| Field | Record |
| --- | --- |
| Found by | `tests/unit/test_prefix_cache_identity_ledger.py` |
| Symptom | The test captured zero events where it expected two |
| Cause | The test monkeypatches `_log_frontier_vllm_v1_schedule_decision` on the main scheduler module. The prefix-cache methods now live in `vllm_v1_prefix_cache.py` and resolve the name in that module's namespace, so the patch no longer intercepts. |
| Resolution | The patch target moved with the methods. This is the one test change in the scheduler step; the test's subject, the ledger's emitted events, is unchanged. |
