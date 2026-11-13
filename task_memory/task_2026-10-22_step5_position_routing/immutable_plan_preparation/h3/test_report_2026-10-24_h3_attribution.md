# H3 exclusive warm-pricing attribution

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Attribute existing profiles; retain one partially qualified control pair. |

Task `D29-IMMUTABLE-PLAN-PREPARATION-H3`; session `ses_e963098ecffebqCOtdysKGY6Xe`. Frozen source `/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes` at `56529fcd` (code `2f2d9339`). Records-only ownership: this `h3/` directory.

## Existing profiles FIRST

Read `key-a/warm.prof` and `key-b/warm.prof` under `/data/ycfeng/tmp/d29_immutable_plan_preparation`. Exclusive self time only; builtin self time is partitioned by immediate caller. Categories never overlap; complete categories sum to each profiler total. These are **instrumented**, not unprofiled wall times.

| Disjoint work | key-a ms | key-b ms |
| --- | ---: | ---: |
| JSON codec | 48.386 | 46.851 |
| Snapshot copy | 33.785 | 32.962 |
| Model configuration/layer consistency | 30.277 | 30.009 |
| Operator family/typed-contract derivation | 27.705 | 27.450 |
| Registry admission, excluding numeric lookup | 22.578 | 22.197 |
| Current duration/count numeric lookup | 7.082 | 7.056 |
| Remaining disjoint categories | 71.067 | 70.062 |
| **Total** | **240.880** | **236.587** |

### Exact JSON callers: both profiles agree

Paths below are relative to `frontier/`.

| loads | Calls | Call site |
| --- | ---: | --- |
| Candidate native-key fields | 272 | `execution_time_predictor/prediction_model_registry.py:409–410` |
| Native binding normalization | 272 | same file `:684,687` (96 nested +176 round-trip) |
| Native selector string fields | 192 | same file `:671–674` |
| CPU plan/context/buffer/count comparison | 192 | `cc_backend/backends/collective_sim_cc_backend.py:818,824–825` |
| Actual plan/compute-context admission | 128 | `execution_time_predictor/moe_operator_times.py:99–100` |
| Fresh cost/count validation | 64 | `cc_backend/collective_plan.py:125` |
| Current actual-point context | 16 | `execution_time_predictor/moe_routing_workload.py:207` |
| **Total** | **1136** | |

**800 dumps:**624 through `profiling/common/runtime_identity.py:96`;96 selected semantic identities at `execution_time_predictor/prediction_model_identity.py:154`;80 current-context binding keys at `execution_time_predictor/step5_cost.py:75`. The624 split into plan-candidate shape/shared/context192 (`moe_routing_workload.py:173–185`), candidate/current-context comparison128 (`:232`), actual operand-context comparison128 (`moe_operator_times.py:100`), and native normalization176 (`prediction_model_registry.py:687`).

## Validation versus fixed derivation

Numeric lookup is not dominant:288 duration/count calls,144 fresh proofs; actual point/context/buffer/count comparisons remain live. Existing marker lookup remains240 calls; no identity-cache reopening.

Repeated fixed derivation:384 family-view contract serialization attempts (96 non-null);304 registry bindings,34544 profiling-name checks (`operators/binding.py:43–75`, `spec.py:128`, `prediction_model_identity.py:113`). Binding's profiling generator makes28880 name checks; typed-family matching5360; final binding304. Registry selection/replacement remains live; no new view cache proposed.

Distinct candidate:144 `_resolve_moe_layer_ids` calls via `_is_mixed_model` (96 layer-contract +48 dense-contract activation calls), scanning92 layers each. Default getters already cache parsed IDs; repeated **consistency scanning**, not parsing, is avoidable (`model_architectures.py:1038–1076`; `config/model_config.py:876–906`).

## Single control pair and limits

Only replace default getter/predicate scanning with their algebraically equivalent canonical IDs; retain all ID checks and original custom/replaced-method scanning. No new memo. Both arms use identical prior key reuse and ONE single-role diagnostic fixed-byte adapter—not P/D/run binding.

Twelve unprofiled warm waves/arm: **0.127015→0.119095s**, saving **7.921ms/6.236%**. Membership calls13264→16; getters13408→160. JSON1136/800, numeric288, point16, fresh proofs144 and all208 event/ledger/full-key values unchanged. Payload SHA256 `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`.

Seven config-only differential guards pass, including changed IDs and custom predicate mismatch with184 real predicate calls. **Control remains UNQUALIFIED:** appended real estimator/row/identity/fresh-layer harnesses exceeded60s and produced no completion receipts. Pricing subphases completed; timeout is not correctness RED. Stop further runs; no allocation recommended now.

Exact protected commands, hashes, profiles, counts and exclusions: `evidence/evidence.json`; runner/log snapshots accompany it. Python3.10.6, BLAS threads1, timeout60s/MemoryMax30G. Cold/setup/profiling/equality/guard work excluded from timing. CPU-fixture reconstructed manager only; no native/fullCLI/GPU/input/source/test/trace-encoding work. Rebase and verify after snapshot delivery. Exclusive attribution/count/equality PASS; full live-replacement acceptance unfinished.
