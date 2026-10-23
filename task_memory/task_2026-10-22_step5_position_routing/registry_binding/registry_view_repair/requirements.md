# D29 registry mixed-view repair requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record root review finding, two-path authority and revised zero-sum allocation. |

TaskID: D29-REGISTRY-VIEW-REPAIR.

## [Original Request]

Root review ses_e9abd8364ffee5sQKdpiQh3yH4 found P2: native hash enumeration
then bare-name resolution over the unrestricted family causes same-name
legacy/non-native plus native coexistence to raise in family/cluster views.
Retain native hash labels and the original non-native bare projection with
existing resolution. Single-model ambiguity remains strict. No fallback,
parallel registry, independent-chain change, caller edit or peer operation.

Exclusive source/test paths: prediction_model_registry.py and
tests/unit/test_prediction_measurement_bindings.py. Remaining incremental gross58;
whole repair <=360 vsd96b3279. Revised caps vs1aee0af: registry120, test200,
untouched predictor118(current113), workload145(current136); caller delta remains35.
Read root core_review_scope_amendment.json.registry_view_review_repair;
CORE4404/109paths/global10000 are unchanged. No new paths or budget borrowing.

Required: real family/cluster coexistence regression in both insertion orders,
RED/GREEN, public ambiguity, legacy/typed/precision defaults and native bundle
retention. Archive local records via gitadd-f and deliver a separate fix commit
for root re-review/integration. Preserve root-owned submodule gitlink.
