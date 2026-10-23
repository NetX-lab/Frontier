# Prepared point reuse interface receipt

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record immutable implementation contract,lifecycle,error behavior and review inputs. |

Task ID **D29-PREPARED-POINT-REUSE**; code `0be11762d0f83e9db77658a1447a672507166872`. Source SHA256 `d9423f3f318cd355f2773ea6a65a9bf9d297cdc3ec7c8c46230b38456ccacbfc`; submodule3dac unchanged. Only `frontier/execution_time_predictor/moe_routing_workload.py` has production changes.

The external interface remains `_step5_position_queries(lane) -> (native_context,operator_queries,collective_plans)`. Callers must still establish current measured layer,active family,shared manager,current routing source and existing artifact config. No new options,cache identity,registry or module. Superseded raw selection scan is removed in the same diff.

Private preparation stores:
- `_step5_physical_points`:actual estimator object -> attached row tuple,measurement-identity string,EP/count/full-source-frame index. Matching rows retain native-context JSON strings only; full current context is decoded fresh for each query. Failed row decoding does not publish partial entries.
- `_step5_collective_receipts`:exact current file text,decoded counts,phase/sizes/graph/tensor index,full shared/compute domains and frozen plans. Read actual current file text each query. Different text prepares anew;failure does not publish it. Same-text candidate additions use private map copies until success.

Model/family/group registry selection remains live. Tuple/string/estimator replacement rebuilds its own state. No global sharing,numerical prediction or whole-proof caching. Source owner/live/physical geometry,unique actual context,full plan shared/policy/compute/buffer/context/key equality remain per-query checks. Plan construction/key decoding is retained because inputs are fixed current-text structural facts; the original reconstructed key and decoded context still compare to current receipt. Numerical duration/kernel counts,source proofs and current `global_layer_id` are produced downstream every invocation.

Lifecycle support is inspected production construction,not invented invalidation: `moe_load_imbalance.py:79-84` returns tuple-of-dicts; shared trainer `prediction_family_trainers.py:1684` and independent trainer `moe_dataset_training.py:428` assign receipts after fit. `simulator.py:152-182` finishes manager construction before P/D predictors. Supported replacement is tested; nested in-place row mutation is not a serving path and not newly supported.

Acceptance inputs:exact three-path diff `prepared_point_repair.diff`,caps `budget.json`,source/test/script/result SHA `source_receipt.json`,focused and full affected logs/commands in dated report. Committed actual layer60 has208/208 direct matching pairs;layer61 remains165/208 captured subset. Matrix uses the existing fixture construction seam and candidate query vs original frozen method;diagnostic preparation disabled.

Original proposal errata are superseded in `progress.md`:runtime before amendment53,not70;include trace20 so allocated-envelope4240/4404,not4220/184. Actual step159/250;retained CORE4111 including trace20 reservation,conservative step-sum4149. Parent final union may substitute actual trace usage; no donor edits or implicit budget borrowing.
