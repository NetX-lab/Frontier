## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded completed fresh CPU uniform replay and request-level comparison. |

# Uniform case error analysis checkpoint

analysis_state: INCOMPLETE
status: INSUFFICIENT_EVIDENCE_FOR_FINAL_RCA
correction_state: pending
next_action: YC decision on D013 scoped activation profiling correction; retain current complete pair as the new baseline.

D012 routing runtime/model identity correction is complete:9e3d1874,74focused checks, successful fresh CPU replay and100formal request joins. The current paired mean is103.510401225ms Frontier versus127.380511761ms official vLLM; absolute error23.870110536ms, relative18.739217017%. This is outside the raw10% threshold and does not close the D006 official-TTFT gate. No historical predictions or previous-version numerical profiles enter this pair.

## Evidence and interpretation

- Clean request evidence: uniform-frontier-01/summary.json and request_comparison.csv, joined to uniform-clean-01/request_id_map.csv. Full reproduction and run validation: ../test_report_2026-09-08_uniform_frontier.md.
- Routing runtime and equal-population distribution are aligned: D012 config and uniform_routing_same_population.json; effective config27checksPASS. First actual Frontier prefill has128experts×256assignments each layer. Cross-side batch composition and dummy participation remain separate workflow questions; deferred trace import is not required to complete D012.
- Frontier first-prefill accounting closes: uniform-frontier-first-prefill.json/.md and uniform_frontier_first_prefill_endpoint.json. The65.790076904ms first request is a validated component endpoint, not the100-request mean. This establishes current accounting, not accuracy of each predicted operator.
- The source-confirmed activation omission remains: moe_profile_contract_review.md refreshed against current uniform rows. W1->slice->W2 profiling differs from vLLM W1->gatedSiLU->W2. Its numerical impact is unknown. Local moe_sum is separately absent and outside the grouped scope.
- Clean vLLM algebra: uniform_clean_engine_prefill_decomposition.json/.md. Remaining40.594977364ms contains queue/outside-engine work; engine86.785534397ms includes host/device activity. Neither component is an independently measured correction for Frontier.
- Same-case TPOT is57.583941406ms versus78.336091547ms (26.4912%rawerror); requestE2E59.011882459s versus80.264999897s. See uniform-frontier-01/secondary_metrics.json. The gap also exists after the first token, so a first-token endpoint adjustment alone cannot close execution accuracy. This does not isolate activation, communication, or host costs.

Current uniform runs do not yet supply aligned three-batch vLLM operator diagnostics. Historical standard-router operator times are not substituted. No complete per-op numeric RCA, residual source classification, or CPU analysis admission is claimed. Existing communication primitive/domain questions remain open.

## Proposed next correction

activation_profile_repair_proposal.md defines D013: restore declared activation inside the existing grouped scope, use identical H200 tensors/routing/kernel settings for controlled verification, regenerate affected fresh MoE profiles and replay this same CPU case. The separate moe_sum operator expansion is excluded. The proposal was presented through the requested grill-me surface; no answer has arrived and no D013 implementation has started. Its shared profiling-interface change is outside the confirmed D012 scope.
