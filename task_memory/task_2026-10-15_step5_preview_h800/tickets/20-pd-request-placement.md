# 20: PD request placement and load feedback

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-20 | Structural part done and merged into feat/step5-preview at 0c92903. |
| 2026-10-20 | Recorded the completed final dense-initialization noise and CPU/wall diagnosis. The unchanged budget remains unverified because CPU confidence intervals do not consistently exclude it; no further measurement loop or product-code change is authorized by this completed round. |
| 2026-10-20 | Released the structural part (DP-lane placement with dummy timing) under decisions D2 and D3. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved PD endpoint and DP-lane placement ticket. |

**What to build:** Complete online PD requests using autonomous endpoint and DP-lane placement driven by the load snapshots the frontend can observe.
Carry the chosen request ownership through target reservation and handoff, validating actual Step behavior through the existing placement owner.

**Blocked by:** structural part: the structural part of `17` (decisions D2 and D3). Native-backed placement criteria: `08`. Trained-predictor acceptance: `09.full_frontier_path[GLM]`.

**External prerequisites:** Verified external09.full_frontier_path[GLM], complete 08/17 and a D-DP-PD disposition are required. Keep the existing co-location guard until a supported PD implementation and its distinguishing evidence exist. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Record distinguishing native evidence and reuse sufficiency in the existing D-* record. Routine incremental changes proceed under standing execution authority; only a demonstrated material core refactor or materially unresolved semantics/acceptance requires explicit approval.

**Status:** in-progress (structural part done at `d65a4b9`, merged at `0c92903`; native-backed placement criteria wait for concurrent native evidence from ticket 08 or 09; trained-predictor acceptance waits for GLM `09.full_frontier_path`; the dense-initialization budget is `UNVERIFIED_UNRESOLVED_MEASUREMENT_CONDITIONS`)

**D31 status (2026-10-25):** Native-backed placement criteria are judged within plan step E.

**Source finding — 2026-10-20 (parent agent, not yet verified by a run):** The Step image's DP lane choice is `DPLBAsyncMPClient.get_core_engine_for_request` (`vllm/v1/engine/core_client.py:1642–1697` in the exported image source). Its score is `max(client_count × inflight, waiting + running) + waiting × 6.0 × max(0, kv_cache_usage − 0.5)`. The scan start rotates after every choice, the chosen lane's local waiting count is raised by `client_count`, and the coordinator publishes load every 100 ms. Frontier already implements the same v0.28 rule (`frontier/scheduler/utils/vllm_dp_load_balancer.py:180–186`, `_v028_score`). The cluster scheduler `vllm_load_balancing` limits it to one co-location Replica (`frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py:16–47`). With `client_count = 1` the two rules agree; a line-by-line comparison of the coordinator and the publish timing remains. The likely structural work is therefore one balancer per role API server: the P lane is chosen when the request arrives at P, and the D lane is chosen when the proxy sends the D request after P's response, before the C16 reservation. This must be confirmed against the native P/D entry points from ticket 08.

**Structural release — 2026-10-20:** Under the user's 2026-10-20 mainline decisions ([requirements](../requirements.md), [spec §6, §9 and §10](../spec.md)), the structural part may start after the structural part of `17`: autonomous endpoint and DP4-lane placement for Step PD with dummy timing. Native-backed tie and update behavior waits for `08`.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S06.
**Shared owners:** C05 and the existing endpoint, cluster-scheduler, and DP-lane placement owners.

## Review disposition — 2026-10-18

NOT STARTED. Source review can precede implementation; native placement evidence and D-DP-PD disposition are required. C05 integrates19/20 serially while their evidence/test design may proceed independently.

## Acceptance

- [ ] Native evidence identifies actual P/D entry points, endpoint selection, DP4 load reports, update timing, and placement behavior; inherited co-location behavior is not assumed equivalent.
- [ ] Online requests autonomously select endpoints and DP lanes from causally available load snapshots, including observed reporting delay, without future batches, completions, or oracle queue state.
- [ ] Request ownership remains one DP lane within a Replica; TP/EP participation does not multiply schedulers or enlarge a Replica-local collective domain.
- [ ] Selected endpoint and DP-lane identity survive reservation, READ, readiness, first decode, and completion without reassignment or duplicate admission during handoff.
- [ ] Controlled arrivals distinguish current from stale load snapshots and validate native-backed tie and update behavior with recorded expected versus actual placements and completions.
- [ ] Unsupported topology or role combinations remain explicit errors; removing the co-location-only guard alone cannot satisfy this ticket's placement or lifecycle acceptance.
- [ ] Existing approved placement remains unchanged outside Step's supported scope; shared semantic differences have distinguishing evidence and a disposition in the existing D-DP-PD record. A material core refactor or materially unresolved semantics/acceptance requires approval; no parallel Step policy authority is introduced.
- [ ] At least 50 baseline-versus-candidate scenarios preserve unchanged fidelity under specification §10; preregistered wall-time/memory budgets are measured, and every excess has an explanation and disposition.
  - **Dense-initialization budget disposition — 2026-10-20:** The final authorized diagnosis is complete and stopped. Existing wall SD/MAD are 0.384537/0.106815 seconds for `8585748` and 0.282295/0.144529 for `d65a4b9`; a ten-pair matched CPU experiment gives mean +1.028279% with paired bootstrap 95% CI [-3.573364%, +7.822538%] and median -0.832985% with CI [-3.414289%, +2.521546%]. The frozen 1.941337% limit is not consistently excluded, so this budget item remains **UNVERIFIED_UNRESOLVED_MEASUREMENT_CONDITIONS**, not PASS and not a proven CPU regression. No favorable block-only CI is selected, no threshold or product code is changed, and no additional loop is started. Full noise-floor, power-planning, wall/CPU/user/system results and evidence are in [the placement report, Dense initialization budget diagnosis](../test_report_2026-10-20_step5_pd_placement.md#dense-initialization-budget-diagnosis).

Specification references: §§3, 5, 7, 9–10.
