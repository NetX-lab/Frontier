# 13: NIXL and shared-network measurements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |

**What to build:** Independent EP-only, NIXL-only, and concurrent network measurements that identify actual component payload costs and contention, supplying evidence to the existing D-PD decision without presuming a simulator contention model.

**Blocked by:** `08` G3 native PD content and lifetime qualification; `12` TP and EP communication measurements.

**External prerequisites:** Qualified08 component/payload facts and12 measurement identity are required. Run only cases needed to explain the represented PD path; raw contention evidence does not require a new generic network model. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.

**Status:** blocked

**D31 status (2026-10-25):** Paused. NIXL contention is reopened only when the TTFT error needs it.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S07.
**Shared owners:** C03, C04; existing D-PD evidence consumer.

## Review disposition — 2026-10-18

NOT STARTED. Keep EP-only/NIXL-only/concurrent measurements distinct and use them only where the workflow exposes a contention effect. A new contention-state abstraction is conditional, not a prerequisite inferred from this ticket title.

## Acceptance

- [ ] Deliver the EP-only/NIXL-only/concurrent measurement set, payload/topology/concurrency manifest, independent coverage report, and same-D-PD contention evidence card or sufficiency finding under §§5 and 7–9.
- [ ] Use G3's actual CUDA-buffer NIXL registration and state dispositions to identify raw KV, side-summary, and any transferred tail components; distinguish logical unique bytes, resident/replicated bytes, and actual wire bytes.
- [ ] Measure NIXL-only transfers independently from EP-only communication, then run controlled concurrent cases with matched relevant identities; retain observed costs and variability for each mode rather than infer concurrence from separate timings.
- [ ] Record endpoint/rank mapping, node/NIC/NUMA placement, backend, precision/packing, payload sizes, useful/padded tokens, source/target page layout, and overlap intervals; concurrency and resource sharing must be observed rather than assumed.
- [ ] Separate submission/host work, network completion, GPU-visible readiness, and source release using validated local intervals and causal correlation; never turn network completion into an unsupported D-ready boundary.
- [ ] Cover independent payload/shape/placement and overlap combinations, including summary/tail boundaries and relevant empty-lane EP work; retain target reservation and capacity-pressure conditions so blocking is not mislabeled link transfer time.
- [ ] Exclude unintended GPU/NIC interference from isolated runs and document deliberate interference in concurrent runs; peak bandwidth, fixed 64-GPU latency, inherited p2p slowdown, or arbitrary contention multipliers cannot replace measurement.
- [ ] Submit the smallest evidence-supported contention/state amendment to the existing D-PD owner with alternatives, omitted detail, distinguishing experiment, and consumers, or document that approved semantics suffice; measurement itself grants no production-semantic approval.
- [ ] Reject wrong payload/backend/topology, missing raw/summary records or participants, ambiguous cross-node timing, missing independent modes, and unsupported extrapolation; no held-out native transfer duration may become a predictive input.
- [ ] Measurement-tool changes satisfy §10's ≥50-scenario fidelity comparisons and preregistered wall-time/memory budgets with over-budget dispositions; ≥32-GPU auxiliary harness errors mark evidence unverified and notify the user without stopping healthy workloads solely for those errors.
