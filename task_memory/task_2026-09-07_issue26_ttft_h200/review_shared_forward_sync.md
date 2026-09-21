## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Rebuilt the failure mechanism and accepted a smaller shared synchronization implementation. |

# D017 second-pass review

## 1. Verdict

ACCEPT after the implementation refinements below. Shared identity plus source-local continuation reaches the observed mechanism; retain existing events as adapters instead of adding parallel event machinery. YC authorized sequential implementation after this review passes and requested concise, precise code without hard-coding or redundant defenses.

## 2. Scope and inputs

Reviewer: primary agent /root, independent second-pass inspection; no delegated review is claimed. Target revision a4a0c496d370c84d83a5b098d3fdd8561af1c74e. Inputs: requirements.md, plan.md, design_shared_forward_sync.md, actual sync-entry/state, wave scheduling, both completion helpers, original request callbacks, current exact-case command, raw three-request sync snapshot and prior control outcomes. Active config remains the frozen H2004096/1024 case. No production diff existed at review entry; three untracked E2E files are preserved.

## 3. Independent evidence ledger

| Claim | Label | Direct evidence | Strength | Gap / next check |
| --- | --- | --- | --- | --- |
| Same admitted group splits by local phase | Evidence | three_requests_MONOLITHIC_sync_state.json: group5, layer0, decode240/lane0 and prefill241/lane1 | Direct | Focused real-handler regression |
| Split prevents EP dispatch | Inference | ForwardSyncState phase partition plus separate sync rooms; mixed control0waves versus same-phase1wave | Strong | Mixed-phase regression must turn green |
| Load-balancer weights directly cause this hang | Inference, rejected | Handler-only control has no load-balancer execution; source rooms still require phase-local peers | Corroborated alternative check | Exact future placement remains runtime-dependent |
| Removing request2 eliminates this failure | Evidence | Two-request4096/1024 completion artifact; three-request original shape fails | Direct counterexample | Repeat three-request run after fix |
| Source timing is currently borrowed from one batch | Evidence | prefill_collective.py/decode_collective.py choose sample_batch before per-lane loop | Direct | Unequal-shape continuation regression |
| Existing predictor supports local mixed input | Evidence | attn_decode_in_mixed feature selection and per-request KV extraction | Direct | Observe intact original batches/features |
| Request completion already supports mixed batch tokens | Evidence | original Batch iterates request/token pairs; Request selects per-request state | Direct | Verify old TTFT and first-token credit |
| Fix closes numerical TTFT gate | Unknown | No repaired numerical output exists | Unestablished | Full fresh100request comparison |

## 4. Reconstructed RCA

Original third request causes a real prefill lane to overlap a decode lane. Both are admitted to group5, but phase-specific step binding allocates240and241. Each separate room holds one of the two required lanes. Neither peer is idle, so dummy synthesis cannot satisfy either room. The event queue ultimately drains while requests remain unfinished. Same-phase controls and the two-request negative control distinguish this from missing profiles or a generic load-balancer failure.

## 5. RCA/design corrections

- Neutral event classes are unnecessary: existing events are valid adapters when MONOLITHIC step resolution and completion share one path. Keep their existing types and ordering; phase names are diagnostic labels, not group identity.
- Reuse existing source-local completion helpers with direct_batch and an explicit already-restored ownership result. Restore the full cohort once before calling those helpers. This avoids duplicating PP/speculative/final-tail logic.
- In mixed sources, advance decode layer counters for the decode subset only. Pure-decode sources already advance through their existing helper. Request terminal callbacks remain untouched.
- Maintain explicit per-source model-component accounting for decode as well as prefill; join wait is wall time, not CPU or operator work.
- Preserve current same-layer rendezvous for dense layers inside MoE models, then execute their FFN locally without creating an EP collective. Changing that admission timing is outside this repair.

## 6. Implementation review

Use existing shared modules and one small cohort-completion helper. Normalize MONOLITHIC sync kind to forward and let existing room attributes reference one room. Aliasing alone is insufficient; wave metadata/ledgers and completion must be source-local. No config, model-name routing, scaling factor, Request contract change, catch-all fallback, or new event enum is needed. Existing PDD phase-specific behavior stays intact. Initial scope is approximately8–10production files plus focused tests; replace duplicated behavior rather than layering a second engine.

## 7. Corrected plan

Actual-handler failing regression -> shared identity/room repair and source-specific wave/completion -> phase/shape/token/ownership regressions -> trained3request replay -> verified code commit -> fresh100request replay -> batch and metric comparison. Existing reference/profiles remain from this fresh current-main task. Deferred items remain unchanged. No implementation authorization blocker remains after ACCEPT; YC explicitly requested advancement.

## 8. Verification record

Review inspected the original artifacts and source; it did not rerun GPU measurement. Handler controls recorded1/1/0waves for P/P,D/D,D/P; raw three-request snapshot records arrival0.2954594669165676/0.29563780122719624s. Trained reproduction command and environment are preserved in test_report_2026-09-08_mixed_phase_forward_stall.md. New implementation checks and exact commands will be recorded separately. Numerical accuracy and complete cross-DP reference-round joins remain unknown.
