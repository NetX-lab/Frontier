# D29 repeated input derivation and parsing

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Allocate disjoint loader and artifact-parser repairs from measured residual costs;preserve live validation and file-update semantics. |

## Authority and dependencies

D29-REPAIR-PATH-AUTHORITY covers these necessary CPU-efficiency repairs inside
the existing111 paths and owner/global limits. Apply the two new overlays after
model_group_reuse_scope_amendment.json, in init-validation then artifact-parse
order. CORE remains4404, PRICING3956, native1582 and global10000.

Run the existing model-group repair, loader repair and PRICING parser repair in
parallel on disjoint files. Independently diagnose the unexplained initial21.5
minutes using a short bounded real-path probe. Deliveries -> parent checks and
independent review -> integrated residual checks -> new source freeze -> fresh
required regressions and unchanged6144/1024 offline/online CLI,2400s per mode.
No partial-point measurement admits the full workflow or native/GPU results.

## CORE: once-per-loader architecture family

Source: profiling_dataframe_loaders.py; regression:
test_moe_grouped_gemm_padded_features.py. Budget source<=6,test<=39,total<=45.
Compute the existing resolve_runtime_attention_family(model_config).family_id
before the typed-column map and pass the existing architecture_family_id keyword.
The symbol is already imported. Keep every row's JSON/schema/ownership checks
before scalar filtering, including rows outside the selected operator/layer.
No cache, API change, model mutation or dataframe sharing is needed.

Parent runs the corrected family_only_loop.py on the real3000-row head slice:
2.837165373s baseline versus0.274640171s hoisted, one config validation and equal
payload SHA256 f460359008814440c80a40fd2e97b557e7bcbb802c72a4076c6ebe3c8a2cdd3f.
Exact command/result: ../resumed_parent/test_report_2026-10-23_input_validation.md.
The full-file observations and their procedural deviation remain in
../init_validation_scope/report.md; no further full scans are authorized here.

Acceptance: a deterministic regression through the actual loader is RED before
the fix and GREEN afterwards; one family derivation per load; equal selected
frames/duration-count bindings; a malformed unselected row must still fail.
Run both existing affected files/suites as appropriate, preserving prior negatives.

## PRICING: exact-text parsed-artifact reuse

Source: collective_sim_cc_backend.py; regression:test_collective_sim_cc_backend.py.
Budget source<=15,test<=35,total<=50. The measured isolated artifact decode is
54.7ms per plan on the retained8,033,853-byte artifact, versus9.8ms to read it.
Five real point calls perform20 decodes. The scout's stat-keyed control proves
parsing is separable but is not the selected implementation.

Keep reading the current text on every plan call. Reuse only the last parsed
artifact when that text compares equal; replace the backend-local pair after a
successful parse. This retains exact file-content visibility, including same-size
rewrites with restored mtime, and avoids a new timestamp/inode invalidation
contract. Keeping one additional8MB text is bounded by the existing one-artifact
backend use; measure the actual control and residency before claiming its gain.
Do not add a hash when direct equality already selects the next action.

First run a single-variable scratch control using the existing actual pricing
loop, with group reuse fixed in both arms. Require the complete costs/proofs/plan
payloads to match and the redundant large decodes to disappear. Then implement
only the two allocated paths. The current dict is read-only in all consumers.
Keep TP/DP placement, CPU/native identity, qualification, count/context/buffer,
receipt composition, numerical cost cache and simulator calls live and unchanged.
No shared accessor, CORE edit or native cache-key optimization belongs to this step.

Acceptance: deterministic decoder-call regression; same-instance changed text
is observed even with the prior size/mtime; invalid current JSON/read failures
remain errors; all45 existing backend tests and relevant trace consumers pass.
Report the new exact-text control separately from the scout's stat-keyed timing.
The remaining CORE-side artifact parse and native whole-artifact serialization
are measured/deferred boundaries, not silently included in this allocation.
