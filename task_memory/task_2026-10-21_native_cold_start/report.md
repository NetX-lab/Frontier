# W128-NATIVE-COLD-START report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Completed approved first-boundary proof, successor artifact and CPU verification. |
| 2026-10-22 | Closed primary P1 worker-readiness rejection with RED/GREEN and complete predicate receipts. |

## Outcome

Implemented D-NATIVE-COLD-START without inventing zero gauges. Only `before_repeat01_warmup` can carry proof `dedicated_services_ready_client_zero_native_counters_zero`. The original root checker is untouched. The task-local successor remains a candidate for parent review/freeze/repoint; no old run is requalified.

### Exact admission rules

1. **Dedicated new services:** `dedicated_services()` reads the existing run manifest and node results/readiness files. Every declared node binds its actual `Popen.pid`, launch UTC, generation and manifest SHA256. Launch follows the timezone-aware prelaunch UTC. P/D DP ownership must cover all lanes without overlap and match the actual requested URLs. Both role heads have `/health` HTTP200 readiness after launch, with the same PID, launch timestamp, generation and manifest hash. These fields are written by the worker at its existing Popen/health seam; they are not an unbound flag. Missing receipts fail the auxiliary observation.
2. **Client zero dispatches:** the first boundary counts actual non-None `dispatch_monotonic` fields across all retained records before replay. Any existing dispatch rejects the first boundary.
3. **Explicit native zero counters:** every role/lane must publish all five v0.28 `request_success_total` FinishReason children (`stop`, `length`, `abort`, `error`, `repetition`), finite and exactly zero. Missing lane/reason, duplicate, wrong engine, negative/NaN/Inf/nonzero samples fail. The five children are native-source-backed (`vllm/v1/metrics/loggers.py:713-724`, `vllm/v1/engine/__init__.py:43-62`), not defaults.
4. **No synthetic gauges:** retained first-boundary lane gauge maps remain empty and `idle=False` for the actual captured publication gap. Either all gauges are unpublished or all original gauges are explicitly zero; partial/nonzero gauge evidence is rejected.
5. **Later six boundaries:** `idle_metrics()` still requires running/waiting/KV samples for every DP lane and all three equal zero; explicit per-lane completion evidence is also required. Role-level completion targets start from the proven zero baseline and increase by each planned phase count, yielding 16/144/160/288/304/432. A failed initial baseline makes later boundaries unverified, even if zero gauges subsequently appear; it no longer silently disables freshness.
6. **Soft failure / native behavior:** auxiliary proof, scrape, checkpoint and metric errors mark artifacts unverified but do not cancel phase replay or suppress native exceptions. Failed warmups keep their repetition and overall client workload unverified. No transport fix, native source/image change or Frontier core change.

### Successor checker

`check_groundtruth_run_successor.py` was copied from the exact frozen root file. Before edits, both SHA256 values were `9116ab7c83da2b4e7cafb871c272af56c8322954eccd56abf2eeac1b55906b55`. AST comparison identifies only two changed top-level functions: `native_idle_evidence` and `check_native_canonical`.

The successor rebinds node receipts through the already source-pinned harness in a fresh interpreter only when the new first-boundary proof is present; checks the retained proof/zero counters/zero client-dispatch declaration; and independently requires every client dispatch to follow that first observation. `native_workers` reuses that validated receipt and compares the complete head readiness object, including every metadata field. A failed rebind rejects the new-proof worker predicate. It does not project to role/url. The six later gauge/completion/timestamp checks are unchanged. Old gauge-only artifacts retain the frozen checker's exact role/url whole-object comparison, including extra-field rejection. Frozen-to-current diff: `checker_successor.diff`. Candidate SHA256: `5efd00ea6544d64440c535d06ad415b6a4d1a25b96ce8863cf4944568f37ec36`.

## Observed verification

- Captured native response RED: actual P/D body replay through `observe_idle` returned False and both missing-running/waiting/KV errors.
- Captured response GREEN: actual `replay_repetitions` and native client SSE parsing executed all 432 canonical records, preserving QPS1 scheduled arrivals and 16 warmups/128 formal requests across three repetitions. Only transport and harness clock are CPU-scripted; no GPU timing claim.
- Nine replay scenarios: healthy, missing lane, missing reason, wrong lane, nonzero counters, existing client, stale completion with explicitly idle gauges, failed repeat02 warmup HTTP502, and missing initial baseline. All expected verdicts observed. Stale completion is isolated from gauge failure and leaves all six later targets enforced.
- Seven readiness mutations: missing PID, mismatched PID/URL/generation/manifest, readiness preceding launch, wrong role. All rejected.
- Successor snapshot mutations: unbound proof, missing counter reason, existing-client declaration, missing later KV gauge, stale completion, first observation after a client dispatch. All rejected.
- Primary P1 RED: adding `native_workers` PASS to the previous complete-checker test exposed `Head readiness endpoint mismatch: host0`, despite passing request/idle. The previous241 PASS/21 SKIP result omitted worker/lifecycle/terminal assertions and did not establish complete checker admission; that claim is withdrawn.
- P1 GREEN: the complete checker now passes all seven expected predicates. All eight readiness fields (`role`, `url`, `server_pid`, `server_started_at_utc`, `run_manifest_sha256`, `run_generation`, `ready_at_utc`, `health_status`) have separate missing/wrong negatives: all16 reject both `requests_and_idle_lanes` and `native_workers`. Two legacy controls retain identical strict verdicts with the frozen checker. `p1_predicates_{red,green}.json`, `p1_readiness_negatives.json` and `p1_review_receipt.json` retain exact outcomes.
- Latest focused regression: **241 PASS / 21 SKIP / 0 failures / 0 errors**, 132.05 s. Nineteen skips require the approved C26 source fixture; two require a legacy checker baseline, neither installed in this worktree. No new cold-start check is skipped.
- Immutable checks: all five canonical files byte-identical to `68f891b`; root checker SHA unchanged; `native_full_structure.py` byte-identical to base; cumulative production/test scope is five files and **495 changed lines** counting only the narrow successor delta, not its unchanged frozen contents. `source_identity.json` records exact hashes. P1 changes only the successor and existing cold-start test.

### Every expected full-checker predicate

| Predicate | Expected / observed | Limit |
| --- | --- | --- |
| `checkout` | PASS / PASS | CPU fixture provenance only. |
| `inputs_unchanged` | PASS / PASS | Actual fixture input hashes checked. |
| `harness_pinned_and_client_rows` | PASS / PASS | Real pinned reader subprocess, Git receipts fixture-backed. |
| `requests_and_idle_lanes` | PASS / PASS | Seven retained boundaries and bound service evidence. |
| `native_workers` | PASS / PASS | Complete new metadata bound to manifest/node receipt; native node/GPU facts fixture-backed. |
| `native_lifecycle` | PASS / PASS | Retained fixture drain/stop ordering. |
| `rank_runtime_graph` | FAIL / FAIL | `KeyError: 'vllm/v1/cudagraph_dispatcher.py'`: required native source hash/rank/graph evidence missing; no native qualification. |
| `terminal_status` | PASS / PASS | Fixture platform receipt, no platform query performed. |
| Overall status | FAIL / FAIL | Intentionally not qualified as native evidence. |

P1 closed the worker's metadata rejection, collateral lifecycle/terminal failures from incomplete nodes, and missing complete-predicate assertions. Native qualification remains unresolved/out of scope; it is not forced to PASS.

## Old V4 and remaining decisions

The actual retained `/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_results` remains UNVERIFIED: first boundary failed, six later boundaries were marked verified, and 431/432 requests completed. The one failure is repeat02 warmup HTTP502. Its old readiness/results lack the new launch-provenance fields, so `dedicated_services` refuses retroactive admission. Later historical verified labels had no initial baseline; this report does not promote them to a freshness proof.

Primary must accept the P1 fix before any freeze/package/repoint. This agent is the sole successor writer and stops editing after this handoff. Future authorized generations require newly bound receipts; no old generation is promoted. No GPU, platform, auth, SSH, cloud access, external network, dependency install, run launch, merge, push or metrics/P90 change was performed. Dedicated-service isolation is established by coordinator-owned fresh run directories, Popen and bound readiness, not by a network ACL or independent server identity endpoint. That is the approved harness trust boundary.

## Deliverables

- Production: `tests/comparison/calibration/{pd_replay.py,native_pd_worker.py}`.
- Tests: `tests/unit/{test_native_cold_start.py,test_native_pd_worker.py}`.
- Candidate successor: `task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py`.
- Evidence in this directory: `checker_successor.diff`, `source_identity.json`, `verification_evidence.json`, `captured_response_replay.json`, `focused_regressions.xml`, `test_report_2026-10-21_native_cold_start.md`.
- Exact cumulative code/test Git diff against parent base `e55c83c263f86f48972f137858723a07c6f5e691`: `cumulative_code_vs_parent.diff` includes the actual successor file (not just its narrow frozen delta). P1-only Git diff against `746b744d3382ad1c49a1741c6dea7f5fdfee4c41`: `p1_fix.diff`. Exact commands and hashes are in `p1_review_receipt.json`; raw diff context whitespace is retained unchanged.
