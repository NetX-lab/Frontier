# W128-NATIVE-COLD-START report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Completed approved first-boundary proof, successor artifact and CPU verification. |

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

The successor rebinds node receipts through the already source-pinned harness in a fresh interpreter only when the new first-boundary proof is present; checks the retained proof/zero counters/zero client-dispatch declaration; and independently requires every client dispatch to follow that first observation. The six later gauge/completion/timestamp checks are unchanged. Old gauge-only artifacts produce the same predicate verdict as the frozen checker. Narrow diff: `checker_successor.diff`. Candidate SHA256: `48e311358aa23ae56e6f89b0b8686dced5d6ff5d7b3bcf10bb638dab9d091771`.

## Observed verification

- Captured native response RED: actual P/D body replay through `observe_idle` returned False and both missing-running/waiting/KV errors.
- Captured response GREEN: actual `replay_repetitions` and native client SSE parsing executed all 432 canonical records, preserving QPS1 scheduled arrivals and 16 warmups/128 formal requests across three repetitions. Only transport and harness clock are CPU-scripted; no GPU timing claim.
- Nine replay scenarios: healthy, missing lane, missing reason, wrong lane, nonzero counters, existing client, stale completion with explicitly idle gauges, failed repeat02 warmup HTTP502, and missing initial baseline. All expected verdicts observed. Stale completion is isolated from gauge failure and leaves all six later targets enforced.
- Seven readiness mutations: missing PID, mismatched PID/URL/generation/manifest, readiness preceding launch, wrong role. All rejected.
- Successor snapshot mutations: unbound proof, missing counter reason, existing-client declaration, missing later KV gauge, stale completion, first observation after a client dispatch. All rejected.
- Successor complete native checker path: actual pinned reader subprocess admits seven boundaries with bound receipts; changing the recorded head PID rejects `requests_and_idle_lanes`. The overall CPU fixture correctly remains FAIL for missing native graph evidence.
- Final focused regression: **241 PASS / 21 SKIP / 0 failures / 0 errors**, 113.62 s. Nineteen skips require the approved C26 source fixture; two require a legacy checker baseline, neither installed in this worktree. No new cold-start check is skipped.
- Immutable checks: all five canonical files byte-identical to `68f891b`; root checker SHA unchanged; `native_full_structure.py` byte-identical to base; production/test scope is five files and **453 changed lines** counting only the narrow successor delta, not its unchanged frozen contents. `source_identity.json` records exact hashes.

## Old V4 and remaining decisions

The actual retained `/data/ycfeng/tmp/glm53_dsv4/v4_p0_native03_results` remains UNVERIFIED: first boundary failed, six later boundaries were marked verified, and 431/432 requests completed. The one failure is repeat02 warmup HTTP502. Its old readiness/results lack the new launch-provenance fields, so `dedicated_services` refuses retroactive admission. Later historical verified labels had no initial baseline; this report does not promote them to a freshness proof.

Parent must review/freeze the successor and repoint only new run generations; freeze the new harness and launch receipts for future authorized runs. No GPU, platform, auth, SSH, cloud access, external network, dependency install, run launch, merge or push was performed. Dedicated-service isolation is established by coordinator-owned fresh run directories, Popen and bound readiness, not by a network ACL or independent server identity endpoint. That is the approved harness trust boundary.

## Deliverables

- Production: `tests/comparison/calibration/{pd_replay.py,native_pd_worker.py}`.
- Tests: `tests/unit/{test_native_cold_start.py,test_native_pd_worker.py}`.
- Candidate successor: `task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py`.
- Evidence in this directory: `checker_successor.diff`, `source_identity.json`, `verification_evidence.json`, `captured_response_replay.json`, `focused_regressions.xml`, `test_report_2026-10-21_native_cold_start.md`.
