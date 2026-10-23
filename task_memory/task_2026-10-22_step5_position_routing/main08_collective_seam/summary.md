# D29-WAVE-COLLECTIVE-SEAM summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive the pre-decision diagnosis; correct the inclusive byte-read comparison to1.528ms and point to YC's later run-fixed decision. |
| 2026-10-23 | Re-measure on the 5e149dd8-equivalent source; replace the shared-accessor proposal with byte admission at both existing current-file checks; add native-key evidence and the remaining decisions. |
| 2026-10-23 | Archive seam analysis, measured differentials, proposed interface and open decisions. |

## Outcome

This is the pre-decision diagnostic record. YC subsequently approved fixed
inputs per simulator run; current ownership, scope and checks are in
`../run_input_snapshot/design.md`. The estimates below do not grant a current
allocation or establish full-workflow acceptance.

There is no need for a wave-level or lane-level collective memo. The waste sits in the two
existing places that check the current `collective_local.json` on every call:

- PRICING backend `CollectiveSimCCBackend.predict_collective_plan`
  (`collective_sim_cc_backend.py:812–817`), 64 calls per wave.
- CORE position query (`moe_routing_workload.py:213`, compared in
  `_step5_collective_points` lines 155–168 at 5e149dd8), 16 calls per wave.

Both already keep the last file content and its parse, and they reparse only when the
content changes. Most of the time is spent turning the 7.66 MB file into text and comparing
two 8 MB strings: about 10 ms per call, against1.528ms to read and compare raw bytes.
The latter already includes the read; adding the separate0.804ms read measurement
again produced the earlier incorrect2.3ms summary. The report's timing table and
raw `micro_admission.json` were correct.
Reading bytes in both places keeps every read, every comparison and every
source/point/family/context/placement/qualification/count/layout check live on each call.
A same-size rewrite with restored mtime is still seen on the next call.

Measured on the actual main08 decode layer 61 wave, with source equivalent to 5e149dd8 and
the payload hash `b2593e72…` unchanged in all 12 runs: warm pricing goes from
0.95–1.33 s to 0.31–0.39 s per wave, and backend time from 0.64–0.89 s to 0.14–0.17 s.
At about 88 routed waves per decode round, that is roughly 84–117 s versus 27–34 s of
wave pricing per round. That figure is extrapolated, not a timed round.

## What is shared and what is lane-local (observed)

- **Same for all 16 lanes and for every layer in a routed group with the same source.**
  This covers the phase, both DP sources' physical sizes, tensors, groups, graph mode, the
  16-rank `context_json`, the artifact and its artifact-level checks, and the cost.
  `CollectivePlanCost` already holds all 16 ranks' local times
  (`collective_plan.py:103–124`). The wave barrier takes the maximum over lanes
  (`expert_parallel.py:241–265`), so equal repeated values are not double-counted.
  Layers 3 and 87 give the same plans and times as layer 61; the group covers layers 3–87.
- **Lane-local.** `buffer_json.compute_context` ties the collective to that lane's expert
  operands. In this wave only `native_padding.num_tokens_post_padded` differs (128 on lane
  11). That gives 8 distinct plans over 64 calls. The checks in
  `moe_operator_times.py:95–125` are per-lane proofs and stay per lane.
- **Lifetime.** There is one backend per `Cluster`, and the predictor keeps CORE's retained
  tuple. No wave-scoped lifetime exists. `predict_ep_wave_phase_times` is a generic per-lane
  loop that DeepSeek-V4 also uses.

## Recommended change (smallest, no new interface)

PRICING, `collective_sim_cc_backend.py`:

```python
data = path.read_bytes()
if data != self._collective_artifact_bytes:
    artifact = json.loads(data.decode())
    self._collective_artifact_bytes, self._collective_artifact = data, artifact
```

Rename the attribute in `__init__` (line 73) as well, because it now holds bytes.

CORE, `moe_routing_workload.py`: change line 213 to `.read_bytes()`, and change line 160
to `json.loads(text.decode())`. The `.decode()` matters: an existing CORE test observer
does `'"source_frames"' in text` on every `json.loads` argument and would break on bytes.

`.decode()` is strict UTF-8, which matches today's `read_text()` under the verified UTF-8
locale and JSON's required encoding. Equal bytes always mean an equal parse, so neither
side can reuse a parse that today's text check would reject.

## Measured alternatives (not recommended now)

- **Shared backend accessor used by CORE.** Measured 0.31–0.44 s, no gain over both
  reading bytes. It would change four CORE test stubs
  (`SimpleNamespace(_config=...)` in `test_moe_routing_runtime.py` and
  `test_prediction_measurement_bindings.py`). It is only worth doing to remove the
  duplicated path rule and about 18 MB of duplicate retention.
- **CORE bytes only.** 0.91–0.95 s; most of the cost is the backend's 64 calls.
- **Wave-scoped plan memo.** On 79f1a3fd it brought the backend to 0.095 s per wave. At most
  about 0.05–0.08 s per wave remains beyond byte admission. It needs a new wave lifetime and
  hides a rewrite in the middle of a wave.
- **Load-time binding** through bundle `input_bindings` (path and sha256,
  `shared_prediction_model_manager.py:140–146`, `prediction_model_registry.py:788–793`).
  It removes all per-call reads, about 0.18 s per wave. It contradicts the tested same-instance
  rewrite contract and needs YC's decision that inputs are immutable during a run.

## Native cache key (line 871), important for the qualified native run

`key = ("plan", plan.cache_key(), json.dumps(artifact, sort_keys=True))` serializes the
whole artifact on every native call: 43.9 ms at 7.66 MB, so about 2.8 s per wave if the native
artifact is the same size. Each cached entry also keeps its own full string, up to
168 × 7.66 MB ≈ 1.3 GB if every plan is cached. This is inferred from the code; the native
artifact's size is not measured.

Using the admitted bytes as the key cannot produce a hit that the old key would reject. The
scratch regression `test_native_key_scratch.py` passes with it and fails at base. The
behavioral difference is that a whitespace-only rewrite recomputes once. main08 is a CPU
fixture and does not reach this branch, so this does not speed up main08.

## Decisions for primary/YC

1. **Existing backend test instrumentation.**
   `test_collective_artifact_reuses_only_identical_current_text` hooks `read_text` and a
   `str` subclass, so it cannot pass unchanged with byte admission. A copy that hooks
   `read_bytes` and keeps every behavioral assertion passes. Those assertions are: two calls
   give two reads and one decode; a same-size rewrite with restored mtime is seen; invalid JSON
   raises every time; the same error objects propagate.
   - (a) Change only the hook lines. This is recommended, at 0.31–0.39 s per wave.
   - (b) Keep the file byte-for-byte and the backend on text. Only the CORE change is then
     possible, at about 0.91–0.95 s per wave, a negligible gain.

   The requirement says this test must keep passing, so (a) needs explicit acceptance.
2. **Native key (line 871).** The earlier parse allocation forbade cache-key changes. This
   needs its own allocation. It is recommended before any native collective run.
3. **Admission granularity.** Keep it per call; no decision is needed unless YC wants the
   wave or load-time trade-off above.

## Affected paths and gross estimate (no allocation assumed)

| Owner | Path | Step gross | Note |
| --- | --- | --- | --- |
| PRICING | `frontier/cc_backend/backends/collective_sim_cc_backend.py` | ~10 (lines 73, 813–816), +2 for native key | path 84/93 at 79f1a3fd, unchanged at HEAD |
| PRICING | `tests/unit/test_collective_sim_cc_backend.py` | ~10 hook lines (1a); ~12 native-key regression | 233/233 used, so a zero-sum PRICING transfer is needed |
| CORE | `frontier/execution_time_predictor/moe_routing_workload.py` | ~4 (lines 160, 213; optional docstring 156) | no CORE test change needed |

PRICING needs about 22 gross with no native key, or about 34 with the native key and its
regression, out of 36 remaining. CORE needs about 4–6 out of 434. Global needs about 28–40
out of 746. Root should confirm the current per-path ceilings before allocating.

## Verification after a source change

1. Run `seam_loop_own.py --variable none` on the changed tree. The payload hash must stay
   `b2593e72…`, with 64 backend and 16 CORE reads per wave. Warm pricing should be about
   0.3–0.4 s.
2. The backend suite should pass in full, with only the hook lines changed under (a).
   Add the native-key regression if allocated.
3. Run `test_moe_routing_runtime.py`, the three position/collective tests in
   `test_prediction_measurement_bindings.py`, `test_step5_measured_ffn.py`,
   `test_step5_moe_pricing.py`, `test_op_trace_utils.py` and `test_step5_comm_cost_export.py`.
4. Full main08 CLI acceptance stays with primary.

## Limits

- One actual wave on a CPU-fixture artifact. The native branch is covered only by the
  scratch regression.
- Runs used `timeout 60s` and `RLIMIT_AS 30GiB`; systemd `MemoryMax` was unavailable.
- Pytest checks use runtime substitution, not edited source.
- The whole `test_prediction_measurement_bindings.py` exceeds 60 s; only 3 of its 12 tests ran.
- The root worktree moved twice during the task, and another process edited one scratch
  script and wrote three runs. Only the `run-own-*` runs are evidence (see the test report).

## Deliverables

- This directory: `requirements.md`, `test_report_2026-10-23_collective_seam.md`, `summary.md`.
- Scratch `/data/ycfeng/tmp/d29_wave_collective_seam/`: `seam_loop_own.py` (current-source
  probe), `seam_loop.py` (79f1a3fd probe), `admission_variant_plugin_head.py`,
  `test_native_key_scratch.py`, `test_admission_bytes_adapted.py`, `micro_admission.py/.json`,
  `results_index.json`, `run-*/result.json` and logs.
- No source, test, input or commit changes. No CLI, GPU, platform, network or agents.
