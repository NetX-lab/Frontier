# D29-COLLECTIVE-JSON-SCOPE diagnosis

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record bounded RED loop, measured artifact read/decode cost, supported reuse seam and remaining checks. |

Scope: read-only diagnosis of the repeated `collective_local.json` read and JSON
decode in `CollectiveSimCCBackend.predict_collective_plan`. No production, test,
CORE (`moe_routing_workload.py`, `sklearn_moe_execution_time_predictor.py`,
`test_prediction_measurement_bindings.py`), CLI, GPU or network change. All new
files live under `/data/ycfeng/tmp/d29_collective_json_scope/`.

Binding: frozen source `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`,
commit `e41abc073500d61f224eda4fca1123e74e40544d`. Retained actual input: main07
CLI command `/data/ycfeng/tmp/d29-parent-position-main07/offline/command.json`,
paired model bundle `d29-parent-position-fixture03/cache/bundle_3e4623...json`,
actual position lane layer3 EP0 (`q6128`, gathered 6129), collective artifact
`d29-parent-position-fixture03/fixture_collectives/h800/collective_local.json`
(8,033,853 bytes, `evidence_kind=cpu_fixture`, `candidate_only=true`,
`native_admitted=false`, 168 count entries, 0 models).

## 1. RED-capable loop and one-variable control

`loop.py` is the existing main07 loop (`runtime_loop.py`, SHA
`bba92e78...344852e`, group-reuse arm as the fixed baseline) copied into scratch
with one flag added, `--artifact-memo`, plus a backend-module-local counter. The
baseline arm keeps the flag off; nothing else differs in the timed region.
Only `frontier.cc_backend.backends.collective_sim_cc_backend`'s own module
bindings of `Path` and `json` are replaced (never global `json` or global
`pathlib`). In the memo arm the module-local `read_text` returns a marker carrying
the file identity `(str(path), st_size, st_mtime_ns, st_ino)` and the module-local
`json.loads` returns the parsed dict recorded for that identity, so the file is
read and decoded once per identity. All plan qualification, `cache_key()`,
receipt and cost construction still execute per plan.

Payload hashes compared per run: the original `result_sha256` (MoE op times,
communication op times, plan keys, proofs) and `extended_sha256` (adds every
`CollectivePlanCost` field through `vars(cost)`). Both are identical across all
arms: `d2c141f088b52063...` and `3888b075b5af3fd6...`.

## 2. Measured evidence

Wall time per position execution, one fixed actual P point, `--execution
--reuse-groups`, 5 iterations, `--no-profile`:

| Arm | iteration walls (s) | mean | steady (iters 3-5) |
| --- | --- | ---: | ---: |
| baseline (`baseline07`) | 0.4126, 0.3621, 0.3613, 0.3457, 0.3467 | 0.3657 | 0.3540 |
| artifact memo (`memo07`) | 0.1581, 0.2293, 0.0956, 0.0977, 0.0965 | 0.1355 | 0.0966 |

Backend-local counters, same runs:

| Quantity | baseline | memo |
| --- | ---: | ---: |
| plan calls (4 per position execution) | 20 | 20 |
| artifact file reads | 20 | 1 |
| artifact reads total time | 0.1969 s | 0.0101 s |
| artifact decodes (input > 1 MB) | 20 | 1 |
| artifact decode total time | 1.0938 s | 0.0348 s |
| per read / per decode | 9.8 ms / 54.7 ms | 10.1 ms / 34.8 ms |
| memo hits / misses | - | 19 / 1 |

Profiled arm (`baseline05`, 2 iterations, cProfile on) reproduces the existing
receipt: 1654 `json.loads` calls; backend module 32 loads (0.4939 s) and 8
`read_text` (0.0849 s); wall 0.4887 s / 0.3792 s. Memo arm: 1647 `json.loads`
(7 artifact decodes removed), wall 0.1741 s / 0.2213 s.

Attribution on the prior `execution-groups01` profile (2 iterations, 8 plan
calls): `predict_collective_plan` 0.542 s cumulative, of which 32 `json.loads`
0.4519 s and 8 `read_text` 0.0779 s; `_step5_position_queries`
(`moe_routing_workload.py:127`, CORE-owned) 1394 `json.loads` 0.1382 s and 2
`read_text` 0.0199 s. The fixture branch makes exactly 4 loads per plan
(artifact, `plan.context_json`, `plan.buffer_json`, receipt); the native branch
makes 3 loads plus 2 `json.dumps`, one of which serializes the whole artifact
for the cache key.

So the repeated artifact parse is 0.25 s of the 0.354 s steady position
execution (~71%); the memo removes all but the first. The residual ~0.096 s/point
is dominated by CORE's own artifact re-read and 168-key scan inside
`_step5_position_queries` (0.138 s per 2 points plus one 8 MB decode per point).

## 3. What is artifact-static and what must stay live

`predict_collective_plan` (`collective_sim_cc_backend.py:801`):

- artifact-static, safe to cache by file identity: `Path(profiling_data_dir)/
  device_type/"collective_local.json"` + `json.loads` (810-811). The parsed dict
  is never mutated afterwards (only `artifact["counts"][key]`,
  `artifact["models"]`, `artifact["qualification"]`, `json.dumps(artifact,...)`
  reads).
- must stay live per plan: `_validate_selected_parallel_dim_sizes` +
  TP8/DP2 placement (806-809); CPU branch identity checks against the plan's
  `evidence_kind`, schema, device, `native_admitted is False`,
  `candidate_only is True` (813-817); exact `artifact["counts"][plan.cache_key()]`
  lookup, `buffer_receipt`/`context`/`plan_key`/`cost`/receipt recomposition
  (818-825); native qualification (`status=PASS`, `independent`, observations,
  `parent_receipt`, `candidate_only` false) (826-833); frozen-scenario equality
  (834-844); `execution_policy` presence and model-context match (845-855);
  count/context/evidence match (856-863); `_get_cached_prediction` /
  `_collective_sim_predict` / `_set_cached_prediction` (865-875).

Duplicate per-plan context decode at 812 and 845 (native branch) is per-plan,
not artifact, and is negligible (~4 KB); not part of this seam.

## 4. Supported lifetime semantics (from actual callers and tests)

- The backend instance is created once per cluster at `Cluster` construction and
  lives for the process; `_prediction_cache` already has that lifetime.
- No runtime writer of `collective_local.json` exists in `frontier/`: the only
  readers are `collective_sim_cc_backend.py:810` and
  `moe_routing_workload.py:161`; writers are
  `tests/integration/step5_comm/export_costs.py`,
  `tests/integration/step5_measured_compute_fixture.py` and the unit fixtures.
- Tests require mutation visibility on a live backend:
  `tests/unit/test_collective_sim_cc_backend.py:89-102` prices, rewrites the
  artifact, and prices again on the same instance (expects `KeyError` before the
  rewrite and the new cost after). `setup_model` writes before construction, so
  load-once-in-`__init__` would fail that test; a per-call stat-keyed reuse does
  not.
- Appending a parsed-artifact memo is compatible with the whole suite: with the
  seam applied through a scratch pytest plugin (no source edit) all 45 tests in
  `tests/unit/test_collective_sim_cc_backend.py` pass, including the 27
  CPU/native fault cases and all old qualification failures; 40 distinct file
  identities produced 40 misses and 1 hit, i.e. every rewrite invalidated.
  Control run without the plugin: 45 passed.

Mutation probe (`mutation_probe.py`, scratch copy of the artifact, same backend
and plan key): after a same-size cost edit the returned cost changes
(`0.03 -> 0.05`), after an immediate same-size edit it changes again, after a
size-changing rewrite the new value (`0.07`) is returned, and with the memo off
each call reads the file. The probe also walks `network_ms`/`remainder_ms`/
`duration_ms` together because `CollectivePlanCost.__post_init__` rejects a
broken composition (`collective_plan.py:122-130`).

Identity granularity on the production filesystem: XFS under `/data` exposes
4 ms `st_mtime_ns` steps; 40 back-to-back 8 MB same-size rewrites all changed
`mtime_ns` (an 8 MB write takes longer than the tick), so no stale read was
observed. `os.stat` costs 2.2 us per call (4 per point, negligible).

## 5. Existing mechanism vs unsupported mutations

- `_prediction_cache` (`OrderedDict`, capacity `prediction_cache_size=4096`) is a
  per-plan cost cache keyed by plan content (plus the full artifact dump in the
  native branch). It cannot hold the artifact: it stores `CollectivePlanCost`
  values, is consulted only after the artifact has already been read and
  decoded, and would push per-plan costs out of LRU (the type annotation is
  stale, but the use is cost-only).
- No generic JSON/artifact loader or file-identity cache exists in `frontier/`;
  `functools.cache` appears only for pure functions. So there is no existing
  mechanism to reuse directly; the seam is new but local.
- Unsupported (would break the tests): parse once in `__init__`; module-global
  cache with no invalidation; keying on path only; caching the cost objects
  instead of the artifact (costs depend on plan).
- Parsed artifact residency is 10.1 MiB (`tracemalloc`), one entry per backend
  instance (at most the prefill and decode clusters), against a 1.19 GiB peak in
  the existing run.

## 6. Proposed owner, paths, estimate

Owner: PRICING stream (`D29-CPU-PLAN-CONSUMER` reserves <=120 gross across these
paths; `pricing_design.md:1163` already assigns this file to pricing). No CORE
mainfile edit is required for this seam.

- `frontier/cc_backend/backends/collective_sim_cc_backend.py`: two instance
  attributes (identity, parsed) in `__init__` and one `_collective_local_artifact()`
  that stats, returns the resident dict when `(st_ino, st_size, st_mtime_ns)`
  matches, otherwise reads and parses; replace 810-811 with the call. ~12 gross
  lines, no new flag, no fallback, no ignored failure (read/parse errors still
  propagate; validation unchanged).
- `tests/unit/test_collective_sim_cc_backend.py`: one regression case asserting
  a second identical call does not re-read and that a rewrite is observed.
  ~10 gross lines.

Total ~22 gross, within the 77 spare in PRICING3956 (3879/3956; effective 111).
CPU/native distinctions and every old qualification failure are preserved
because only read+parse are memoized and every check still runs on the parsed
dict; the 45-test run above is the direct evidence.

## 7. Remaining falsifiable checks and limits

1. Scale: 4 plans per point at one P point layer3; a full CLI multiplies by
   layers x positions. The per-point saving is proven, the whole-run saving is
   not, and CORE's per-query re-read (one 8 MB decode per point) is untouched by
   a backend-local seam.
2. Native branch: `json.dumps(artifact, sort_keys=True)` at line 865 costs 46 ms
   per plan for this 8 MB artifact. It is not on the measured fixture path; a
   native artifact run must be measured before claiming native-path savings.
3. Identity on another filesystem: a same-size rewrite inside one mtime tick on
   a coarse-timestamp filesystem would be missed. Not reproduced here (40/40
   attempts changed the tick); the existing tests all change size or run far
   from the tick.
4. Cross-consumer reuse: `moe_routing_workload.py:161` decodes the same file per
   query. Sharing one accessor would need CORE's call site and a shared API,
   which this scope does not authorize; report as an ownership dependency only.
5. Limits: single position execution at layer3 EP0 of one retained fixture
   artifact; cProfile arm included for call counts; construction seam and
   partial model registry inherited from `runtime_loop.py`; no full CLI, no
   GPU/network, no workflow admission.

## 8. Artifacts

- Scratch: `/data/ycfeng/tmp/d29_collective_json_scope/` - `loop.py`,
  `plugin_memo.py`, `mutation_probe.py`, runs `baseline03/04/05/06/07`,
  `memo03/04/05/06/07`, `probe01/`, `pt_memo/`, `pt_control/`.
- Baseline loop preserved: `/data/ycfeng/tmp/d29_cli_timeout_repro/runtime_loop.py`.
