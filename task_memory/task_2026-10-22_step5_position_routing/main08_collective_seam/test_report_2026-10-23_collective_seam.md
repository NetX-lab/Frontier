# D29-WAVE-COLLECTIVE-SEAM test report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive existing observations unchanged; root checks that bytes_equal_ms already includes the read and corrects only the summary's double-counted figure. |
| 2026-10-23 | Add re-measurement on the 5e149dd8-equivalent source (CORE prepared points landed), runtime-substituted pytest checks of that source and a note about foreign runs in the scratch directory. |
| 2026-10-23 | Record the scratch micro-timing, single-variable wave differentials and pytest variant checks. |

Task ID **D29-WAVE-COLLECTIVE-SEAM**. Python 3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`,
cwd = frozen tree `verify-step5-d29-integrated-06` at `79f1a3fd` (clean before and after).
Every probe: `timeout 60s`, `RLIMIT_AS 30GiB`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
systemd `MemoryMax` was unavailable, so RLIMIT_AS bounds address space, not resident memory.
Scratch: `/data/ycfeng/tmp/d29_wave_collective_seam/`, index `results_index.json`.

## Commands

```
E="env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u"
timeout 60s python3 micro_admission.py
timeout 60s $E seam_loop.py --output run-<name> --variable <none|backend-bytes-admission|wave-plan-memo|shared-text-admission|shared-bytes-admission> --iterations 3
timeout 60s $E seam_loop.py --output run-layers02 --variable none --iterations 1 --other-layers 3 87
timeout 60s $PY -m pytest -q -p no:cacheprovider [-p admission_variant_plugin] tests/unit/test_collective_sim_cc_backend.py
```

`seam_loop.py` reuses `wave_loop.py`'s unchanged setup (lines 1–104). Each variable
replaces exactly one code region at runtime through `inspect.getsource` plus an asserted
single string replacement. Script SHA256 values are in `results_index.json`.

## Micro timing on the retained 8,033,853-byte artifact (`micro_admission.json`)

| Operation | ms/call |
| --- | --- |
| `Path.read_text()` (current backend/CORE admission) | 10.42 |
| `Path.read_text() == retained_text` | 10.19 |
| `Path.read_bytes()` | 0.80 |
| `Path.read_bytes() == retained_bytes` | 1.53 |
| `json.loads(text)` (CORE line 168 per lane) | 35.47 |
| `json.dumps(artifact, sort_keys=True)` (native key, backend line 871) | 43.91 |

## Real main08 DECODE layer61 wave (16 lanes, 64 backend calls)

Expected: payload SHA256 `b2593e723ee095488ff213e84ddcd6e85391c11f42e092a7a5ba8b77a38ffc3a`
for every arm (complete costs, proofs and plan keys of all 16 lanes). Warm = iterations 2–3.

| Arm (single variable) | Warm pricing s | Backend s | Artifact reads backend/CORE | JSON loads | Payload | Result |
| --- | --- | --- | --- | --- | --- | --- |
| none (01,02,03) | 2.716–2.954 | 0.690–0.821 | 64/16 text | 28131 | same | baseline reproduced |
| backend bytes admission | 2.191–2.367 | 0.220–0.228 | 64/16 | 28131 | same | PASS |
| shared text accessor (backend owns; CORE uses it) | 1.792–1.873 | 0.643–0.708 | 64/16 text | 28115 | same | PASS |
| shared bytes accessor | 1.149–1.212 | 0.147–0.178 | 64/16 bytes | 28115 | same | PASS |
| wave plan memo (diagnostic ceiling only) | 2.106–2.160 | 0.094–0.098 | 8/16 | 27795 | same | ceiling, changes admission granularity |

Every arm keeps one current-file read for each backend call and each CORE query, except
the diagnostic memo. Reductions are measured differentials on one wave. The per-round
values below are extrapolations, not timed rounds.

## Cross-lane and cross-layer invariance

- Dispatch plan for lane 0 vs lane 11 (`run`: `/data/ycfeng/tmp/d29_main08_residual_repro/wave-final-red02/op_traces.jsonl`):
  the only difference is `buffers.compute_context.native_padding.num_tokens_post_padded` (0 vs 128).
  Phase, sizes, tensors, groups, graph mode and `context_json` are equal. Cost fields
  other than `plan_key`/`count_receipt_json` are equal. Each phase has 2 distinct plans
  (lanes {0–10,12–15} and {11}), so the wave has 8 distinct plans across 64 calls.
- `run-layers02`: layers 3 and 87 are in layer 61's routed group (layers 3–87, 85 layers).
  All 16 lanes' plan keys, collective times and MoE times equal layer 61's.
  `run-layers01` used a self-comparison bug for collective times and is superseded.

## Pytest variant checks

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Unchanged source, `test_collective_sim_cc_backend.py` | all pass | 46 passed | PASS |
| Plugin bytes accessor + native key on admitted content, unchanged test file | evidence of contract | 45 passed, 1 failed: `test_collective_artifact_reuses_only_identical_current_text` hooks `read_text`, sees 0 reads | Unchanged test cannot pass with bytes |
| Same plugin, adapted copy hooking `read_bytes` with identical behavioral assertions | pass | 1 passed; fails `0 == 2` on unchanged source | Behavior preserved |
| Plugin text accessor + native key on admitted content, unchanged test file | all pass | 46 passed | PASS |
| Bytes plugin: `test_op_trace_utils`, `test_step5_moe_pricing`, `test_step5_measured_ffn` | same as baseline | 105 passed, 1 skipped (baseline identical) | PASS |

The adapted test asserts the same contract: two calls read twice and decode once; a same-size
rewrite with restored mtime is observed; invalid JSON raises on each call and is not
admitted; restoring the admitted content does not decode again; `FileNotFoundError` and
`PermissionError` propagate as the same objects.

## Limits

- One actual wave, CPU-fixture artifact (`evidence_kind=cpu_fixture`). The native branch
  (lines 832–882, including the line 871 key) is not reached by main08. The 43.9 ms
  dumps cost is measured on the fixture-sized artifact, and the native artifact size is unknown.
- No complete CLI, no 1024-token decode, no 2400 s acceptance. Initialization, attention,
  scheduler and trace costs are outside this loop.
- Pytest plugin variants are scratch runtime substitutions, not source changes.

## Re-measurement on the current CORE source (5e149dd8-equivalent)

The root worktree advanced during this task: CORE's prepared row/plan selection landed in
5e149dd8, and HEAD then moved to docs-only 48d34106. Every run below checks that
`git diff --quiet 5e149dd8 HEAD -- frontier tests` holds and that the working tree for
`frontier/tests` is clean, and it records the actual HEAD. The source under test was
equivalent to 5e149dd8 before and after the runs.

```
cd /data/ycfeng/Frontier/.worktrees/step5-preview
timeout 60s $E seam_loop_own.py --output run-own-<arm>-0N --variable <arm> --iterations 3
```

The arms were run interleaved (none, backend bytes, both bytes, shared accessor), twice,
then core-only bytes, none and both bytes once more. Warm means iterations 2–3.

| Arm (single variable) | Runs | Warm pricing s | Backend s | Warm wall s | Reads per wave backend/CORE | Payload |
| --- | --- | --- | --- | --- | --- | --- |
| none | 3 | 0.951–1.333 | 0.640–0.892 | 1.034–1.522 | 64/16 text | b2593e72… |
| backend reads bytes (lines 813–815) | 2 | 0.460–0.507 | 0.191–0.209 | 0.544–0.591 | 64/16 | same |
| CORE reads bytes only (line 213) | 2 | 0.906–0.949 | 0.666–0.703 | 0.990–1.035 | 64/16 | same |
| both read bytes | 3 | 0.312–0.386 | 0.137–0.173 | 0.397–0.478 | 64/16 bytes | same |
| shared backend accessor, CORE keyed on object identity | 2 | 0.311–0.435 | 0.142–0.154 | 0.396–0.517 | 64/16 bytes | same |

JSON loads per warm wave are 2096 in every arm. Each of the 12 runs reproduced the full
payload hash `b2593e723ee095488ff213e84ddcd6e85391c11f42e092a7a5ba8b77a38ffc3a`.
Per-run values and script hashes are under `head_source_equivalent` in `results_index.json`.

### Pytest with runtime substitution (`admission_variant_plugin_head.py`)

Variant: backend reads bytes and parses `json.loads(data.decode())`, the native key uses the
admitted bytes (`NATIVE_KEY=1`), and CORE reads bytes and parses `json.loads(text.decode())`.
The swapped functions are rebound to the real module globals, so tests that monkeypatch
`module.Path` or `module.json` still observe them. The first plugin draft copied the
globals; its adapted-hook result was invalid and was rerun.

| Suite | Base | Variant |
| --- | --- | --- |
| `test_collective_sim_cc_backend.py` | 46 passed | 45 passed, 1 failed: the `read_text`-hook test |
| scratch `test_admission_bytes_adapted.py` (46 originals + bytes-hook copy) | copy fails | copy passes, original text-hook test fails |
| `test_moe_routing_runtime.py` (CORE "{"-error / `saved+" "` reparse contract) | 8 passed | 8 passed |
| `test_prediction_measurement_bindings.py -k` 3 position/collective tests | 3 passed | 3 passed |
| `test_step5_measured_ffn.py` | 15 passed | 15 passed |
| `test_step5_moe_pricing.py` + `test_op_trace_utils.py` | 90 passed, 1 skipped | same |
| `test_step5_comm_cost_export.py` | 54 passed | 54 passed |
| scratch `test_native_key_scratch.py` | FAIL: 2 full-artifact serializations | PASS |

The whole `test_prediction_measurement_bindings.py` exceeds 60 s in both arms, so only
its three collective/position tests ran. The native-key scratch test proves the following.
Identical content hits the cache, and nothing serializes the full artifact. Changed content
recomputes. Restoring the earlier content hits again. A whitespace-only rewrite recomputes
once, where the old key would have hit.

Locale encoding is UTF-8 (`locale.getpreferredencoding(False)`), so `read_text()` and
`read_bytes().decode()` give the same parse for this JSON.

### Memory measurement on the retained artifact

The file is 7.66 MB. The parsed object is 10.1 MB, with a 17.8 MB parse peak (tracemalloc).
The sorted dump is 7.66 MB, and the file has 168 distinct count keys.

### Foreign runs in the scratch directory

`seam_loop_head.py` was edited at 22:19:53, and `run-head-both01`, `run-head-bytes02` and
`run-head-both02` were written at 22:19:58–22:20:13. This task did not make that edit or
those runs. The edit fixed the same bug that had made this task's first both-bytes runs fail
(an over-broad string replacement put the variable into the shared-accessor branch).
Their values agree with the own runs, but they are not used as evidence.
All evidence above comes from `seam_loop_own.py` and the `run-own-*` directories.
