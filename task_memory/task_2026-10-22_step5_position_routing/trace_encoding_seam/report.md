# D29-TRACE-ENCODING-SEAM report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Hypothesis ranking, one-variable controls, contract negatives, actual-wave and 5000-event comparisons, scratch proposal and verdict. |
| 2026-10-24 | Correct Git-derived 15-gross accounting, supported/RED case wording and repeated-wave buffer qualification; replace cleanup advice with read-only exact inventory and attribution limits. |

Task ID **D29-TRACE-ENCODING-SEAM**. Read-only analysis. Source frozen at
`7540414341d5b81a86b18b9562d1506c0ba74285`
(`/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields`, `git status` clean before
and after every run; `frontier/metrics/trace_store.py` SHA256
`07d7cbbb081829f558b6a4c101201baf112e5387e2045ee244436853fb463a0d`). Scratch
`/data/ycfeng/tmp/d29_trace_encoding_seam`. No production, test or retained-input
edit. Exact commands and hashes are in
`test_report_2026-10-24_trace_encoding_seam.md`.

## Verdict

**Conditional SAFE_SEAM.** The flush can reuse the stdlib JSON string escape of
each exact string value within one flush through the existing CPython
`json.encoder.c_make_encoder` string-encoder argument. Every container, number,
separator, key-order, `None`, error and circular-reference decision stays in the
unchanged C encoder, which still runs once per row with the `json.dumps` defaults.
The output is byte-identical on every timed run, including a real-sized
5000-event buffer of freshly parsed repetitions of one wave. Cross-layer equality
was not established. Error text, the 3902-byte late-failure prefix and buffer
restoration are identical.

- Gain: the flush of one wave falls by **5.3-6.1 ms** on freshly parsed events
  (about 23-25 %) and by **8.9-9.4 ms** on real `record_ep_wave` events (about
  33 %). The real-sized 5000-event repeated-wave buffer falls from **535.7 ms to
  375.3 ms** (-30 %), or 22.3 -> 15.6 ms per 208-event wave; cross-layer equality
  is unproven.
- Cost: one file, `frontier/metrics/trace_store.py`, **15 gross lines** (14+, 1-,
  net +13), measured by `git apply --numstat` on the actual artifact,
  `evidence/proposal.diff`. No flag, no cache outside one `flush()` call, no
  format change, and no change to writes or to the emission path.
- Preconditions that the reviewer must accept:
  1. The proposal uses `json.encoder.c_make_encoder` and
     `c_encode_basestring_ascii`, which are undocumented module attributes
     (private API). The 9-argument contract was verified on CPython 3.10.6 (the
     C encoder returns a `list`) and 3.12.13 (it returns a `tuple`); later
     versions are untested. Without the `_json` accelerator,
     `c_make_encoder is None` and the first flush would raise. No supported
     environment lacks it, so the proposal adds no fallback.
  2. Reuse is keyed by string equality. It is exact for `str` and for every str
     subclass that keeps `str.__eq__`/`__hash__`. A pathological subclass that
     overrides them inconsistently changes the output (RED negative below). All
     208 real events contain only exact `str` keys and values. The only str
     subclasses in `frontier/` (`MeasurementType`, `SequenceMixerType`, both
     `(str, Enum)`) override neither method, and in their MRO `str` precedes
     `Enum`.
  3. An exact-type guard removes the gain (see H5), so the proposal has none.
- **Budget statement: this seam does not close the gap.** With pricing at zero,
  the whole-output cost per wave is now emission ~50 ms + flush ~24-29 ms. With the
  seam it is about 68-70 ms per wave. The target is 12-17 ms. Emission alone
  (49.9-50.5 ms, flush excluded) is about 3x the whole budget, so even a
  zero-cost flush cannot meet it. Over 90,200 waves the seam saves about
  0.48-0.85 ks per mode. The zero-pricing projection is still about
  6.1-6.3 ks of waves plus 0.84-1.29 ks of startup, or 7.0-7.6 ks (2.9-3.2x the
  2400 s cap). This is a current-implementation projection, not an impossibility
  proof.

## 1. Hypotheses ranked before any control

These come from the byte/work composition (`evidence/composition.json`) and the
phase-only split (`evidence/phases-py310.json`; medians over 15 runs, each on
freshly parsed events).

| Fact | Value |
| --- | --- |
| body | 2,889,655 B, 208 rows; parsed rows re-encode byte-identically |
| string values | 18,112 occurrences, 459 distinct, 2,149,828 encoded B (74.4 %); escaping adds 351,360 B |
| strings >= 256 chars | 19 distinct values, 256 occurrences, 1,946,896 encoded B (67.4 % of body) |
| largest string paths | `collective_cost.count_receipt_json` 1,086,408 B; `collective_cost.plan_key` 598,656 B; `collective_plan.context_json` 156,352 B; `collective_plan.buffer_json` 105,480 B |
| keys | 39,136 occurrences, 155 distinct, 491,552 B (17.0 %) |
| numbers / bool / null / structure | ints 33,235 B; floats 22,240 B; bool/null 3,088 B; separators/brackets/newlines 189,712 B |
| actual flush | 23.52 ms |
| row dicts | 0.79 ms |
| `json.dumps` of all 208 rows | 22.19 ms |
| text-mode write of pre-encoded rows | 2.95 ms |
| standalone escape of the 256 long strings | 6.74 ms (Python list loop; over-states in-encoder cost) |
| hash of 256 fresh long strings | 2.50 ms |

Ranking:

1. **H1: repeated escaping of identical long string values within one flush.**
   Two thirds of the bytes are 19 values repeated about 13x each, all JSON text
   that needs quote escaping. This is the largest separable term (~6.7 ms) and is
   testable by replacing only the escaper.
2. **H2: per-row Python wrapper overhead** of `json.dumps` ->
   `JSONEncoder.encode/iterencode` over 208 rows. It looked small, but its upper
   bound was unknown.
3. **H3: file write**, 2.9 ms. Joining writes is forbidden, so this is out of
   scope.
4. **H4: row-dict construction**, 0.8 ms. Negligible.
5. **H5: callback overhead.** A non-C escaper turns 57,248 C fast-path escapes
   (keys and values) into Python-level calls. This could cancel H1, so it is a
   real trade-off.

## 2. Controls and results

`A` is the one-variable control. It runs the actual unmodified `TraceStore.flush`
and patches only the module global `json.encoder.encode_basestring_ascii`, which
`iterencode` reads per call, with a per-flush `EscapedStrings().__getitem__`
(a `dict` whose `__missing__` calls the stdlib C escaper). `P1` is the
production-shaped variant derived textually from the actual `flush` source
(`evidence/derived_flush_P1.py.txt`).

| Variant (py310, fresh-parse wave unless noted) | Median ms | vs actual | Hypothesis verdict |
| --- | --- | --- | --- |
| actual flush | 23.46 / 24.03 | - | baseline |
| A (only the escaper replaced) | 18.14 / 18.31 | **-5.33 / -5.72** | H1 confirmed |
| P1_c (bypass `json.dumps` wrapper, C escaper kept) | 23.59 | -0.44 | H2 small |
| A_guard / P1_guard (Python `type(v) is str` guard) | 23.63 / 23.44 | -0.40 / -0.59 | H5: Python callback cancels H1 |
| P1 (proposal shape) | 17.96 | **-6.07** | combined |
| real emitted wave, actual vs P1 (11 reps) | 26.60 -> 17.70 | **-8.91** | holds on actual events |
| real emitted wave, actual vs proposal module | 28.97 -> 19.57 | **-9.41** | proposal file itself |
| Real-sized 5000-event buffer of freshly parsed repetitions of one wave, A (3 reps) | 535.72 -> 375.32 | **-160.41 (-29.9 %)** | cross-layer equality unproven |
| Same real-sized repeated-wave buffer, P1 (3 reps) | 532.85 -> 370.99 | **-161.86** | cross-layer equality unproven |
| stdlib replica, py3.12.13 (11 reps) | 23.25 -> 16.46 | -6.79 | contract holds on 3.12 |
| stdlib replica, py3.10.6 reruns (21 reps) | 26.3/27.2 -> 20.2/21.5 | -6.09 / -5.71 | replica only |

Call-count differential on one wave (`contracts-py310.json`): escaper calls
57,248 in both variants. Escapes performed fall from 57,248 to **597**, and
escaped input characters fall from 2,175,524 to **179,538** (-91.7 %).

Memory: on the 5000-event flush, the tracemalloc peak is 382,366 B for actual and
620,077 B for the control (+237,711 B). The map holds 597 entries,
`sys.getsizeof` 269,273 B. Process peak RSS (`ru_maxrss`) reached 472,924 KiB after
the first buffer pair and did not grow afterwards. It is a whole-process maximum,
so it cannot separate the variants, and it is dominated by the 5000 freshly
parsed events held in memory.

Conditions on the timings:

- Every buffer wave is parsed again, so no string object is shared between
  waves.
- The buffer repeats equal content, so cross-wave value reuse happens there:
  6.67 ms per wave versus 5.3 ms for a single wave. A real buffer covers about 24
  different layers. The long strings carry no layer token, but cross-layer
  equality is not proven from one wave, so **the single-wave figure, which needs
  only intra-wave repetition (about 15x per value), is the conservative
  estimate.**
- The real emitted wave has 256 long-string occurrences that point to only 32
  distinct objects (19 distinct values), so their hashes are cached. That partly
  explains the larger real-event gain.
- The first py310 replica run measured `dumps` at 71.3 ms. This outlier was not
  reproduced: reruns measured 26.3 and 27.2 ms, and an isolated probe measured
  20.7 ms.

## 3. Correctness, error and snapshot negatives

Eight supported contract cases (`contracts-py310.json`) are equal to the actual
flush for P1 on bytes written, exception text and restored buffer identity/order.
The ninth case is a deliberately lying str subclass and goes RED for P1:

| Case | Actual outcome | A | A_guard | P1 | P1_c | P1_guard |
| --- | --- | --- | --- | --- | --- | --- |
| late JSON failure | `TypeError: Object of type object is not JSON serializable`, 3902 B prefix, buffer restored in order | = | = | = | = | = |
| circular reference | `ValueError: Circular reference detected`, 3902 B prefix | = | = | = | = | = |
| unsupported key | `TypeError: keys must be str, int, float, bool or None, not tuple` | = | = | = | = | = |
| Unicode (surrogate, controls, U+2028, emoji, quotes/backslash), 3 rows | 705 B | = | = | = | = | = |
| NaN/inf/-0.0/1e300/1/3 floats | 174 B | = | = | = | = | = |
| non-str keys and order | 173 B | = | = | = | = | = |
| top-level `None` fields omitted | 86 B | = | = | = | = | = |
| str-Enum keys and values next to equal plain strings | 298 B | = | = | = | = | = |
| **lying str subclass (`__eq__` always True)** | 247 B | **RED** | = | **RED** | = | = |

- Mutating pending metadata between log and flush (two flushes, one shared
  `meta`) gives output equal to actual for every variant. A deliberately wrong
  meta-identity cache is detected as **RED**.
- The accepted, unchanged probe body (`probe_body.py`, SHA256 `1cabc04f...`) was
  rerun with only the bound `TraceStore` class swapped to the scratch proposal
  (`evidence/probe-body-with-proposal.json`). All checks pass:
  - 208 events, body SHA256 `18d32043...`;
  - whole JSONL SHA256 `1850a9b6...`, identical to the parent run;
  - ledger 2,309,545 B, SHA256 `006dc7e8...`, identical;
  - getter/pending snapshot isolation, independent trace/ledger maps, 64
    wrong-plan-key negatives and the 3902 B late-JSON prefix.

  Emission median is 49.86 ms (flush excluded; the emission path is unchanged).
- Leaf types of the real emitted wave are only `key:str` (39,552, including
  top-level fields later filtered as `None`) and `value:str` (18,112).

## 4. Proposal shape (scratch only, not applied)

`evidence/proposal.diff` against `frontier/metrics/trace_store.py`:

- Import `c_encode_basestring_ascii` and `c_make_encoder` from `json.encoder`.
- Add a module-level `EscapedStrings(dict)` with `__missing__` calling the stdlib
  escaper.
- In `flush()`, before the existing `try`, create
  `escaped = EscapedStrings()` and `default = json.JSONEncoder().default`.
- Replace `f.write(json.dumps(row) + "\n")` with a per-row
  `c_make_encoder({}, default, escaped.__getitem__, None, ": ", ", ", False, False, True)`
  and `"".join(encode(row, 0))`.

What stays unchanged: one encoder with a fresh `markers` dict per row, as in
`json.dumps`; the per-row write boundary; the exception handler and buffer
restoration. The map lives for exactly one `flush()`, so the existing
5000-event limit bounds it without a new cap.

`git apply --numstat evidence/proposal.diff`: 14 additions, 1 deletion, **15 gross,
net +13**. The earlier 12-gross count incorrectly omitted blank added lines.
**No allocation is granted by this report**; CORE299 headroom and
PRICING36 are unchanged.

## 5. Remaining gap (no implementation)

Emission is the term that binds the budget. The accepted probe counters at
`75404143` (`d29_cost_fields_parent01/run.log`) show, per wave, 21,472 `deepcopy`
calls, 208 `asdict`, 192 `to_dict`, 64 `cache_key` and 208
`compute_op_trace_meta`, for a 50.5 ms emission median. The requirements keep the
getter deep snapshots and the independent mutable trace/ledger maps, so they are
not a free target. Because the encoding verdict is not `NO_SAFE_SEAM`, the
conditional emission-duplication study was not performed here.

## Limits

- CPU timings are sequential on a shared 64-core host (load average about 5).
  Medians, minimums and raw samples are retained.
- The py3.12 evidence comes from a stdlib replica of the per-row loop, because the
  Frontier import needs `plotly`, which is absent in the py312 environments.
- The full CLI, strict77/full-unit, GPU and native paths were not run. Five
  root-owned, hash-identical 69 MB buffer copies remain in scratch (see
  `evidence/buffer-body-hashes.txt`); removing them needs `sudo rm`, which was not
  authorized.
- Filesystem side effect differs from tracked source changes: the earlier runs
  reported 284 root-owned git-ignored `.pyc` files and 33 `__pycache__` directories
  in the source worktree. Tracked source stayed unchanged. The read-only manifest
  `evidence/pyc_manifest_2026-10-24.json` now lists exactly 284 files and their 33
  parent directories with absolute path, owner/uid, size, mode, mtime, ctime and
  SHA256. Directory SHA256 is explicitly the digest of sorted immediate entry
  names, not file content. File mtimes range from 2026-10-23 18:06:20.662846 UTC to
  18:13:08.374932 UTC (2026-10-24 local time).
  The matching counts/timestamps are consistent with attribution to these runs,
  but no pre-run pathname/hash manifest exists, so exclusive provenance cannot
  be proven. The unsafe timestamp-only recursive cleanup advice is withdrawn.
  No deletion or ownership change is performed or recommended. New runs use
  both `python -B` and `PYTHONDONTWRITEBYTECODE=1`.
