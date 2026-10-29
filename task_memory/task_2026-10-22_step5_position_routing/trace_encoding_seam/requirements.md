# D29 trace encoding seam requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record the parent assignment, constraints and acceptance deliverable. |

## [Original Request]

TaskID: `D29-TRACE-ENCODING-SEAM`. Read-only analysis with scratch controls; no
source allocation.

- Source frozen at `7540414341d5b81a86b18b9562d1506c0ba74285` in
  `/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields` (accepted trace +
  prepared + 8-gross cost fields; both reviews no findings; parent 208 full-byte /
  ledger / snapshot / 64 wrong-key checks pass).
- Only create an isolated records worktree at `75404143` under
  `.worktrees/d29-trace-encoding`; records in
  `task_memory/task_2026-10-22_step5_position_routing/trace_encoding_seam/`.
  Scratch `/data/ycfeng/tmp/d29_trace_encoding_seam`.
- Never edit primary/peer/source/test/retained inputs. Rank hypotheses before
  controls. Require an actual RED-capable output/call-count differential, not only
  timing.
- Goal context: output construction/serialization dominates the whole workload even
  with zero pricing (92 layers, 88 MoE layers, ~90,200 waves for unchanged
  6144/1024; 208 events/wave, ~2.9 MB). TraceStore flushes every 5000 events with
  per-row JSON writes and no sampling. Emission 51 ms, flush 24-26 ms; the
  whole-output budget must get near 12-17 ms/wave after startup (projection, not an
  impossibility proof).
- Forbidden: dropping/sampling/compressing/changing format, joining writes,
  borrowing mutable source components, caching numerical results, sharing whole
  layer-specific proofs, metadata-object/identity caches, partial manual JSON
  protocols. Existing getter deep snapshots and independent mutable trace/ledger
  maps must remain. Late JSON failure keeps the 3902 B first-row prefix and restores
  the buffer.
- Study output-preserving pure encoding first, without duplicating pricing or
  collective reads. Candidate hypothesis: long immutable string leaves
  (`count_receipt_json`, identities) are repeatedly JSON-escaped; per-flush
  exact-string-value reuse through the CPython `json.encoder.c_make_encoder`
  string-encoder callback, bounded by the existing 5000-event flush lifetime (no
  new flag, no LRU cap). Investigate versions/contracts; private API compatibility
  and callback overhead are real trade-offs. Do not force a proposal. The callback
  must not mask errors or change Unicode/escaping/floats/order/None/circular
  references. If a safer existing seam exists, measure that instead.
- Reuse the retained 208 events
  (`/data/ycfeng/tmp/d29_immutable_cost_fields_repair/candidate/iteration-0/op_traces.jsonl`,
  fixed header + 2,889,655 body bytes, body SHA256
  `18d320438ee1fb5f077b12aaa83da8055b47d8cf256d88aedcca159a5d7b819c`), core probe
  `/data/ycfeng/tmp/d29_cost_fields_parent01/run.py`, original body
  `/data/ycfeng/tmp/d29_immutable_cost_fields_repair/probe_body.py`. Preserve
  originals; derive hashed scripts.
- Timing compares the actual unmodified `json.dumps` / per-row flush against a
  one-variable control. Include a 5000-event buffer or one 88-wave point only within
  60 s / 30 GiB, with fresh parsed string objects per repeated wave; record memory
  and exact bytes.
- Every command <= 60 s under `sudo -n systemd-run --scope -p MemoryMax=30G`;
  existing py310/py312 only; no dependencies/GPU/platform/network/further agents.

## Acceptance deliverable

Exact control/source hashes; correctness/error/snapshot negatives; phase-only
timing/counters/peak memory; comparison on a real full buffer; the safe smallest
existing path with a gross estimate, or `NO_SAFE_SEAM`. No production edit. CORE299
headroom current; PRICING36; allocation not granted. Full CLI 2400 and proof
contracts unchanged. If encoding is `NO_SAFE_SEAM`, identify only the source-backed
largest remaining emission duplication, without implementation and without
speculative cross-layer numeric reuse.

## [Original Request] Parent correction before emission continuation

2026-10-24: derive the trace patch count from `git apply --numstat` (14 additions,
1 deletion, 15 gross); describe 8 supported cases equal for P1 and the deliberate
ninth lying-str RED case. Describe 5000 events consistently as a real-sized
buffer of freshly parsed repetitions of one wave, cross-layer equality unproven.
Replace timestamp-only recursive cleanup advice with an exact read-only manifest
of the 284 files and 33 directories (path/owner/size/timestamp/SHA), explicitly
record incomplete attribution. No deletion/chown. Distinguish tracked-source
cleanliness from filesystem writes. Commit corrections separately before new
`D29-EMISSION-IMMUTABLE-WORK` records. The trace candidate stays unallocated.
