# D29 current-query JSON reuse investigation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Add one60s alternating timing-only process after materially inconsistent independent-arm timings. |
| 2026-10-24 | Record the bounded follow-up to accepted named querying and YC's continue request. |

## Requirements and authority

[Original Request] YC: `continue` after the named-query short-regression checkpoint.
The parent requirements' D29-NAMED-MODEL-QUERY decision preauthorizes analogous
in-scope performance work. This investigation has no production allocation.

Preserve current-model/actual-point checks, dynamic family/model/row/identity
selection,288 numerical calls,144 fresh proofs,64 collective calls,16 points,
complete208-event costs/keys/proofs/order/JSONL/ledger,64 wrong-key rejections,
mutable snapshot isolation and3902-byte failure prefix. Keep the6144/1024/2400s
workflow gate. Do not cache numerical values, whole proofs or model views.

Use immutable source dbe19f53 and the accepted wave/fixture/protected body.
Candidate changes are explicit in-memory function replacements from scratch
copies, not committed production methods. No original source/input/helper is
edited. Raw files go only to `/data/ycfeng/tmp/d29_runtime_json_reuse01`.

Ceiling: one baseline and at most three candidate wave invocations, each
timeout60s/MemoryMax30G/single-thread math/python-B/runuser brainpp. Each uses the
existing five-wave timing/profile loop and complete protected output checks.
The three candidates isolate registry decoding, point encoding and their union.
No initialization, CSV scan, full CLI, GPU, platform or network run is included.

Measured timing discrepancy: baseline warm207/243/203ms, registry86/87/87ms,
point87/87/88ms and combined162/162/149ms despite additive160-load/160-dump
reductions and complete output parity. These are not accepted speedup numbers.
The primary extends only diagnostic timing by one process, timeout60s/30G,
one cold wave plus24 order-balanced warm waves across the same four methods.
Keep GC enabled, record its callbacks plus CPU/wall times, and verify the full
pricing payload every time. Do not rerun the protected output suite or claim
that moving a GC pause outside pricing improves whole CLI time.

Allocation requires meaningful observed savings plus preserved predicates.
Small/noisy savings do not justify an interface expansion. CORE has143 lines
remaining; PRICING has2, and owner/global caps remain binding.
