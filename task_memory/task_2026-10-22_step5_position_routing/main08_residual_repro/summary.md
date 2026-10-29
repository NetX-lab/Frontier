# D29-MAIN08-RESIDUAL-REPRO summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive the completed read-only actual decode-wave diagnosis. |

## Outcome

A deterministic seconds-scale RED exists at the actual main08 DECODE
layer61/16-EP wave after all three earlier fixes. Frozen79f1a3fd/3dac remains
clean. Original wave cold/warm pricing3.387/2.735s, total3.597/2.926s,208 events;
warm28003 JSON parses and80 unchanged-artifact reads. It is not the old initial
P or D23 state. Main08 offline completed four decode batches and timed out.
Online was never accessed.

Actual tail comparison **PASS**:165 observed pairs match complete duration,
model/count proofs, native source/context and full collective plan/cost. Payload
SHA256 `b2593e723ee095488ff213e84ddcd6e85391c11f42e092a7a5ba8b77a38ffc3a`.
43 generated pairs lie outside the bounded actual tail sample.

## Causally supported residual costs

- Same-text artifact-read differential: warm2.735 ->1.887s,80 ->0 reads.
- Same exact JSON text decode differential: warm1.321s,0 decodes,80 reads.
- CORE plan-key decode only: warm2.188s,28003 ->17251 decodes; same full payload.
  The profile shows168 unique keys decoded21504 times across two waves.
- Captured trace replay:598 events/8.35MB, original profiled1.202/1.175s;
  shallow-field diagnostic0.081/0.076s, bytes identical. This is a smaller
  hotspot and not proof that arbitrary dataclass metadata may be shallow-copied.

## Bounded repair proposal, not authorization

Primary should consider a CORE predictor-local prepared row/plan-key selection
seam in `moe_routing_workload.py:144–190`, using existing consumer-local lifetime.
Retain per-layer group selection, actual source-frame/ep/live counts, family/
native binding, full context/buffer equality, exact duration/count proofs and
all negative paths. No global cache or prediction bypass. Estimate60–100 gross
source/test lines for a narrow prepared-plan differential, subject to root path
allocation and current ceilings. A full row/index reuse adds scope and requires
independent tests for cross-source/model/family separation; do not assume it fits.

Existing PRICING backend `collective_sim_cc_backend.py:812–817` owns64 of80
reads per wave. CORE's query owns16. Any I/O reuse needs an explicitly supported
lifetime/change-detection contract; do not remove current exact-text detection
or invent a fingerprint. Delegate that decision separately only after root
authorization. TraceStore is another separately owned seam; defer until its
serialization/mutation semantics and allocation are explicit.

The measured single-variable results do not establish a sufficient repair for
1024 decode tokens or2400s.88 routed layers×2.735s gives240.7s as extrapolation,
roughly matching observed~220s decode rounds, not a timed whole-round benchmark.
Initialization (14–18 minutes from root log evidence) remains a separate cost
outside this decode-wave loop.

## Deliverables and next actions

- This directory: requirements, plan, progress, test report and checkpoint.
- Scratch `/data/ycfeng/tmp/d29_main08_residual_repro/`: final `wave_loop.py`,
  `trace_loop.py`, `sample_offline.py`, `compare_actual_wave.py`; result/profile/
comparison JSON, raw logs, samples, source/script hashes and failed setup logs.
- Primary independently reproduces final RED with a fresh output path, allocates
  any narrow source repair, reviews it, then owns full frozen-source acceptance.
- No code/tests/inputs/cache/timeout/retained outputs were changed. No complete
  CLI, platform/network/GPU or active-online operation occurred. No agents used.
- Requested systemd MemoryMax scope was unavailable; probes used timeout60s+
  RLIMIT_AS30GiB, not an equivalent cgroup resident-memory guarantee.
