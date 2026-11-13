# D29-MAIN08-RESIDUAL-REPRO progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Complete actual decode-wave/trace loops, causal probes and partial-artifact matching. |

Task ID: **D29-MAIN08-RESIDUAL-REPRO**. Status: **completed diagnosis**;
no implementation or complete-workload acceptance claimed.

## completed — observations and tight loops

- Frozen79f1a3fd/3dac stayed clean. Offline2400.175252s timeout confirmed;
  scheduler has52 records, last DECODE iteration4, only four completed decode
  batch-end records. The trace is1447809841 bytes; only bounded head/tail samples
  were read. No online access occurred.
- Recorded DECODE batch spans are214.734s,221.392s,220.158s and224.602s.
  This is new decode progress, not the old initial-P/D23 state and not completion.
- Real TraceStore replay:598 captured DECODE events,8353308 bytes per iteration,
  profiled1.202/1.175s. Actual `asdict`/to_dict consumes2.208/2.376s across both
  calls;453008 deepcopy calls. Byte equality is checked, not only exit success.
- Actual wave loop uses main08 source position6147, layer61, FULL live1/idle0,
  gathered2, all16 participants, real paired-cache registry and Cluster backend.
  Public stage calls, per-point admission, selected bindings, exact duration/count
  proofs and complete plans remain live.208 events are generated per wave.

## completed — setup diagnosis and correction

The initial probe had a CLI sys.argv slice error; it failed before parse and
never entered prediction. An older bundle's recorded hashes mismatched seven
current pickles; strict verification rejected it. A compatible existing bundle
was selected only after all retained pickle hashes matched, with no cache edit.

The first wave setup inherited test two-stream pricing and selected KERNEL_ONLY.
An actual-artifact comparison found compute proof modality/model-hash mismatch,
even though durations and plans matched. Those runs are superseded and retained
only as setup evidence. Final setup uses both verified bundles, actual CLI
predictor config, the existing constructor's forward-start lifetime marker and
the existing two-stream classifier (False because skip_cpu_overhead_modeling).
It removes the inherited calibration overrides. The actual public call selects
CUDA_EVENT FFN proofs just as main08 does; attention remains excluded at this
post-attention wave seam. Actual model/count/plan hashes are now matched.

## completed — causality and final RED

Four hypotheses were presented before variable tests: repeated JSON parsing;
repeated artifact text I/O; trace recursive copying; residual estimator dominance.
Final original cold/warm pricing is3.387/2.735s; total wave3.597/2.926s.
Each warm wave performs28003 JSON parses and80 reads of the same artifact.
Only168 unique plan-key texts account for21504 parses across two profiled waves.

Single variable results with actual binding:
- Retain exact artifact text only: warm pricing1.887s,0 reads,28003 parses.
- Retain exact JSON text decode only: warm pricing1.321s,0 decodes,80 reads.
- Restrict text decode reuse to CORE plan-key loop line178: warm2.188s,
  17251 parses,80 reads. Full key/cost/proof payload is unchanged.
- Captured trace `to_dict` shallow-field diagnostic:0.081/0.076s,0 deepcopy,
  exactly identical8353308-byte serialized outputs. This does not establish
  correctness for other nested dataclass metadata or permit a production change.

All actual-binding wave variants preserve complete payload SHA256
`b2593e723ee095488ff213e84ddcd6e85391c11f42e092a7a5ba8b77a38ffc3a`.
Final original wave matches165 captured layer61 operator pairs: duration,
compute/count model hashes, native context, immutable source, collective plan
and cost all equal; no mismatch. The remaining43 pairs are outside the8MiB tail
sample, not independently observed in that sample.

Final counting boundary is before the diagnostic payload hash verification.
Earlier wave-final-red01/actual02 counted128 additional JSON parses from
out-of-timing plan.cache_key verification. Final red02/read02/json02/plan02
exclude them without changing the business path, timing or payload; all earlier
artifacts remain retained. The diagnostic source receipt is updated separately
as final_receipt.json, preserving the earlier script-hash receipt.

## limits and next authorized frontier

Requested `systemd-run --user --scope -p MemoryMax=30G true` failed. Every
inspection/probe is bounded by timeout60s plus RLIMIT_AS30GiB; this is not a
cgroup resident-memory guarantee. No source/test/cache/retained-run edit, full
CSV scan, active-online operation, full CLI or platform work occurred.

The scheduler executes88 routed layers with16 participant calls each; one-wave
cost multiplied by88 is an extrapolation, not a timed full decode loop. At warm
2.735s it gives240.7s versus observed completed-round~220s; agreement in scale
supports relevance but cannot prove all layers/remaining1020 tokens or2400s PASS.

Primary next actions:
1. Independently reproduce the final bounded wave with fresh scratch output.
2. Allocate a narrow CORE actual row/plan preparation reuse seam if accepted;
   preserve model/layer/source/point selection and every admission/identity check.
3. Coordinate artifact I/O with PRICING separately; do not relax exact-text
   change detection. Keep trace-copy semantics investigation separate.
4. Only after scoped implementation/review, primary owns full frozen-source CLI.
