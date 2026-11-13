# D29 call-local JSON controls

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive canonical parity, codec reductions and inconclusive timing; decline production allocation. |

## Outcome

No production change is allocated. Registry decoding once per key and encoding
the current context once per point preserve the tested canonical output, but
same-process pricing medians improve by only1.7–3.9ms/wave. Combined mean time is
worse with enabled GC. This does not establish a whole-wave or full-CLI speedup.

All four arms use committed frozen source `dbe19f53` plus explicit candidate-only
in-memory method substitutions.23 source bindings and the protected-body hash
remain unchanged. Codec calls are1136/704 baseline,976/704 registry,1136/544 point,
976/544 combined.288 numeric lookups,144 fresh proofs,64 collective calls and16
points remain live.19 complete-file comparisons per candidate and the protected
208-event/64wrong-key/snapshot/3902-byte-prefix checks pass.

The first archive checker failed because it included real timestamps from two
auxiliary initialization headers. Raw failure and headers are retained. The
corrected checker follows the original probe:3902 failure-body bytes identical,
profile body empty, canonical full-wave headers pinned and compared exactly.

## Deliverables

- `evidence/verification.json`: comparisons, call counts, source/raw hashes and limits.
- `evidence/{baseline,registry,point,combined}_wave.json`: completed arm receipts.
- `evidence/paired_timing.json`: all24 CPU/wall/GC samples and their raw callbacks.
- `evidence/candidate_source.diff`: exact unallocated scratch changes.
- `evidence/artifact_sha256.json`: retained artifact bindings.
- `test_report_2026-10-24_json_reuse.md`: commands, expected/actual results and disposition.
- `prepare.py`, `install_control.py`, `paired_timing.py`, `verify.py`: diagnostic scripts.

Raw logs and profiles remain at `/data/ycfeng/tmp/d29_runtime_json_reuse01`.
Only the canonical FP8 DECODE1/layer60/16EP scenario and protected negatives were
run; extended replacement/error tests, native and fullCLI were not. Independent
arm timings remain unexplained and are not reused as accepted speedup evidence.
Owner/global quotas and the6144/1024/2400s gate are unchanged.
