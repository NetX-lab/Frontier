# D29-MAIN08-RESIDUAL-REPRO requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record the delegated read-only main08 residual diagnosis and its limits. |

## [Original Request]

Task ID: **D29-MAIN08-RESIDUAL-REPRO**. Diagnose remaining full CLI timeout
after the three reviewed repairs on frozen source
`79f1a3fd35d42b751cd1520c764ddac717b923d9`, submodule `3dac53c`.
Read current [requirements](../requirements.md), [plan](../plan.md), GLOSSARY
and related ADR; use the diagnosing-bugs workflow. Establish a seconds-scale,
deterministic RED-capable loop at the actual full layer/EP-wave or trace-emission
boundary using real main08 decode frames. Measure before hypotheses, test one
variable at a time, and preserve costs, proofs and full plan identity.

Main08 offline is immutable and timed out at 2400.175 seconds. Online was active
when delegated: do not inspect, poll, attach, instrument or stop it. Do not
substitute the earlier initial-P or D23 state, run a complete CLI, scan complete
large CSVs, edit production/tests, weaken checks, or change retained inputs,
outputs, timeouts, hashes or serialization identity. No GPU/platform/network or
subagents. Writes are restricted to the new diagnostic scratch and this new
task-record subdirectory. Memory-heavy inspections use timeout60s and the
requested 30GiB bound; disclose unavailable systemd scope rather than inventing
a successful control.

## Acceptance boundary

Deliver the actual short-loop command/verdict, timings and correct multiplicity,
symptom correspondence, decisive source lines, single-variable evidence,
artifact paths, limitations and a bounded proposal only where causally supported.
This diagnosis does not accept the original 6144/1024 CLI or native workflow.
Primary owns the next repair authorization and final regression acceptance.
