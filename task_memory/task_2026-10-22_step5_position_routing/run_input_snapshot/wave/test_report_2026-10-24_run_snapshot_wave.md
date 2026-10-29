# D29 run snapshot: actual-wave verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Verify real input ownership, complete replay fidelity and source freeze. |

**Task:** `D29-RUN-SNAPSHOT-WAVE`

**Session:** `ses_e963098ecffebqCOtdysKGY6Xe`

## Result

**PASS**, limited to the actual layer60/16-EP replay and stated constructor evidence. Candidate source: `/data/ycfeng/Frontier/.worktrees/step5-d29-input-snapshot`, actual HEAD `ecccf83fa6f2c24a14fa5397ef567fbb32b06bef` with uncommitted snapshot changes. The withdrawn `b3178400` pointer was not used. Baseline: frozen byte/cost-fields worktree `step5-d29-artifact-bytes` at `56529fcd`.

| Invariant | Result |
| --- | --- |
| All208 event costs, actual-layer proofs, full keys and ordering | PASS, complete payload bytes equal |
| Whole JSONL and ledger bytes | PASS, complete files equal |
| Getter/source/pending-event/ledger snapshot isolation | PASS, unchanged protected probe |
|64 wrong-key negatives and exact error | PASS |
|3902-byte late-JSON error prefix and restored buffer | PASS |
| Real shared P/D input owner and one successful main-run file read | PASS, binding1 + cold0 + warm0 + profiled-wave0 |
| New-run JSON/UTF8 errors are uncached; later valid retry | PASS, repeated failures reread; first success binds |
| Standalone backend public call shape | PASS, two calls/one read |
| Numeric lookup/proof work remains live | PASS,288 lookups/144 fresh operator proofs/64 collective prices per wave |
| Source hashes and original fixture/cache preserved | PASS,21 candidate/20 baseline source hashes unchanged |

Warm operation counts are identical: `loads1520`, `dumps992`, `cache_key192`, actual-point queries16. Receipt preparation is the candidate's real four-tuple, not the old five-tuple.

## Executed timing and fidelity

Unprofiled warm samples (seconds): baseline **0.341421,0.339053,0.340867**; candidate **0.162715,0.162417,0.165457**. Medians **0.340867→0.162715**. Candidate binding **0.089499s**; cold pricing **1.080278s** excluding binding, baseline cold **1.154228s**. Setup, output verification/emission, profiling and lifecycle negatives are excluded from warm timing. Baseline reads80/wave (400 over five waves); candidate reads1 across binding/five waves.

The original full-wave assignment and accepted cost-fields body are unchanged. External `fixture_wiring.py` changes only source pins, scratch paths/config input location and explicit owner binding. Its SHA256 is `bffe25039ecb0aae94b5eab905aff0c4cecb150eabbfcf4912aa6d5f9e7b95f1`; protected body SHA256 `1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`.

The cached manager/predictor use the accepted `__new__` helper. Initialize the real `CollectiveProfileInputs` field explicitly, then bind actual P/D backends through public methods. No diagnostic fixed-file adapter, prediction replacement or numerical/result/proof cache. Mid-run file changes touch scratch copies only. Actual role-constructor wiring is separate inherited82PASS evidence (`d29_snapshot_green01.xml`, named test at `tests/unit/test_moe_routing_runtime.py:13`), not full Simulator validation.

Initial attempt01 failed output equality in both arms because the adapter selected the last warm reporting plan; request IDs1 became1409. Attempt02 restores the accepted driver's cold-plan selection. No source or protected-body edit; all hashes match across attempts.

## Durable evidence and limits

Exact protected commands, all source/script/result hashes and raw artifact paths: `evidence/evidence.json`. Executed under `sudo systemd-run`, `runuser -u brainpp`, timeout60s/MemoryMax30G, Python3.10.6 `-B`, BLAS threads1. Scratch: `/data/ycfeng/tmp/d29_run_snapshot_wave_20261024_02`.

Payload SHA256 `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`; whole JSONL `1850a9b6a90cc7bb087b881e72d8cc00f5a669fa0be0f54bb25f3361a38d7e11`; ledger `006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf`.

No production/test/root-worktree edits, training/full-fixture content scan, full CLI, GPU or native qualification. H3 config candidate remains UNQUALIFIED and unallocated. Parent retains integration, independent review, allocation audit and full-workflow acceptance.
