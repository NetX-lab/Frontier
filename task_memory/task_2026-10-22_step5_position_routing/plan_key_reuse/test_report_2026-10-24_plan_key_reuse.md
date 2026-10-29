# D29 plan-key reuse verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record public-interface RED/GREEN, production-wave parity and bounded accounting. |

Task `D29-PLAN-KEY-REUSE`; session `ses_e963098ecffebqCOtdysKGY6Xe`. Candidate branch `fix/d29-plan-key` starts at `087db1bfcb3dbc2377d4032f650b29c48fb104fd`; protected snapshot baseline remains `6c7121d819e679054285bb3be5cb117a7e77bc4e`.

Verified source/test commit: `c4e06a61`. Records are committed separately.

## Implementation and tests

Only `CollectivePlan.cache_key()` retains its successfully encoded full canonical string. Encoding finishes before publication. No import/helper/API, `to_dict`, deepcopy/protocol, backend numerical-key, plan-object, numerical/proof cache or admission change.

The existing identity regression gained11 lines: exact canonical equality, repeated retained string identity, detached public context dictionary, and two repeated TypeErrors each for constructor-accepted bytes and bytearray replacements. RED: canonical content passed, then repeated identity failed (1FAIL,1.57s, exit1). GREEN:1PASS. Affected suites (`test_step5_moe_pricing`, `test_collective_sim_cc_backend`, `test_step5_contract_reuse`):124PASS/1SKIP in13.11s; baseline124PASS/1SKIP in13.09s. The same torch-dependent test is skipped because torch is unavailable; no failed/error cases.

Commands, Python3.10.6/GCC11.3/platform, raw logs/XML and SHA256 manifests are archived under `evidence/`. Tests used240s; CPU wave probes used60s, MemoryMax30G, runuser brainpp, Python-B and single BLAS threads. A local shared clone supplies the unchanged3dac53c submodule; no network/install/build/pin change.

## Actual-wave results

Accepted body SHA256 `1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555` is unchanged. External wiring pins source/fixture/output locations and initializes the real input owner and actual P/D backends for a cached manager/predictor helper. The original cold plan remains the reporting plan. Observers delegate real reads/encodings; no diagnostic key/byte adapter.

- PASS: all208 event/cost/proof/key/order values, whole JSONL and ledger bytes equal. Pricing SHA256 `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`; whole JSONL `1850a9b6a90cc7bb087b881e72d8cc00f5a669fa0be0f54bb25f3361a38d7e11`; ledger `006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf`.
- PASS:64 wrong-key negatives,3902-byte error prefix/recovery and public/pending/source/ledger snapshot isolation. New-run/standalone lifetime checks retain production behavior.
- Both arms: one collective read across binding+cold/warm, zero cold/warm reads;192 live key calls,288 numeric lookups,144 fresh proofs,64 collective prices and16 actual-point queries/wave.
- Actual encodings: cold200->8; warm192->0/wave; emission64->0. Warm pricing median0.161006936->0.119742442s (3 samples/arm). Separate emission median0.051601212->0.038155627s (15 samples/arm). Equality precedes these timing observations; profiled samples are excluded.

## Scope and limits

Git from087db1bf: source4 added/1 removed (5gross,+3); test11 added (11gross,+11). Total16gross/+14, within17/15. From D29 origin1aee0af1: plan139/140, test311/311; donors93/14/236 unchanged. Owner ceiling3956 unchanged; full union accounting remains primary-owned.

Source SHA256: plan `3158bc296ebb607c97d4305825a3b98e1111ac847e91483f8d362599597a6e06`; test `86985242ad39898819a0a3f954b0e6450ab58de0bc28ea4305484d4f8b07fb21`. Sources, original fixtures/caches and input hashes are unchanged during verification.

This is CPU-fixture pricing/emission verification, not full Simulator/full CLI/native qualification. H3 remains UNQUALIFIED/unallocated. Parent dual reviews, integration, final union accounting and broader workflow gates remain pending. See `evidence/implementation.diff` and `evidence/changed_files.txt` for review.
