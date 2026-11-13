# JSON reuse investigation progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete24 paired samples and19 per-arm byte comparisons; retain auxiliary header timestamps and decline production allocation. |
| 2026-10-24 | Four full-output arms pass; encoding/decoding counts change as predicted, but wall-time attribution remains unqualified. |
| 2026-10-24 | Read the accepted profile, bind two call-local candidates and prepare bounded scratch controls. |

- completed: inspected source and retained named-query profile. Source remains
  unchanged; profile counters identify272 registry-key decodes and repeated
  canonical context encoding. An initial profile lookup under the records
  directory failed; the report identifies the retained raw path under
  `/data/ycfeng/tmp/d29_named_model_query01`, which was then read successfully.
- completed: scratch-only source copies and explicit method installation;23
  source bindings and protected-body hash verified. All four arms complete
  208-event/full JSONL/ledger/64wrong-key/3902-byte-prefix checks. The preserved
  failure-injection ERROR log is expected, not a failed run.
- completed:24 balanced same-process samples, six per arm, with GC enabled and
  CPU/wall clocks retained. Median pricing90.982/89.274/87.092/88.348ms for
  baseline/registry/point/combined; mean101.850/91.365/104.933/124.540ms. No sample
  was excluded and no GC time subtracted. First independent-arm207/87/87/162ms
  medians are not accepted speedup evidence. Remaining GC-sensitive mean changes
  prevent a whole-wave or CLI improvement claim.
- completed: final source23/body-hash checks,19 per-candidate exact file comparisons
  (15 complete wave JSONLs, two pinned auxiliary headers, ledger, pricing payload),
  protected negatives and count checks. Registry loads1136→976; point dumps704→544.
  Numerical288/fresh-proof144/collective64/point16 calls remain unchanged.
- completed/iteration: the first archive checker incorrectly required equality
  of two real-clock auxiliary headers. `verify.log` retains its AssertionError.
  Source `probe_body.py:123-150` establishes that the profile store is unflushed
  and the failure assertion compares the body, not initialization metadata.
  Raw headers/timestamps are retained; all four3902-byte failure bodies match,
  and all profile bodies are empty. Corrected `verify02.log` passes without a
  probe rerun or any original artifact mutation.
- completed/disposition: NO_PRODUCTION_ALLOCATION. Paired median savings are
  only1.7–3.9ms; the point control expands a private helper signature and neither
  control resolves the workflow gap. Extended model replacement/error tests were
  not run. Source quotas,9565source/9579global and all acceptance gates stay unchanged.
- completed/record hygiene: five diagnostic scripts parse and all seven retained
  evidence hashes match. Full staged `git diff --check` exits2 only on the two
  unified-diff blank-context markers at candidate_source.diff:29,47. The raw diff
  is preserved; the remaining scoped staged changes pass the whitespace check.
