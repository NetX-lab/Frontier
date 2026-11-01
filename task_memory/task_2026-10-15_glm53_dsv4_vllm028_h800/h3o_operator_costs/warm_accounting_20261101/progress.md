# WARM-ACCOUNTING progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Classified both mechanisms; CPU validation passed. |

- **completed — NIXL:** current structural bytes39,521,664 versus native packed
  bytes46,103,040 per handle; derived difference6,581,376. The helper is correct
  for its documented structural meaning, but that size is used as the wire
  pricing input. Eight-handle aggregation is not an eightfold byte multiplier.
  Per-worker post/poll/get_finished timestamps are unavailable; lifecycle timing
  residual remains unclassified rather than filled with a constant. Final byte
  evidence uses all native windows, not an inherited warm-ID filter: wave64 has
  127 windows with8 samples and one ambiguous16-sample aggregate window;
  lifecycle32 has384 windows with8 samples. All samples have identical wire bytes.
- **completed — P-step:** retained three real NONE forwards have129 EP dispatch
  kernels over43 layers. Registry default1 feeds _operator_gaps; source says
  three input tensors and three NCCL calls. Fixture count3 adds0.046836ms
  device pricing. No macro P-step host/overlap correction is supported.
- **completed — validation:** gap_verify.py and validate.py,7/7 PASS, Python3.10.6.
  Current and native segment checks are kernel-launch joins, not clean TTFT
  measurements. Initial payload import failed because memory.py imported the
  package-only type annotation; scoped AST definitions resolved the inspection
  without changing source. An ad hoc root-level predictor-config lookup failed;
  predictor config is under cluster_config. These errors did not affect final
  outputs. Several guessed native paths were absent; decisive source paths are
  recorded in the report and validation.json.
  Final unfiltered native-window validation exposed one16-sample aggregate window;
  the final check explicitly requires127 eight-sample/one16-sample wave64 windows
  instead of treating all scheduler stats as request-unique. Final validate.py:
  7/7 PASS; no duration statistic is promoted as a warm input.
- **completed — delivery:** mechanism report, summary and progress checkpoint.
  Source/comparison diff is empty. Parent mainline files show primary-session
  modifications and were not written here; old reports remain untouched.
  Native checkout is clean at pinned2cf0a691; four supplied source hashes match,
  three inspected sources have no individual worker hash. New artifacts are
  ignored by .gitignore:171; primary must explicitly force-stage selected files.

Commands already run, with ulimit-v31457280 and requested Python/env:
inspect_current.py; payload.py; segments.py p2_64 and p2_32 (independent calls
run concurrently); gap_verify.py; validate.py. Each was bounded by timeout60s.
run.sh records the exact reproducible sequence; the full run.sh itself has not
been rerun because its already-completed reads need no repeat.
