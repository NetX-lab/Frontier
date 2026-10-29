# Step canonical client metrics completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive the executable CPU implementation and verification limits. |

## Overview

D20-CLIENT delivers a working Step source adapter through the existing canonical
normalize/gate CLI, not a preparation-only report. The exact shared five-file
delta from `9f58dbe^1..9f58dbe` and the shared canonical workload reader were first
reused in `ac92396`. Step reuses that reader while preserving D18's fixed pins.
The executable Step extension and its verified tests are committed in `4d62590`.

## Deliverables

- `tests/comparison/calibration/client_metrics.py`: native source-format registry,
  exact Step raw-schema mapping, repeat/population/token/prompt/clock validation.
- `tests/comparison/calibration/pd_replay.py`: unchanged shared canonical reader.
- `tests/comparison/e2e_gate/{admission,normalize,gate,run_gate}.py`: actual source/
  runtime/image evidence admission, per-repeat four-metric comparisons, pooled and
  output/request-throughput diagnostics; legacy internal-TTFT behavior is unchanged.
- `tests/comparison/e2e_gate/cases/step5_canonical.json`: frozen common workload
  pins and truthful Step runtime declaration, intentionally unqualified routing.
- `tests/comparison/e2e_gate/client_metrics.md`: executable command, native schema,
  receipt contract and pooling semantics.
- `tests/integration/step5_pd/canonical_replay.py`: use the common workload reader
  while retaining D18 pins; no frozen stage was rewritten.
- `tests/unit/test_{canonical,step5}_client_metrics.py`: both schemas, existing CLI,
  bad-round retention and missing/wrong provenance/count/repeat negatives.
- `test_report_2026-10-21_client_metrics.md`: exact commands and remaining real inputs.

## Validation

106 focused/native-consumer tests PASS on Python 3.12.13; 148 internal/legacy tests
PASS on Python 3.10.6; 77 native P/D harness/worker tests PASS on Python 3.12.13.
Sets overlap. Explicit normalize+gate shell calls for both synthetic raw schemas
produce client TTFT 2000 ms, TPOT 2 ms, E2E 4046 ms and THPT 7001.3888 token/s.
The shipped Step matrix correctly leaves all four `INSUFFICIENT_EVIDENCE`.

## Open work and limits

No real native run, trained prediction, profile admission or formal accuracy claim
is accepted. The parent still needs actual canonical native artifacts, original
stage config, actual source/image/qualification receipts and a passing existing
run-check-owner result, measured Frontier completion truth, and routing/coverage
admission. The CPU implementation did not submit jobs or modify external systems.
Historical full native Git build SHA remains unknown; no GLM v0.28 identity or
invented clean native Git manifest was substituted.
