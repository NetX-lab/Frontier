# P0 useful-token accounting and remaining G4 coverage

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Reconcile useful scheduled tokens, retain missing dummy/rank evidence and identify actual MRv2 profile-binding requirements. |

## Inputs and method

Parent local analysis of completed P0 job `exp-1021-205523-285491`; no platform operation or new workload. The existing99-row `generation03_runtime_rows.csv` is the input, with prior native request verification in `generation03_receipt.json`. New derived receipt: `generation03_useful_accounting.json`. It binds the CSV, prior receipt, frozen native config and four current source files by SHA256. All four native source hashes match the frozen config.

Native `compilation/cuda_graph.py:65–85,120–124` aggregates observed records per logging interval then resets, so interval counts can be summed. MRv2 `v1/worker/gpu/cudagraph_utils.py:83–91` defines the unpadded and padded fields. Each row also satisfies padded minus unpadded equals paddings.

From repository root, reproduce the arithmetic with host Python3.10.6; this command performs no native execution:

```bash
python3 - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path
p = Path('task_memory/task_2026-10-21_step5_priority_runs/native_replay/generation03_runtime_rows.csv')
totals = defaultdict(int)
for row in csv.DictReader(p.open()):
    assert int(row['padded']) - int(row['unpadded']) == int(row['paddings'])
    totals[row['role']] += int(row['count']) * int(row['unpadded'])
for role, expected in [('prefill', 24 * 6128), ('decode', 24 * (16 + 1023))]:
    observed = totals[role]
    print(role, observed, expected, observed - expected)
    assert observed == expected
PY
```

The24 requests include16 warmups and8 concurrent requests. P computes6128 prompt tokens per request. D computes16 prompt-tail tokens and1023 subsequent decode inputs to produce1024 output tokens. These are scheduled-token totals, not E2E throughput numerators or elapsed times.

## Results

| Role/lane/mode | Logged forwards | Unpadded scheduled tokens | Padded tokens |
| --- | ---: | ---: | ---: |
| P0 NONE |12|73536|73536|
| P1 NONE |12|73536|73536|
| D0 FULL |2224|12208|12483|
| D0 PIECEWISE |20|260|448|
| D1 FULL |2152|12217|12411|
| D1 PIECEWISE |18|251|416|

| Conservation check | Expected | Observed | Absolute error | Relative error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| P useful scheduled tokens |147072|147072|0|0%|PASS|
| D useful scheduled tokens |24936|24936|0|0%|PASS|

Final artifact validation rechecked all7 input hashes successfully. There is no predicted latency or measured latency comparison here. Equal aggregate useful-token totals do not establish individual request/forward/rank attribution.

## What the existing logs cannot prove

Locked MRv2 `v1/worker/gpu/model_runner.py:1832–1837` only constructs graph statistics under `if not dummy_run`. Thus these99 rows cannot count required idle-DP dummy forwards. Do not derive dummy counts from differences between lanes or claim every TP rank has an observed forward. Full G4 remains unverified, as do G3 and formal clean-reference admission.

## Source-backed producer and consumer corrections

The initial P0 producer assignment repeated a MRv1-derived no-metadata/fixed-one-slot assumption. The producer handoff at a7551a3/4173fba cites `v1/worker/gpu_model_runner.py:5805ff`; the actual runtime is MRv2:

1. `v1/worker/gpu_worker.py:1467–1474` calls `_dummy_run` with uniform decode.
2. `v1/worker/gpu/model_runner.py:821–911` defaults `skip_attn=False`; skipping attention is limited to initial memory profiling.
3. `:1810–1825` obtains a DP-synchronized execution descriptor.
4. `:1863–1888,1926–1953` prepares descriptor-sized dummy attention inputs/metadata and restages FULL capture buffers.

This does not by itself measure the reached dummy distribution. Producer owner `step5-p0-profiles` is correcting the actual MRv2 binding; prior stages are immutable and unsubmitted. Independently supported no-metadata isolated cells must not be labelled serving idle coverage.

Native `h800_32_qualification03/stage/config.json:89–92` explicitly uses FULL_AND_PIECEWISE, consistent with the observed D modes above. Current Frontier `config/config.py:168–175` has none/full_decode_only/piecewise policies; `execution_time_predictor/step5_cost.py:20–34` maps full_decode_only and treats D tails/misses as no graph. This is an unresolved serving-semantic binding, not a measured error percentage. Align the consumer after the producer's actual MRv2 identity is settled; do not relabel NONE/PIECEWISE measurements or change native runtime policy to fit the simulator.

Next G4 implementation should extend the existing native observer at the source-backed execution/DP-dispatch seams to record useful versus dummy calls and rank/descriptor identity. It remains separate from clean timing and uses non-blocking auxiliary checks. No new submission or diagnostic-only repeat is authorized by this report.
