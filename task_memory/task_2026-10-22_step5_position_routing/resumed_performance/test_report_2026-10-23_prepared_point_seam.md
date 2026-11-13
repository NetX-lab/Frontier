# Parent prepared-point seam verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Independently replay the prepared wave and replacement/negative matrix before allocating implementation. |

## Scope and environment

CPU prototype only, frozen source `79f1a3fd35d42b751cd1520c764ddac717b923d9`
in `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`.
Python `/data/ycfeng/envs/frontier-py310/bin/python`, CPython3.10.6. Scripts are
under `/data/ycfeng/tmp/d29_prepared_point_seam`; no source, fixture or dependency
changes. Their internal RLIMIT_AS is30GiB; this is an address-space bound, not a
cgroup RSS measurement. Both commands have the original60-second outer limit.

## Exact commands

Run from the frozen worktree:

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_prepared_point_seam/wave_probe.py --output /data/ycfeng/tmp/d29_prepared_point_seam/parent-prepared01 --prepared > /data/ycfeng/tmp/d29_prepared_point_seam/parent-prepared01.log 2>&1
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_prepared_point_seam/selection_matrix.py --output /data/ycfeng/tmp/d29_prepared_point_seam/parent-matrix01 > /data/ycfeng/tmp/d29_prepared_point_seam/parent-matrix01.log 2>&1
```

Parent also recomputed every script/result/evidence SHA256 listed in
`prototype_receipt.json`: all matched. Raw receipts remain in the two output
directories; the script identities are recorded by the owner's unchanged receipt.

## Expectations and observations

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Fixed preparation | No repeated full row/key scan on the warm wave | One16464-row/168-key preparation across32 lane queries | PASS |
| Live current-file reads | Existing80 reads per wave retained |80/80 reads; cold3258/warm1968 JSON parses | PASS |
| Full replay payload | Baseline costs, proofs and full plan keys unchanged | Both wave hashes `b2593e723ee095488ff213e84ddcd6e85391c11f42e092a7a5ba8b77a38ffc3a` | PASS |
| Multiple point/group/family selection | Same contexts, queries, plans and selected duration/count results |36 positive comparisons | PASS |
| Replacement and corruption | Existing unsupported/ambiguous source, model-row replacement and buffer/context/policy failures retained |24 negative records, including both-path records where applicable | PASS |
| Timing | Observe rather than impose a universal threshold | Cold pricing1.868104s; warm1.236643s; whole warm wave1.427166s | OBSERVED |

The previous parent original-path warm result was2.653s; the owner prepared
measurement was0.951s. Timing varies with I/O, including observed slow reads in
this parent run. The deterministic acceptance is preparation/parse multiplicity
and payload equality, not selection of the fastest observation.

## Limits and next action

The wave is the actual retained layer61/position6147/16-EP replay. Its complete
208-event generated payload matches the original replay; direct retained-main08
comparison covers165 captured pairs. The separate layer60 record already verifies
208/208 actual pairs, but this prototype was not relabeled as that separate run.
The matrix uses the existing predictor fixture seam; it is not a second full
public-predictor construction, native admission or complete CLI run. It includes
live selected estimator/count checks, and does not cache numerical results.

Implement within the new250-gross allocation, verify supported replacement and
current-text behavior on actual source, then review and combine with other
measured repairs. Full1024-token execution, startup and the2400-second per-mode
gate remain unresolved.
