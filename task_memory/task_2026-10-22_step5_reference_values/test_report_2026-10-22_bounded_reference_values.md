# D24 bounded reference values: CPU test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded real-collector verification, commands, observed results, payload budget and source-pin handoff. |

## Environment and scope

- Worktree `/data/ycfeng/Frontier/.worktrees/step5-reference-values`, branch `feat/step5-reference-values`, base `0c9d1ef299e27d0e9bddbe4eb888527f966ba718`.
- Python `/data/ycfeng/envs/asyncopd-py312/bin/python`, 3.12.13 (Clang 22.1.3); Torch 2.9.1+cu128; pytest 9.0.2. No conda environment change.
- CPU tensors only; no CUDA operations or GPU/platform/auth/cloud/SSH calls. The test replaces only diagnostic synchronization and counts real `Tensor.cpu()` calls. Native-like tensor geometry and worker arguments exercise the unchanged production collector.
- Each suite ran through `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s`; temporary files, logs and JUnit XML stay under `/data/ycfeng/tmp`. No native package was staged or restaged. The existing `stage_capture` regression creates synthetic source files in pytest scratch only, as before.

## Exact commands and results

Working directory for every command below: the worktree above.

Environment inspection:

```sh
/data/ycfeng/envs/asyncopd-py312/bin/python -c 'import sys, torch, pytest; print(sys.version); print("torch",torch.__version__); print("pytest",pytest.__version__)'
```

First real-seam iteration:

```sh
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_reference.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d24_reference_values_first01 --junitxml=/data/ycfeng/tmp/d24_reference_values_first01.xml > /data/ycfeng/tmp/d24_reference_values_first01.log 2>&1
```

Actual: **83 PASS, 0 failed, 22.29 s**.

Combined caller/observer/canonical regression after completing later-checkpoint and verdict checks:

```sh
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_canonical_g3.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d24_reference_values_iteration02 --junitxml=/data/ycfeng/tmp/d24_reference_values_iteration02.xml > /data/ycfeng/tmp/d24_reference_values_iteration02.log 2>&1
```

Actual: **235 PASS, 0 failed, 36.18 s**.

Final reference/observer/canonical/verdict/native regression:

```sh
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_verdict.py tests/unit/test_step5_native_pd.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d24_reference_values_final03 --junitxml=/data/ycfeng/tmp/d24_reference_values_final03.xml > /data/ycfeng/tmp/d24_reference_values_final03.log 2>&1
```

Actual: **539 PASS, 0 failed, 41.09 s**.

Final superset, adding the existing R2 stage regression; no code changed after the preceding run:

```sh
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_verdict.py tests/unit/test_step5_native_pd.py tests/unit/test_step5_native_pd_r2_stage.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d24_reference_values_final04 --junitxml=/data/ycfeng/tmp/d24_reference_values_final04.xml > /data/ycfeng/tmp/d24_reference_values_final04.log 2>&1
git diff --check
```

Actual: **550 PASS, 0 failed, 41.21 s**. Diff check PASS (no output). None of the iterations produced a test failure or required an error-suppressing fallback. Deliberately injected caps produce `observer_budget_exceeded`; those are expected witnesses and their tests pass.

## Expected versus observed acceptance

| Criterion | Required output | Observed evidence | Result |
| --- | --- | --- | --- |
| Actual and independent full-local callers, all TP ranks | Both use real `worker_before/worker_after` -> `local_tail/checkpoint`; first-tail values for both components | 3 prompt lengths × 8 ranks × 2 caller paths = 48 real collector cases. Actual starts from native prefix formula; reference starts at 0 with reference_start/resident scan. | PASS |
| Only two logical pages at first-tail | Current computed-token page and predecessor, valid units only; no later-checkpoint hex | 7 computed-token counts × 3 boundaries = 21 cases; real later worker forwards cover region crossing/final input. Older full-history pages remain hash-only; partial raw/mean units are exact. | PASS |
| Complete hash/geometry census | All original logical components present; hash-only projection equal | Existing `checkpoint_census` PASS; exact SHA/shape check for every item; removing hex leaves `logical_state` unchanged. Physical remapping regression still passes. | PASS |
| No extra GPU copy/sync/reserve | Same existing calls and byte reservations | Per caller: exactly 2 original synchronizations; sample count = complete checkpoint item count + 10 original tail samples; exactly one reservation and one CPU copy per sample. Reserved bytes equal full checkpoint bytes + 5,144 B existing two-page accumulator/marker/mean samples. | PASS |
| Bounded mean wire retention | Original two mean rows, first-tail only, no numerical conversion | Each row 512 B uint8, raw_hex SHA verified; later/read-ready rows hash-only. | PASS |
| Numerical verdict unchanged | Non-identical state UNVERIFIED, no tolerance/waiver | Real `r2_controls.reference` compares 24 pairs; one changed BF16 byte yields one `missing` entry, 0 failures, UNVERIFIED. Removing retained bytes leaves the whole result identical. | PASS |
| Inactive predecessor ignored | `last_slot=-1` does not compare positional accumulator values | Deliberately different stale accumulator hashes remain ignored; no policy code changed. | PASS |
| Cap failures isolated | Request/file/time errors cannot escape native return | Three cap modes × actual/reference clients: two native calls both return 41, terminal observer_budget_exceeded, observer disabled. Existing wrapper is unchanged. | PASS |
| Existing staging and frozen inputs | Collector source pins refreshed by normal builder, no changed engine/cap/workload | Synthetic stage includes exact collector bytes, correct reference_helper_sha256; verifier remains `g3_qualified=false`, `clean_timing_eligible=false`; frozen 7168/8192/128, 8 content IDs and 1 GiB/600 s limits unchanged. | PASS |

## Payload arithmetic

Actual local evidence shapes, not toy-fixture sizes: 92 BF16 raw layers, 384 channels × 1 head × valid tokens; 23 mean layers, 256 uint8 bytes × valid eight-token regions. Full logical page: **1,142,272 B** across components; 15-token page: **1,071,616 B**; 1-token page: **76,544 B**. Only first-tail hex is newly serialized; source copies/reservations already existed.

- canonical6144: 2,284,544 B/side, 9,138,176 B hex/pair.
- boundary6143: 2,213,888 B/side, 8,855,552 B hex/pair.
- boundary6145: 1,218,816 B/side, 4,875,264 B hex/pair.
- These three paired checkpoints × TP8: **182,951,936 B** extra hex.
- Existing full mean wire rows at the same boundary: **2,260,992 B** extra hex.
- Normal-pair total: **185,212,928 B (176.6328125 MiB)** plus small raw_hex JSON key/quote overhead.
- Each existing successful cancel/reuse6144 control adds **36,929,536 B** across TP8. All 8 current content IDs reaching first-tail: **259,072,000 B (247.0703125 MiB)** plus key/quote overhead. This corrected upper bound reflects unchanged selection, not a new workload.
- No whole-checkpoint hex (560,726,016 B for a 6144 pair), no expansion to other checkpoints, no change to diagnostic caps.

## Source-pin handoff

At this branch, before parent combination:

| File | SHA256 |
| --- | --- |
| `tests/integration/step5_pd/r2_reference.py` | `30fbfd0a9eb440cc5dd42e3fc550eff2c2f1782fabdbce1bea92212bca8f7698` |
| `tests/integration/step5_pd/r2_observation.py` | `774280921282530c3df3eb22d8f20435d8b99766ffe412f7cafac20a045e5b81` |
| Unchanged `r2_controls.py` | `7d0d6840a8605bd1cbb3bc25979435b5d125aa354ea36fc0ba76cbc758672bf7` |
| Unchanged `r2_runtime.py` (includes parent's startup census) | `b71bc59620648ea2d575d1c32906b3da7b682ba2353efd56a0614f1c2900326a` |
| Unchanged `r2_canonical.json` | `e82615f44923b5d4567be59f506281f8befeb3a2ec09d1d463de0e59cf7477f0` |

`r2_stage.stage_observation` already copies both files. It recomputes `reference_helper_sha256`; the existing builder recomputes tool/observation/config/package hashes. Parent must combine the HTTP/input/census corrections first and use those normal mechanisms for one fresh generation. Do not copy these hashes into an old package or edit old verdicts. A combined `r2_observation.py` hash will differ from this branch-only value.

## Verification limits

Observed: all CPU tests above and exact metadata/byte-count conditions. Inferred budget: native layer/shape counts from the previously retained local capture, assuming all current content controls reach first-tail; large-model latency/native serialization has not been measured. No native value evidence was collected in this slice. FP8 remains uint8 wire storage; no numerical decoding or tolerance is supplied. Native evidence and any proposed numerical rule still require the parent diagnostic and separate YC approval.

Task/session `D24-G3-STATE-VALUES` / `ses_ea074b7bfffe5UImN2k3jg0Uax`.
