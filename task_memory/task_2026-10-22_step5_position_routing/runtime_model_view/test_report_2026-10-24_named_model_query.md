# D29 named-model query production verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete focused/committed-source wave, byte parity, scope and preserved-input verification. |

## Scope and environment

Task/session: `D29-NAMED-MODEL-QUERY` /
`ses_e93af1ff7ffe0YosowRNrc24Xa`. Worktree/branch:
`/data/ycfeng/Frontier/.worktrees/step5-d29-named-query` / `step5-d29-named-query`.
Allocation base617d832b; source-step comparisonde2d818f; cumulative comparison
1aee0af11cd58691e64603b99eca0ac459a5b350. Accepted unchanged key baseline:
`/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key`, d1e2fb8c.
Production source commit:1737ce9c4a761d2c57ee617f5f3a78d7bcfd9a58, committed
after focused verification and before wave preparation/execution.

Python: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6;
NumPy2.2.6, pandas2.3.3, scikit-learn1.7.2. Every Python
command uses `-B`, `PYTHONDONTWRITEBYTECODE=1`, `runuser -u brainpp`, and
`sudo -n systemd-run --scope -p MemoryMax=30G`. Numerical work uses one thread
for OPENBLAS/OMP/MKL/NUMEXPR. Scratch:
`/data/ycfeng/tmp/d29_named_model_query01`; no writes to original fixtures.

Optional runtime module is cloned locally with `--shared --no-checkout` from
`verify-step5-d29-integrated-06/frontier/cc_backend/backends/collective-sim`, then
detached at3dac53c544e546574cb6e8078cd14053d0916113. The retained correct-pin
`sim/datacenter/htsim_ndp` is copied with `cp -p`, not built or source-edited:
SHA2568818d1146c32505707dee41147cfe17de46a0c9f3cf6bb8fc7b1b3fe30e15637.
`ldd` resolves all libraries; submodule source remains clean and gitlink unchanged.

## Commands

Exact shell prefix for the numerical commands below:

```sh
timeout <seconds>s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_named_model_query01/cache /data/ycfeng/envs/frontier-py310/bin/python -B
```

The RED/GREEN commands use180s, candidate CWD, and these exact Python arguments,
with `red` or `green` substituted for `<phase>`. RED ran with unchanged production
source and new tests; the later import-only test cleanup reduced runtime allocation
from32 to30 lines before GREEN without changing assertions.

```sh
-m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_routing_runtime.py -k 'named or mixed_native_non_native' --basetemp=/data/ycfeng/tmp/d29_named_model_query01/<phase>
```

Focused comparison uses300s, baseline/candidate CWDs respectively, with these
identical test paths and side-specific scratch/XML/log paths:

```sh
-m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_training.py tests/unit/test_moe_position_routing.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_model_identity_reuse.py --basetemp=/data/ycfeng/tmp/d29_named_model_query01/focused-<side> --junitxml=/data/ycfeng/tmp/d29_named_model_query01/focused-<side>.xml
```

Governance comparison uses180s and the same two CWDs:

```sh
-m pytest -q -p no:cacheprovider tests/unit/test_profiling_governance_minimal_red.py -k 'manager_' --basetemp=/data/ycfeng/tmp/d29_named_model_query01/governance-<side> --junitxml=/data/ycfeng/tmp/d29_named_model_query01/governance-<side>.xml
```

Each log is redirected to `/data/ycfeng/tmp/d29_named_model_query01/<phase>.log`
or `<suite>-<side>.log`, preserving stderr. The manifest archives exact logs/XML.

## Observed checks

| Check | Passing condition | Observed | Verdict |
| --- | --- | --- | --- |
| Meaningful RED | Demonstrated pre-change selection failure | 10FAIL/2PASS/17deselected,63.78s | PASS RED |
| GREEN | Same selected cases pass | 12PASS/17deselected,61.12s | PASS |
| Typed/public registry governance | No new baseline failure | Both14PASS/1FAIL/37deselected;1.94s/1.92s | PASS parity; existing failure remains |
| Focused runtime/row/count tests | No new failures, new cases PASS | Baseline117PASS/1SKIP165.09s; candidate125PASS/1SKIP195.81s | PASS |
| Final selected/native role check | Final test bytes pass | 12PASS/17deselected65.92s | PASS |
| Committed-source208-event wave | Full output/calls/guards retained, no method replacement | Real source1737ce9c;19 byte comparisons and all protected guards | PASS |
| Scope delta | Four paths only;15/5/50/30 caps and20/80/100 aggregate caps | Final14/3/49/30=96gross(source17/tests79) | PASS |

After the focused comparison, three additional assertion lines cover native-only
DECODE_ATTN exclusion, complementing the typed-role case. Final selected checks
passed with `final-green` scratch/XML/log paths before source commit. No source
behavior changed after the broad focused runs.

Measured cumulative gross against1aee0af: registry104/120, workload196/223,
bindings348/353 and runtime147/151. Cumulative growth is88 rather than96 because
some source changes replace lines already added by D29. Incremental totals over
accepted union9491: CORE4261/4404, PRICING3954/3956, native1350/1582,
source9565/9942, global9579/10000. This is an incremental receipt, not the final
composed parent audit. Donor files are unchanged.

Nine RED failures are `unexpected keyword argument 'model_name'`. The tenth is
the approved unrelated native gating collision, previously raised by the GEMM
point query. Selected GEMM collisions and the retained cache-name guard already
pass RED and remain mandatory GREEN cases.

The unchanged governance failure is
`test_real_linear_producer_csv_round_trip_reaches_manager_loader`:
`ModuleNotFoundError: No module named 'torch'`. No package installation, skipped
test substitution or source fallback is used to hide this failure.

## Production-wave acceptance and limits

**COMPONENT VERIFIED; PARENT ACCEPTANCE PENDING.** Both wave sides completed
their unchanged output oracle within60s. Baseline warm samples are
0.122730910/0.132523966/0.122182456s; candidate samples are
0.088554673/0.088141062/0.088166638s. The full values are recorded in
`repair_evidence/{baseline,candidate}/wave_result.json`. Median0.122730910→0.088166638s,
−28.1626%. This is three unprofiled warm samples per side, not a CLI speedup.

| Production check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Unfiltered view and selected entries | Same order/objects; no filtering for public/default callers | 38 full entries and2 selected GEMM objects | PASS |
| Contract derivations | Remove unrelated work | 384→0 | PASS |
| Numerical/fresh proof/collective/point/key calls | Unchanged | 288/144/64/16/192 each side | PASS |
| Input binding | One read, no pricing reads | 1 binding;0 cold/warm/profile reads | PASS |
| Full pricing payload | All costs/proofs/keys unchanged | SHA2563d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83 | PASS |
| Complete JSONL/order | 15 full208-event files identical | Each SHA2561850a9b6a90cc7bb087b881e72d8cc00f5a669fa0be0f54bb25f3361a38d7e11 | PASS |
| Ledger | Byte identity | 2309545B;SHA256006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf | PASS |
| Snapshots and negatives | Preserve getter/pending/ledger isolation and errors | Isolation guards pass;64 wrong-key negatives;3902B restored prefix | PASS |
| Input/source guards | Original metadata/hashes/body unchanged | 262 fixture files,23 source bindings per side and retained input hashes | PASS |
| Emission observation | No output-path optimization claimed | Medians37.372→42.533ms,15 samples/side | Observed;not attributed |

The runtime error-policy difference is explicitly tested: a registered unrelated
`moe_gating_linear` native-label collision still raises in public full views, but
the named GEMM point query succeeds. A selected GEMM collision still raises. No
claim is made that the omitted unrelated error necessarily reappears in a later
context-qualified numerical lookup. Existing live estimator/row tuple/identity,
family/source/point/buffer/count failures remain in the passing focused suites.

New `repair_evidence/prepare.py` rebinds only source/paths/owner. It removes the
old `from narrow_view import install` and `install(...)` entirely, checks real
method code filenames against committed source, and leaves the original evidence
and protected body unchanged. `repair_evidence/verify.py` compares all19 eligible
whole files and separately compares the3902B late-error prefix and two unfrozen
metadata timestamps. The oracle SHA256 remains
1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555.

Exact wave commands from candidate CWD (prepare/verify use the same60s/30G
wrapper; raw wave additionally isolates all caches):

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/repair_evidence/prepare.py /data/ycfeng/tmp/d29_named_model_query01 > /data/ycfeng/tmp/d29_named_model_query01/prepare.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_named_model_query01/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_named_model_query01/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_named_model_query01/cache/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_named_model_query01/cache/numba /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_named_model_query01/run.py baseline > /data/ycfeng/tmp/d29_named_model_query01/baseline.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_named_model_query01/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_named_model_query01/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_named_model_query01/cache/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_named_model_query01/cache/numba /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_named_model_query01/run.py candidate > /data/ycfeng/tmp/d29_named_model_query01/candidate.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/repair_evidence/verify.py /data/ycfeng/tmp/d29_named_model_query01 > /data/ycfeng/tmp/d29_named_model_query01/verify.log 2>&1
```

The inherited protected body still reports its historical
`D29-IMMUTABLE-COST-FIELDS` task label. New outer receipts identify
`D29-NAMED-MODEL-QUERY`, with `view_control.applied=false` and verified real method
filenames. This is production verification, not the original in-memory control.
The intentional late serialization fault logs `Object of type object is not JSON
serializable` in both arms; the expected buffer/prefix/error assertions pass.

Closing read-only hash inspection (bounded60s/30G, the same Python3.10.6 `-B`
environment) verifies all22 original diagnostic manifest files unchanged, all21
new archived files,23 source files against1737ce9c Git blobs, clean3dac module and
the retained runtime binary. `repair_evidence/final_guards.json` retains these
counts and script hashes. Verification SHA256:
d5f2c5939f6279f460469a048b32636402ff0ee1a4636c4475d6531ac1c9c6c5.

This component does not close full-output/current-model validation for the whole
workflow,6144/1024/2400s CLI, frozen regressions, strict77/full-unit, native/GPU,
G3 or E2E acceptance. Primary owns both review axes, final scope audit, acceptance
and integration. No donor/source paths outside the four-file allocation change.
