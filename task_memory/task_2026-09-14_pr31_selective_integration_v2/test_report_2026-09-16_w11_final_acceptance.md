## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded final paired execution, diagnostic call counts, native collection and source-bound evidence integrity checks. |

# W11 final acceptance evidence

Source: `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`; fixed main `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. Python `/usr/bin/python`3.12.3, no conda, CPU-only; final measurement pins numerical-library threads1. Full script paths, exact command and all18 samples are maintained in `performance_rca.md` and `w11_final_paired_samples.json` rather than duplicating the numerical record here.

## Criteria and observed result

- Each of three fixed workloads runs three interleaved pairs, with identical flags/event/request/token work and no concurrent agent benchmarks: **18/18 successful executions**. Source unchanged throughout.
- Run/init/process times are separate and all samples retained. Dense residuals remain, so the raw residual remains measured while **D02 is explicitly accepted for this scope**. Final paired medians1.775045x/1.679023x/0.978816x. MoE is not a universal speedup claim.
- Diagnostic profiling executes separately. Homogeneous small-dense block calls1440→36, mutable snapshots1440→36;72 aggregate reads are cached scalar accesses. Required physical identities remain. Profile times are excluded from acceptance.
- Native capability verification on final source: **19 SKIP in2.57s**, exact reasons in `/data/ycfeng/tmp/pr33-w10-native-capabilities-final.log`. CPU-safe collection produces19 native and8 non-dummy exact node IDs, retained in `validation_manifest.json`.
- Final source/runtime and tests have no uncommitted edits; task documents remain a separate handoff. `validation_manifest.json` binds selected output bytes/hashes, all lane commands, failure/skip nodes and execution tiers. Later documentation commits do not change the source tested.

## Additional exact commands

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/integration/test_device_timer_native.py tests/integration/test_gdn_native_acceptance.py tests/integration/test_pr33_native_profiling_acceptance.py -q -rs -p no:cacheprovider > /data/ycfeng/tmp/pr33-w10-native-capabilities-final.log 2>&1
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/integration/test_device_timer_native.py tests/integration/test_gdn_native_acceptance.py tests/integration/test_pr33_native_profiling_acceptance.py --collect-only -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w10-native-nodes-final.log 2>&1
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/integration/test_pr33_nondummy_acceptance.py --collect-only -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w10-nondummy-nodes-final.log 2>&1
```

Full CPU retained failures and the58-case raw comparator failures are documented separately in the W10 reports; they are not hidden by the successful benchmark process exits. No new unclassified defect was observed. The scoped D02 acceptance and native/production-data limits remain explicit.

## Evidence-integrity verification

Final direct verification parsed the manifest and re-read all812 selected artifact byte counts/SHA-256 values: **PASS**. Git diff and the ledger agree on675 current production hunks;471 original IDs remain covered. All13 applicable remediation commits have the required co-author trailer; original W00 harness/freeze commit10326ce4 is outside the material-PR31-adaptation criterion. Production/tests are clean. The initial overly broad trailer assertion and one Markdown EOF whitespace failure were recorded in `progress.md`, corrected at their actual scope, and rechecked. No runtime change or history rewrite resulted.
