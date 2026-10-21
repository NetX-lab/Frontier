# D23 forward agreement: primary integration verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Primary finishes fixed-capacity verification and resumes integration after provider interruptions. |
| 2026-10-22 | Final source review passes; merge32fb934 and clean baseline/candidate matrix started. |
| 2026-10-22 | Clean strict77 comparison passes with zero artifact/cache/provenance differences. |

## Authority and scope

Approved D23 in `requirements.md` covers pre-first-prediction P/D useful and
dummy agreement using the existing stage group/events, FULL_AND_PIECEWISE,
NONE local widths and strict typed coverage. No native or accuracy admission
is implied by this CPU integration.

## Completed source verification

- Forward implementation3cad8dbe: primary previously verified444 affected checks.
- Capacity correctionc9e9388f: native FULL buckets cannot exceed request capacity;
  cap3/decode3+dummy1 uses PIECEWISE4 or NONE3/1 according to policy.
- Fixed config bindingef476fe: use `_config.batch_size_cap` directly. This states
  native graph creation capacity. The prior field equals it on supported Step
  requests because thinking is rejected; no current Step arrival-order defect
  is claimed for that final one-line clarification.
- Primary final affected stage/engine/public predictor/Step structural-PD run:
  **242 PASS in33.92s**, zero failure/error/skip. Python3.10.6, pytest9.1.1;
  worktree `step5-mrv2-forward`. Exact command is retained in that task's
  `test_report_2026-10-22_forward_agreement.md`; raw log/XML:
  `/data/ycfeng/tmp/d23_fixed_parent_takeover01.{log,xml}`.

## Source review and merge: PASS

Reviewer `ses_ea077118cffeNBkYQnihHe1SZ7` inspected the complete
`/data/ycfeng/tmp/d23_forward_final_ef476fe.diff` (`5c2319cd..ef476fe`), task
records, design and native/current code. Standards/Spec: **No findings**.
The original cap3 finding is resolved; the hidden-phase example is withdrawn
as unreachable for current supported Step requests. Full review is in
`../task_2026-10-22_step5_mrv2_forward/review.md`.

Merged source: `32fb934bafcd5d422555f23534b99609d03d540b`. Primary byte-compared
all16 changed core/test files with `ef476fe`; all match. Independent fixture
work remains isolated; no merge conflict or unrelated file overwrite occurred.

## Clean legacy matrix: PASS

Fresh detached worktrees were clean before running: baseline
`/data/ycfeng/Frontier/.worktrees/fidelity-baseline-9dcd876`, candidate
`/data/ycfeng/Frontier/.worktrees/fidelity-candidate-32fb934`.
Both commands use the candidate's same driver/case table and separate caches:

```bash
# cwd=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-32fb934
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-baseline-9dcd876 --label baseline --output-root /data/ycfeng/tmp/step5_d23_forward_fidelity_20261022 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure > /data/ycfeng/tmp/d23_fidelity_baseline01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-candidate-32fb934 --label candidate --output-root /data/ycfeng/tmp/step5_d23_forward_fidelity_20261022 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure > /data/ycfeng/tmp/d23_fidelity_candidate01.log 2>&1
```

Comparison command (same candidate CWD):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/step5_d23_forward_fidelity_20261022 > /data/ycfeng/tmp/d23_fidelity_compare01.log 2>&1
```

Expected and observed: **77/77 identical**, zero mismatches, baseline/candidate
failures, missing evidence, differing definitions, skipped comparisons or
predictor cache-name differences. All three commands exit0. Both sides carry
clean source provenance and use the same32fb934 harness. Python3.10.6,
numpy2.2.6, pandas2.3.3, sklearn1.7.2, scipy1.15.3 and plotly7.1.0 match.

Primary validated every zero-finding list and the complete/clean cache flags
in `comparison.json`. Retained comparison, both manifests and six evidence
hashes: [d23_forward_fidelity.json](d23_forward_fidelity.json).
Full raw artifacts remain at the output root above. This passes the required
legacy-fidelity check; full MRv2 CLI and native accuracy are independent.

## Full CLI: in progress

The owner reuses already-fitted fixture02 and existing caches in
`step5-mrv2-main`. No duplicate fit or old isolated-shape run was started.
Offline/online completion, actual tail/read/release evidence and final metrics
remain unverified under the fresh MRv2 input contract.

## Legacy cancellation

The cancelled old D20 agent left a completed isolated offline fixture and a
still-running online child. Primary preserved source/receipts and checked
6144/1024, tail16/cached6128,1024 D forwards and ready/release equality.
[Cancellation audit](../task_2026-10-21_step5_d20_integration/actual_main/cancellation_20261022/test_report_2026-10-22_legacy_cancellation.md).
That result is historical and does not satisfy the current MRv2 acceptance.
