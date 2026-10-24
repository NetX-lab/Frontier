# Named-query parent verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accept composed scope and integrated source after a fresh124-binding final-head guard. |
| 2026-10-24 | Verify final-source focused tests and complete two-arm wave replay; retain independent reviews and audit dependency. |

## Scope and result

Requirements: parent D29-NAMED-MODEL-QUERY decision and
`repairs/named_model_query_scope_amendment.json`. Source `1737ce9c`, records
`3d75c786`, isolated worktree `/data/ycfeng/Frontier/.worktrees/step5-d29-named-query`.

**PASS for component replay and focused tests.** Parent independently executed
the final committed tests and both real production wave arms. Both review axes
report No findings in `review.md`. Composed budget audit and final acceptance
were pending at this first check; the final section below closes them. This
report does not qualify the full CLI.

| Check | Expected | Observed |
| --- | --- | --- |
| Final-source focused suite | No new failure; added cases pass | 125 PASS / 1 existing Torch SKIP, 190.33 s |
| Retained baseline IDs | Same statuses for all 118 | Exact equality |
| New cases | All pass | 8 PASS |
| Owner artifact integrity | Match recorded sizes/hashes | All 21 archived artifacts match |
| Full wave outputs | Costs/proofs/keys/order/ledger and eligible JSONL match | 208 events and all 19 whole-file comparisons match both baseline and owner receipt |
| Error/ownership behavior | Original snapshots, wrong-key errors and partial prefix | 64 wrong keys per arm; 3902-byte prefix and buffer restoration; protected body assertions pass |
| Live work | Same numerical/proof/collective/point/key calls | 288 / 144 / 64 / 16 / 192 per arm |
| Contract preparation | Remove unrelated derivation only | 384 to 0 |
| Input/source guards | One input binding read; no mutation | One binding read per arm, 262 fixture metadata entries and 23 source bindings preserved |
| Parent warm medians | Observe without asserting full-workflow speed | 0.121525 to 0.090143 s |
| Governance evidence | Preserve owner baseline/candidate statuses | Both 14 PASS / 1 unchanged missing-Torch FAIL; not rerun by parent |

Both wave arms ran concurrently with isolated caches while the focused suite was
active. Their timing is observational; owner serial samples independently show
0.122731 to 0.088167 s. Neither result predicts that the 2400-second full CLI
will pass. Two diagnostic-only headers retain their different real timestamps;
all remaining header values and respective 3902/0-byte payloads match. All 15
full 208-event JSONL files are byte-identical without timestamp normalization.

## Exact commands and environment

Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6; NumPy 2.2.6,
pandas 2.3.3, scikit-learn 1.7.2. The numerical commands run as `brainpp` with
30G memory scope, `-B`, `PYTHONDONTWRITEBYTECODE=1`, one BLAS/OpenMP/NumExpr
thread, and `TMPDIR=/data/ycfeng/tmp`. Candidate worktree is the numerical CWD.

```sh
timeout 300s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_training.py tests/unit/test_moe_position_routing.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_model_identity_reuse.py --basetemp=/data/ycfeng/tmp/d29_named_query_parent01/pytest --junitxml=/data/ycfeng/tmp/d29_named_query_parent01/pytest.xml > /data/ycfeng/tmp/d29_named_query_parent01/pytest.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-d29-named-query/task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/repair_evidence/prepare.py /data/ycfeng/tmp/d29_named_query_parent01/wave > /data/ycfeng/tmp/d29_named_query_parent01/prepare.log 2>&1
```

The following command was executed once for each `SIDE=baseline` and
`SIDE=candidate`, independently in parallel:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp XDG_CACHE_HOME=/data/ycfeng/tmp/d29_named_query_parent01/wave/cache-${SIDE} MPLCONFIGDIR=/data/ycfeng/tmp/d29_named_query_parent01/wave/cache-${SIDE}/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_named_query_parent01/wave/cache-${SIDE}/numba /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_named_query_parent01/wave/run.py ${SIDE} > /data/ycfeng/tmp/d29_named_query_parent01/${SIDE}.log 2>&1
```

Parent receipt check, CWD main worktree:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/parent/verify.py > /data/ycfeng/tmp/d29_named_query_parent01/verify.log 2>&1
```

All commands completed within their limits and the receipt check passed first
attempt. No production method was monkeypatched; preparation verifies real
method filenames and removes the earlier candidate installation. The protected
body is unchanged. Parent reuses every owner wave/source/input assertion before
the owner-only suite/archive phase, then independently compares its own JUnit.
Existing owner/session labels in inherited wrappers remain explicitly attributed
to the original scripts; this report describes the parent execution.

## Artifacts and limits

- Machine receipt: `verification.json`, SHA256
  `0d9095925263f9beab801a738a41561efa8ccdffa8f391e669036b1efb22d8f5`.
- Full raw artifacts and exact hashes: receipt `parent_artifacts`, rooted at
  `/data/ycfeng/tmp/d29_named_query_parent01/`.
- Independent review records: `review.md`.
- Global/cap figures remain owner incremental evidence until the independent
  composed audit returns. No full CLI, frozen strict77/full-unit, GPU/native,
  G3 or E2E qualification is claimed for the named-query successor.

## Final integration and scope acceptance

Independent audit `de5eeab0`, archived as `c6b1f3b8`, verifies all114 caps and the
65-line zero-sum allowance transfer. Step usage is96gross (17source/79tests),
net84; cumulative budget growth is88. The accepted union is CORE4261/4404,
PRICING3954/3956,native1350/1582,source9565/9942 and global9579/10000, with zero
violations. Source `ffb435ca` equals owner `1737ce9c`; owner records are `8ecb705e`.

Parent verified all six audit artifacts against hashes and original committed
blobs, then rechecked all124 bindings on integrated `c6b1f3b8`: four query deltas,
120 unchanged non-query bindings, fixed native source bytes, pinned submodule
blobs and exact non-task source equality. Non-query digest:
`19c5718dfc1aa4cb11eef0306b8b877dbd9f234f1192053cb77ce4de7c9337fb`.
The audit's small specified reads used30GiB RLIMIT_AS after transient scope
denial; parent used the required working30G cgroup wrapper below.

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/parent/final_scope_guard.py > /data/ycfeng/tmp/d29_named_query_parent01/final_scope_guard.log 2>&1
```

PASS first attempt. Machine receipt `acceptance.json` SHA256
`442847ed6cb9526b53ebb562689821a93f30dc6c7698f967b05c176fe91c0793`, status
`ACCEPTED_COMPONENT_AND_COMPOSED_SCOPE`. The successor's frozen regression and
whole-workflow gates remain required; component acceptance does not close them.
