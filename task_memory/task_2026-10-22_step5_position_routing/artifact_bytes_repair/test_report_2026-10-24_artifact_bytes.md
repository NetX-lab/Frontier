# D29 artifact bytes verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive allocated repair, paired actual-wave evidence, original tests and Git accounting. |

Task `D29-COLLECTIVE-ARTIFACT-BYTES`, session `ses_e9669de64ffeH614F6iNYIqgeb`.
Code commit `2f2d9339d5bce600fbb0a3f78aebbcf6f71be696`; base
`2cfbb34474c1d4d31f6daf8f9c626d03ad972f58`. Tests measured the candidate working
files at the base HEAD, with exact file SHA checks and an exact three-path diff
guard. `receipt.json` then verifies every committed file matches those measured
bytes. The recorded base HEAD is not relabeled as a candidate commit.

## Environment and commands

Local CPU, CPython 3.10.6 (`/data/ycfeng/envs/frontier-py310/bin/python`), pytest
9.1.1, numpy 2.2.6, pandas 2.3.3, sklearn 1.7.2. No conda executable was used.
Every successful probe uses a 60s outer timeout and MemoryMax=30G; the original
setup also keeps RLIMIT_AS=30 GiB. The four-file pytest selection uses 240s.
All commands below ran from `/data/ycfeng/tmp/d29_artifact_bytes_repair` unless
a Git command names the source worktree. The variables expand to the exact
executed argv values, not alternate flags or a changed workload.

```bash
CODE=/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes
SCRATCH=/data/ycfeng/tmp/d29_artifact_bytes_repair
PY=/data/ycfeng/envs/frontier-py310/bin/python

# Actual unchanged-source baseline05 and final candidate, sequential source states.
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=$SCRATCH $PY -B $SCRATCH/run_wave.py baseline > $SCRATCH/baseline_wave_05.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FRONTIER_TMP_ROOT=$SCRATCH $PY -B $SCRATCH/run_wave.py candidate > $SCRATCH/candidate_wave.log 2>&1

# Original affected selection before edits; then unchanged selection after hook adaptation.
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 240s env PYTHONPATH=$CODE:$CODE/tests/unit FRONTIER_TMP_ROOT=$SCRATCH TMPDIR=$SCRATCH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 $PY -B -m pytest $CODE/tests/unit/test_collective_sim_cc_backend.py $CODE/tests/unit/test_prediction_measurement_bindings.py $CODE/tests/unit/test_moe_routing_runtime.py $CODE/tests/unit/test_step5_measured_ffn_replay.py -q -p no:cacheprovider --basetemp=$SCRATCH/pytest-baseline-03 --junitxml=$SCRATCH/baseline_03.xml > $SCRATCH/baseline_pytest_03.log 2>&1
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 240s env PYTHONPATH=$CODE:$CODE/tests/unit FRONTIER_TMP_ROOT=$SCRATCH TMPDIR=$SCRATCH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 $PY -B -m pytest $CODE/tests/unit/test_collective_sim_cc_backend.py $CODE/tests/unit/test_prediction_measurement_bindings.py $CODE/tests/unit/test_moe_routing_runtime.py $CODE/tests/unit/test_step5_measured_ffn_replay.py -q -p no:cacheprovider --basetemp=$SCRATCH/pytest-candidate --junitxml=$SCRATCH/candidate.xml > $SCRATCH/candidate_pytest.log 2>&1

# Instrumentation-only old hook after source edit, before adapting the test.
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s env PYTHONPATH=$CODE:$CODE/tests/unit FRONTIER_TMP_ROOT=$SCRATCH TMPDIR=$SCRATCH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -B -m pytest $CODE/tests/unit/test_collective_sim_cc_backend.py::test_collective_artifact_reuses_only_identical_current_text -q -p no:cacheprovider --basetemp=$SCRATCH/pytest-old-hook --junitxml=$SCRATCH/old_hook.xml > $SCRATCH/old_hook.log 2>&1

# Evidence/AST/Git verification before committing; no workload rerun.
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- timeout 60s $PY -B $SCRATCH/verify_delivery.py > $SCRATCH/verification.log 2>&1
```

The two candidate commands ran in parallel. Baseline unit tests and baseline03
wave also ran in parallel; baseline05 is the final corrected instrumentation.
Fresh setup/output directories intentionally reject reuse. After the code
commit, replay requires a fresh output root and an explicit source-HEAD/hash
rebinding; the retained wrapper is an exact measurement artifact, not a runner
that silently follows current HEAD.

Submodule preparation, from the sole-writer source worktree:

```bash
git clone --shared /data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields/frontier/cc_backend/backends/collective-sim frontier/cc_backend/backends/collective-sim
git -C frontier/cc_backend/backends/collective-sim switch --detach 3dac53c544e546574cb6e8078cd14053d0916113
git diff --numstat HEAD -- frontier/cc_backend/backends/collective_sim_cc_backend.py frontier/execution_time_predictor/moe_routing_workload.py tests/unit/test_collective_sim_cc_backend.py
git diff --numstat 1aee0af -- frontier/cc_backend/backends/collective_sim_cc_backend.py frontier/execution_time_predictor/moe_routing_workload.py tests/unit/test_collective_sim_cc_backend.py
git diff --check
git add -- frontier/cc_backend/backends/collective_sim_cc_backend.py frontier/execution_time_predictor/moe_routing_workload.py tests/unit/test_collective_sim_cc_backend.py
git commit -m "perf(cc): compare current collective artifacts as raw bytes"
```

## Passing conditions and observations

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Current reads | 16 query +64 backend reads/16-EP wave | Both methods counted; baseline80 text/candidate80 bytes in each of four waves | PASS |
| Parse reuse | Two existing independent caches; changed bytes decoded only on a miss | Two artifact parses cold, zero warm on both sides; total JSON calls2810 cold/1520 warm unchanged | PASS |
| Complete actual layer60 | All208 event costs/proofs/full keys and order equal | 2,889,655 event bytes identical to original reference; complete ledger and647,510-byte pricing payload equal | PASS |
| Existing affected tests | Every original case/assertion retained | Baseline78 PASS/127.120s; candidate78 PASS/124.657s; zeroFAIL/ERROR/SKIP | PASS |
| Backend cache/error contract | Two reads/one parse, same-size/restored-mtime rewrite, repeated invalid JSON, exact read-error identity | Original assertions all pass; backend file's20 assert ASTs unchanged | PASS |
| Prepared replacement/error seams | Original tuple/model/identity replacement and JSON/buffer/context errors | Existing binding/runtime tests pass without edits; invalid preparation retains prior successful state | PASS |
| Existing output/key negatives | Unchanged wrong-key rejection, snapshots and original late-JSON prefix | 64 wrong-key negatives, detached trace/ledger/source mappings, restored buffer and3,902B prefix | PASS |
| Source scope | Only existing cache representations/read sites change | Exact transformation matches both full source files; every other byte and native cache key unchanged | PASS |
| Fixture preservation | No original fixture03/cache writes | All262 file names/sizes/mtimes unchanged across both final runs; selected model bytes verified against original bundle SHA | PASS |
| Old hook observation | read_text no longer intercepts byte reads | One assertion FAIL/1.49s, observed zero intercepted reads | Expected instrumentation evidence, not baseline correctness failure |

Pricing observations (seconds), without claiming a full-run speedup:

| Phase | Baseline | Candidate |
| --- | --- | --- |
| Cold wave |1.759869 |1.123448 |
| Warm samples |1.132800 /1.037333 /1.031934 |0.421348 /0.310525 /0.325204 |
| Warm median |1.037333 |0.325204 |

Observed median speedup3.1898x and time reduction68.65%. Timing includes concurrent
pytest CPU load and all existing source/numerical checks; only pricing is timed,
not setup/ledger serialization/flush. The deterministic acceptance is read count
and exact output equality, not the fastest sample. The unchanged parent emitter
also checks15 sequential emissions per side; no extra source behavior is installed.

## Exact source, evidence and budget bindings

Parent `setup.py` does not exist; actual file is `current_setup.py`, identified by
the supplied `run.py`/`binding.json`. The original setup SHA in that parent binding
is `c93b9d286342fc9b854ccb6692630542bf306a856ab4254cd84feddfaf7eb0d7`.
The parent-derived setup used here is
`b6c869b2b01bd118b02aaf0a9755ee6b87e25a7f00c9667cbd52d50a2cca2b80`.
Both local derived setups are
`b88526c33caf5f53e5852262ac8701bf0482ff1b49d9af4987aa7f648cd20091`:
only two source-root occurrences and the HEAD assertion change. The unchanged
body SHA is `1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`;
wrapper SHA is `2efbb9cface31e73173dc70f2ad626d14db7770f3a36c9467d28435336611056`.
The unchanged body retains its historical D29-IMMUTABLE-COST-FIELDS result label;
the enclosing bindings/receipt correctly identify this task and layer60.

The scratch cwd holds only the needed unchanged Step5 model JSON at its existing
relative path; SHA `e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63`.
All output/cache/runner/log CLI values relocate before `SimulationConfig` parsing.
The original main08 command/capture and seven actual source hashes are in
`baseline_binding.json` and `candidate_binding.json`. Their raw-binding links
retain the full76-model manifests; compact archives keep the exact manifest SHA.

| Changed path/lines | Step added/deleted/gross/net | Retained added/deleted/gross | Cap |
| --- | --- | --- | --- |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py`:73,813–816 |5/5/10/0 |84/0/84 |12 step;91 retained |
| `frontier/execution_time_predictor/moe_routing_workload.py`:155–165,213–214 |7/7/14/0 |191/6/197 |14 step;323 retained |
| `tests/unit/test_collective_sim_cc_backend.py`:200,207–213 |5/3/8/+2 |235/0/235 |10 step;235 retained |

Total +17/-15=32 gross, source24/test8, net+2; retained growth+2 against `1aee0af`.
No owner/global allowance changes or additional source/test paths. Final owner/
global union accounting remains root-owned, not inferred from these three rows.

Committed source SHA256: backend
`e763fb0523c21b947faf5e15749824461cd1b2f4ffec464e76d15d16be312e49`, workload
`9942587be5803af284e1e5b581f3096d87de3f70453c477d793e4fe344701017`, test
`56017c6e476b88e38f7f3dabdd3cfc3a373b76df7372e4cf147a057f506caa37`.
Review-ready `code.diff` SHA:
`51a6171e385223c3c64003361c05cc2679f97c74ab6b76b1526e707305d1b8ea`.

Selected equality hashes:
- Pricing payload: `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`.
- 208 event bytes: `18d320438ee1fb5f077b12aaa83da8055b47d8cf256d88aedcca159a5d7b819c`.
- Full ledger: `006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf`.
- Baseline/candidate JUnit: `73e04347cf8eb7f18d94b635d108e2c4fc03ead6415e0997c53c490c878d917a` / `9ede88492b42744f050ca2b26a317ccdd107375f5295e50edaeeb792bd91d696`.

`receipt.json` inventories every selected raw log/XML/setup/body/trace/ledger
path and hash. Raw logs and large outputs remain in fresh scratch, not duplicated.
Before success, unprivileged scopes failed authentication; scratch-relative model
lookup failed; a wrapper global `before` collided with the unchanged body; and
an optional counter assumed schema position. Their original logs are preserved
and resolutions are in `progress.md`. None is treated as a product correctness
regression or a passing acceptance attempt.

## Limits

Read/JSON error behavior is executed in original tests; UTF8-decode failure state
is inspected structurally (decode occurs before any cache-state assignment), not
separately fault-injected. Native/CPU qualification, source/placement/context/
buffer/count and numerical checks are byte-for-byte unchanged outside the named
representation lines. No diagnostic byte patch, stat/shared/global/wave cache,
API/flag/helper, native key change, suppressed read/check or new persistent test.
No full CLI/trainer/CSV scan/GPU/network/dependency run. Root still owns independent
review, final union/regressions, source freeze, full workflow and native/G3/E2E.
