# D29 plan-key primary verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accept the component, replay exact output, and bind final union accounting. |

## Result and environment

**ACCEPTED_COMPONENT_NOT_WORKFLOW.** Owner source `c4e06a61` is integrated as
`30fb6a30`; final checked source/records revision is `46fe83ac`.
The entire non-`task_memory` tree matches the verified owner revision. The
existing submodule remains `3dac53c`.

Linux host `kun-workspace-vgen2`; `/data/ycfeng/envs/frontier-py310/bin/python`,
Python 3.10.6. Commands run as `brainpp`, with `python -B`, single BLAS threads
for runtime tests, and systemd `MemoryMax=30G`. Tests are capped at 240 seconds;
each wave/check command at 60 seconds. Raw evidence root:
`/data/ycfeng/tmp/d29_plan_key_parent01_wave/`.

| Check | Expected | Observed |
| --- | --- | --- |
| Existing affected unit suites | No new failure/error/skip or missing ID | PASS: 124 PASS / 1 existing Torch SKIP, 11.83 seconds; all 125 IDs/statuses equal retained baseline |
| Owner evidence/source binding | Exact hashes, unchanged protected body and driver | PASS: 38 manifest files and 22 source hashes per arm; body `1cabc04f…`, driver/wiring copied without behavioral edits |
| Complete real FP8 wave | 208 costs/proofs/keys/order and exact output | PASS: pricing 647,510 B, ledger 2,309,545 B, whole JSONL 2,889,764 B byte-identical |
| Negative and ownership checks | Preserve errors, prefix and independent snapshots | PASS: 64 wrong-key checks; 3,902 B late-error prefix; getter, pending metadata, source and ledger isolation |
| Work and input lifetime | Keep live checks; remove only repeated encoding | PASS: 192 pricing key calls, 288 numerical lookups, 144 fresh proofs, 64 collective prices, 16 point queries; one binding read and zero cold/warm reads per arm |
| Actual key encoding | One encoding per retained immutable plan | PASS: cold 200→8, warm 192→0 per wave, emission 64→0 |
| Integrated union | Correct two-path delta and all other bindings unchanged | PASS: 16/17 gross, 14/15 retained; 122 non-key bindings unchanged; 9,477 source / 9,491 global, no violations |

Final owner totals: CORE 4,173/4,404; PRICING 3,954/3,956; native 1,350/1,582.
There are 114 allocated / 113 changed paths. Acceptance combines the previously
accepted complete snapshot audit, independently recomputed two-path delta, and
the guard rerun on integrated `46fe83ac`. It does not claim that unchanged
historical checkpoint measurements were rerun.

Owner sequential timing remains the performance comparison: pricing median
0.161007→0.119742 s; emission 0.051601→0.038156 s. Parent baseline/candidate
processes ran concurrently for parity: their timing observations in
`verification.json` are not an independent sequential speedup estimate.

## Exact commands

Affected-test working directory: `/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key`.

```bash
timeout 240s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key:/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key/tests/unit PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -ra tests/unit/test_step5_moe_pricing.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_contract_reuse.py -o cache_dir=/data/ycfeng/tmp/d29_plan_key_parent01_pytest_cache --basetemp=/data/ycfeng/tmp/d29_plan_key_parent01_pytest --junitxml=/data/ycfeng/tmp/d29_plan_key_parent01.xml > /data/ycfeng/tmp/d29_plan_key_parent01.log 2>&1
```

Remaining commands ran from `/data/ycfeng/Frontier/.worktrees/step5-preview`.
The two wave commands were launched independently in parallel; the loop below
lists their exact per-arm arguments. Preparation and verification ran once.

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_plan_key_parent01_wave/prepare.py > /data/ycfeng/tmp/d29_plan_key_parent01_wave/prepare.log 2>&1
for arm in baseline candidate; do
  timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_plan_key_parent01_wave/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_plan_key_parent01_wave/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_plan_key_parent01_wave/cache/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_plan_key_parent01_wave/cache/numba /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_plan_key_parent01_wave/run.py "$arm" > "/data/ycfeng/tmp/d29_plan_key_parent01_wave/$arm.log" 2>&1
done
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_plan_key_parent01_wave/verify_parent.py > /data/ycfeng/tmp/d29_plan_key_parent01_wave/verify_parent.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 D29_GUARD_ONLY=1 D29_CANDIDATE=46fe83ac D29_CANDIDATE_RECORDS=46fe83ac /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/repairs/evidence/plan_key_scope/recompute_plan_key_delta.py > /data/ycfeng/tmp/d29_plan_key_parent01_wave/parent_union.json 2> /data/ycfeng/tmp/d29_plan_key_parent01_wave/parent_union.log
```

`prepare.py` verifies owner hashes, creates fresh scratch collective copies and
refreshes only source HEAD/status metadata. The accepted driver's pricing,
output and mutation checks are unchanged. The verifier relocates output and
retained owner-test paths, then independently checks the fresh parent JUnit.
The existing owner RED/GREEN receipts were checked, not rerun by the parent.

## Evidence and remaining work

`acceptance.json` binds nine selected raw artifacts, both review tasks and the
audit inputs. `verification.json` distinguishes retained owner tests from fresh
parent tests. `union_guard.json` retains the guard output without relabeling its
own two-path status; primary union acceptance is the composed evidence above.
The non-key digest is `fe524bcd1e36662489cb74c64c0bc4f071de3c107bebb65179a758bc4b2e81fe`.

No new correctness failure was observed. The known missing-Torch skip remains.
This canonical cached-manager replay is not full Simulator/CLI/native
qualification. Fresh frozen regressions are assigned separately; the full
6144/1024 workflow still needs to meet 2400 seconds per mode and resolve output
storage. T2 staging/seal/platform, native, G3 and E2E gates remain open.
