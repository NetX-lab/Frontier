# D29 Ledger Production Integration Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | Integrate final verified ledger code/records and pass the complete parent source/binding/budget guard. |

## Result

ACCEPTED_PRODUCTION_SOURCE_ONLY at2ca5e57113b0bc99ab84e926e5bb492b3670072e,branch feat/step5-preview. Tested candidate1461573de388b72450189bc0c1b88b4ac32491f0 is a merge parent; all non-task tree bytes match it. Source9682/9942 and global9696/10000 are accepted; previous9566/9580 usage remains historical. Full D29 workflow is not yet accepted.

## Commands and merge evidence

`git merge --no-ff --no-commit fix/d29-ledger-stream` returns1 with six task-record conflicts and no source conflict. Each conflict is resolved against stage1/2/3: retain candidate delegated-owner additions in ledger requirements/plan and latest parent review/regression/archive/current-state records in the four root files. Exact conflict paths, hunk counts, stage hashes and resolution hashes are in evidence/production_integration_2026-10-25/resolution.json. No behavior is invented.

All four source files match146 byte-for-byte. `git diff --cached --quiet 1461573de388b72450189bc0c1b88b4ac32491f0 -- . ':(exclude)task_memory'` returns0. Required regressions have already run on that identical source and are not repeated. `git commit -m 'merge(step5): integrate verified complete EP ledger persistence'` creates2ca5e571 with parents094ecce3 and1461573d. Source/resolved-document staged whitespace check returns0. Original pytest-log/XML whitespace is retained from the preceding archive, not rewritten.

## Fresh parent guard

Exact command:

```bash
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/merge_ledger/parent_guard OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/merge_ledger/parent_guard/audit.py
```

The retained complete audit is bound to actual production2ca5e571. Only known unrelated untracked .cortexkit/.pi entries are permitted at primary; full statuses and index hashes are preserved and unchanged. Submodule must remain clean at3dac53c544e546574cb6e8078cd14053d0916113. All scope/numerical accounting predicates are unchanged.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Complete allocation | 116 paths; owner caps unchanged | 116 paths,18 zero-sum overlays | PASS |
| Previous actual usage | Independently recompute114 rows | All114 exact | PASS |
| Source/native/image binding | All126 equal final frozen candidate | All126 exact | PASS |
| CORE | <=4404 | 4377 | PASS |
| PRICING | <=3956 | 3955 | PASS |
| Native | <=1582 | 1350;actual native patch reconstructed | PASS |
| Source/global | <=9942/10000 | 9682/9696 | PASS |
| Ledger step | gross<=135;caps36/20/65/14 | 124gross,96net,cumulative increment116 | PASS |
| Individual step usage | <=36/20/65/14 | 33/15/62/14 | PASS |
| Source/index immutability | Before/after identical | Identical | PASS |

Guard subprocessexit0; audit duration2.113390676s. Environment is Python3.10.6,/data/ycfeng/envs/frontier-py310/bin/python,-B,single math threads,configuredMemoryMax30GiB,timeout60s. Effective cgroup peak is not measured; process-maxrss in the receipt is not a whole-workload measurement. No Conda environment is selected. Full command/exit/source metadata and every Git/native-patch operation are retained under evidence/production_integration_2026-10-25/parent_guard/.

## Remaining limits

Canonical raw8028PASS/279FAIL/32ERROR/281SKIP and the historical approved old-ID rawpresenceFAIL remain unchanged. Existing frozen evidence is reused only because all source bytes are identical; no suite count is summed. This source-admission result does not establish cohesive pricing+emission+flush, worker lifecycle/append throughput/full-workload RSS, complete6144/1024 CLI<=2400s, finalT2/package,native/G3 or E2E<=10%. Required-shape6/6 remains exhausted. No new GPU/cloud/shape/pricing/fullCLI operation occurs here.
