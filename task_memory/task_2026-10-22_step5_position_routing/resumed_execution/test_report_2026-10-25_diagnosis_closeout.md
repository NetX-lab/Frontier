# D29 finite diagnosis integration verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 HKT | Verify and consolidate copy/init records without new controls or source changes. |

Both finite diagnoses are complete. Snapshot-copy is NO_ALLOC from supported static copy/error counterexamples, with0/1 controls and0 launch attempts. Init is ALLOC_RECOMMENDATION_ONLY, with0 new/recovered controls and0 source allocation. Its126.58s modeled removed work is historical inference, excludes the accepted loader saving, and is not current startup qualification.

Parent metadata verification passes25 current copy bindings/seven owner records and13 current init bindings/14 retained receipts/six owner records. The init owner6335f581 has the expected886608ea parent and a clean retained worktree. Mergec6b3da15 completes without conflicts; its six records equal the owner bytes. Main non-task source equals production2ca5e571 and tested146. CORE4377/4404,PRICING3955/3956,native1350/1582,source9682/9942,global9696/10000 remain the accepted production accounting; no composed audit or suite is rerun.

Exact verification command, from `/data/ycfeng/Frontier/.worktrees/step5-preview`:

```sh
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_ledger_final_1461573d_20261025/diagnosis_integration/resume_parent.py
```

Observed: Python3.10.6,metadata exit0,verification body0.424885142s. Configured MemoryMax is32,212,254,720B; cgroup peak/RSS is not measured. The exact executed script and native metadata result are byte-identically retained in `evidence/diagnosis_closeout_2026-10-25/`. Initial generic `python` reads exited127 because the alias is absent; using the specified interpreter resolves only that invocation choice.

Exact records integration command:

```sh
git -c core.hooksPath=/dev/null merge --no-ff 6335f581e0129ddb273836f058f4e84fcbb35c22 -m 'docs(step5): integrate the closed init preparation diagnosis'
```

Observed merge exit0,0.020264751s,HEADc6b3da155f920f264749e497f884722ee75d78f1. The command-local hook override prevents unassigned runtime operations and changes no configuration file. Snapshot merge52831c6e and its prior parent receipt are archived exactly, without reissuing that merge. Six exact parent artifacts plus archive.json record all paths, sizes and SHA256 values.

Record staging initially returned exit1 because task_memory is ignored by the existing Git rules. Use git add -f only on the exact authorized record files; retain the ignore configuration unchanged. This is a staging correction, not a runtime retry.

PASS conditions: records-only owner scope; expected parent/clean status; exact current source and historical receipt bindings; zero consumed controls; no source allocation; unchanged non-task source; all integrated owner records exact. Observations satisfy every condition. No application initialization, runtime control, pricing wave, required-shape call, fullCLI, platform, GPU or cloud operation ran. Required-shape usage remains6/6. Both diagnosis agents are closed; all worktrees and unrelated.cortexkit/.pi files are retained.

The remaining execution conditions are current-source joint pricing/emission/flush, worker interpreter/image/source mount/lifecycle/append, and full-workload memory/runtime. Complete6144/1024 offline/online must still retain every output/proof/key/check and pass2400 seconds separately for each frontier.main subprocess. FinalT2/source/package/native/G3/E2E remain open. The init recommendation has no patch/allowance and returns to the existing bounded allocation process only if current evidence warrants implementation.
