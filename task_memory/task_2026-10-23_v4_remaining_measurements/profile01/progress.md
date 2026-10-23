# W128-V4-PROFILE01-PACKAGE

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Began CPU-only package preparation using the existing builder and accepted runtime05952396. |

## Status

- completed: Read local instructions, root requirements/plan/proposal/driver design and acceptance/input pins. Confirmed clean baseline1dc93c6f and all13 selected runtime modules equal git05952396.
- in-progress: Add one optional profile-plan input to the existing builder. Preserve the unselected06 branch, frozen checkers, original packages and all five workload files. Stage a diagnostic-only profile01 package after committing source.
- pending: Actual generation; CPU config/manifest/source/tool/model/workload verification; both-role/DP/environment/command checks; final delivery and writer stop.

## Scope and implementation reason

Only the allocated builder and profile_plan.json are source changes (300 gross-line maximum). The optional plan selects NativeServeProfile, records native ProfilerConfig fields and the existing row/byte limits, adds source pins for profiler/RPC/hook callsites and removes historical four-metric admission assertions from diagnostic case/readiness. Existing staging, launch and seal mechanisms remain authoritative; no new runtime path or test machinery.

No seal, launch, platform, SSH, cloud, GPU or Docker operation is permitted to this owner. Root owns image/source runtime checks and external destination/quota inspection. Configuration cloud paths are strings only and have not been accessed.

## Preflight evidence

CPU preflight rehashed all8 proposal input files, all5 workload files against FINAL_FREEZE_R4 and git68f891b, and all4 model metadata files. Native checkout is clean at2cf0a6915ce544dc493a0990f2ea38d81601128a. Source inspection confirms13 additional profiler/RPC/hook source files beyond74 original pins,13 runtime modules after adding the actual NativeServeProfile class, and current worker-derived role/DP directories.

Environment probe failure, preserved verbatim (not a package-generation failure):

```text
$ /data/ycfeng/envs/asyncopd-py312/bin/python -I -B -c 'from frontier.config import SimulationConfig'
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'frontier'
```

Isolated mode omits the repository from sys.path. The frozen checker explicitly adds the worktree, and its actual reconstruction passed with /data/ycfeng/envs/frontier-py310/bin/python (Python3.10.6),84 imported source files and P/D world_size16 each. Use that unchanged interpreter for the actual builder; use the approved py312/PYTHONPATH environment for package CPU controls.

## 1. Planning completion

Summary: PASS1, TODO3. Source/input preflight PASS; scoped source commit TODO; actual generation and package controls TODO; delivery/writer stop TODO. These are the owner's I1/I2 portions of root plan.md:24.

## 2. Drift detection

Window: task start through preflight. Current frontier: source commit then package generation.

| Pattern | Verdict | Evidence |
| --- | --- | --- |
| D1 Tooling polish | CLEAR | Existing builder and actual checker controls only; no new persistent tests. |
| D2 Retry loop | CLEAR | One isolated-import probe failed; the actual supported checker passed. |
| D3 Side quest | CLEAR | No unrelated source or external operations. |
| D4 Stalled frontier | CLEAR | One allocated writer; root owns independent image/platform work. |
| D5 Gold-plating | CLEAR | Optional plan and bindings directly satisfy package criteria. |
| D6 Analysis sprawl | CLEAR | Inspection produced the bounded source change and concrete preflight results. |
| D7 Off-route | CLEAR | CPU-only; generation06 and accepted runtime/frozen workload unchanged. |

Spec checks S1-S6: no material defect in the active allocation. Historical GPU-approval wording is superseded by the explicit current requirements/proposal decision and user delegation. Attribution: none.

## 3. Route correction

Branch A: no spec/plan change. Stop deviations: none. Next route: commit only allocated source/records, generate profile01 exactly once, inspect actual package and CPU paths, record delivery and stop all writes. A failed generation must remain immutable and be reported to root rather than refreshed.

## 4. Three points to watch

1. Build only after the source commit leaves a clean worktree.
2. Verify runtime bytes against05952396 and all5 workload pins, not a historical package's admission verdict.
3. Root still owns independent image/destination/runtime verification and sealing; CPU checks make no GPU/native/reference claim.
