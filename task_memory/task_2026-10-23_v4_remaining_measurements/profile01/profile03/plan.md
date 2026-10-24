# W128-PROFILE03-PACKAGE plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Added the records-only P3 correction, fixed-base diff and unchanged-input rehash follow-up. |
| 2026-10-24 | Defined the source, generation, CPU verification and stop milestones. |

## Dependency and ownership

`A current acceptance + fresh worktree -> B profile-only builder/plan + fresh bindings + source commit -> C one immutable generation -> D {integrity/manifest/domain checks, staged entry/node/native-config CPU checks} -> E records commit + delivery -> STOP`.

No subagents. Root image/SDK/review/seal/launch work is outside this slice, not a dependency on CPU-only delivery. Only existing builder PROFILE branch and new profile03 area are maintained; generated candidate03 is a separately counted artifact.

| Step | Done condition |
| --- | --- |
| A | Current PRIMARY documents read; branch/worktree created from1437ace6, no prior worktree reused. |
| B | Actual integration acceptance copied byte-identically; fresh freeze binds13 Git module bytes; proposal binds original workload/reference/matrix; builder guard uses current status/source_revision; role override and case generation come from committed plan; combined gross<=140; native06 behavior unchanged. |
| C | Clean committed source produces candidate_h800_p0_profile_03 once, unsealed/unsubmitted; package inputs remain immutable afterwards. |
| D |13 runtime,87native,20 retained binary pins; all hashes and manifest closure; four role/lane plans;512/64,600000/268435456; balanced canonical workload; actual staged launcher/bootstrap import paths verified with pre-write stop controls; prior files/checkers/sealed inputs unchanged;77-case/domain binding valid. |
| E | Exact commands, counts, source/records SHAs, budgets, receipts, hash inventory and root gates retained under profile03; owned-only commit, clean tree, STOP. |

## Verification

Review follow-up: `P3 reference correction -> exact1437-to99467718 maintained-two-file diff/numstat -> unchanged package/manifest/prior-pin rehash -> record-only commit -> STOP`;<=25gross Markdown lines, rawdiff/JSONevidence separate. No source/package operation or new qualification; root closes reviews and performs actual image/SDK/seal checks.

Use existing CPU interpreters and prior package entry/config check mechanisms. Launcher executes only on a disposable copy with a clearly marked test-only manifest and tracing that exits before write_run_manifest's body. Bootstrap executes only against explicitly labeled fake installed-source/binary/PathFinder stand-ins and a fresh child import. No real credentials, platform modules, cloud access, native constructors, GPU or network. Raw logs and disposable scripts stay under `/data/ycfeng/tmp`; selected receipts/scripts are retained here for reproducibility. Validate actual installed image/binary files only by root later.
