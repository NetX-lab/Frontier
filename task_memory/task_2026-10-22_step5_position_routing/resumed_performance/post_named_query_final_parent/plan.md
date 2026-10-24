# D29 final parent audit plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Define bounded receipt verification and binding checks before parent admission. |

Dependency: exact committed inputs -> {receipt/JUnit/strict audit, source/input/script/scope hashes} -> acceptance/report -> records-only commit.

1. Bind owner commits, root requirements/plan, prior parent mechanisms, and
   partial short admission. Verify every durable owner artifact and retained/raw
   copy without rerunning suites. Reuse the existing exact-ID parser and strict
   side auditor; independently recover the eight/eleven additions and raw gates.
2. Stream-hash exactly listed source/input/script files; verify source tree,
   gitlink, clean state, and absent canonical binary. Reuse the accepted scope
   guard, changing only its output destination. Retain every hash in phase records.
3. Compose a parent receipt only after both phases pass. Report actual raw exit 1,
   79 presence FAIL, short-suite independence, environmental reconciliation,
   exact counts/hashes, commands and limits. Re-read requirements, check progress,
   and commit only this subtree.

Verification command envelope: `timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /data/ycfeng/envs/frontier-py310/bin/python -B <checker> <phase>`.

Each phase must independently finish inside 60 seconds. No raw gate is converted
into PASS; parent admission distinguishes the approved replacement from raw
comparator failure. No test stack, source, data, or native package is changed.
