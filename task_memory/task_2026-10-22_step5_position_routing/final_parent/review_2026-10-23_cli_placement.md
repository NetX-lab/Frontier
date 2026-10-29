# CLI placement repair closure

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record independent No findings and parent57PASS/6SKIP; preserve the fresh full-CLI acceptance boundary. |

TaskID D29-CLI-PLACEMENT-REPAIR-REVIEW. Independent reviewer inspected frozen
75ad26f9dfcd2467dde8912856333592c7c600c5,codeae1fb721,complete two-file
placement_repair.diff,root requirements/plan/allocation and owner records/logs.
Verdict: **No findings; bounded repair accepted**.

run_step5_measured_main.py157–161 adds the existing globalTP,DP selector only to
the position branch. Legacy argv,backend/gates,numerical behavior,timeouts and
cache hashes are unchanged. test_step5_measured_main.py291–326 exercises actual
command→SimulationConfig→Cluster/lazy factory→predict_collective_plan and checks
dimensions,full key,0.03ms and exact count receipt. It neither presets placement
nor converts exceptions to success. Metrics/cache outputs are relocated before
parsing; imports and the writer are from the current repository.

Reviewer read original4-case RED,5PASS focused,57PASS/6unchangedSKIP whole-file
logs and retained fixture03 constructor results for both modes and roles. It did
not run tests or platform operations. Parent then independently confirms the full
affected files:57PASS,6SKIP,0FAIL/ERROR,63.956s. Parent XML and actual40-gross delta
are bound in cli_placement_acceptance.json. Exact command:

```bash
timeout 180s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/tests/unit" TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_cli_placement_parent10 --junitxml=/data/ycfeng/tmp/d29_cli_placement_parent10.xml tests/integration/test_step5_measured_main.py tests/integration/run_step5_measured_main.py > /data/ycfeng/tmp/d29_cli_placement_parent10.log 2>&1
```

Workdir:/data/ycfeng/Frontier/.worktrees/step5-cli-placement-repair.
This closes the active constructor-wiring repair. The user requests code/docs
delivery and handoff now; the next session must run fresh full6144/1024CLI and
final-source checks. The prior1be full-CLI failure remains preserved. Neither this
review nor feature-branch consolidation grants complete D29 or GPU acceptance.
