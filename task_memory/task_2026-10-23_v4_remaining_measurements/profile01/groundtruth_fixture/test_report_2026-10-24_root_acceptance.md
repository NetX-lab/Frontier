# Root CPU fixture acceptance

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accepted source after dual review, independent owner/main reruns and32-rank artifact checks. |

Result: **ACCEPTED_CPU_FIXTURE_RESTORATION** in [primary_acceptance.json](primary_acceptance.json). Source6abd2d8e/records0d5b0d6e integrate as0dbb617f/3c3fdce4; only the existing test file changes37/60gross lines. All17original test-function ASTs remain equal.

Both commands execute the same78-line verifier, SHA recorded in each receipt:
- `/data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/glm53_dsv4/groundtruth_fixture_root_rerun_20261024.py /data/ycfeng/Frontier/.worktrees/v4-groundtruth-fixture`
- `/data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/glm53_dsv4/groundtruth_fixture_root_rerun_20261024.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`

Environment: asyncopd Python3.12.13; actual retained native/C26 fixtures; CPU only, at most two concurrent processes. Exact child argv/environment,60s/30G per-process limits, logs/XML and freshly collected node IDs are in [owner rerun](root_worktree_verification.json) and [merged rerun](root_integration_verification.json).

| Disjoint check | Expected | Owner rerun | Merged rerun |
| --- | ---: | ---: | ---: |
|32-rank positive|1 PASS|1 PASS|1 PASS|
|Non-V2 groundtruth|80 PASS|80 PASS|80 PASS|
|V2 negatives A/B|9+9 PASS|9+9 PASS|9+9 PASS|
|V2 inventory/oracle|46 PASS|46 PASS|46 PASS|

Each execution covers145unique tests,0FAIL/ERROR/SKIP. Fresh99-node collection equals the original set and every groundtruth node executes exactly once. Both reruns rehash189protected files;15owner-to-main files and the existing matrix tree hash also match. The77-case matrix is an existing binding, not a new capture.

`python3 -B /data/ycfeng/tmp/glm53_dsv4/groundtruth_fixture_root_archive_20261024.py` validates [32snapshots](root_32rank_audit.json):16world ranks per role,14identical hooks and six warmup/formal phases per rank; prefill3456forwards/0replays, decode3456/3456, no observation errors/drops. [Local archive failures](root_archive_failures.json) preserve the mistaken engine-local/global-rank assertion and nonexistent rank8 lookup; the correction reads `groups.world.rank`, without changing tests or native behavior.

Standards `ses_e9296e63bffeBzg6BbptUCa2fF` and Spec `ses_e9296e680ffeJ02WsBKt7RqDp6` report **No findings** after complete source/spec/record review. Reviewer execution/hash claims remain supplied evidence; the two root executions above are independent.

Limits: original RED/setup errors and one-process60s timeout remain preserved. The exhaustive grouped runs do not claim a new combined466-test execution. Synthetic host returns prove neither GPU/distributed correctness nor serving/reference/operator cost. Profile04's45sealed inputs/runtime1437ace6 and existing queued job remain unchanged.
