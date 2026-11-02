# Shared routing primary acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Verified final shared-source fidelity/provenance and merged accepted source. |

## Environment and commands

Primary worktree: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
Inspection interpreter: `/data/ycfeng/envs/frontier-py310/bin/python -B` (Python3.10.6; no conda activation).
The existing root test command/result is recorded in `test_report_2026-10-22_primary.md`; its XML is `/data/ycfeng/tmp/glm53_dsv4/root_balanced_callers_186dfef.xml`.

Git inspection commands executed from the primary worktree:

```bash
git diff --name-status df99909 186dfef
git log df99909..186dfef --oneline
git diff --numstat 9d44675 186dfef -- <the twelve paths in callers_final_source_inventory.json>
git show 186dfef5503ecb5329bd0fa96f9e978cc813276a:<each inventoried path>
git merge --no-ff 186dfef5503ecb5329bd0fa96f9e978cc813276a -m 'feat(moe): integrate verified per-case balanced routing identity'
```

The Python inspections read the original final `comparison.json`, candidate `manifest.json` and all77 `results.jsonl` rows under `/data/ycfeng/tmp/glm53_dsv4/balanced_callers_186dfef_20261022/`. They asserted the exact predicates below before allowing the merge; no capture or broad test suite was rerun. `callers_primary_final_acceptance.json` is the exclusively created primary receipt.

## Observed results

| Expected predicate | Actual result | Verdict |
| --- | --- | --- |
| Comparison bytes match final owner receipt | SHA256 `adafca736b29735060bb29949acd51837de8f48990a298e35ec54e167a00bfd8` | PASS |
| Complete strict comparison | 77 compared,77 identical; no failure/mismatch/missing evidence/definition/provenance findings | PASS |
| Predictor cache identity | Clean initial cache; zero baseline-only/candidate-only filenames or findings | PASS |
| Every raw case pins final clean source | All77 source/harness revisions are186dfef5503ecb5329bd0fa96f9e978cc813276a, source_dirty=false, returncode=0 | PASS |
| Root focused suite already completed | XML218 tests,0 failures,0 errors,0 skipped | PASS |
| Full approved shared source inventory | Exactly12 files,428 added+4 deleted; all working source bytes equal the committed revision | PASS |
| Primary merge contains only reviewed source changes | Eleven changed paths relative todf99909; shared strategy already present; merge292c489f302e45f868a709fd9e24e5f52d5e505e | PASS |

## Limits and continuation

Accepted scope is shared source and CPU correctness/default-fidelity only. The strict77 baseline is retainedbfa82de, as recorded in the unchanged comparator; it is not a new native performance reference. B/C integration, final merged-source checks, package/image admission, actual balanced H800 execution and numerical calibration remain pending.
