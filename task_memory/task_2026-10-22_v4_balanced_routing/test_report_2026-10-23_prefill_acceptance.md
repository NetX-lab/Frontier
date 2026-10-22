# Balanced prefill primary source acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Verified C runner-value fix, actual consumer controls and affected regression before integration. |

Source:8c2502c; records:b9b90e1. Worktree `/data/ycfeng/Frontier/.worktrees/v4-h800-prefill`; interpreter `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6. No conda activation or dependency changes.

## Commands and results

Root executed the exact bash block in `test_report_2026-10-23_runner_values.md` from that worktree using `zsh -c`. Producer and actual `read_isolated_scope` rejected the same-name wrong runner class; valid balanced receipt, balanced consumer and absent-selector legacy consumer passed. All five values were true and the command exited0. Raw output: `/data/ycfeng/tmp/glm53_dsv4/root_prefill_runner_r1_20261023.log`.

The outer reader initially attempted `json.loads` on all stdout and failed because the existing architecture warning precedes the JSON. Root parsed the retained object beginning after the first newline followed by `{`; it contained exactly five true booleans. The warning and original bytes were preserved; the control was not repeated. This was a receipt-reader error, not a production failure.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider -k 'exact_prefill or exact_isolated or attention_rejects_unqualified or attention_does_not_median or attention_layout' --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_prefill_runner_r1_20261023.xml > /data/ycfeng/tmp/glm53_dsv4/root_prefill_runner_r1_affected_20261023.log 2>&1
```

Expected all22 affected checks to pass; XML observed22 tests,0 failures,0 errors,0 skipped: **PASS**.

Root also ran `git show 8c2502c:<path>` for all five C-owned files and compared the output bytes to the working files, then `git diff --numstat a71fcd3 8c2502c -- <the same five paths>`: exact bytes and five files/190 routing changed lines, **PASS**. The five-path list is in `diff_count.json`; final builder selections remain outstanding.

## Review and limits

Independent `W128-V4-BALANCED-PREFILL-R1-REVIEW` APPROVE closes the original runner-values P2. Accepted scope is source/CPU integration. Fixture0.125ms is a correctness control, not measured performance. Fresh package, installed-image checks, actual H800 routing/model execution and measurement-table acceptance remain pending. B's schema/DP findings do not block this independent prefill path.

## Merged-source and builder selection

Root merged source and resolved only three task-document add/add conflicts, preserving the primary approvals and the C owner history. All five C implementation files remained byte-identical to8c2502c before final builder selection. Combined A/C affected command added `tests/unit/test_balanced_routing.py` and `selected_producer_rows or disaggregated_roles` to the command above, with `PYTHONPATH` set to the primary worktree plus `/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages`. XML `/data/ycfeng/tmp/glm53_dsv4/root_balanced_ac_merged_20261023.xml`:26 tests,0 failures/errors/skips. Merge84c806f.

Root then selected six existing builder values: package `h800_prefill_balanced`, name `PREFILL_WARM_6144_C0_BALANCED`, current-clean-HEAD revision via the existing Git snapshot mechanism, revision-derived job/cloud names, and task identity `W128-V4-BALANCED-PREFILL`. This is exactly12 additional baseline changed lines; C totals five files/202 lines. The whole delivered feature with retained B is887/900. AST/import/current-HEAD/name/fresh-path assertions and source whitespace check passed; builder main was not invoked. The previously checked model/geometry/measurement arguments are unchanged.
