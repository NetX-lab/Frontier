# Primary review of the sparse metrics timer

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Accepted the retained timer evidence, preserving one malformed aggregate receipt field; no control rerun. |

The timer is ACCEPTED_TIMER_FOR_ONCE_ONLY_DIAGNOSIS in [source_review.json](primary/source_review.json). Source is98/140gross lines, SHA256fdce5eca10e0cea28af12b87d25a94ed68f0ce0de2eda77c6b5fa2ccb5b541b5;owner Markdown88/90lines. The allocated full timing diagnosis is NOT_RUN.

Primary read the whole timer, accepted adapter, frozen stage/loading-GC pipeline and owner control source. The source inserts sparse batch/group/sort/union observations and delegates the existing sorted binding once. Memberships, count/sum order, union recurrence and all other generated statements remain unchanged. Retained AST strings equal the accepted function/generated module/pipeline;all original source pins match.

The owner executed the sole tiny control using `timeout 60s /usr/bin/python3 -B task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/metrics_tail/owner/tiny_control.py`,cwd this worktree,Python3.10.6/GCC11.3.0,no conda,0.290513119s. It reported63runtime checks PASS across five fixtures. Primary did not execute/import this control or any spent full control/driver.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Retained source/AST closure | Exact pins and stripped AST | Eight source pins;function/module/pipeline equal | PASS evidence review |
| Tiny numerical outputs | Exact count/sum/union and float hex | Five fixture expected/actual arrays equal | PASS retained outputs |
| Aggregate check-row consistency | Every expected equals actual |62of63;stable_tie_order has a mutated list | Initial FAIL preserved |
| Stable tie order |[1,2,0,3] |Independent deep-copied fixture has[1,2,0,3] | PASS fixture binding |
| Runtime invocation limits | Tiny1/1,diagnostic0/1 | No new control execution | PASS |

Primary's initial command was `timeout 60s /usr/bin/python3 -B -`;its `assert len(tiny['checks']) == 63 and all(r['status'] == 'PASS' and r['expected'] == r['actual'] for r in tiny['checks'])` failed. [Failure](primary/source_review_failure_01.json) preserves the actual mismatching row. `check()` kept actual=orders by reference;later orders.clear()/append() changed that field to the final float fixture's[2,1,0]. The fixture record used copy.deepcopy(orders) and retained the stable order. [Separate binding](primary/tiny_receipt_binding.json) keeps the raw field invalid and derives the order from existing fixture evidence. Original receipt/control source are untouched;there is no new executed check.

The frozen loader/stage paths and API expressions were inspected without invoking their pipeline. Initial shell `python` was unavailable;all task commands use the existing `/usr/bin/python3`,as the host binding specifies. No dependency/environment change was needed.

The timer's batch_start follows caller-side groups construction;kernel_pass_done→batch_start measures that separate gap. Original _obs clocks are rounded to milliseconds and have log overhead. This source/evidence review supports only the one allocated diagnosis,not R5,performance/non-perturbation,native ownership,dependencies,cost or reference acceptance.
