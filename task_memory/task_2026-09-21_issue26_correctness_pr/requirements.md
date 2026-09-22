# Issue 26 Correctness PR — Requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Recorded the original request, the specification hand-off, and the decisions from the planning interview. |

## [Original Request] 2026-09-21

"this is a new task and current is plan mode, we need to discuss and land key docs first. plz read and analysis draft `.local-draft/Frontier_Issue26_Correctness_PR_Execution_Spec_2026-09-21.md` and land doc. if there is something need to consult me, use grill-me skill."

The draft specification is landed verbatim in `plan.md` together with an Amendments table.

## Decisions from the planning interview (2026-09-21)

| Question | Decision |
| --- | --- |
| Q1/Q7 Task document location | `task_memory/` inside the PR worktree; narrow `.gitignore` exception so the records are published with the PR. No `docs/development/` tree. |
| Q2 Step 7 collective-sim zero-payload | Decide at Step 7 (facts: candidate gitlink `e564935d…` unreachable on the configured remote; no local clone). |
| Q3 GPU path | StepMind runbook exists; GPU workers available. Simulator runs on local CPU; GPU only for Step 6 native check and additional profiling CSVs. |
| Q4/Q8 2,000-line gate | Full cleanup + split for the four touched oversized modules only. |
| Q9/Q10 PR organization | Stacked: `refactor/oversized-module-split` first, `fix/issue26-correctness-pr` based on it; both draft PRs open, correctness base = refactor branch, retarget after merge. |
| Q11 Refactor acceptance | ≥50 scenarios; per scenario `request_metrics.csv` value-identical and `system_metrics.json` identical after removing timestamps/run ids; any difference is FAIL unless an explicitly approved fidelity fix. |
| Q12 Task directories | Two: `task_2026-09-21_oversized_module_split` and `task_2026-09-21_issue26_correctness_pr`. |
| Q5 Publication | Authorized: push both branches to `origin`, create/update draft PRs. Not authorized: merge, force-push, history rewrite, closing Issue 26. |
| Q6 Stop point | Stop after the Step 0 push for user review. |
| vLLM reference | Clone into `.real-engine/` (local exclude), pinned to `ea95f57`. |

## Decisions taken during execution

| Date | Question | Decision |
| --- | --- | --- |
| 2026-09-22 | W5 open item 3: the single public name for the routing implementation identity, asked under the AGENTS.md naming gate with three materially plausible spellings. | **`moe_routing_runtime_path`**, a `ReplicaConfig` field, CLI `--replica_config_moe_routing_runtime_path`. It shares the prefix of the neighbouring `moe_routing_seed`, `moe_routing_trace_path` and `moe_routing_distribution_type`. The profiling CSV column stays `routing_runtime_path` and the standalone trainer flag stays `--routing_runtime_path`; neither is renamed. |
| 2026-09-22 | W5 scope, asked after the naming decision: global field only, or global plus per-role overrides. | Global + per-role overrides (`prefill`, `decode`, `decode_ffn`). Superseded the same day by the row below. |
| 2026-09-22 | `[Original Request]` "我在rethink添加 frontier.moe_routing_runtime的必要性。我的concern是，该部分的align是否对模拟准确度意义重大？如果对fidelity的影响不到0.5%,我认为完全无需引入如此复杂的变量和setting,这会使得可用性和可读性变得很困难" -> after the explanation of what the module decides and the measured 3-8% per-layer cost of a wrong path versus the 0% change of the unset override: "我认为采用 当前默认的 两个 cluster 都只能读同一个全局 moe_routing_distribution_type，永远解析出同一个 path ；进行回退" | **W5 not ported.** Keep the existing contract: one global `moe_routing_distribution_type`, one derived routing path per run, no override field, no registry routing axis. The drafted implementation was reverted before any commit and archived as `w5_reverted_moe_routing_runtime_path.patch`. Both earlier W5 decisions above are void. |
