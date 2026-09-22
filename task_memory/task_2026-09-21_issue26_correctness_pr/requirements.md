# Issue 26 Correctness PR — Requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Recorded the original request, the specification hand-off, and the decisions from the planning interview. |
| 2026-09-22 | Recorded the W6 artifact-identity decision and the native GPU validation instruction. |

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
| 2026-09-22 | W6 artifact identity: should the profiling metadata gain a column that separates an incomplete legacy `moe_grouped_gemm` measurement from a complete one? | **"不改 metadata，只记录限制"** — do not change the profiling metadata; record the limitation only. Written into `docs/profiling/README.md` under the MoE producer. No code change, no new column, no admission gate. |
| 2026-09-22 | W6 native validation: how should the GPU parity check be run? | **"请从 dockerhub 中找到 v0.10.2 的官方镜像（如果没有，fallback 到 >=0.10, <0.11），然后参考 hand-book 中对 docker 的使用在 gpu worker 上使用该镜像。如果你需要使用原来的 benchmark ... 中的测试 suits 和插桩，你需要 mount 该 repo 到 gpu worker。如果需要进行对比验证，则 follow skill：/home/brainpp/.claude/skills/frontier-calibration"** — `vllm/vllm-openai:v0.10.2` exists on Docker Hub, so no fallback was needed. It is pulled through the company docker.io proxy as `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2`. The instrumented benchmark repository is not mounted: the parity check compares tensors from vLLM's own `fused_experts` inside one process and needs no serving instrumentation. The `frontier-calibration` skill is not invoked for the same reason — its workflow is E2E simulator-versus-served-vLLM calibration, not a kernel-level tensor comparison. |
| 2026-09-22 | W7: authorize a companion-repository branch in `fwyc0573/frontier-htsim`, or record W7 as `EXCLUDED`. Asked with the option spelled out as "授权我在那个仓库建分支、提交测试和源码、推送并开 companion draft PR，然后再动 Frontier 的 gitlink。按计划这是 backend 先发布、Frontier 后跟进的顺序。" | **"1.授权"** — companion-repository work authorized. Delivered in that order: branch `fix/zero-payload-input-handling` pushed to `fwyc0573/frontier-htsim` at `eb7bc4f`, companion **draft** PR 1 opened, then Frontier's gitlink moved. PR 35 stays draft; nothing was merged. |
| 2026-09-22 | `[Original Request]` "对于当前 task 的 gpu worker 集群应该使用 codesign 而不是 step_main" | All GPU submissions for this task use `charged_group="codesign"`. Applied from `exp-0922-142415-796404` onward; the first attempt `exp-0922-140423-075005` had used `steptron_ci`, found no capacity after ~20 minutes Pending, and was stopped. The successful native parity run `exp-0922-145047-660565` is on `codesign`. |
