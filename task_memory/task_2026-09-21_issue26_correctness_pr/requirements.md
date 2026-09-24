# Issue 26 Correctness PR — Requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Recorded the calibration-tool follow-up decision (Q8 round 2) and the user's merge of companion PR 1. |
| 2026-09-24 | Recorded the follow-up request for items 1-7 and its interview decisions (Q1-Q9). |
| 2026-09-24 | Recorded the W7-R4 decision (option a) and the removal of the final-validation worktrees. |
| 2026-09-21 | Recorded the original request, the specification hand-off, and the decisions from the planning interview. |
| 2026-09-22 | Recorded the W6 artifact-identity decision and the native GPU validation instruction. |
| 2026-09-22 | Recorded the PP>1 `vllm_load_balancing` request, the codesign-only GPU instruction, and the open Step 9 decisions. |
| 2026-09-23 | Recorded the W9-01 merge-forward request after PR 36 merged. |
| 2026-09-23 | Recorded the D9-2 decision (group-anchored report key). |
| 2026-09-23 | Recorded the W9-04, W9-05 and P5-worktree decisions. |
| 2026-09-23 | Recorded the G3–G5 GPU authorization and the decision to advance W9-05. |

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
| 2026-09-24 | Recorded the C4 disposition, the S43/S42 task split, the W9-05 worktree removal, the dummy-mode question and the fix-review request. |
| 2026-09-22 | W5 scope, asked after the naming decision: global field only, or global plus per-role overrides. | Global + per-role overrides (`prefill`, `decode`, `decode_ffn`). Superseded the same day by the row below. |
| 2026-09-22 | `[Original Request]` "我在rethink添加 frontier.moe_routing_runtime的必要性。我的concern是，该部分的align是否对模拟准确度意义重大？如果对fidelity的影响不到0.5%,我认为完全无需引入如此复杂的变量和setting,这会使得可用性和可读性变得很困难" -> after the explanation of what the module decides and the measured 3-8% per-layer cost of a wrong path versus the 0% change of the unset override: "我认为采用 当前默认的 两个 cluster 都只能读同一个全局 moe_routing_distribution_type，永远解析出同一个 path ；进行回退" | **W5 not ported.** Keep the existing contract: one global `moe_routing_distribution_type`, one derived routing path per run, no override field, no registry routing axis. The drafted implementation was reverted before any commit and archived as `w5_reverted_moe_routing_runtime_path.patch`. Both earlier W5 decisions above are void. |
| 2026-09-22 | W6 artifact identity: should the profiling metadata gain a column that separates an incomplete legacy `moe_grouped_gemm` measurement from a complete one? | **"不改 metadata，只记录限制"** — do not change the profiling metadata; record the limitation only. Written into `docs/profiling/README.md` under the MoE producer. No code change, no new column, no admission gate. |
| 2026-09-22 | W6 native validation: how should the GPU parity check be run? | **"请从 dockerhub 中找到 v0.10.2 的官方镜像（如果没有，fallback 到 >=0.10, <0.11），然后参考 hand-book 中对 docker 的使用在 gpu worker 上使用该镜像。如果你需要使用原来的 benchmark ... 中的测试 suits 和插桩，你需要 mount 该 repo 到 gpu worker。如果需要进行对比验证，则 follow skill：/home/brainpp/.claude/skills/frontier-calibration"** — `vllm/vllm-openai:v0.10.2` exists on Docker Hub, so no fallback was needed. It is pulled through the company docker.io proxy as `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2`. The instrumented benchmark repository is not mounted: the parity check compares tensors from vLLM's own `fused_experts` inside one process and needs no serving instrumentation. The `frontier-calibration` skill is not invoked for the same reason — its workflow is E2E simulator-versus-served-vLLM calibration, not a kernel-level tensor comparison. |
| 2026-09-22 | W7: authorize a companion-repository branch in `fwyc0573/frontier-htsim`, or record W7 as `EXCLUDED`. Asked with the option spelled out as "授权我在那个仓库建分支、提交测试和源码、推送并开 companion draft PR，然后再动 Frontier 的 gitlink。按计划这是 backend 先发布、Frontier 后跟进的顺序。" | **"1.授权"** — companion-repository work authorized. Delivered in that order: branch `fix/zero-payload-input-handling` pushed to `fwyc0573/frontier-htsim` at `eb7bc4f`, companion **draft** PR 1 opened, then Frontier's gitlink moved. PR 35 stays draft; nothing was merged. |
| 2026-09-22 | `[Original Request]` "对于当前 task 的 gpu worker 集群应该使用 codesign 而不是 step_main" | All GPU submissions for this task use `charged_group="codesign"`. Applied from `exp-0922-142415-796404` onward; the first attempt `exp-0922-140423-075005` had used `steptron_ci`, found no capacity after ~20 minutes Pending, and was stopped. The successful native parity run `exp-0922-145047-660565` is on `codesign`. |

## [Original Request] 2026-09-22 — PP>1 support for `vllm_load_balancing`

"添加需求：我需要在当前pr中补全 vllm_load_balancing_cluster_scheduler.py 的模拟支持，使得其不被限制在pp=1；你需要基于frontier 和vllm的codebase进行充分调研（codebase design skill)和设计，并且运行vllm v0.10.2进行实际调度结果的对比（pp>1情况下的dp 调度策略；调用calibration来确保参数设定一致）。请你先设计落地该子任务的plan，在我批准之前暂不执行"

Reading: (a) remove the PP1 restriction of `VllmLoadBalancingClusterScheduler` inside PR 35; (b) research both codebases and design first; (c) compare against a real vLLM v0.10.2 PP>1 DP-scheduling run, using the `frontier-calibration` skill to keep settings consistent; (d) deliver the plan first and do not execute before approval. Plan: `plan.md` §18 (Step 9) and amendment A12.

## [Original Request] 2026-09-22 — GPU charged group

"ps：后续的gpu worker集群只允许使用 codesign（暂停对steptron_ci的使用，直至得到我允许）"

Rule: every GPU submission uses `charged_group="codesign"`; `steptron_ci` is suspended until the user allows it again. Also saved as a durable memory note.

## [Original Request] 2026-09-22 — Step 9 decisions

"d-a: /home/brainpp/.claude/skills/frontier-calibration 依据该skills   D-b：授权  D-c：tiny Qwen3-MoE + dummy 权重 + 跳过 tokenizer（无需下载）；  D-d：采用hook 名 on_replica_batch_scheduled  D-e：暂时使用 dummy模式验证，如果过程中存在无法解决block转为h800 profiling模式； D-f：选择worker 挂载：code_mount_point=/data/ycfeng/Frontier（父目录，同时覆盖 .real-engine/vLLM-BS；  D-g：/home/brainpp/.claude/plugins/cache/claude-plugins-official/mattpocock-skills/1.2.3/skills/engineering/codebase-design  完成上述问题确认，统一更新docs（确保上述执行plan和思路和已有观察被清晰记录）并提交push"

| Id | Question | Decision | Recorded in |
| --- | --- | --- | --- |
| D-a | Calibration helper archive absent. | Follow `frontier-calibration` v2 as written. The case path (`parity-run` → `workflow-gap-analysis`) binds no pinned helper; entries that do (`e2e-metrics-gap`, `op-supplement`, `dispatch-align-trace`) are off-path and would be `FAIL` if needed. | `plan.md` §18.7, §18.9 |
| D-b | vLLM-BS instrumentation. | Authorized: local commit on `feature/frontier-comparison-instrumentation` (engine identity in schedule rows; per-engine publication log). Push not requested. | `plan.md` G1 |
| D-c | Model and tokenizer. | Tiny Qwen3-MoE config, dummy weights, tokenizer skipped. | `plan.md` §18.6 |
| D-d | Hook name and key rule. | `on_replica_batch_scheduled`. Key rule left to the P1 probe (K3 recommended). | `plan.md` D9-1, D9-2 |
| D-e | Frontier timing for the placement check. | Dummy mode first; switch to H800 profiling mode only on an unresolvable blocker. | `plan.md` §18.6, §18.7 |
| D-f | Worker mount. | `code_mount_point=/data/ycfeng/Frontier`. | `plan.md` §18.6 |
| D-g | "codebase design skill". | The `codebase-design` skill at the path above; applied in `plan.md` §18.10 and `design.md` W9. | `plan.md` §18.10 |

Also requested: update the records so the execution plan, reasoning, and observations are clearly recorded, then commit and push. Not yet given: an explicit start signal for P1/G1.

## [Original Request] 2026-09-22 — External review corrections

"充分阅读理解/data/ycfeng/Frontier/.local-draft/Frontier_PR34_PR35_Current_Code_and_PP_Extension_Review_2026-09-22.md，逐条校对，采纳正确和高价值建议，修正补充代码和docs。暂不开启new subtask的执行。"

Reading: read the external review in full, verify each finding against the source, adopt the correct and high-value ones, and fix or supplement code and docs. Do **not** start the new sub-task, i.e. the Step 9 / W9 implementation (the review's package F).

| Item | Decision / disposition |
| --- | --- |
| Scope | Review packages A (PR34 cache eligibility), B (mixed-batch dense-layer credit), C (FP8 test wiring, optional-torch skip), D (evidence wording, records consistency, PR bodies) and E (Step 9 plan corrections, records only) are adopted. Package F (W9 implementation) is excluded by the user's instruction; no W9 source change and no GPU submission. |
| Findings verified before editing | C34-01, C35-01..05 confirmed from source; P9-01..P9-06 accepted (P9-02 reproduced on the real balancer; P9-05 divisibility guard and layer counts confirmed). The review's "PR35 mergeable=false" was stale: both PRs report `MERGEABLE`; PR34 is not a draft on GitHub. |
| GPU | The corrected FP8 native check (package C) was **not** re-run on a worker; it needs one H800 under `codesign` and a fresh go from the user. Recorded as `NOT_RUN`. |
| Publication | PR34 correction committed and pushed first, then merged (not rebased) into PR35; PR35 commits pushed; both PR bodies updated through the API. Draft state of PR35 untouched. |

## [Original Request] 2026-09-22 — FP8 native rerun authorization and second Step 9 plan review

> 授权：FP8 native 重跑（1×H800 codesign）；再次review plan §18，review的核心为： 确保当前计划的代码模块的实现/改动/重构是基于整体codebase的，记住，对frontier 核心模块的代码的修改和实现上，确保可读性和可维护，任何引入的修改和实现都应该是高价值的（要么对fidelity有收益，要么与模拟功能直接相关，不可替代），禁止hard-coding，禁止临时补丁，禁止过度防御，禁止冗余性设计和实现，禁止使用ai味命名函数和变量。

| Item | Decision / outcome |
| --- | --- |
| FP8 native rerun | Authorized and executed: `exp-0922-202645-561899`, `codesign` / 1×H800, 8 passed in 14.27 s, exit 0 (W6 report §8). |
| Plan §18 second review | Performed against `c231322` with the stated gates; findings R9-01..R9-08 in `plan.md` §18.12, amendments to D9-1, D9-2, P1, §18.10, §18.11 and `design.md`. Records only; no Step 9 source change; execution still awaits the user's start signal. |

## [Original Request] 2026-09-23 — W9-01 merge-forward after PR 36

> 我已经完成 PR 36 merge，把 origin/main merge 进 fix/issue26-correctness-pr（用 merge，不 rebase），重跑 G3b、G9、G10 作为 composition check。通过后恢复 Step 9 的 P1(b) 和 D9-2；暂不处理 pr34和35的 gitingore（由我在merge前人工处理）

| Item | Decision |
| --- | --- |
| Merge | PR 36 was squash-merged into `main` as `4ab1964` by the owner. `origin/main` is merged into this branch with a merge commit; no rebase. |
| Composition check | PR 36 matrix groups G3b, G9 and G10 rerun on the merged tree. |
| Next | On a pass, resume Step 9 P1(b) and the design checkpoint D9-2. |
| `.gitignore` | The `task_memory` exceptions of PR 34 and PR 35 stay as they are; the owner removes them before those merges. |

## [Decision] 2026-09-23 — D9-2 report key

Question (AskUserQuestion): "D9-2：P2 应实现哪条 report key 规则？"

Answer: "Group-anchored (Recommended)", selected together with its preview.

| Item | Decision |
| --- | --- |
| Key rule | `key(l) = max(C.joinable_forward_group_id, last_admitted_key[l] + 1)`. An admission stores its key and reports it while the pipeline has room; otherwise the key is held. A completion reports the held key, or `key(l)` without storing it (`design.md` "Design checkpoint D9-2", plan §18.15). |
| Included with the option | A read-only `StageExecutionContext` property and two per-lane dicts in the policy scheduler. The policy's use of the `ForwardSyncState` key is removed. C1 regains the MoE `attn_dp=2, PP=3` row on the analytical backend. |
| Not decided | W9-03 (reference DP lockstep under PP) stays an observation outside Step 9. GPU runs G3/G4 still need their own authorization (manifest BLOCKED). |

## [Decision] 2026-09-23 — W9-04, W9-05 and the P5 worktrees

Questions (AskUserQuestion) after Step 9 P6. Answers, verbatim:

- W9-04 (stale first-layer placeholder deadlock): "本 PR 修复 (Recommended)".
- W9-05 (requests lost mid-decode under KV pressure): "暂缓，单独立项 (Recommended)".
- `.worktrees/p5-fidelity-{before,after}`: "删除 (Recommended)".

| Item | Decision |
| --- | --- |
| W9-04 | Fix in this PR with option 1 of `issues.md` W9-04: in `enter_layer_sync`, drop an idle placeholder whose lane's stage has since become busy. Add a regression test, then rerun the deadlock sweep, the PP=1 policy C2 matrix, the 71-case fidelity matrix, stage-admission groups G3b/G9/G10 and the suites. |
| W9-05 | Deferred as a separate correctness item; recorded in `issues.md` only. No source change here. |
| P5 worktrees | Removed 2026-09-23 with `git worktree remove` (the `after` tree held two generated `config.json` run outputs only, so `--force`). |

## [Decision] 2026-09-23 — G3–G5 GPU authorization and W9-05

Message, verbatim: "授权上述1-2，推进W9-05". It answers the numbered next steps
after the W9-04 publication: (1) remove the validation worktrees
`.worktrees/w9-04-{before,after}`, (2) run the Step 9 vLLM-side GPU comparison
G3–G5.

| Item | Decision |
| --- | --- |
| Validation worktrees | Removed 2026-09-23 with `git worktree remove` (`w9-04-after` held two generated `config.json` run outputs only, so `--force`), then `git worktree prune`. |
| G3–G5 | Authorized. Manifest decision `D-gpu-authorization` is answered. Scope is plan §18.5/§18.6/§18.9: S0 smoke on 2×H800 and S1 on 4×H800, `codesign` only, within the §18.9 budget of at most three jobs of at most one hour each. G2 must complete before the first submission. |
| W9-05 | Superseded the earlier deferral: diagnose and fix in this branch under the approval rules. A change to a shared interface or beyond the step's size limits still needs approval. |

## [Decision] 2026-09-24 — C4, S43/S42, worktrees, dummy mode and a fix review

Message, verbatim: "C4 选择a; S43 和 S42 各开一个单独的校准和修复任务； 允许删除 .worktrees/w9-05-{before,after}；当前仅用dummy模式来进行逻辑侧的矫正和修复是否合理？是否存在遗漏情况？ 对已有修复进行一次review并修正review中发现的问题；"

It answers the numbered decisions after G5 (plan §18.21).

| Item | Decision |
| --- | --- |
| C4 | Option (a): accept `SCENARIO_NOT_REACHED` as the C4 outcome. The per-burst qualification is not changed after the fact, and the last authorized GPU job is not used. |
| S43, S42 | Each becomes its own calibration and repair task, outside this PR. Any Frontier change there follows the calibration contract ("Analysis Before Code Change"). |
| Validation worktrees | `.worktrees/w9-05-{before,after}` removed 2026-09-24 with `git worktree remove` (`--force` for `after`, whose uncommitted files are byte-identical to `75c1140` or already committed), then `git worktree prune`. |
| Dummy mode | Question: is correcting and repairing the logic in dummy mode alone sound, and is anything missed? Answered with evidence in `progress.md` and plan §18.22. |
| Fix review | Review the existing fixes of this branch and correct the confirmed findings, under the quality gates and approval rules. |

## [Decision] 2026-09-24 — W7-R4 and the final-validation worktrees

Message, verbatim: "W7-R4 选择a；允许删除 .worktrees/review-final-{base,head}"

| Item | Decision |
| --- | --- |
| W7-R4 | Option (a): the companion runner rejects `tensor_bytes=0` for every collective kind except all-to-all, instead of clamping it to one byte (b) or pricing it at 0 ms (c). |
| Final-validation worktrees | `.worktrees/review-final-{base,head}` removed 2026-09-24 with `git worktree remove --force` (each held only the unit suite's untracked `outputs/metrics/meta_llama_llama_2_7b_hf/`; the evidence is under `/data/ycfeng/tmp/issue26-correctness-pr/final_20260924/`), then `git worktree prune`. |

## [Request] 2026-09-24 — follow-up items 1-7

Message, verbatim: "规划上述1-7，要求在符合依赖条件的情况下并行处理，对于重要决策和修改（你难以抉择），先grill-me，对frontier/关键模块的代码修改方式必须符合~/.claude下的CLAUDE.md"

The items are the open proposals after the W7-R4 fix: (1) F-R5/F-R6
preemption fidelity, (2) W3-R5 strict `sync_entry` predicate, (3) W3-R6
sync-room alias refactor, (4) S44, (5) the native W6 FP8 rerun, (6) the pinned
calibration tools, (7) the collective-sim gitlink re-point after companion PR 1
merges. Before the interview, the F-R5 magnitude was measured on a 96-cell
KV-pressure grid (`/data/ycfeng/tmp/issue26-correctness-pr/followups_20260924/fr5/`).

## [Decision] 2026-09-24 — interview answers for items 1-7

Message, verbatim: "1。当前周限额已经刷新，请对处于limit中的subagent额外提交一条消息来刷新（如果依旧失败则重开一个subagent尝试）2。Q1/Q2/Q3/Q4/Q5/Q6采纳你的推荐, A7:尝试重新提交，如果依旧失败，反馈给我原因（是否之前有提交成功，但最近几次失败？） Q8:不再使用path /data/ycfeng/frontier-calibration-old-20260831/，替换为当前claude code的skill：/home/brainpp/.claude/skills/frontier-calibration，Q9：a"

| Item | Decision |
| --- | --- |
| Rate-limited subagents | Send each a message to resume it; where that fails, start a new agent for the same work. The 123 agents of workflow `wf_7606e14e-f10` that stopped on the weekly limit cannot be resumed by message (no transcript is found for a workflow agent id), so their work is re-run by new agents against the current branch. |
| Q1 F-R5 | Approved as a fidelity fix: the preemption victim may be the requester, as in vLLM `scheduler.py:470-548`. |
| Q2 F-R6 | Option (b): measure replay of a resumed decode victim together with F-R6 on a CPU prototype, then return with a design. No F-R6 code enters a branch before that. |
| Q3 Branches | Option (b): the W3 change goes to this branch (PR 35); preemption fidelity (F-R5, later F-R6 and replay) goes to a new branch stacked on this one, with its own draft PR. |
| Q4 W3-R5 | Folded into W3-R6: one change. |
| Q5 W3-R6 | Option (a): `initialize_sync_waiting_rooms` sets `_sync_waiting_room` and `_sync_kind` only; the alias attributes, `uses_shared_forward_room`, the three-way switches and the per-kind open-step partition are removed. Approved beyond five files. No numeric change is allowed. |
| Q6 S44 | Option (a): recorded as the second row of the S43 task; no code before the S43 analysis. |
| Q7 W6 native rerun | Resubmit. The retry was accepted; its job is `exp-0924-114241-429126`. |
| Q8 Calibration tools | Stop using `/data/ycfeng/frontier-calibration-old-20260831/`; use the current Claude Code skill `/home/brainpp/.claude/skills/frontier-calibration` instead. |
| Q9 Companion PR 1 | Option (a): mark it ready, merge it with a merge commit, re-point this branch's gitlink at the merge commit and rerun the collective-sim tests. |

## [Decision] 2026-09-24 — calibration tools, round 2; companion PR 1 merged

The user merged companion PR 1 themselves (`gh pr ready 1 --repo fwyc0573/frontier-htsim && gh pr merge 1 --repo fwyc0573/frontier-htsim --merge`). It merged into `main` as `d28fe917` at 2026-09-24T04:25:59Z.

Question (Q8 follow-up): how do the four helper-bound calibration entries run without the retired archive? Recommended option (a): write the needed tools in the Frontier repository under `tests/comparison/calibration/`, with unit tests. `tool-boundaries.md` then pins them by repository path and commit/blob hash, and the skill stays prompt-first. The first tool is the request-metrics normalizer that the S43/S42 E2E gate needs; op-supplement and dispatch tools follow only when a case needs them. `tool-boundaries.md`, `pressure_scenarios.py` and the four entries' wording change with it.

Message, verbatim: "continue；Q1采纳你的推荐"

| Item | Decision |
| --- | --- |
| Q8 round 2 | Option (a) as recommended. |

