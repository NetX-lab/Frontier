# D29 Frontier position-routing requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record bounded residual CPU repairs and carry forward YC's task-required H800 authority. |
| 2026-10-23 | YC defers handoff and resumes all remaining work in the current session;existing acceptance/resource limits remain. |
| 2026-10-23 | YC requests closure of the currently executing work, code/document commits and a prompt for continuation in another session. |
| 2026-10-23 | YC approves two extra test paths and analogous necessary repair-path adjustments within the existing budgets. |
| 2026-10-23 | Clarify both-trainer coverage and shared-manager P/D consumer scope; withdraw an unrequested independent public-consumer prerequisite. |
| 2026-10-23 | Root selects the existing constructor seam for explicit T2 engine limits within the approved scope. |
| 2026-10-23 | YC approves complete D29-SCOPE-107-10000 and the fixed collective model; begin phased implementation. |
| 2026-10-22 | Bind the design artifact to the approved parent decision and complete workflow scope. |

## [Original Request]

YC's routing request and position-cyclic choice are recorded verbatim in
`../task_2026-10-15_step5_preview_h800/requirements.md`, D29.
The parent delegated the Frontier/profiling design for that decision.

Use `(logical_token_position * top_k + slot) % num_experts`, preserve P/D and
independent-reference routing, retain physical rows and masked assignments as
distinct quantities, and derive positions from Frontier's own scheduled work.
Reuse the native router in profiling and bind a fresh policy/source identity.
The actual-main fixture is required for the complete parent task.

Current active native scope is Step P/D, PP1, TP8/DP2/EP16, without SP/MTP.
Existing other-model behavior remains covered by the legacy regression set.
Workflow precedes GPU/operator-gap alignment; native CPU-gap work remains last.

The earlier design-only restriction is superseded by the following explicit
implementation decision. Its file/domain/line limits remain binding.

## [Original Request / Decision] D29-SCOPE-107-10000

Parent question:

> 是否批准上述完整D29方案：按scope.json的107个代码/测试/配置文件，加现有尺寸说明和submodule引用，总新增＋删除≤10000行，分步实施原生路由、Frontier位置计数、profiling来源绑定、DP通信模型与验收链？通信采用固定的tensor/root顺序、通道分片、8条TP链路共同模拟、实测copy/reduce代理及拟合剩余项，按无重叠相加。模型参数须来自冻结配置和独立GPU数据，保持原G3标准与四项E2E≤10%；额外采集的资源和运行批准在封包就绪后确认。

YC selected:

> 批准完整方案 (Recommended)

Selected description:

> 立即按单文件负责人和依赖顺序实施；各步验证、提交并整合，随后准备真实采集。

The approved ledger is scope.json at318ca264d5b2b42fc56eeaec9778182931a3af2a:
107 source/test/config paths,9942 conservative gross lines after the H11
deduction, plus the named existing size note (up to12 added lines) and submodule
gitlink. Total ceiling10000 insertions/deletions. No scope expansion is implicit.
Shared paths have one writer. Existing native/profile/core/pricing designs and
their closed review corrections define behavior; missing measured parameters do
not permit invented values. Preserve D30, legacy artifacts/defaults, original
G3 criteria, separate numerical-state acceptance and each E2E error<=10%.

Implement the fixed ordered-stage/channel/8-rail model, measured copy/add proxies
and affine effective remainder with network+max-rank-local+remainder composition,
no overlap. The remainder is a modeled fit, not isolated protocol measurement;
retain signed residuals and independent qualification. Full-plan lookup is not
selected. Existing internal derived-image publication authority continues;
additional collection resources and run authorization are settled at readiness.

## [Implementation Decision] D29-T2-ENGINE-LIMITS

Root-directed in-scope correction, not a new user decision or resource approval.
T2 inputs6ebb038 correctly disclose that positions below7168 do not bind either
existing EngineArgs construction. Preserve full official model metadata bytes.
Read the established serving limits7168/8192/128 from
`/data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage/engine.json`:
max_model_len, max_num_batched_tokens, max_num_seqs respectively. The published
native fused_moe/layer.py:274,361 derives MoE max_num_tokens from the scheduler's
max_num_batched_tokens, so this is a real configuration/capacity binding.

Use a mandatory engine_limits object in existing controlled cases.json. Pass it
before construction to both existing T2 config paths; retain their distinct eager
module/graph behavior. The PRICING-owned create_native_config accepts an optional
engine_limits keyword for these three supported fields; omission preserves old
kwargs/defaults. No second builder, post-construction mutation or metadata edit.
Observe final model/scheduler values after with_hf_config; CPU checks must establish
both paths receive the intended limits. Runtime qualification remains separate.

Root reserves up to100 additional gross lines in the two existing native T2/script
test paths and60 in PRICING's existing dispatch probe/test paths, within unchanged
native19/1582 and PRICING44/3956 totals. Sole-writer rules and stop/report on quota
exhaustion apply. No new source paths, image change or allocation is authorized.

## [Implementation Clarification] D29-TRAINER-CONSUMER-BOUNDARY

This clarifies the existing approval, not a new user request. Both independent
and shared trainer entrypoints must exercise the selected binding, validation,
duration/count fit and cache contracts in design.md869–944. The approved runtime
consumer scope remains Step P/D position routing. Its product entrypoint always
constructs and supplies the shared manager (simulator.py152–174); the position
scope validator rejects monolithic routing (moe_routing_policy.py194–207).

Adding an independent public-predictor binder or monolithic measured Step mode is
not required or allocated. Root withdraws its earlier extra prerequisite while
retaining the observed latent None-manager/name-projection issue and the explicit
test limits in repairs/independent_consumer_scope.md. Do not weaken the two-trainer
fit/cache checks or claim independent public-consumer qualification. All approved
actual-main, source/routing, legacy, G3 and numerical gates remain unchanged.

## [Original Request / Decision] D29-REPAIR-PATH-AUTHORITY

Parent asked whether to add tests/unit/test_moe_share_expert_operator_families.py
and tests/unit/test_dsv4_precision_metadata.py for the six observed fixture
failures, with total additions/deletions<=60 inside PRICING3956 and global10000.

YC answered:

> 批准，类似问题都授权

These two paths are approved (effective111). Similar necessary repair/test-path
adjustments for already-approved D29 behavior may be allocated by root within
the existing owner/global budgets without repeated approval. Record each path,
reason, bounded delta and verification. This does not authorize unrelated
features, budget increases, relaxed gates or new GPU allocations. The specific
zero-sum allocation is repairs/legacy_output_scope_amendment.json.

## [Original Request] D29-SESSION-HANDOFF

YC requests:

> 请你完整当前正在执行任务的收尾，然后完提交更新docs和codes，生成一段handoff prompts，我将在其他session中继续执行

Finish the active work at a verified boundary,commit the delivered code and
documents,and provide a self-contained continuation prompt. Preserve observed
failure evidence and explicitly identify any remaining acceptance work. This
handoff request does not turn pending CLI/native/GPU checks into passed gates or
authorize additional GPU allocations. Record unfinished work and next commands
in progress.md as well as the handoff prompt.

The verified handoff boundary is the active two-file CLI placement repair:
parent57PASS/6unchangedSKIP and independent No findings. Consolidate reviewed
implementation and records on feat/step5-preview for the next session,while
keeping full CLI replay,final-source qualification and GPU work explicitly open.
This delivery merge is not the final D29 workflow admission described earlier.

## [Original Request] D29-RESUME-CURRENT-SESSION

YC requests:

> 暂不执行handoff，继续在当前session执行剩余任务。

Continue from the verified delivery in this session. Freeze e41abc07 for fresh
full CLI and required regressions;rebind T2 records in parallel without modifying
the active frozen worktree. The handoff document remains a historical checkpoint.
This request does not change budgets,workload,gates or GPU approval boundaries.

## [Implementation Decision] D29-RESIDUAL-INPUT-WORK

Under D29-REPAIR-PATH-AUTHORITY, allocate the measured once-per-loader family
resolution to two existing CORE paths (<=45 gross) and conditional exact-text
artifact parsing reuse to two existing PRICING paths (<=50 gross). These are
separate from the active105-gross model-group repair. The two ordered overlays
and acceptance rules are in repairs/input_validation_and_artifact_parsing.md.
Owner/global ceilings, per-row/per-plan checks and full-run gates stay unchanged.

## [Original Request / Cross-task Decision] H800 resource authority

YC's2026-10-23 cross-task approval, carried by project memory394:

> 批准；h800集群的的运行无需向我过问，允许你提交符合任务需求的任意规模的rjob（在h800上）

Task-required H800 runs no longer require another resource approval. Actual
quota/capacity, platform/storage rules, D29 scope and workflow/package readiness
remain binding. This does not approve H200 resources or qualify a pending run.
Earlier unapproved-resource statements below historical checkpoints retain only
their historical meaning; final T2 staging must carry this current authority.
