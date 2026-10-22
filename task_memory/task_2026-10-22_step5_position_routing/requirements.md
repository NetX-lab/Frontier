# D29 Frontier position-routing requirements

## Modification History
| Date | Change |
| --- | --- |
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
