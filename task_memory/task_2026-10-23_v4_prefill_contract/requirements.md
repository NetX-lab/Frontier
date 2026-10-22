# Isolated V4 prefill contract requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded delegated approval, exact scope and immutable accepted-input bounds. |

Task ID: `W128-V4-PREFILL-CONTRACT-DQ10`. Base: `b62c155abd981a71e12e62aa0c0fe3990741d2f8`; worktree: `/data/ycfeng/Frontier/.worktrees/v4-prefill-contract`.

## Original requests and approval

[Original Request] Implement the bounded CPU-only isolated prefill contract/accepted-input consumption step. Own only `frontier/execution_time_predictor/deepseek_v4.py`, `tests/integration/check_dsv4_measured_scopes.py`, and one existing affected unit-test file (prefer `test_dsv4_graph_measured.py`). Initial total <=3 maintained files/200 added+deleted lines, <=150 core lines. Stop before any material seam or scope expansion. No further delegation.

[Original Request / Approval Q&A] Primary `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/requirements.md:1063-1069`, D-Q10-V4-GRAPH-MEASURED-AND-NIXL:
> 是否批准 V4 后续两项 Q10 修改？A：依据 native V2 证据，接入 graph padding 与实测计时读取，缺表或缺 scope/phase 证据仍报 typed error。B：扩展现有 NIXL 到 V4 grouped cache，先核实原生 SWA/state/compressed 分组页规则，复用已有分配和字节计算，保持 GLM 行为及完整 prompt 传输。两项先做 CPU 验证，不重启 CUPTI 调查、不改变四指标标准；H5、V4 准入和 profiling 继续并行。
> YC：批准两项（推荐）。

The delegated step is approved unfinished measured-consumer implementation, not routing implementation or a GPU campaign. Primary ticket `tickets/09-predictor-bundle.md` remains OPEN. Primary `v4_prediction_next_20261022.md` section 7 explicitly requires independent non-timing engine/step metadata and accepted-file provenance, preserving the legacy checker default.

## Acceptance and exclusions

- Build flat layouts independently from the accepted producer plan and engine/step raw metadata, per variant/planning role; never derive expected identity/layout from candidate timing rows.
- Explicitly bind source revision, runtime fingerprint, model-config hash and CSV hash from pinned accepted receipts/artifacts. Use the existing reader API and metadata/routing derivation.
- Drive actual `read_isolated_scope` for both methods. Compare every qualified scope exactly to its CSV median; `serving_admission=false` throughout.
- Typed negatives: source/hash mismatch, missing/wrong metadata, wrong variant/planning role, historical unbalanced q1 under explicit balanced selection, graph and PD.
- Preserve absent-selector legacy prefill, H200/DP4/EP32 checker defaults and 102-positive/72-negative behavior, decode provenance and unrelated exception types. No forced rejection of valid non-balanced legacy cases.
- No producer/instrumentation, new schema/registry/shared config, hardcoded H800 production defaults, wrapper hierarchy, full non-dummy constructor admission, PD/graph admission, invented prices, CPU-overhead/gap study, GPU/platform/SSH/cloud/Docker/network action, full strict77 rerun before source/affected checks pass, or primary-worktree edits.
- Commit verified scoped code and task records; return exact diff, tests, real-table comparisons, source hash and open qualifications to root for review.

## Immutable inputs

Primary accepted receipt: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json` and its `accepted/` artifacts. Frozen package: `/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced`. Source `39b2bf8088dff40dc8b695b6927d6e6322c337e9`, policy `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`. Preserve original artifacts byte-for-byte; do not copy/relabel them.

[Observed enumeration] Each table has six rows. CUDA_EVENT has 50 scope entries; KERNEL_ONLY has 44 (no enclosing layer scope), total 94. The requested expected50 is the event-only count, not both methods combined. Acceptance follows actual independent producer enumeration, without inventing six kernel-only layer timings.
