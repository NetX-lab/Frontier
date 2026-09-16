## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Started independent diff-driven quality cleanup. |

# Progress

## P0 — in-progress

- Read all 817 lines of AGENTS.md, prior requirements/plan/progress/current status, and relevant validation records. No nested AGENTS.md found in source/test/data/docs search.
- Pinned `main=0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`, `candidate=c288a19f59bec09529ee18d782fa57218da2c781`, identical merge-base/main. Initial tracked worktree clean; 266 changed files including substantial historical evidence.
- Local clock confirms 2026-09-17 Asia/Hong_Kong. No new remote access or publication performed.
- Planning-with-files applies using this task directory; code-review supplies independent Standards/Spec reviews; codebase-design informs invariants/ownership, subordinate to user scope.
- Read-only Standards worker owns profiling/training/config review; Spec worker owns prior behavior/verification boundary check. Main owns runtime inspection and all implementation.
- Previous task reported 18 known main-equivalent CPU failures and explicit native hardware skips. These are leads for fresh verification, not current PASS evidence.

## Pending work

1. Build ranked module inventory and preserve executable pre-refactor reference.
2. Establish fresh baseline evidence and inspect runtime changed hunks.
3. Execute P1–P5 in dependency order.

New unresolved issues: none established.

## P1a — completed: construction and finalized snapshot contracts

- Preserved pre-refactor candidate in `../quality-baseline-c288a19f`; pinned main reference remains `../pr33-r12-baseline-20260915`.
- Default Python 3.12.3 lacks pytest/dependencies. Consulted environment/package-mirror handbooks, then installed isolated CPU/test requirements with uv in `/data/ycfeng/tmp/quality-review-env`. No shared environment was modified.
- Fresh full-unit collection failed on eight missing-torch imports and one missing-matplotlib import. Public CPU Torch index timed out after three retries; logs: `/data/ycfeng/tmp/quality-pre-unit-env.log`, `/data/ycfeng/tmp/quality-env-torch.log`. Matplotlib installation through the internal mirror succeeded. Torch resolution remains pending.
- Motivation: Simulator duplicated predictor construction; ExecutionTime treated constructor-owned components as absent; StageExecutionTime repeated filtering after rejecting missing IDs.
- Method: one predictor construction loop, direct required component access with isolated operator-map copies, uniqueness check on validated IDs. Ordinary monolithic manager/path remain None; disaggregated/hybrid sharing and physical-layer identity remain unchanged.
- Checked CC backend lazy construction: backend factory reads cluster configuration/runtime topology; manager owns training/cache state, not backend configuration. Shared initialization remains before predictor creation.
- Verification: frozen candidate eight non-dummy cases passed (65.34 s). Cleaned candidate 78 focused/non-dummy tests passed (71.00 s). Existing comparator found 90/90 stable artifacts equal across eight cases. Bidirectional artifact inventory and acceptance/stage-summary comparison follow the independent Spec review recommendation.
- Editing command `apply_patch` was absent from PATH; no patch was applied on the first attempt. Resolved by invoking the installed Codex binary in its apply_patch mode, without changing system configuration.

## P1b — completed: canonical memory/model contract

- P1a committed as `48a4093b`; supplemental symmetric metrics inventory and acceptance/stage-summary comparisons also PASS for all eight cases (106 metric files total).
- Removed planner-local hybrid layer scanning and used `BaseModelConfig.get_attention_family`, the same runtime KV owner as Replica/head metadata. All-GDN Qwen is already rejected by construction, so the removed zero-KV fallback represented no admitted model.
- ParamCounter now calls the required per-layer and GDN methods directly; genuine `get_gdn_config() is None` remains supported for homogeneous models.
- Focused before/after suites: 47 PASS / 47 PASS. Four TP/PP memory snapshots match exactly, covering weight bytes, recurrent state bytes, page size and available block count.
- Found usable same-interpreter system Torch 2.12.0+cu132. Enabled standard system-site-packages inheritance only in the dedicated venv; no target overlay or different Python package tree was injected. Fresh full frozen-candidate unit run is in progress.

## P1c — completed: GDN batch and scheduler invariants

- Batch/Request expose all required feature properties; GDN feature extraction no longer fabricates missing counts/state or copies the request list. Positive query lengths establish a nonzero mean.
- Runtime guards consume the required BaseModelConfig method. Preemption reads constructor-owned replica configuration; the incomplete PD-AF preemption fixture now provides a real non-GDN model.
- Focused verification: 46 PASS, including feature vectors, mixed-batch rejection, admission/continuation/completion/rollback and preemption progress. Transactional exception cleanup remains unchanged.
- Full frozen-candidate baseline initially finished 3584 PASS / 21 FAIL / 25 SKIP. Three added environment failures came from example shells resolving `/usr/bin/python`, which lacks plotly. Re-running with the dedicated venv on PATH; no production workaround added.

## P1d — completed: MoE routing construction and layer state

- Removed the unreferenced private routing-generator alias (only test caller updated), None-then-dict initialization, repeated validation of internally generated maps and unsupported `_cluster_num_replicas` alternate state.
- Reused subclass `_get_cluster_replica_config` through normal override dispatch instead of reflective capability discovery. Preserved optional actual replica IDs, standalone local IDs and per-replica map isolation.
- Removed unused ExecutionTime layer-count attributes: neither production nor tests read them; constructor argument validation remains compatible.
- Verification: 116 focused/non-dummy PASS; 90 stable artifact comparisons PASS, symmetric 106-file metrics inventory and supplemental summaries match. No expected value changed.
- Corrected baseline environment: 3587 PASS / 18 FAIL / 25 SKIP. The same 18 failure nodes also fail on pinned main. Source causes are missing legacy debug/analysis assets and outdated release-document expectations. README remains untouched.

## P1e — completed: lowest-free-slot ownership

- Replaced list front removal and full release-time sorting with the standard heap priority queue. The observable rule remains allocation of the lowest available slot ID.
- 16 lifecycle/scheduler tests PASS; added one out-of-order release regression. A deterministic 1,000-request trace compared directly against the frozen state manager produces identical allocation/release records.

## P1 full regression follow-up — completed

- Full candidate run returned 3583 PASS / 22 FAIL / 25 SKIP. Four newly exposed failures are incomplete fixtures: the shared routing model omits required `is_moe`; two transfer model doubles omit `get_num_gdn_layers`. The production constructor/accessor contracts were verified before adding those members to the doubles. Expected routing and transfer values remain unchanged.
- The remaining 18 nodes match the freshly established frozen-candidate/main failures. Detailed log: `/data/ycfeng/tmp/quality-p1-unit.log` (132.10 s).
- Experimental routing integerization characterization completed 3,456 generated cases with zero admitted histogram mismatches. This is sampled evidence, not a proof for arbitrary ratios; unification remains under investigation.
- Focused fixture/transfer/GDN checks now PASS: 25 tests in 5.62 s, with unchanged expectations. No production accommodation was reintroduced.
