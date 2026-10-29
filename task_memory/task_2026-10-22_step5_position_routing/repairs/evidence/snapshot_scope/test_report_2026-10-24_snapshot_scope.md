# D29 snapshot baseline scope audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Independently replay the original 107-path ledger and all 15 overlays; measure the 114-path snapshot, pinned submodule and native patch; retain incremental drift bindings. |

Task: **D29-SNAPSHOT-SCOPE-AUDIT**

Session: `ses_e95ac82fdffeSeMEUfR9SxvW03`

Kind: **SNAPSHOT_BASELINE_AUDIT**, not final key or full-workflow acceptance.

Records branch/worktree: `d29-snapshot-scope-audit`, `/data/ycfeng/Frontier/.worktrees/d29-snapshot-scope-audit`.

## Result

**PASS: no unallocated changed path, per-path overage, owner overage, overlay arithmetic/order mismatch or non-zero owner transfer. No source edit or reallocation.**

| Owner | Allocated paths | Changed paths | Actual gross | Cap | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| CORE/SOURCE | 49 | 49 | 4173 | 4404 | 231 |
| PRICING | 46 | 45 | 3940 | 3956 | 16 |
| native | 19 | 19 | 1350 | 1582 | 232 |
| source/test/config | 114 | 113 | 9463 | 9942 | 479 |
| gitlink + existing size note | 2 extra bindings | 2 | 2 + 12 | Named allowances | — |
| global | — | — | 9477 | 10000 | 523 |

`tests/unit/test_step5_contract_reuse.py` has zero actual change and retains a **1-line allocation**, not one measured line. The 114 allocated paths comprise 105 superproject paths, five submodule paths and four native-image paths. Actual changes comprise 104 + 5 + 4 = 113 paths. Changed superproject Python paths number 101; the other three are the shell script, Dockerfile and JSON configuration.

Measured composition, with no gitlink or documentation double counting:

| Component | Changed paths | Added | Deleted | Gross |
| --- | ---: | ---: | ---: | ---: |
| superproject source/test/config | 104 | 8740 | 291 | 9031 |
| pinned collective-sim | 5 | 331 | 2 | 333 |
| native-image source patch | 4 | 98 | 1 | 99 |
| source/test/config total | 113 | 9169 | 294 | 9463 |
| gitlink | 1 | 1 | 1 | 2 |
| existing size note | 1 | 12 | 0 | 12 |

## Frozen source and authority

- Source commit: `e200bb780d3b20ec6f5ff0a067ed8b9ddd505662`.
- Source tree: `8c66dff99cfe53d7396b258d405ce176a6ef2006`.
- Baseline and merge-base: `1aee0af11cd58691e64603b99eca0ac459a5b350`.
- `frontier` tree: `ae5025067c9d98dbc6c97b249023065514b491fb`.
- `tests` tree: `82bd11e02ae7b3e52190f0d65b4bea9595a4b848`.
- Non-record tracked bytes are identical to snapshot `db581ff6ad43e1ce57361ed00fc4be0256e85a45`, integration `64c83ac4786cd76d2cd248e1149ce2adf4c4fa7f` and allocation `087db1bfcb3dbc2377d4032f650b29c48fb104fd`. Full commit trees differ because task records differ.
- Original scope at `318ca264d5b2b42fc56eeaec9778182931a3af2a` hashes to `12b6f02b1df18f03d8b7a4c004f8d9c62af5946bc28a322bc4f0d661e62a2652`. Its 138 allowance rows over 107 paths are unchanged in current `scope.json`.

Whole-file owners are derived from `shared_scope.md`, not allowance-majority guessing. H04 `moe_operator_times.py` and H17 `fp8_block_experts.py` belong to PRICING. All other CORE/SOURCE shared paths retain CORE ownership. The initial whole-file allocation independently reproduces CORE44/4404, PRICING44/3956, native19/1582.

## Ordered overlay replay

The receipt records every individual `before/delta/after`, owner, original allowance row and final cap. Every row's arithmetic, declared effective path count, ordering and unchanged owner total was checked.

| Order | Overlay file in `repairs/` | Effective paths | CORE transfer | PRICING transfer | native transfer |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `registry_scope_amendment.json` | 109 | 300 | 0 | 0 |
| 2 | `core_review_scope_amendment.json` | 109 | 310 | 0 | 0 |
| 3 | `legacy_output_scope_amendment.json` | 111 | 0 | 60 | 0 |
| 4 | `prepare_metadata_scope_amendment.json` | 111 | 0 | 20 | 0 |
| 5 | `final_scope_reconciliation.json` | 111 | 0 | 1025 | 233 |
| 6 | `cli_placement_scope_amendment.json` | 111 | 40 | 0 | 0 |
| 7 | `model_group_reuse_scope_amendment.json` | 111 | 100 | 0 | 0 |
| 8 | `init_validation_scope_amendment.json` | 111 | 27 | 0 | 0 |
| 9 | `collective_artifact_parse_scope_amendment.json` | 111 | 0 | 50 | 0 |
| 10 | `trace_serialization_scope_amendment.json` | 112 | 20 | 0 | 0 |
| 11 | `prepared_point_scope_amendment.json` | 112 | 212 | 0 | 0 |
| 12 | `immutable_cost_fields_scope_amendment.json` | 112 | 0 | 0 | 0 |
| 13 | `collective_artifact_bytes_scope_amendment.json` | 112 | 0 | 2 | 0 |
| 14 | `run_input_snapshot_scope_amendment.json` | 114 | 100 | 24 | 0 |
| 15 | `plan_key_reuse_scope_amendment.json` | 114 | 0 | 15 | 0 |

Transfers are allowance movements, not measured code consumption. Each transfer is matched by equal withdrawals within the same owner. The new paths are registry/model-binding tests (+2 CORE), legacy catalogue tests (+2 PRICING), trace store (+1 CORE) and shared input module/manager (+2 CORE).

The latest key overlay **only reserves** 17 step-gross lines (source 6/test 11), at most 15 retained. It sets plan cap 140 and pricing-test cap 311, funded by backend 99→93, base 16→14 and backend-test 243→236. Their current actuals independently remain 136/300 and 93/14/236. No key implementation code or results were inspected. Future retained growth of 15 would bound source/global at 9478/9492; these are bounds, not final measurements.

## Git-derived step measurements

Caps are allowances. Actual source/test columns count additions **plus deletions**, separately measured using the implementation commit's parent (not its later documentation-only branch tip).

| Step/source commit | Gross cap | Source/test caps | Actual source | Actual test | Actual gross | Step net |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| model group `3763e5b8` | 105 | 45 / 60 | 17 | 54 | 71 | 59 |
| loader `ee7e66ee` | 45 | 6 / 39 | 4 | 34 | 38 | 38 |
| artifact parse `51e1f720` | 50 | 15 / 35 | 8 | 35 | 43 | 41 |
| trace `390bb5a2` | 20 | 20 / 0 | 14 | 0 | 14 | 0 |
| prepared `0be11762` | 250 | 180 / 70 | 92 | 67 | 159 | 121 |
| cost fields `75404143` | 8 | 8 / 0 | 8 | 0 | 8 | 0 |
| bytes `2f2d9339` | 36 | 26 / 10 | 24 | 8 | 32 | 2 |
| snapshot `db581ff6` | 340 | 120 / 220 | 73 | 139 | 212 | 86 |

Snapshot's ten paths exactly match its amendment and all ten per-step caps pass:

| Path | Owner | Step added/deleted | Step gross/cap | Snapshot gross/final cap |
| --- | --- | --- | --- | --- |
| `frontier/cc_backend/collective_profile.py` | CORE | 20 / 0 | 20 / 24 | 20 / 24 |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | CORE | 6 / 0 | 6 / 12 | 6 / 12 |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | CORE | 2 / 0 | 2 / 8 | 113 / 128 |
| `frontier/execution_time_predictor/moe_routing_workload.py` | CORE | 8 / 10 | 18 / 30 | 195 / 223 |
| `frontier/cc_backend/base_cc_backend.py` | PRICING | 8 / 0 | 8 / 10 | 14 / 14 |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py` | PRICING | 14 / 5 | 19 / 36 | 93 / 93 |
| `tests/unit/test_collective_sim_cc_backend.py` | PRICING | 28 / 27 | 55 / 100 | 236 / 236 |
| `tests/unit/test_moe_routing_runtime.py` | CORE | 59 / 17 | 76 / 100 | 117 / 128 |
| `tests/unit/test_prediction_measurement_bindings.py` | CORE | 3 / 3 | 6 / 14 | 299 / 311 |
| `tests/unit/test_step5_measured_ffn.py` | CORE | 1 / 1 | 2 / 6 | 105 / 110 |

Snapshot has 149 additions/63 deletions, source 48/25 and tests 101/38. Its 212 step-gross lines are **not** 212 new baseline-relative scope lines.

| Cumulative checkpoint | CORE actual | PRICING actual | native actual | Source/global |
| --- | ---: | ---: | ---: | --- |
| accepted `79f1a3fd` | 3970 | 3920 | 1350 | 9240 / 9254 |
| trace merge `64881373` | 3984 | 3920 | 1350 | 9254 / 9268 |
| prepared merge `eece87b2` | 4105 | 3920 | 1350 | 9375 / 9389 |
| cost-fields merge `1b6dfd6a` | 4105 | 3920 | 1350 | 9375 / 9389 |
| byte `2f2d9339` | 4105 | 3922 | 1350 | 9377 / 9391 |
| snapshot `db581ff6` and current source | 4173 | 3940 | 1350 | 9463 / 9477 |

The accepted audit's 111 path actuals were all re-derived and match. Trace/prepared/cost merges and later replacement blobs are recorded under `source_step_receipts[].blob_inheritance`; superseded blobs are explicitly distinguished from unchanged inherited blobs. No step gross was summed as though it were retained scope growth.

Historical registry cap 360 applies to `d96b3279..a5189e19`: measured **331**. Its current cumulative same-path figure is 517 across later separately authorized changes. Historical fixture cap 80 applies to `e49b26f8..f03a1157`: measured **72**; current cumulative is 174. Neither cumulative number is a violation of the earlier checkpoint cap. Historical core-review 228/230 remains **carried, not independently recomputable**; the current 114 per-path accounting is independently measured nonetheless.

## Non-superproject evidence

Submodule Git objects, not a mutable checkout/index, were read from the existing integrated-06 submodule object database:

- Baseline `d28fe917370c2d55230c3ea226e5660785b44aca`, tree `63e37c262ca5bc4c6524bd15b960c5658a7e1fa6`.
- Pin `3dac53c544e546574cb6e8078cd14053d0916113`, tree `17b3985bce53f5aa7eebab33b391915526049d6c`.
- Exactly five changed paths: runner 54, intra-server model 77, predictor 30, schema 53, test 119; total 333 gross.
- Recomputed Git patch SHA256: `c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d`.

Native review patch SHA256 is `66608153204012927e8ef678378d95ca1ef93dab9e2f504a82a5318b9828bbab`. Exactly four files change: forward context 15, Step model 11, router factory 3, routing simulator 70, total 99. Normalizing only the `native-image/` header prefixes reproduces build patch SHA256 `7a03c3de88329cd6123b00a9519849b4434b79411f7f14500a9bc02ee2d73df7` exactly.

The existing patch was applied **in memory**, with exact hunk positions/context, no fuzz or offset, to four pinned base files at `/data/ycfeng/tmp/step5_profiling_20261020/image_src`. All four outputs match both the manifest pins and the corresponding exported files at `/data/ycfeng/tmp/d29_published_sources01/source`. No native package, source directory, artifact or image was rebuilt.

| Native path under `vllm/` | Derived SHA256 |
| --- | --- |
| `forward_context.py` | `7d9f19ad2797fc8c9fdfdef3f3d739156702d2f6a6d584f4391f7875f7011930` |
| `models/step5/nvidia/model.py` | `693e24e3cd9e86708a1363d42101eebcbb47c7ea84185bcc88aaf0ad6c189269` |
| `model_executor/layers/fused_moe/router/router_factory.py` | `c1b24be354a16ad7914725a1e220c4e4113407aa72502e344c5f47295daf8fef` |
| `model_executor/layers/fused_moe/router/routing_simulator_router.py` | `9ee0db5084b41c9fdbc56f432c255c136315fb03e1ffb77cda47637e55abc725` |

All eight named image evidence files remain blob-identical to accepted `79f1a3fd`. Derivation manifest hash `561c5a87518cd22b02e8ee47ec396860ad0af84548921874c8f3557e170c32ec`, installed-source receipt hash `b99bc33b7e17f1256e970a93be5698544a2b52c11ab0683029b2c75e18f87a09`, and OCI index bytes hash `10cdd7af56c77b244f70f280133464cce59573b0bcaa58adba6f39342ed1e94d` were recomputed. Their image binding remains `hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee`.

The existing size note is unchanged, SHA256 `b9897e3be8f2f04048dcb3769c5d277ef46f5938582bcdf496259b56718429e5`, 12 added/0 deleted lines. Its inventory sentence explicitly describes frozen `1be91bad`; its 98/5/4 count is historical, not the current snapshot's 101/5/4 count. This audit does not rewrite that checkpoint document.

## Verification commands, outcomes and limits

Environment: `/usr/bin/python3`, Python 3.10.6, GCC 11.3.0; Git 2.34.1. No conda or product imports were used.

```bash
timeout 60s python3 -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_path("task_memory/task_2026-10-22_step5_position_routing/repairs/evidence/snapshot_scope/recompute_snapshot.py", run_name="__main__")'
```

**PASS**, exit 0, 114 allocated/113 changed, 9463 source/9477 global, empty violations. The JSON receipt retains exact Git argv, exit codes, output SHA256 and sizes for each verification call. Decisive command forms:

```bash
git diff --numstat --no-renames 1aee0af11cd58691e64603b99eca0ac459a5b350 e200bb780d3b20ec6f5ff0a067ed8b9ddd505662 -- . ':!task_memory/**'
git diff --numstat --no-renames 2f2d9339 db581ff6 -- frontier tests
git diff --numstat --no-renames 2f2d9339^ 2f2d9339 -- frontier tests
git --git-dir=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06/frontier/cc_backend/backends/collective-sim/.git diff --numstat --no-renames d28fe917 3dac53c -- .
git ls-tree <commit> -- <allocated-path>
git cat-file blob <blob-oid>
```

Two cgroup wrappers failed before audit execution: `systemd-run --user --scope -p MemoryMax=30G ...` returned `Job failed`; system scope returned `Interactive authentication required`. The successful command enforced a 60-second timeout and a 30-GiB **virtual-address-space** limit (`RLIMIT_AS`), not an unavailable cgroup RSS limit. Initial receipt development also used two documentation-only branch tips for trace/prepared; scoped Git path history corrected these to the actual implementation commits before the accepted receipt was generated.

No source edit/reallocation, tests, fullscans, Docker, network, GPU, cleanup, native package rebuild or peer checkout/index mutation occurred. Only scoped Git metadata/blobs and named native files were read. The image inventory is a 340,881-byte existing receipt; no source archive or unrelated artifact tree was scanned. Historical publication/source receipts are not fresh registry inspection or runtime/G3/E2E qualification.

## Deliverables and next acceptance boundary

All paths below are relative to this `snapshot_scope/` directory:

- `snapshot_baseline_receipt.json`: 114 owner/cap/actual rows, every overlay transition, source/step/checkpoint totals, non-source bindings and exact verification commands.
- `snapshot_paths.tsv`: directly inspectable 114-path ledger, with caps before/after key reservation and independently measured additions/deletions.
- `snapshot_blob_manifest.json`: baseline/current blob IDs and SHA256, gitlink and note, four native source pins and fixed image-evidence hashes for future unrelated-drift detection.
- `recompute_snapshot.py`: stdlib-only bounded accounting script; not a product test.
- `test_report_2026-10-24_snapshot_scope.md`: this report.
- `check_progress_2026-10-24_snapshot_scope.md`: audit route checkpoint.
- `artifact_sha256.json`: computed sizes/SHA256 for all six deliverables above, including the script and Markdown records; excludes its own recursive hash.

Residual checks belong to the later parent acceptance:

1. Compare final key commit against this frozen source with the manifest's exact Git name-set guard. Only `frontier/cc_backend/collective_plan.py` and `tests/unit/test_step5_moe_pricing.py` may change outside task records. Recompute both against `1aee0af`, enforce 6/11 step caps and ≤15 retained growth, and recheck owner/global totals. A mere net-line addition is not sufficient.
2. Preserve every other source/test/config blob, the submodule pin, native patch/derived pins and named image evidence. Any unrelated drift requires a fresh audit of the affected scope, not reuse of this receipt as final acceptance.
3. Key behavioral/review checks, final integration/freeze/regressions, full CLI, T2/native/G3 and four-metric E2E acceptance remain outside this audit. The historical core-review 228/230 remains a carried checkpoint receipt.
