# Balanced prefill wiring CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded existing regressions, direct controls, exact scope and retained-package checks. |

Task: W128-V4-BALANCED-PREFILL-FINAL. Working directory: `/data/ycfeng/Frontier/.worktrees/v4-h800-prefill`. Python: `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6, GCC11.3.0. No conda invocation, dependency changes, GPU/Docker/platform/SSH/cloud operations or CPU-overhead work.

## Commands and observed results

| Command / execution | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| `env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider` | Existing prefill/legacy/decode regressions pass |184PASS, initial76.78s; after identity controls184PASS95.81s |PASS |
| Same pytest command plus `-k 'exact_prefill or exact_isolated or attention_rejects_unqualified or attention_does_not_median or attention_layout'` | Affected final source regression subset passes |22PASS,162deselected,2.12s |PASS |
| `env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -` with adapter subprocess controls (exact stdin is in the session tool transcript; no new maintained harness) | Real adapter registers before fixture CLI import/model stub and rejects invalid config/import/hash |5PASS; `cpu_adapter_receipt.json` |PASS |
| Same Python stdin execution with native2cf AST function/code-origin fixture | Real `_routing_identity` observes constructed MoERunner and independently rejects env/origin/registry/source mismatches |9PASS,9 native_sources; `cpu_identity_receipt.json` |PASS |
| Same Python stdin execution using existing prefill row and real isolated consumer fixtures | Both methods retain six rows; old rows not relabeled; explicit-balanced missing/wrong/mixed identity typed rejects |16PASS; `cpu_rows_consumer_receipt.json` |PASS |
| Same Python stdin execution reading native git objects, parser/plan and both retained package manifests | Native2cf pins preserved,7168/8192/128/128 plan, fixed strategy SHA, immutable retained packages |9 pinned sources;527files verified per retained package; `cpu_builder_receipt.json` |PASS |
| `git diff --check` | No whitespace errors |No output, exit0 |PASS |
| Python `ast.parse` for all five maintained files | Source parses in Python3.10 without GPU imports |All5parse |PASS |

## Scope and baseline

- Builder baseline was9086bytes with SHA256 `3e2f462748490b4e53c88f506a1db0faf697a9223a462d9152cc1ca1d1de3471`. Exact bytes retained as `builder_before_routing.txt`; baseline Git blob `1453487f3cc5f701b24a490ce45d3f549c2ebe02`, audit commit `a71fcd3`.
- The builder was originally ignored/untracked. Audit commit tracks the unchanged141-line historical baseline; those lines are not new routing implementation. Worktree edits were never overwritten. Routing diff against that baseline contains exactly five maintained files,182added+7deleted=189changed lines; see `diff_count.json`.
- Native strategy remains SHA256 `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`. C26 and native bootstrap were untouched.
- Old failed-generation manifest SHA256 `f198ecece51961457ac69a66d8f2cf086f54301401ccc66dad0acbbefc57711e`; configured-capacity manifest SHA256 `ff7f37b0f7c78468a500bdedd20f3c168127820082c723a346898d40a253dff1`. All527manifest-bound files of each remain valid.

## Failures diagnosed during direct control setup

Initial native AST fixture attempted to execute BaseRouter without its FusedMoERouter base: NameError, fixture limitation, no production behavior was changed for it. Subsequent real registered strategy inspection exposed that get_forward_context/get_ep_group are closure-bound, not function globals: changed actual source inventory to inspect closure bindings. Source inspection also showed select_experts is defined by fused_moe_router.py, not base_router.py: included that native module pin. Native owner's final receipt added runners/native_sources and actual factory/env object evidence: aligned prefill receipt and independently verified all9native origins.

## Limits

CPU fixture registrations, router instances, timing0.125ms and native AST code are source/ordering/rejection controls, not a native model or GPU measurement. No package was built/sealed and no installed H800 image was admitted. Root owns fresh generation/source fields, joint review, persistent integration tests by native owner, strict77, image/source freeze and the approved GPU run. Observed GPU routing counts/kernel/backend behavior and latency remain unverified. No CPU overhead or gap analysis was performed.
