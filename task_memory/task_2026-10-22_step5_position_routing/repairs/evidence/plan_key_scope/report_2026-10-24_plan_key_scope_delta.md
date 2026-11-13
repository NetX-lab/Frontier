# D29 final two-path snapshot delta audit (plan key reuse)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Initial independent two-path delta audit from audited snapshot `e200bb78`; receipt, guard, and recomputation script added. |

## Verdict

`PASS_TWO_PATH_DELTA_ONLY`. The non-`task_memory` diff from audited snapshot
`e200bb78` to candidate source `c4e06a61` is exactly the two allocated Python
paths. All 122 non-key bindings, the pinned submodule, the native derived
files, the module-size note, the gitlink, and the eight fixed image evidence
blobs are unchanged. No violations were found.

This is the final two-path snapshot delta only. It is not the final 114-path
union acceptance and not full-workflow acceptance.

## Inputs

| Item | Commit | Tree |
| --- | --- | --- |
| Audited snapshot | `e200bb780d3b20ec6f5ff0a067ed8b9ddd505662` | `8c66dff99cfe53d7396b258d405ce176a6ef2006` |
| Step base (snapshot source + integration docs) | `087db1bfcb3dbc2377d4032f650b29c48fb104fd` | `d7f5c365f11696e1a9dd8af034e81bc382f77952` |
| Candidate source | `c4e06a6159ee3b1e63c50abaa346066ae8b6da31` | `6e16a43d59ccfa012e50d00eb06383aa811b5b16` |
| Candidate records / worktree head | `d1e2fb8c0162c58e7e694fab2d3a900638f0d50a` | n/a |
| Source snapshot | `db581ff6ad43e1ce57361ed00fc4be0256e85a45` | n/a |
| Source merge | `64c83ac4786cd76d2cd248e1149ce2adf4c4fa7f` | n/a |
| Original source base | `1aee0af11cd58691e64603b99eca0ac459a5b350` | `3fbf44dc1075040588c3ba54f271c5ce3085c7b1` |
| Accepted snapshot audit (records) | `d267db2d7d337dc237c85ea63f16cfa0290eb42d` | n/a |
| Accepted snapshot audit (merge) | `ae55a5f573962528a20bd850ca65c7e50747614d` | n/a |
| Parent acceptance | `1a365ebc8f27090e48f9223de80807fc68fe13b0` | n/a |

Records worktree head at execution: `d267db2d7d337dc237c85ea63f16cfa0290eb42d`.
Primary repository head at execution: `e4b949f1b6b60503c5d5f9bd0221210059b35db4`.

The snapshot manifest used for all baseline bindings is
`snapshot_blob_manifest.json`, SHA256
`807fe446a7ec87c94b5b6e9f0b128ea6ffc41d7e90f11f80c03abfdf3ed6a4e0`.
The accepted baseline receipt is SHA256
`d39ab6ee4401368685dcb60ad6259814a488ac38c2e0642d5c3d66597b203f9e`.

## Exact diff guard

Command:

```text
git diff --no-renames --name-status e200bb78 c4e06a61 -- . ':(exclude)task_memory'
```

Result (exactly two modified paths):

```text
M	frontier/cc_backend/collective_plan.py
M	tests/unit/test_step5_moe_pricing.py
```

The full diff from the snapshot also contains ten `task_memory/` record paths
(nine deletions in `detached_emission_ownership/` and `progress.md`). Those are
records, not production scope. Candidate source to candidate records
(`c4e06a61..d1e2fb8c`) changes 37 `task_memory/` files and zero non-record
files. Both two-path modes are `100644`; no mode change.

The candidate worktree reports no local modification for the two audited paths.

## Two-path accounting

| Path | Owner | Cap before | Cap after | Actual lines | Headroom | Step added | Step deleted | Step gross | Step net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frontier/cc_backend/collective_plan.py` | pricing | 136 | 140 | 139 | 1 | 4 | 1 | 5 | +3 |
| `tests/unit/test_step5_moe_pricing.py` | pricing | 300 | 311 | 311 | 0 | 11 | 0 | 11 | +11 |
| Total | pricing | 436 | 451 | 450 | 1 | 15 | 1 | 16 | +14 |

Against original source `1aee0af`, both files are additions: 139 and 311
added lines, matching the total-line caps `140` and `311`. Against step base
`087db1bf` the source blob is `57b53fa6...` and the test blob is
`2213df1f...`; both are identical to `64c83ac4` and `db581ff6`.

Changed blobs:

| Path | Snapshot OID | Candidate OID | Candidate SHA256 | Bytes |
| --- | --- | --- | --- | ---: |
| `frontier/cc_backend/collective_plan.py` | `57b53fa6d428aa67b987396b97859e1e697f8872` | `d4246ea93869a4b41746085436d3e95c3ec32ba5` | `3158bc296ebb607c97d4305825a3b98e1111ac847e91483f8d362599597a6e06` | 7144 |
| `tests/unit/test_step5_moe_pricing.py` | `2213df1fc9fee0bb1c43b978672853b48c5221bc` | `036bb10dab716cec7df2f4d9492d26e8e4ac96b5` | `86985242ad39898819a0a3f954b0e6450ab58de0bc28ea4305484d4f8b07fb21` | 19907 |

The diff content is limited to memoizing the canonical JSON key in
`CollectivePlan.cache_key()` and eleven test lines covering key identity,
public-dictionary isolation, and repeated `TypeError` for non-serializable
buffers. No import, helper, API, donor, or unrelated change appears.

## Step ceilings

| Check | Actual | Ceiling | Pass |
| --- | ---: | ---: | --- |
| Step gross lines | 16 | 17 | yes |
| Source gross lines | 5 | 6 | yes |
| Test gross lines | 11 | 11 | yes |
| Retained growth (net) | 14 | 15 | yes |
| Per-path total lines | 139 / 311 | 140 / 311 | yes |

Allocated paths for this step: 2. Changed paths: 2. Total snapshot allocation
remains 114 paths with 113 changed; the only unchanged allocation is
`tests/unit/test_step5_contract_reuse.py` (cap 1, actual 0), whose one-line
allowance is retained.

## Owner and global totals

| Owner | Paths | Cap | Actual before | Actual after | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| CORE | 49 | 4404 | 4173 | 4173 | 231 |
| PRICING | 46 | 3956 | 3940 | 3954 | 2 |
| native | 19 | 1582 | 1350 | 1350 | 232 |
| Source total | 114 | 9942 | 9463 | 9477 | 465 |
| Global (source + 12 doc + 2 gitlink) | 114 | 10000 | 9477 | 9491 | 509 |

Pricing actual `3954`, source actual `9477`, and global actual `9491` match the
requested expectations. Owner path counts and caps are unchanged.

### Overlay transition

Only `plan_key_reuse_scope_amendment.json` is applied on top of the accepted
snapshot. The 15 earlier overlays are carried from the accepted baseline, not
replayed.

| Path | Before | Delta | After | Actual | Slack |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frontier/cc_backend/collective_plan.py` | 136 | +4 | 140 | 139 | 1 |
| `tests/unit/test_step5_moe_pricing.py` | 300 | +11 | 311 | 311 | 0 |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py` | 99 | -6 | 93 | 93 | 0 |
| `frontier/cc_backend/base_cc_backend.py` | 16 | -2 | 14 | 14 | 0 |
| `tests/unit/test_collective_sim_cc_backend.py` | 243 | -7 | 236 | 236 | 0 |
| Sum | | 0 | | | |

The transfer is zero-sum: `+15` pricing allowance to the key paths and `-15`
from the three donors. Donor actuals remain `93`, `14`, and `236`; their new
caps equal their actuals.

Earlier checkpoint figures, such as the carried core-review `228/230` and the
E41 `152`-line step, were measured at their own checkpoints. They are part of
the accepted cumulative baseline and are not re-measured by this delta audit.

## Non-source bindings

| Binding | Evidence | Result |
| --- | --- | --- |
| Superproject-tree non-key entries | 105 paths, including 103 source/test/config, the doc note, and the gitlink | 0 mismatches |
| Fixed image evidence | 8 blobs | 0 mismatches |
| Submodule blobs | 5 pinned files | 0 mismatches |
| Native derived files | 4 files, base and derived hashes | 0 mismatches |

### Submodule

Pinned submodule `3dac53c544e546574cb6e8078cd14053d0916113` versus original
`d28fe917370c2d55230c3ea226e5660785b44aca`:

- 5 files, 331 added, 2 deleted, 333 gross, 329 net.
- Clean worktree.
- Blob OIDs match the accepted snapshot exactly:
  `htsim_runner.py` `e4561cb7...`, `intra_server_model.py` `28349d77...`,
  `predictor.py` `bd1f7ad0...`, `schema.py` `89d0ca54...`, and
  `tests/test_grouped_plan_input.py` `a0a75962...`.

The submodule remains separately counted at 333 gross lines.

### Native image patch

The four pinned image files produce 98 added, 1 deleted, 99 gross lines:

| Path | Added | Deleted |
| --- | ---: | ---: |
| `vllm/forward_context.py` | 15 | 0 |
| `vllm/models/step5/nvidia/model.py` | 10 | 1 |
| `vllm/model_executor/layers/fused_moe/router/router_factory.py` | 3 | 0 |
| `vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py` | 70 | 0 |

All four base and derived SHA256 values match the accepted manifest. No
native package or filesystem source rebuild was performed.

### Module note and gitlink

- `docs/general/module-size-exceptions.md` OID `c080d1317cbdf6dd43f76c2a1cd8152f516c2140`,
  SHA256 `b9897e3be8f2f04048dcb3769c5d277ef46f5938582bcdf496259b56718429e5`,
  12 added lines versus `1aee0af`, unchanged from the snapshot.
- Submodule gitlink: 1 added, 1 deleted, 2 gross, unchanged at `3dac53c`.

The note and gitlink remain the separate 12 + 2 global component.

## Drift guard

The guard covers 124 bindings: 107 superproject-tree entries, 5 pinned
submodule blobs, 8 fixed evidence blobs, and 4 native derived files.

| Digest | Value |
| --- | --- |
| Baseline, all bindings | `8a24f28e4b27adb6d2fccd77aadf79e3993f3e05492e89a636dca9c77c964c0b` |
| Candidate, all bindings | `b869dadda64d7a61360f75af8ac24d64a9d7d1dcc75d524ce0082304553ef50e` |
| Baseline, 122 non-key bindings | `fe524bcd1e36662489cb74c64c0bc4f071de3c107bebb65179a758bc4b2e81fe` |
| Candidate, 122 non-key bindings | `fe524bcd1e36662489cb74c64c0bc4f071de3c107bebb65179a758bc4b2e81fe` |
| Two-path delta | `67208a7d9b9c996f371f0158b31baec47332f841dbf57c5c824083237cd2f387` |

The non-key digests match. Future final-union checks must reproduce the
non-key digest and may differ only on the two paths captured by the delta
digest.

## Verification commands

```text
git diff --no-renames --name-status e200bb78 c4e06a61 -- . ':(exclude)task_memory'
git diff --no-renames --name-status c4e06a61 d1e2fb8c -- . ':(exclude)task_memory'
git diff --no-renames --numstat 087db1bf c4e06a61 -- frontier/cc_backend/collective_plan.py tests/unit/test_step5_moe_pricing.py
git cat-file blob <rev>:<path>   # bytes, SHA256 and newline counts
git ls-tree <rev> -- <path>      # all manifest bindings
git -C <submodule> diff --numstat d28fe917 3dac53c
git diff --no-index --numstat <native-base> <native-derived>
sha256sum <eight evidence blobs and eight native base/derived files>
D29_GUARD_ONLY=1 D29_CANDIDATE=<final-head> D29_CANDIDATE_RECORDS=<final-records> \
  python3 repairs/evidence/plan_key_scope/recompute_plan_key_delta.py
```

All commands were read-only and bounded to Git metadata, small JSON/TSV
receipts, and scoped file hashes.

## Violations

None.

## Limits and residual checks

- Read-only only: no tests, imports, probes, code execution, model/CSV/trace
  reads, Docker, network, GPU, package operations, or native rebuild.
- Candidate, primary source, tests, peers, and the source worktree were not
  modified or checked out.
- Behavioral correctness of the cache-key change is not proven here. Dual
  reviews and primary runtime checks remain required.
- Cumulative checkpoint figures from earlier authorized work are carried, not
  re-derived.
- The final union must keep source `<=9942`, global `<=10000`, pricing
  `<=3956`, CORE `<=4404`, and native `<=1582`, and must reproduce the 122
  non-key binding digest.
- The two key paths must remain at 139/140 and 311/311 lines unless a separate
  allocation is approved.
- Full-workflow acceptance remains outside this audit.
