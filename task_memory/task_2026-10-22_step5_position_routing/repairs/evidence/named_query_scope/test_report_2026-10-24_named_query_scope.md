# D29 named-model-query scope delta

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Independently compose the four-path query delta onto the accepted key ledger; bind all 124 entries and verify 114 effective caps. |

## Result

**PASS_NAMED_QUERY_ACCOUNTING_ONLY — zero violations.**

| Owner | Allocated paths | Actual cumulative gross | Cap | Headroom |
| --- | ---: | ---: | ---: | ---: |
| CORE | 49 | 4261 | 4404 | 143 |
| PRICING | 46 | 3954 | 3956 | 2 |
| native | 19 | 1350 | 1582 | 232 |
| Source | 114 | 9565 | 9942 | 377 |
| Global, including doc 12 and gitlink 2 | 114 | 9579 | 10000 | 421 |

114 paths remain allocated; 113 have actual changes. The unchanged contract-reuse
test retains its one-line allowance. The new query does not add paths to the
allocation: all four files already existed and had cumulative changes.

This audit measures accounting and byte bindings only. It does not accept
implementation behavior, parent pytest/wave results, a final integrated-head
freeze, or the full workflow.

## Bound inputs

| Input | Commit | Tree |
| --- | --- | --- |
| Accepted key baseline | `46fe83ac55876c10f6dd8ff792705e87efd1905f` | `bbbca98802261662e02d32a59c1a4d6ab20a12e5` |
| New approved allocation | `617d832bed8f52d6b240e4e33023ed3e5d06dacd` | `1626ce9ba5466a0476b68da0310c4b86bde8b1d5` |
| Explicit step base | `de2d818fc49613e5b7fc5c41b4e49691b2a36bd6` | `c7447d0c49f363fea7b4329a55ba207acb90d8c4` |
| Query source | `1737ce9c4a761d2c57ee617f5f3a78d7bcfd9a58` | `80f390f84f63c28e3edb876177945592f96aa0d4` |
| Query records / candidate HEAD | `3d75c786781802ad2753cbdd81d989c9b694660c` | `5196ecef472c6dcb474c92fddcef5fe1aeed5639` |

Authority is primary `requirements.md:338-362` and
`repairs/named_model_query_scope_amendment.json` at `617d832b`. The latter is
SHA256 `88e6deb069b2c7736ee688b33847b2e78b06af504766c5f07d8a0fb394ca647d`.
The existing CORE 4404, PRICING 3956, native 1582, source 9942, and global 10000
ceilings remain unchanged.

The prior snapshot and key artifacts were read from their accepted Git commits,
matched against the local records, and matched against `46fe83ac`:

| Prior artifact | SHA256 |
| --- | --- |
| Snapshot manifest (`d267db2d`) | `807fe446a7ec87c94b5b6e9f0b128ea6ffc41d7e90f11f80c03abfdf3ed6a4e0` |
| Snapshot receipt (`d267db2d`) | `d39ab6ee4401368685dcb60ad6259814a488ac38c2e0642d5c3d66597b203f9e` |
| Key receipt (`6a980a51`) | `b483694024bb4e32aed051e494018e0005824656492b4801032f48351f9e1c07` |
| Key guard (`6a980a51`) | `77f0601e0164fc522f8fef0daebcc3a0fcd9fc91f1b4f2746d9ecd49880f4015` |
| Key checksum inventory (`6a980a51`) | `cde6ff23ef1d12e3fd4a9dcd4f66eb1c33c961762999d8627001a77ef8f0a9a2` |

`30fb6a30` and `46fe83ac` have the same non-record source as owner `c4e06a61`.
`617d832b`, `de2d818f`, and primary `e4a7d7b2` / `2d417982` have no non-record
source delta from `46fe83ac`.

## Exact changed-file guard

From each of `46fe83ac`, `617d832b`, and `de2d818f` to query source:

```text
M	frontier/execution_time_predictor/moe_routing_workload.py
M	frontier/execution_time_predictor/prediction_model_registry.py
M	tests/unit/test_moe_routing_runtime.py
M	tests/unit/test_prediction_measurement_bindings.py
```

Source to records changes 30 paths, all under `task_memory/`. No source mode
change or unrelated source/test/config drift was found. Every bound candidate
working-file SHA256 matches its pinned Git blob, including both accepted key
files. Primary and candidate HEAD/index were preserved.

## Four-path usage: Git gross, not physical line counts

Cumulative values below are `git diff --numstat 1aee0af <source>`. Step values
are measured against the explicit `de2d818f` base.

| Path | Baseline gross | Candidate + / − | Candidate gross / cap | Headroom | Step + / − | Step gross / cap | Step net |
| --- | ---: | --- | --- | ---: | --- | --- | ---: |
| `frontier/execution_time_predictor/prediction_model_registry.py` | 96 | 91 / 13 | 104 / 120 | 16 | 9 / 5 | 14 / 15 | +4 |
| `frontier/execution_time_predictor/moe_routing_workload.py` | 195 | 190 / 6 | 196 / 223 | 27 | 2 / 1 | 3 / 5 | +1 |
| `tests/unit/test_prediction_measurement_bindings.py` | 299 | 348 / 0 | 348 / 353 | 5 | 49 / 0 | 49 / 50 | +49 |
| `tests/unit/test_moe_routing_runtime.py` | 117 | 147 / 0 | 147 / 151 | 4 | 30 / 0 | 30 / 30 | +30 |

| Check | Actual | Ceiling / expected | Result |
| --- | ---: | ---: | --- |
| Step source gross | 17 | 20 | PASS |
| Step test gross | 79 | 80 | PASS |
| Step gross | 96 | 100 | PASS |
| Step retained net | 84 | No separate retained ceiling in amendment | Observed |
| Cumulative budget-gross growth | 88 | 88 hypothesis | PASS |

Source step additions are 11 and deletions 6; tests add 79. Step additions total
90 and deletions total 6: 96 gross and 84 net. Budget
growth is independently recomputed as `(104−96)+(196−195)+(348−299)+(147−117)`
= 88. Re-diffing modified files against the original base is not additive in
step gross or step net. Therefore accepted global 9491 becomes **9579**, not
9587 (adding step gross) or 9575 (adding retained net).

## Eight overlay adjustments

The amendment transfers 65 allowance lines from six unedited CORE donors to
two existing tests. Transfer sum is zero; no owner or global reallocation occurs.

| Path | Before cap | Delta | After cap | Actual gross | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tests/unit/test_prediction_measurement_bindings.py` | 311 | +42 | 353 | 348 | 5 |
| `tests/unit/test_moe_routing_runtime.py` | 128 | +23 | 151 | 147 | 4 |
| `frontier/execution_time_predictor/step5.py` | 51 | −15 | 36 | 34 | 2 |
| `frontier/scheduler/utils/ep_trace.py` | 28 | −15 | 13 | 11 | 2 |
| `frontier/scheduler/utils/ep_wave.py` | 22 | −15 | 7 | 5 | 2 |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | 128 | −13 | 115 | 113 | 2 |
| `frontier/execution_time_predictor/prediction_family_trainers.py` | 88 | −6 | 82 | 80 | 2 |
| `frontier/metrics/trace_store.py` | 20 | −1 | 19 | 14 | 5 |

All six donor cumulative numstats were freshly recomputed against `1aee0af`
at both accepted baseline and candidate; they match exactly. The prior key
donors remain 93 / 14 / 236 under the unchanged byte bindings and ledger.
All 114 effective caps, owners, and actuals appear in `named_query_paths.tsv`
and `named_query_scope_receipt.json`.

### Applied sequence

The following 15 overlays are carried from the accepted, hash-bound snapshot
audit; their before/after owner totals are recorded in the new receipt:

1. `registry_scope_amendment.json`
2. `core_review_scope_amendment.json`
3. `legacy_output_scope_amendment.json`
4. `prepare_metadata_scope_amendment.json`
5. `final_scope_reconciliation.json`
6. `cli_placement_scope_amendment.json`
7. `model_group_reuse_scope_amendment.json`
8. `init_validation_scope_amendment.json`
9. `collective_artifact_parse_scope_amendment.json`
10. `trace_serialization_scope_amendment.json`
11. `prepared_point_scope_amendment.json`
12. `immutable_cost_fields_scope_amendment.json`
13. `collective_artifact_bytes_scope_amendment.json`
14. `run_input_snapshot_scope_amendment.json`
15. `plan_key_reuse_scope_amendment.json`

Overlay 16 is `named_model_query_scope_amendment.json`, applied to the accepted
key ledger. Historical repair ceilings remain checkpoint-specific ceilings,
not measured cumulative usage; historical diagnostics were not rerun.

## 124-binding reconstruction and 120-binding guard

The accepted baseline was reconstructed from the snapshot manifest plus the
two accepted key OIDs. It reproduces the key audit's full binding digest.
Every source/test/config/doc/gitlink and fixed evidence tree entry was compared
through allocation, step base, candidate source, and candidate records. Fresh
SHA256 values were checked for named working files and all requested pins.

| Binding group | Count | Result |
| --- | ---: | --- |
| Superproject source/test/config | 105 | Only four query entries change |
| Module-size note and gitlink | 2 | Unchanged |
| Pinned submodule blobs | 5 | Unchanged |
| Fixed image evidence entries | 8 | Unchanged |
| Native derived files | 4 | Unchanged; all four base files also freshly hashed |
| Total | 124 | 120 non-query entries unchanged |

| Digest | SHA256 |
| --- | --- |
| Reconstructed accepted key, all 124 | `b869dadda64d7a61360f75af8ac24d64a9d7d1dcc75d524ce0082304553ef50e` |
| Query candidate, all 124 | `e8c3240577b726dae2f9f3bbc4c64ed0286f52afd5474d788fafe589bbfff694` |
| Accepted and candidate, 120 non-query | `19c5718dfc1aa4cb11eef0306b8b877dbd9f234f1192053cb77ce4de7c9337fb` |

Algorithm: SHA256 of sorted `path oid` lines joined by LF without terminal LF.
Pinned submodule entries use `submodule:<superproject-path>` and blob OID;
native derived entries use `native:<path>` and derived SHA256. Base-native
hashes are separate checks, not extra digest entries.

### Separate non-superproject accounting

- Submodule gitlink stays `3dac53c544e546574cb6e8078cd14053d0916113`.
  Against original `d28fe917370c2d55230c3ea226e5660785b44aca`: five files,
  331 additions / 2 deletions = **333 gross**, 329 net. Five pinned blob hashes
  and candidate local bytes match the snapshot.
- Native image patch: four files, 98 additions / 1 deletion = **99 gross**,
  97 net. All eight base/derived file hashes match; no rebuild or patch
  application was performed.
- Module-size note: **12 gross**; gitlink: **2 gross**, still separately added
  to the source total. Neither changed.
- Eight fixed image/source evidence blobs and candidate local files match
  their snapshot hashes. This binds retained image evidence; it is not a live
  registry, installed-package, or worker inspection.

## Verification and limits

Executed successful checker:

```text
timeout 60s python3 -B task_memory/task_2026-10-22_step5_position_routing/repairs/evidence/named_query_scope/recompute_named_query_scope.py
```

Environment: `/usr/bin/python3`, Python 3.10.6, Linux 5.10.25, glibc 2.35;
bytecode disabled. The checker enforces `RLIMIT_AS = 32212254720` bytes
(30 GiB) and uses a 60-second timeout on every Git child command. Largest
named input receipt is 427461 bytes; largest bound blob is 372643 bytes.
No memory-heavy read or artifact/data scan occurred.

Exact Git command arguments are retained in the receipt. They cover scoped
`ls-tree`, `cat-file`, `diff --no-renames --name-status/--numstat`, four native
`diff --no-index --numstat` comparisons, ancestry metadata, and HEAD/index
read-only checks. No tests, probes, project imports, CSV/pickle/trace/model
reads, network, GPU, package operations, or production edits were performed.

Three execution issues were resolved without changing source or behavior:

1. `systemd-run --scope -p MemoryMax=30G ...` did not start the checker:
   `Failed to start transient scope unit: Interactive authentication required.`
2. The user-scoped variant also did not start it:
   `Job failed. See "journalctl -xe" for details.` Cgroup `MemoryMax` is therefore
   unverified on this host. Small scoped reads used the stated 60s/30GiB address-space
   bound instead; no memory-heavy inspection was substituted.
3. The first checker run incorrectly expected the new amendment at the explicit
   pre-allocation step base. Git returned `path ...named_model_query_scope_amendment.json
   does not exist in de2d818f`. The checker was corrected to verify step-base ancestry
   and bind the amendment at allocation/source/records, where it actually belongs.
   The next accounting run passed with zero violations.

After the first successful accounting window, parent independently advanced
primary from `e4a7d7b2` to `2d417982` (`docs(d29): record named-query parent checks
and independent reviews`). A closing assertion that assumed the old HEAD
remained current failed. Read-only Git inspection confirmed zero non-record
changes; the checker stopped requiring that unrelated parent records remain
frozen and was rerun successfully against the new primary state.

Primary `/data/ycfeng/Frontier/.worktrees/step5-preview` was preserved throughout
the final checker window at `2d417982b2b9851bc781c9b201a3a5b4495c8a60`; index
SHA256 was unchanged at `7442976baad3a35b134f5845a59df745d233a1c493032d63d10c52f6e78e541d`.
Candidate remains at `3d75c786781802ad2753cbdd81d989c9b694660c`; index SHA256
remains `eba11ebc9ba894ea34705c703531d97921551c5dc708850b3b67cdf709948341`.
Previous audit files were not edited.

## Residual checks

- Parent pytest/wave checks and both review axes are separate evidence; no
  execution results were read or accepted by this accounting audit.
- The final integrated head must retain the exact four-path changed-file set,
  the captured query blobs, and the 120-binding digest. It must stay within all
  existing path/owner/source/global caps.
- A fresh final source freeze, required frozen regressions, and full-workflow
  acceptance remain outside this scope. No current CLI success is inferred.
