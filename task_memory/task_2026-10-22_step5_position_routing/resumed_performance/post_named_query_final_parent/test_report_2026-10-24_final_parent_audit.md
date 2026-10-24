# D29 post-named-query final parent audit

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Independently admit the completed successor receipts while preserving raw exit1, the approved79 ID-loss FAIL, and all qualification limits. |

## Outcome

**ACCEPTED_NO_NEW_CODE_FAILURE_WITH_UNWAIVED_APPROVED_79_ID_REPLACEMENT.**
No new source failure, status drift, scope violation, missing artifact, or unfinished
regression check was found. This is receipt admission, not an all-green unit run
or whole-CLI/native qualification.

| Independent selection | Observed counts | Raw exit / result |
| --- | --- | --- |
| Canonical full unit | 8028 PASS / 279 FAIL / 32 ERROR / 281 SKIP; 8620 unique IDs | **1**, preserved |
| Strict fidelity | 77 compared / 77 identical; every difference count zero | Retained run0/compare0; fresh original evidence auditor0 |
| Supplement | 101 PASS | 0 |
| Metadata | 197 PASS | 0 |
| Catalogue | 53 PASS | 0 |
| Governance | 52 PASS | 0 |

Counts are not summed across overlapping selections. Shorts remain separate
same-source environmental evidence; no full-unit failure/error/skip was relabelled.

## Frozen inputs and owner archive

- Source: `dbe19f531ff68d9edafc480418e8927c40eb1c60`.
- Tree: `6c43090097e92c6fc0c33aa7358e69c2cf092705`.
- Source worktree: `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08`.
- Submodule: `3dac53c544e546574cb6e8078cd14053d0916113`; canonical binary remains absent.
- Owner: `/data/ycfeng/Frontier/.worktrees/d29-post-named-query-regressions`.
- Archive commits: `0a42e56ef53769a85a999b652cd9d03973d1f727` (34 scoped files), then
  `39f0c1000a96c10f186cabe5f77dfc3058f86549` (five scoped status/manifest updates).
- Audit records base: `86b9923f8c29f1c58ba9f8df23a396c70a4d1caa`.

All 33 current manifest entries plus the manifest itself were checked against
their hashes and owner commit bytes. Retained/raw archive copies, the existing
27-hash short parent admission, and exact completed execution/comparison receipts
were independently rebound. No owner or primary file was edited.

## Raw identity/status comparisons

| Baseline | Common retained IDs | Missing | New, all PASS | Shared-status changes | FAIL/ERROR/SKIP set changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 46fe | 8612 | 0 | 8 | 0 | 0 |
| 79 | 8609 | 1 approved old ID | 11 | 0 | 0 |

The missing79 ID is exactly:

`tests.unit.test_collective_sim_cc_backend::test_collective_artifact_reuses_only_identical_current_text`

The approved new lifetime test is a genuinely different ID, not an alias:

`tests.unit.test_collective_sim_cc_backend::test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published`

The eight named-query additions are exactly the three parameterized
`test_named_point_keeps_selected_collision_and_cache_name_guard` IDs, one
`test_named_native_view_keeps_selected_objects_and_unfiltered_order` ID, and four
`test_named_typed_view_preserves_contract_errors_and_role_exclusion` IDs. Their
full exact names and PASS statuses are retained in `parent_acceptance.json`.
AST inspection independently confirms each added function is absent from the
46fe test source and present in the frozen source. No tests were executed.

Raw strings remain unchanged:

- 79 exact presence/status: `FAIL_RETAINED_ID_LOSS_OR_STATUS_CHANGE`.
- Legacy comparator exit: **1**.
- Legacy verdict: `INCOMPLETE_OR_FAILED_REQUIRED_GATE`.
- Required-gate verdict: `FAIL_OR_INCOMPLETE_REQUIRED_GATE`.

Parent admission recognizes the approved input-lifetime replacement separately;
it does not waive, rewrite, filter, or alias these raw gates.

The original locked267 failures and31 errors remain. Twelve missing-aiohttp
failures join the exact passing IDs in the independent101 supplement; one
missing-Torch collection error joins16 passing IDs from its module. Five
historical missing-Torch failures separately join52 governance PASS. All197
metadata and53 catalogue IDs also PASS in full unit. Raw279/32/281 stays intact.

## Strict evidence and namespace reproduction

The original evidence auditor is SHA256
`8d985f0f964c5ec6b16b4a0c3963490db38de26b9e0ec664927479ca849b4cb9`.
Its predicates, case definitions, original task ID, and comparator are unchanged.
Only source/output bindings are supplied to its function globals; SHA256 reads
use the same digest algorithm with streaming I/O.

Fresh checks independently verify **451 artifacts per side (902 total)** and
**519 cache filenames per side**, exact manifest/results hashes, 77 definitions,
clean full captures, complete comparison, and zero provenance/cache/difference
findings. The recomputed raw postrun receipt is byte-identical to the owner's
retained postrun receipt, SHA256
`db19519102c2f6217d22900e553b9f0e74379b983a9982c7d0fbce7de5fca767`.

The first attempt used the host namespace and returned FAIL because the recorded
`.../verify-step5-d29-integrated-08/cache` path is intentionally absent outside
the run's `bwrap` namespace. Its unmodified failed diagnostic is retained as
`diagnostic_outside_recorded_namespace.json`, SHA256
`51d79dc0a926ac2a89ce5fec64f18470ef7123b5adf01f2708799919913b4ee0`.

The actual retained cache is:

`/data/ycfeng/tmp/d29_post_named_query_regressions_20261024/strict77/generated_repo_root/cache`

It contains exactly519 recorded filenames with no additions or missing files.
The checker therefore reuses the **original recorded bwrap bindings**, adding
only this audit output subtree as writable, and runs the evidence auditor there.
It does not edit the cache, manifest, output, or any predicate. Exact argv is
retained in `receipt_namespace_command.json`. Fresh auditor exit is0; the earlier
outside-namespace diagnostic is not hidden.

## Final hash and scope checks

| Check | Actual | Result |
| --- | ---: | --- |
| Source files, exact frozen Git blob path set | 2755; 368850959 bytes streamed | PASS |
| Explicit guarded inputs | 14 | PASS |
| Explicit guarded scripts | 12 | PASS |
| Scope bindings | 124 | PASS |
| Unchanged non-query scope bindings | 120 | PASS |
| Unique file paths rehashed across the two phases | 3762 | PASS |
| CORE / PRICING / native cumulative gross | 4261 / 3954 / 1350 | Within4404 / 3956 / 1582 |
| Source / global cumulative gross | 9565 / 9579 | Within9942 / 10000 |
| Source/input/script/scope drift or violations | 0 | PASS |

Non-query digest:
`19c5718dfc1aa4cb11eef0306b8b877dbd9f234f1192053cb77ce4de7c9337fb`.

The accepted scope guard is reused unchanged except its output destination.
Source tree/gitlink/working bytes and all native base/derived pins are checked
by that guard; this is not a native rebuild or runtime qualification. Full
per-file hash lists are retained in `receipt_verification.json` and
`binding_verification.json`. All output artifact hashes are in
`artifact_sha256.json`.

Primary remains at86b9923f, owner at39f0c100, frozen source atdbe19f53 in every
verification window. Their index SHA256 values remain unchanged:

| Worktree | Index SHA256 |
| --- | --- |
| Primary | `26603ed9ca380850ed26c789d1ebd164bcaa86c8662df2bfec4fa225aa1b7f13` |
| Owner | `b00c97f2a69d59dec8ae6fd26a726ae51e6c918ae75573cd4ff5699e8ab70e42` |
| Frozen source | `c1d51d9d3edf483d79baad9f640c2730c590840e7ca9bdbc0299bf88d2bdf1c0` |

## Decisive raw hashes

| Artifact | SHA256 |
| --- | --- |
| `/data/ycfeng/tmp/d29_unit_parent07.xml` (2111062 bytes) | `64371d6394026aedb9fa8d9c409a9263bfc2f23d72a063886581b61827b8b5b3` |
| `/data/ycfeng/tmp/d29_unit_parent07.log` (4614412 bytes) | `9d055a740ae01a6428c5c13daca60cf3ee15bad934e6bcf61b73d017549a873b` |
| Strict `comparison.json` | `680c535ee73b14fffb1ed7e0d622b547ad6e8ac341502f9e70fce2510e51a447` |
|46fe baseline XML | `9d5ab2bcd0d5e969c4281b24f7f6b0fcde84a86ea344fbeef1b3173d93511df7` |
|79 baseline XML | `cb64bf1bfbba142b5ad7b3fbad90343d1f9134e6a627c35af5017bfc3654a4d8` |
| Short101 XML | `1463ac9747cc4af431d881c917c60748c5032688c023a5e7273e396becced384` |
| Metadata197 XML | `69ce0103e28663ba975ba420c53e554983529802906e2c034ad724ffd42df80c` |
| Catalogue53 XML | `56d26f9844c7e34ea482653f5426cefbc69364e5609c820b3e50efd1a73059ec` |
| Governance52 XML | `c146cb4b6fbe92dc974d16d63b6a98978d20bc13882ceacb6696532aa69592e4` |

## Commands, environment and execution limits

Checker path relative to this records worktree:

`task_memory/task_2026-10-22_step5_position_routing/resumed_performance/post_named_query_final_parent/verify_final_parent.py`

Each phase uses the following exact envelope, with final argument `receipts`,
`bindings`, or `acceptance` respectively:

```bash
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/resumed_performance/post_named_query_final_parent/verify_final_parent.py receipts
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/resumed_performance/post_named_query_final_parent/verify_final_parent.py bindings
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/resumed_performance/post_named_query_final_parent/verify_final_parent.py acceptance
```

Observed phase durations: receipts2.255898s, bindings2.838707s,
acceptance0.067456s. All independently verified systemd MemoryMax32212254720
bytes (30GiB), UID10250/brainpp, Python3.10.6, bytecode disabled. Git children
have60-second bounds and optional locks disabled. Streaming chunks are1MiB.
Fresh original-auditor scratch is
`/data/ycfeng/tmp/d29_post_named_parent_strict_dmzfusxi`; no cleanup was performed.

The unchanged exact-ID parser is reused without calling its test runner/main.
JUnit files are small retained XML; no pandas CSV read, pickle/model loading,
large-trace parsing, test/probe rerun, simulation, native package operation,
GPU/network/platform operation or production edit occurred.

## Remaining gates

No receipt-audit verification remains unfinished. Parent must merge the owner
and this audit and update current acceptance/root docs separately. Whole
6144/1024/2400s CLI completion, native runtime/source staging, package readiness,
G3, and independent four-metric E2E<=10% remain outside this admission. Historical
full-unit failures/errors/skips and the approved79 old-ID presence FAIL remain
visible and must not be described as all-pass.
