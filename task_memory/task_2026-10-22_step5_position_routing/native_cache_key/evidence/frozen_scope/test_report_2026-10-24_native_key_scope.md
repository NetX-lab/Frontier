# D29 native-key frozen scope delta

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Independently close the committed two-path accounting/binding delta without a native, regression or CLI acceptance claim. |

## Verdict

**PASS_FROZEN_NATIVE_KEY_SCOPE_DELTA; zero violations.**

- Candidate/source: `376983d2ecc441d1e0a8cb8b943e65a2dcb217b5`.
- Tree: `c8f8657f650e4645ec0fc0c86b84283766c09b3c`.
- Fixedpoint/actual source parent: `9870597776229d84af5eb58d49bf9b1756092999`.
- Amendment step base: `18cc3e9e1008d796a08f3b5637c9b10311979735`.
- Original gross-billing base: `1aee0af11cd58691e64603b99eca0ac459a5b350`.

Candidate is exactly the requested commit and clean before/after. Its index
SHA256 is unchanged:
`08d7365de779d28d77af692ee2460f4ec5d8bf043044cd90e14a187f03a5d007`.
Primary remains ateda4413b during the final checker, with unchanged index SHA256
`aedb2d96abe4badafc41a9f6d33d224b321b03ee2c23c8275f2063f5ca9628d6`.
The initial checker observed98705977/index00b587db unchanged. Parent subsequently
staged and committed its native-key regression/review archives ateda4413b;
31 record paths changed, zero non-task_memory differences from98705977. A closing
inspection's old-primary-fingerprint assertion consequently failed; this was
external record progress, not candidate source drift or an audit edit. The final
checker uses current authority, still requires the frozen candidate376983d2,
and preserves the new primary HEAD/index throughout its own execution window.
The current primary task documents are read as authority; no whole dirty primary
snapshot is required, used as source, edited, or cleaned.

## Exact changed-file guard

All accepted named-source/step-base/fixedpoint predecessor comparisons have
**zero non-task_memory differences**. Each predecessor-to-candidate name/status
set is exactly:

```text
M frontier/cc_backend/backends/collective_sim_cc_backend.py
M tests/unit/test_collective_sim_cc_backend.py
```

No donor, other source/test/config, mode, doc, gitlink or bound evidence changes.
Both affected files remain100644. Native key behavior/quality is reviewed by
other agents; this report evaluates numerical accounting and identities only.

## Measured step versus cumulative billing

| Path | Step added/deleted | Step gross/net | Accepted cumulative gross | Candidate gross/cap |
| --- | --- | --- | ---: | ---: |
| `frontier/cc_backend/backends/collective_sim_cc_backend.py` | 1/1 | 2/0 | 93 | 93/93 |
| `tests/unit/test_collective_sim_cc_backend.py` | 1/0 | 1/1 | 236 | 237/237 |
| Unedited `frontier/cc_backend/collective_plan.py` | 0/0 | 0/0 | 139 | 139/139 |

Step is **3 gross (2 source/1 test), net1**, measured identically against
fixedpoint98705977 and explicit amendment step-base18cc3e9e. The backend replacement
changes a previously billed line, so its cumulative gross stays93. Only the test
adds one cumulative billed line: **9565+1=9566**, not9565+3.

The sole new overlay transfers one PRICING allowance:

- plan140->139, delta-1; donor source unchanged and actual139.
- backendtest236->237, delta+1; actual237.
- backend93, baseCC14 and testpricing311 are unchanged, at93/93,14/14 and311/311.

## All114 caps and composed totals

| Owner | Allocated paths | Actual gross | Effective cap | Headroom |
| --- | ---: | ---: | ---: | ---: |
| CORE/source | 49 | 4261 | 4404 | 143 |
| PRICING | 46 | 3955 | 3956 | 1 |
| native | 19 | 1350 | 1582 | 232 |
| Source | 114 | **9566** | **9942** | **376** |
| Global, source+doc12+gitlink2 | — | **9580** | **10000** | **420** |

114 allocated/113 changed. The only unchanged allocated path is
`tests/unit/test_step5_contract_reuse.py`, actual0/cap1. Per-path cap/actual/basis
rows are in `frozen_scope_paths.tsv` and `frozen_scope_receipt.json`.

The original107-path billing is reconstructed from the approved scope rows and
its declared approval SHA, using the original shared-owner rules. All17 overlays
are replayed for **allocation only**, including every before/delta/after, new
path, owner cap and source cap. Every owner transfer is zero-sum and every
effective cap matches the bound accepted ledger plus only the new1-line transfer.
Step reservations/checkpoint ceilings are not actual usage measurements.

## Fresh124/122 binding closure

The accepted named-query receipt, TSV, recompute script and artifact inventory
are byte-bound to `de5eeab05a2f312da415a9e6557847ab583b4990`. Existing helper
functions `entries`, `numstats`, `digest` and the snapshot `allocation` function
are reused without running their historical main functions. Source snapshots,
current candidate working bytes and actual native base/derived bytes are checked
fresh. The unchanged112 billed rows are carried only after the exact name guard
and all124 bindings pass. Only the two affected paths plus unedited donor receive
fresh cumulative superproject Git counts; no full114 historical re-audit occurs.

All124 binding identities pass; exactly2 change and **122 remain unchanged**:

| Digest | SHA256 |
| --- | --- |
| Reconstructed accepted all124 | `e8c3240577b726dae2f9f3bbc4c64ed0286f52afd5474d788fafe589bbfff694` |
| Candidate all124 | `aa9cdac571d68e2b758d916c5977a76c37a40d95f797a468dccd708c87798ae0` |
| Baseline=candidate non-key122 | `eaf97ec1746ad70a1e2bb97fcd9d9e4a8f176dcb166c09c65e6ac26d74ba01b6` |

Algorithm: SHA256 of sorted `path oid` lines joined by LF, no terminal LF;
submodule labels use pinned blob OIDs; native labels use derived SHA256.
The122 digest was recomputed independently and then compared with the earlier
primary/owner record, not copied as a presumed measurement.

Changed blobs:

| Path | Old Git blob | New Git blob | New SHA256 |
| --- | --- | --- | --- |
| backend | `7284d21c2b559bf9ee6c796030446239b8c7dd07` | `743958cba327ff9c3342bf8a31b7df6e27be69b4` | `e4a77ab21176e4e29fe65e6cdae1919a3ab2a9a659ea3d9bc49408a06a037cb5` |
| backend test | `476c36452e0ceceb5d84110a5c4566300f7b137f` | `504dc97c09a03660c8c528c9eb745303ff8931f2` | `f20ecc4ba95b2ff5d63c6ae199774bff27642d509f26f5a56e7fb2190c7fd041` |

Native/submodule/fixed evidence:

- Pin `3dac53c544e546574cb6e8078cd14053d0916113`, five pinned blobs, local bytes and
  clean submodule freshly verified. Git against originald28fe917 gives **333 gross**.
- Four native base/derived source pairs freshly hashed; no-index comparisons
  reproduce **99 gross** and the accepted per-file additions/deletions.
- Eight immutable patch/derivation/image evidence entries match their Git
  identities and SHA256 values; source doc12 and gitlink2 remain unchanged.
- No submodule/image rebuild, live image registry inspection or runtime admission.

## Verification command and execution

Working directory:
`/data/ycfeng/Frontier/.worktrees/d29-native-key-scope-audit`.

```bash
timeout 60s sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/native_cache_key/evidence/frozen_scope/recompute_frozen_scope.py
```

Final successful run: PASS,0.698536s (initial PASS0.708119s), Python3.10.6, UID10250/brainpp, Python-B,
verified MemoryMax32212254720 bytes (30GiB),60-second outer and Git limits,
optional Git locks disabled. **Fallback not used.** The receipt retains41 small
input bindings and189 exact Git verification commands, including output hashes.

First metadata attempt used the records-only de5 audit commit to locate every
overlay and returned128 for `named_model_query_scope_amendment.json`, which was
introduced on the separate accepted source branch. Correction: bind historical
overlay bytes to accepted named-query source1737ce9c, while keeping the prior
receipt/TSV/recompute bound to de5. This is a commit-location correction, not an
overlay edit or relaxed numerical test. The same bounded command then passes.
The earlier owner dirty-patch scope receipt/script remain unchanged and are
retained as hash-bound corroboration, not used as the final committed guard.

## Artifact hashes and limits

The compact parent receipt is `frozen_scope_summary.json`; its evidence section
hash-binds the detailed receipt and TSV. All output hashes are in
`artifact_sha256.json`; all124 fresh binding identities,
41 input hashes, overlay transitions, counts and limits are in
`frozen_scope_receipt.json`. The report/TSV do not introduce another allocation.

No tests, waves, benchmarks, probes, production imports, largeCSV/fulltrace/model
or pickle reads, network/GPU/platform operations, source/test/donor edits, cleanup
or publication occurred. No code-review verdict, new regression admission,
native qualification or6144/1024/2400s CLI acceptance is claimed. Those checks
remain separate with their owners. The only remaining action for this audit is
the records-only delivery; no accounting/binding verification remains unfinished.
