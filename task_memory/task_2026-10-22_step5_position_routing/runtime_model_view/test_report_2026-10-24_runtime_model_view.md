# D29 named-model registry-view control

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record current-source timing, complete output comparisons, contract differences and the pending policy decision. |

## Result

**QUALIFIED CONTROL; IMPLEMENTATION PENDING ERROR-POLICY DECISION.** The cancelled
remaining-runtime design worker delivered no artifact. Primary identified the
whole-family query mismatch, ran the bounded control and retained the evidence.
An independent read-only inspection by `ses_e951e411cffenineDsU35LlI6M` confirmed
that view checks are per-name and identified the cache-name/writer invariant.
This was an exploration, not a Standards/Spec implementation review.

Both arms use owner source `c4e06a61`, records HEAD `d1e2fb8c`, byte-equivalent
outside task records to accepted union `46fe83ac`. The candidate replaces only
two method bodies in memory. Neither production tree is modified. Twenty-three
source files match their committed/accepted bytes at final verification; the
wave's existing runtime guards cover its22 frozen files, with the registry
additionally checked against its Git blob. All262 fixture/cache filenames,
sizes and mtimes remain unchanged.

## Measurements and expectations

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Warm pricing | Material reduction with work preserved | 0.122211045→0.089446953s, −26.8094%;3 samples/arm | PASS control |
| Requested models | Same objects selected | 38 full-view entries;2 identical GEMM objects, same order | PASS |
| Contract derivation calls | Remove unrelated per-point work | 384→0; this fixture has native-only GEMM entries | PASS |
| Family model lookups | Fewer unrelated lookups | 544→160 | PASS |
| Numerical/proof/collective/point/key calls | Unchanged | 288/144/64/16/192 each arm | PASS |
| JSON codec calls | Explain changed work | loads1136 unchanged; dumps800→704 | Observed |
| Collective input reads | One successful binding read | 1 total, zero cold/warm reads each arm | PASS |
| Full208 pricing payload, proofs, keys and order | Exact bytes | SHA256 `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83` | PASS |
| Full JSONL | Exact bytes | 15 complete208-event files per arm; each SHA256 `1850a9b6a90cc7bb087b881e72d8cc00f5a669fa0be0f54bb25f3361a38d7e11` | PASS |
| Ledger | Exact bytes | 2,309,545B; SHA256 `006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf` | PASS |
| Snapshot isolation / wrong keys | Existing behavior | Getter/pending/ledger isolation and64 wrong-key failures retained | PASS |
| Late serialization failure | Same event prefix, error and buffer | Exact3,902B event prefix; failure buffer restored | PASS |
| Emission timing | No emission change proposed | Medians37.383→37.191ms,15 samples/arm | Not attributed to this control |

There are19 eligible whole-file byte comparisons, including the15 complete
wave files. Two auxiliary files intentionally have current wall-clock metadata
headers: `profile/op_traces.jsonl` contains only that header, and
`late-json-failure/op_traces.jsonl` adds the3,902B event prefix. Their only
header difference is `meta.timestamp`; the latter's event prefix is compared
byte-for-byte. These diagnostic headers are not silently normalized into a
whole-file equality claim. Exact values are in `evidence/verification.json`.

## Contract results and decision

`evidence/contract_cases.json` contains eleven case groups, not eleven pytest
tests. Ten groups match, and one group deliberately demonstrates non-equivalence.

- Unfiltered order/object identity, selected objects, missing name, count-model
  exclusion and logical registry/cache names agree on the real manager.
- Normal actual-point results agree. A collision for the selected
  `moe_grouped_gemm` raises the same ValueError in both point paths.
- Replacing rows, measurement text, the estimator through the production
  registrar, or the active KERNEL_ONLY/CUDA_EVENT family stays live. Empty rows
  and malformed routing JSON retain their exact ValueError/JSONDecodeError.
- **Policy difference:** a second native `moe_gating_linear` artifact with the
  same flattened label but different native source makes the old GEMM query
  raise `Conflicting native model artifacts for 'moe_gating_linear:852eb571'`.
  The narrowed GEMM query succeeds and returns the normal result. Both
  unfiltered views still raise. The fault uses `_store_model_precision`, not a
  replacement validator or suppressed exception.

The candidate therefore does not preserve every old error at every call site.
Initialization and public full-family views remain unfiltered. A full-view
collision is not interchangeable with context-qualified numerical admission;
this report does not claim every removed unrelated-view error reappears during
that model's later numerical lookup. YC must accept this policy before source
allocation. Selected-model/native-count/context/proof checks remain mandatory.

Known writers use the logical cache name as registry key, and bundle publication
rejects divergence. The control checks the actual manager's names and retains
the existing `_frontier_cache_model_name` predicate. It is not a new global
guarantee that arbitrary private registry mutation cannot violate that relation.

Still required before implementation acceptance: filtered mixed native/non-native
orders, typed ambiguity/missing-contract cases, DECODE_ATTN filtering, applicable
public-view regressions and affected test stubs. Normal-wave parity alone does
not settle those cases. The scratch method diff is12+2=14 gross / +4 retained.
The proposed100-line ceiling(source20/tests80) is unallocated and needs detailed
path caps after the behavior decision. Current accounting stays9491/10000.

## Exact commands and environment

CWD: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Scratch: `/data/ycfeng/tmp/d29_model_view_control01`.
Python: `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6;
NumPy2.2.6, pandas2.3.3, scikit-learn1.7.2. Each command used
`timeout60s` and `systemd-run --scope -p MemoryMax=30G`, running asbrainpp.
All Python runs used`-B`/`PYTHONDONTWRITEBYTECODE=1`.

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_model_view_control01/prepare.py > /data/ycfeng/tmp/d29_model_view_control01/prepare.log 2>&1
```

The following exact wrapper was used sequentially for `run.py baseline`,
`run.py candidate`, and `contract_cases.py`, with logs `baseline.log`,
`candidate.log`, and `contract_cases.log` respectively:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_model_view_control01/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_model_view_control01/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_model_view_control01/cache/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_model_view_control01/cache/numba /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_model_view_control01/run.py baseline > /data/ycfeng/tmp/d29_model_view_control01/baseline.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_model_view_control01/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_model_view_control01/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_model_view_control01/cache/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_model_view_control01/cache/numba /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_model_view_control01/run.py candidate > /data/ycfeng/tmp/d29_model_view_control01/candidate.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_model_view_control01/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_model_view_control01/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_model_view_control01/cache/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_model_view_control01/cache/numba /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_model_view_control01/contract_cases.py > /data/ycfeng/tmp/d29_model_view_control01/contract_cases.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_model_view_control01/verify.py > /data/ycfeng/tmp/d29_model_view_control01/verify04.log 2>&1
```

Initial archive-check iterations failed on the two unfrozen diagnostic metadata
headers (`verify.log`, then `verify02.log`) and on the wrong receipt field name
`warm_seconds` (`verify03.log`; actual schema is`samples[].pricing_s`). These
were checker defects. The verifier now explicitly separates the two known
diagnostic headers and derives warm values from the recorded schema. Final
verification passes. No wave or contract case was rerun, and no production
behavior was changed to fix the checker. All three original error logs remain
in the evidence manifest.

## Artifacts and limits

`evidence/manifest.json` binds22 compact files, including original/control driver,
two candidate method texts, source/input freeze, raw wave results, contract
results, command logs and the verifier. Verification SHA256 is
`31d72236ba3dc5067189b55e4291876bce8e46be8e82b525b4c1c5d9887d4a12`.
Large raw JSONLs remain under the scratch directory; no full CSV/trace corpus
was scanned. The protected body remains exactly
`1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`.

That protected body retains its old D29-IMMUTABLE-COST-FIELDS task label and
the phrase "No diagnostic reuse control or unallocated source behavior". Those
inherited strings do not describe this new control; `view_control.applied` and
the two candidate method files describe the actual in-memory replacements.
The unchanged body is only the output/error/snapshot oracle here.

This is a cached-manager, actual snapshot/backend FP8 decode wave, not a full
Simulator constructor or6144/1024 CLI/native run. P6128/D16/BF16 output and total
wall time remain unmeasured. The sampled emission cost alone still exceeds the
26.608ms average whole-wave budget. This one change cannot qualify the complete
2400s workflow, and storage remains unresolved. Full-unit completion and its
dependent checks remain separate work under `post_key_regressions/`.
