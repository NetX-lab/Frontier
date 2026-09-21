# Test Report 2026-09-21 — Step 1 Fidelity Matrix and Step 2 Cleanup

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Initial report: harness self-check, baseline capture, and the first cleanup comparison. |
| 2026-09-21 | Added section 6, the `config.py` split. |

## Environment

| Field | Value |
| --- | --- |
| Host | `kun-workspace-vgen2`, CPU only |
| Python | `/data/ycfeng/envs/frontier-py310/bin/python`, CPython 3.10.6 |
| Packages | numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, plotly 7.1.0, pytest 9.1.1 |
| Baseline checkout | `/data/ycfeng/Frontier/.worktrees/fidelity-baseline-main`, detached at `1f694f7c549aa3aeeb7c5bbae04e119c09167a77`, never edited |
| Candidate checkout | `/data/ycfeng/Frontier/.worktrees/oversized-module-split` |
| Output root | `/data/ycfeng/tmp/issue26-correctness-pr/refactor-fidelity` (scratch, not committed) |
| Harness | `tests/e2e/refactor_fidelity/` (`cases.py`, `compare.py`, `run_matrix.py`) |

## 1. What the matrix compares

Each case runs a checked-in example wrapper with the working directory set to the checkout under test, which is what makes the shipped configuration's relative profiling-data and `cache` paths resolve. The driver always comes from the candidate checkout, so one case table and one comparator measure both sides.

Every file the run writes into the normalized metrics directory is compared, and the file sets must match:

| Artifact | Comparison |
| --- | --- |
| `request_metrics.csv` | Every cell, exactly. First differing cell is reported with its column name. |
| `system_metrics.json` | Every leaf, exactly. |
| `frontier_stage_batch_ledger.jsonl` | Record count, then every leaf of every record. |
| `op_precision_metadata.csv` | Every cell, exactly. |
| `config.json` | Every leaf, exactly, after path substitution. |

The only normalization is the substitution of three run-specific absolute paths (the checkout root, the output root, the label root). That list is short because it was checked against the real artifacts: on `1f694f7` the metrics files carry no timestamps, wall-clock durations, hostnames or paths, and `config.json` carries exactly one absolute path, the metrics output directory. Anything the substitution list fails to cover shows up as a difference rather than being silently accepted.

There is no tolerance. A behavior-preserving refactor has no reason to change a simulated number.

In addition, the names of the predictor cache files each side produces are compared. Retraining from the same CSV reproduces the same numbers, so a changed training identity or cache key would otherwise leave no trace in the outputs.

## 2. Case coverage

67 cases, all driven through checked-in wrappers:

| Group | Cases | What it covers |
| --- | --- | --- |
| `trained_predictor` | 6 | Dummy mode disabled; checked-in CSVs for dense and MoE on two devices; the kernel-only CUDA graph path |
| `colocation_offline_dense` | 13 | Request counts 4/8/16/64, prompts 128/512/1024/3584, TP1/TP2, PP1/PP2, one and two replicas, attention DP2, all three decode CUDA graph modes, chunked prefill on and off, the Sarathi and SGLang schedulers, two dummy latencies |
| `colocation_offline_moe` | 11 | EP1/EP2/EP4, MoE TP1/TP2, all four routing distributions, top-k 2 and 4, long prefill, 32 requests, CUDA graph off |
| `colocation_online` | 6 | Dense and MoE at 0.5, 2 and 8 requests per second |
| `colocation_features` | 7 | Thinking mode, speculative decoding (two token counts), prefix caching, each offline and online |
| `pd_disaggregation` | 14 | Dense and MoE, replica counts per role, TP2, chunked prefill off, EP1, skewed routing, thinking mode, speculative decoding, prefix caching, two online arrival rates |
| `pd_af_disaggregation` | 10 | Dense and MoE, EP1 and EP2, the global CUDA graph contract, offline and online |

## 3. Harness self-check

Running the same 13 co-location dense cases twice from the **same** checkout, in two separate processes, must produce identical artifacts; otherwise the comparator cannot distinguish a refactor regression from ordinary run-to-run noise.

| Field | Value |
| --- | --- |
| Command | `run_matrix.py run --repo-root <baseline> --label selfcheck_a|selfcheck_b --case-filter coloc_dense_offline` then `compare` |
| Expected | 13 identical, 0 mismatched |
| Actual | identical 13, mismatched 0, cache file names identical |
| Result | **PASS** |

This covers the 343 KB stage ledger and `config.json` as well as the metrics files.

## 4. Baseline capture

| Field | Value |
| --- | --- |
| Command | `run_matrix.py run --repo-root <baseline> --label baseline --output-root <scratch> --jobs 6 --clean-cache` |
| Expected | Every case completes and writes a metrics directory |
| Actual (first attempt) | 65/67. Two cases failed. |
| Actual (after case corrections) | **67/67**, zero failures |
| Result | **PASS** |

Both first-attempt failures were invalid case definitions, not simulator defects, and were corrected in the case table before any source change:

| Case | Diagnosis | Correction |
| --- | --- | --- |
| `coloc_dense_offline_long_prefill` | 4096 prefill tokens plus 16 decode tokens exceeds the 4096-token context of Llama-2-7b, so no request is ever admitted and the run ends with a non-empty scheduler state | prefill reduced to 3584 tokens |
| `coloc_moe_offline_ep1`, `coloc_moe_offline_attn_dp2` | Violated the shared-domain invariant `attn_tp * attn_dp == moe_tp * moe_ep`; the MoE wrapper additionally enforces the stricter `ATTN_TP == MOE_TP * MOE_EP`, which cannot express `attn_dp > 1` | EP1 case set to `ATTN_TP=1`; the MoE DP case was replaced by an EP4 topology, since the dense matrix already covers DP lanes |

## 5. Cleanup comparison (Step 2, first part)

Source changes under test, in `frontier/config/config.py` only:

1. Removed `BaseExecutionTimePredictorConfig.validate_linear_op_input`, which had no caller anywhere in `frontier/`, `tests/`, `examples/` or `docs/` (re-verified by grep immediately before the edit).
2. Replaced the two parallel CC-backend dispatch ladders with one ordered `(type key, config class, creator)` table, preserving the original evaluation order. The five backend config classes are siblings of `BaseCCBackendConfig` with no cross-inheritance, so the order is not load-bearing either way.
3. Replaced four copies of a field-resolution closure with one `_cc_backend_value_reader`, and three identical `hasattr`-guarded field triplets with one `_shared_cc_backend_fields`. `hasattr(x, "a") and x.a or default` and `getattr(x, "a", default)` are equivalent, so this is a textual deduplication.
4. Removed a method-local re-import of `dataclasses.replace`, which the module already imports.

Net effect: 5720 to 5687 lines.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Generated CLI flag set | Identical to the baseline | 753 flags, `diff` empty | PASS |
| Fidelity matrix | 67 identical, 0 mismatched | 67 identical, 0 mismatched, 0 candidate-only failures | PASS |
| Predictor cache file names | Identical | 138 of 426 names differed; see below | Investigated, harness cause, corrected |

### The cache-name difference was a harness artifact

The differing names were the `.pkl`, `_predictions.csv` and lock files of the two `examples/profiling/smoke_simulator_*_csv.sh` cases. Cause: `ExecutionTimePredictionModelManager._get_hash_relevant_config` includes the profiling input file paths in the model hash, and those wrappers default `DATA_DIR_BASE` to an absolute path under their own repository root, which differs between the two checkouts by construction. The architecture wrappers were unaffected because they leave the relative path templates in place.

This is not a training-identity change: the simulated outputs of both smoke cases were already identical, and the CSV contents are the same file in both checkouts.

Correction: both smoke cases now pass `DATA_DIR_BASE=data/profiling`, a repository-relative base that resolves identically from either checkout, so the cache-name check becomes meaningful again. Both sides were recaptured with a cleaned cache after the correction.

## Limits

- The matrix establishes that the outputs of 67 supported configurations do not change. It does not prove that no unsupported or unexercised configuration changes, and it makes no accuracy claim.
- Dummy-mode cases exercise structure and lifecycle, not realistic latency. Six cases use the checked-in profiling CSVs, which is what covers dataset loading, training identity and the persistent cache.
- The predictor cache comparison compares file names, not pickle contents, because pickle bytes are not required to be reproducible.

## 6. `config.py` split (Step 3)

`config.py` went from one 5,687-line module to twelve modules, the largest 1,888 lines. The full boundary table is in `plan.md` section 3.1; `issues.md` I1 records the one real defect this step introduced and how it was found.

### 6.1 Compatibility gates

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Generated CLI flag set | Identical to the base commit | 753 flags, `diff` empty | PASS |
| Public import surface | All 36 names other modules import resolve from both `frontier.config` and `frontier.config.config` | none missing from either | PASS |
| `ClusterConfig` dataclass fields | Unchanged | 181 fields; the two mixins contribute none, as intended | PASS |
| `ClusterConfig` MRO | `ClusterConfig`, `ClusterRoleConfigBuilder`, `ClusterTopologySummary`, `object` | as expected | PASS |

### 6.2 Unit suites

Selection: every file under `tests/unit/` that mentions `ClusterConfig`, `frontier.config.config` or `from frontier.config import`, 62 files.

```bash
python -m pytest $(grep -rln --include="*.py" \
  "ClusterConfig\|frontier.config.config\|from frontier.config import" tests/unit | sort) \
  -q -p no:cacheprovider --tb=no
```

| Side | Result |
| --- | --- |
| Base commit `1f694f7` | 10 failed, 671 passed |
| This branch | 10 failed, 671 passed |

The failing test identities are **byte-identical** between the two sides: all ten are in `tests/unit/test_colocation_release_review_contracts.py` and are the pre-existing drift recorded in the Step 0 baseline report (the `tests/debug/` scripts they open are not tracked on `main`, and one test spawns a bare `python` executable absent from this host's PATH).

The first run of this selection found **7 additional failures** that the fidelity matrix would not have caught at that point. They are recorded as I1 in `issues.md`: `get_cluster_configs_for_disaggregation` constructs `ClusterConfig` objects at runtime, and the extraction had imported the name only under `TYPE_CHECKING`. Fixed with a lazy import inside the method, matching the package's existing `_get_cc_backend_configs` pattern.

### 6.3 Fidelity matrix

Both sides recaptured with a cleaned predictor cache.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Cases producing artifacts | 67/67 on both sides | 67/67 | PASS |
| Cases identical | 67 identical, 0 mismatched | identical 67, mismatched 0 | PASS |
| Candidate-only failures | none | 0 | PASS |
| Predictor cache file names | Identical | 0 baseline-only, 0 candidate-only | PASS |

Compared revisions: baseline `1f694f7c549a`, candidate `56efdf765ee0` plus the working tree of this step.
