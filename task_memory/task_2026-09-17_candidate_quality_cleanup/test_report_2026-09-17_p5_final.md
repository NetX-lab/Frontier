## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded final full-unit, exact skipped-node, non-dummy artifact and 58-scenario fidelity evidence. |

# Final correctness verification

## Source and environment

- Frozen main/merge-base: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Frozen pre-cleanup candidate: `c288a19f59bec09529ee18d782fa57218da2c781`, checkout `../quality-baseline-c288a19f`.
- Final production source: `5cb8794f`; test-only global isolation correction: `ff2b6cdf`. Documentation checkpoint `567742f3` has identical production and test source to that corrected tree. `git diff --exit-code 5cb8794f HEAD -- frontier` passed before timing.
- Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
- Python: `/data/ycfeng/tmp/quality-review-env/bin/python`, version 3.12.3, uv venv with same-interpreter system packages. No conda activation, GPU execution or production-profile accuracy claim.
- All runs used fresh paths under `/data/ycfeng/tmp`. Correctness jobs may overlap each other; paired timing runs separately with no concurrent correctness job or worker.

## Reproducible execution

Prefix each Python command below with:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python
```

| Check | Arguments following the prefix | Raw evidence |
| --- | --- | --- |
| Full unit | `-m pytest tests/unit -q -rfs -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p5-unit --junitxml=/data/ycfeng/tmp/quality-p5-unit.xml` | `/data/ycfeng/tmp/quality-p5-unit.log`, corresponding XML |
| Eight non-dummy cases | `-m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p5-nondummy` | `/data/ycfeng/tmp/quality-p5-nondummy.log` and fresh run directories |
| Fidelity | `tests/integration/run_scheduler_refactor_fidelity.py --baseline ../quality-baseline-c288a19f --candidate . --output /data/ycfeng/tmp/quality-p5-fidelity --workers 2` | `/data/ycfeng/tmp/quality-p5-fidelity.log`; output manifest/results and per-case invocation/log/artifacts |

The full script paths are the working directory above joined with the exact `tests/integration/...` paths. Fidelity records revision `5cb8794f` at launch; the later fixture-only change is not imported by that isolated subprocess harness. Full unit/non-dummy began at `ff2b6cdf`; subsequent documentation-only commits do not change executed code.

## Full unit: no new failure or skip

Observed: **3813 PASS / 18 FAIL / 25 SKIP**, 575 warnings, 132.23 s. The full suite is **not all green**. Fresh frozen-candidate baseline was **3587 PASS / 18 FAIL / 25 SKIP**, 134.21 s. The cleanup adds 226 passing parameterized checks; no existing failed test is relabeled or weakened.

Both commands exit 0, proving identical failed-node sets against frozen candidate and pinned main:

```bash
diff -u <(sed -n 's/^FAILED /FAILED /p' /data/ycfeng/tmp/quality-pre-unit-path.log | sort) <(sed -n 's/^FAILED /FAILED /p' /data/ycfeng/tmp/quality-p5-unit.log | sort)
diff -u <(sed -n 's/^FAILED /FAILED /p' /data/ycfeng/tmp/quality-main-failures.log | sort) <(sed -n 's/^FAILED /FAILED /p' /data/ycfeng/tmp/quality-p5-unit.log | sort)
```

| Existing failed group | Count | Observed cause / retained boundary |
| --- | ---: | --- |
| `test_colocation_release_review_contracts.py` | 10 | Missing legacy config_optimizer/debug scripts; e.g. `ModuleNotFoundError: No module named 'frontier.config_optimizer'` and missing `tests/debug/e2e-level/monolith_mode/scripts/test_base.sh` |
| `test_examples_documentation_contracts.py` | 2 | Existing README/public-release text assertions; README intentionally untouched |
| MHA/MQA stage2 CLI tests | 2 | Missing `flashinfer-python`, independently reproduced in P3; unchanged scripts |
| MLA stage3 trace-builder CLI | 1 | Existing legacy analysis/artifact dependency failure, same node/cause on frozen/main |
| `test_pdd_public_surface_docs.py` | 3 | Existing public documentation expectations, outside cleanup scope |

The exact **25 skipped final nodes** were extracted from final JUnit and rerun on the frozen candidate, with the same Python/environment and its own PYTHONPATH: **25 SKIP**, 7.08 s, no executed/passing/failing node substituted. `/data/ycfeng/tmp/quality-p5-baseline-skips.log` records the complete command; `/data/ycfeng/tmp/quality-p5-baseline-skips.xml` records outcomes. Reproduction extraction:

```python
import xml.etree.ElementTree as ET
nodes = [c.attrib['classname'].replace('.', '/') + '.py::' + c.attrib['name']
         for c in ET.parse('/data/ycfeng/tmp/quality-p5-unit.xml').iter('testcase')
         if c.find('skipped') is not None]
assert len(nodes) == 25
```

Run `python -m pytest -q -rfs -p no:cacheprovider --junitxml=/data/ycfeng/tmp/quality-p5-baseline-skips.xml` with those exact nodes from the frozen checkout. Three skips require external dataset tools, three require missing live-probe artifacts, and nineteen cover explicitly retired PD-AF lane contracts. These are not native-device PASS evidence.

## Non-dummy: complete reporting artifact preservation

Observed: **8 PASS**, 68.21 s. Real Simulator/profile loaders use deterministic synthetic trained profiles; operation/utilization metrics, op traces, physical-layer expansion and ledger/summary are enabled. Dense co-location/PDD/PD-AF, MLA PP2, MoE EP2/PDD/PD-AF and hybrid GDN are included.

Compared `/data/ycfeng/tmp/quality-pre-nondummy/test_nondummy_simulator_accept<N>/run` with `/data/ycfeng/tmp/quality-p5-nondummy/test_nondummy_simulator_accept<N>/run` for N=0..7:

- Existing `tests.integration.test_pr33_nondummy_acceptance.compare_baseline` reports **90/90 stable artifacts PASS**.
- Exact bidirectional file-name sets are equal: **106 metrics files**, per-case counts `[10, 14, 17, 10, 11, 15, 18, 11]`.
- Assert exactly one `acceptance_evidence.json` and one `frontier_stage_batch_ledger_summary.json` on each side per case; all **16 nonempty pairs PASS** through the original comparator.
- Log: `/data/ycfeng/tmp/quality-p5-artifact-comparison.log`.

Reproduction uses the existing `compare_baseline`, `compare` and `load_artifact` functions; it retains the existing trace-header treatment and numeric tolerances. After each comparison, assert every returned status is PASS, recursively compare both metric-directory path sets, then compare the two supplemental JSON files. Require final counters exactly `{cases: 8, stable_artifacts: 90, symmetric_metric_files: 106, supplemental_pairs: 16}`. No expected output or tolerance changed.

## Fidelity: 58 scenarios, zero finite numeric drift

Observed **58 PASS / 0 FAIL** against the pre-cleanup candidate. The existing matrix covers dense/MoE, co-location/PDD/PD-AF, online/offline, request/QPS/length variations, chunking, prefix/spec/thinking and DP/EP/graph features. Both revisions complete **122 requests**. The driver compares **304 stable artifacts** and requires nonempty stage ledgers.

The unchanged comparator uses `rel_tol=1e-12`, `abs_tol=1e-9`, exact field sets, identity, ordering and nonnumeric values. An additional recursive inspection of all compared artifacts found **902,628 finite numeric leaf pairs; 0 unequal pairs; maximum absolute difference 0.0; maximum relative difference 0.0**. Numeric leaves include numeric-string identity/metadata as well as timing, not 902,628 independent predictions. Matching non-finite fields retain their original schema semantics and are not converted to zero. Log: `/data/ycfeng/tmp/quality-p5-fidelity-numeric.log`.

For reproduction, load `results.json`, require exactly 58 PASS rows, locate one request-metrics directory for each run, and iterate every name in each row's `compared_artifacts`. Recursively traverse `load_artifact` outputs, checking key sets/list lengths/non-numeric values; for finite float-convertible leaf pairs accumulate `abs(candidate-reference)` and `abs(candidate-reference)/abs(reference)` for nonzero reference. Zero-reference leaves must remain zero. This is prediction-to-prediction preservation, not error against hardware ground truth.

## Separate corrected discrepancy and limits

Metrics S1 is explicitly a correctness restoration: frozen candidate crashes on positive residual traces because complete metadata was conflated with additive layer identity. Six reproduced failures and three scalar main controls established the existing supported behavior; the local emitter now preserves that behavior without tensor-shape fallback. It is not claimed to be equal to the broken frozen path. See `test_report_2026-09-17_p2_residual_metadata.md`.

No new unresolved candidate regression remains after the test-isolation correction. Native CUDA/ROCm/SGLang/GDN kernels, production-profile accuracy and live vLLM parity are not established by these CPU checks. [Final wall-clock results](test_report_2026-09-17_p5_timing.md) record the completed isolated paired campaign; correctness PASS does not predict a speedup.
