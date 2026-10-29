# Fresh position fixture and actual-main boundary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record full source tables, fixture/runner opt-in checks and actual-main prerequisites. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Environment as runtime_points report: frontier-py310 Python3.10.6, thread caps, bytecode/cacheprovider disabled, local /data/ycfeng/tmp only.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m tests.integration.step5_measured_compute_fixture --output-dir /data/ycfeng/tmp/d29-position-actual-main-fixture01 --device h800 --gpus-per-role 16 --position-cyclic > /data/ycfeng/tmp/d29-position-actual-main-fixture01.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m tests.integration.step5_measured_compute_fixture --output-dir /data/ycfeng/tmp/d29-position-actual-main-fixture02 --device h800 --gpus-per-role 16 --position-cyclic > /data/ycfeng/tmp/d29-position-actual-main-fixture02.log 2>&1
```

First generator auxiliary timeout120s occurred before full CSV completion. Contract resolution repeated for each physical row; fixture-only invariant typed contracts/metadata now resolved once per group, no production algorithm/performance change. Second generator auxiliary timeout600s occurred after both full **37,044-row** MoE tables, source/attention/linear/gap/READ generation and attention duration/count fitting; last observed stage was actual MoE preflight. No third whole-generation retry or CPU-overhead campaign. Original actual CLI timeout remains **2400s**; neither run reached it. Actual-main and metrics/ledger qualification therefore remain **NOT RUN/BLOCKED**, not PASS.

Tables retain6112/6128 P prefixes,31/16/17 D tails, all1124 decode contexts6143..7166, both actual source descriptors,16 EP owners, FP8/BF16 groups, graph-synchronized physical rows and explicit positive empty scope durations/counts. All timing/layout values are explicitly cpu_fixture, not native. Persistent small seven-source/252-row producer tests run actual preflight/shared+independent cold/warm caches for both measurement types; **58 PASS/3 SKIP** affected union and **3 PASS** command/producer correction in runtime_points report. Old balanced schema/default command remains intact; position schema is a fresh opt-in. Runner enables existing op tracing and validates full source/collective/count/ledger proofs,6144/1024,D26 P6128+D16,D27 READ payload and D30 readiness.

Dependency boundary: existing unowned prediction_model_registry stores only name/typed identity/precision and overwrites selected FP8/BF16 router contexts. Existing unowned production collective backend/config only imports native-qualified collective_local parameters; it has no explicit CPU plan-cost CLI field/consumption seam. Supplying fake native parameters or a second backend/registry is prohibited. PRICING's delivered CollectivePlanCost can be used by real metrics/unit consumers, but does not itself expose this missing production runner entry. Root coordination is required before complete paired actual-main.

Submodule development use: original CORE checkout was empty. Authorized existing PRICING submodule3dac53c was added as an isolated detached git worktree at CORE's expected submodule path; PRICING checkout/history preserved, parent gitlinkd28fe917 remains deliberately unstaged for root integration. No reset/deletion/push/platform operation.
