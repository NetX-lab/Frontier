# Scoped pricing delivery validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record final owned delivery, regression correction and exact budget. |

TaskID: D29-PRICING-IMPLEMENTATION. Baseline:1aee0af.
Working directory: `/data/ycfeng/Frontier/.worktrees/step5-position-pricing`.
Python: frontier-py3103.10.6; asyncopd-py3123.12.13. All tests are CPU-only.
Thread caps OMP/OPENBLAS/MKL=1; no bytecode/cacheprovider; TMPDIR=/data/ycfeng/tmp.

## Commands and observed results

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_contract_reuse.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_package.py tests/unit/test_step5_comm_transport.py tests/unit/test_step5_dispatch_probe.py tests/unit/test_step5_moe_profiling.py tests/unit/test_step5_fixed_probe.py tests/unit/test_execution_time_op_times.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_delivery_final3
```

PASS438/SKIP89 in29.71s. Log `/data/ycfeng/tmp/d29_delivery_final3.log`.
Expectation: complete selected producer keys pass; missing/duplicate/extra keys,
rank/repeat/local queue defects, identity defects and unqualified imports fail;
legacy registry/defaults and phase accounting remain valid. Observed PASS.

Submodule cwd `frontier/cc_backend/backends/collective-sim`:
```sh
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-position-pricing:$PWD/python PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/test_grouped_plan_input.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_sub_delivery_final
```
PASS7 in1.70s. Log `/data/ycfeng/tmp/d29_sub_delivery_final.log`.
Actual HTSIM binary executed: exact element stripes/bytes, rail-local barriers,
all eight joint rails, completed flow count and N+max(local)+remainder validated.
Fixture local/remainder inputs are cpu_fixture, not GPU measurements.

Original published native methods executed under AST isolation:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp FRONTIER_STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm /data/ycfeng/envs/asyncopd-py312/bin/python - <<'PY'
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
path=Path('tests/unit/test_step5_moe_pricing.py')
fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='test_native_inclusive_scope_calls_real_source_methods_with_prepare_excluded')
namespace={'pytest':pytest,'SimpleNamespace':SimpleNamespace}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
namespace[fn.name]()
print('PASS: published original native prepare/modular/finalizer/E8M0 OR shape/coordinator/capacity/source-layout methods; actual adapter verifier seam; CPU only')
PY
```
PASS. Log `/data/ycfeng/tmp/d29_published_native_delivery.log`.
Actual public adapter used original native prepare/modular/finalize/selector/
coordinator/capacity functions with CPU tensors and narrowly substituted leaf
operations. This is neither complete installed runtime nor GPU numerical proof.
Native physical-capacity function: M17/K8/local22/alignment128 ->2944;
M1/K8/local22/alignment128 ->1024. Actual selected alignment remains runtime input.

Budget/parse check: `git diff --numstat 1aee0af`, plus isolated submodule
`git diff --numstat d28fe917370c2d55230c3ea226e5660785b44aca`, grouped by retained
scope.json ownership (pricing-only plus H04/H17); ast.parse/json.loads all changed
owned Python/JSON files. PASS43 changed paths within44; +2555/-71=2626 gross
within3956, remaining1330. Submodule312 included. Peer imports, task records and
root-owned gitlink excluded; H11 removed only once. Exact paths in summary.md.
`git diff --check` passes parent and submodule.

## Failures diagnosed and affected reruns

- Final command initially named nonexistent test_step5_profiling.py: no tests ran
  (exit4). Corrected to actual test_step5_moe_profiling.py/test_step5_fixed_probe.py.
- Additional legacy test exposed missing ExecutionTime.moe_input_quantize_time:
  AttributeError in metrics_store's family projection. Added the owned accessor
  following existing MoE getters; explicit1ms and legacy0ms assertions pass. The
  final14-suite run contains the originally failing legacy test.
- Exploratory request_free_profiling suite had four ModuleNotFoundError:torch
  failures under py310. No peer test edited or dependencies installed; these are
  environment limits, not passing or skipped qualification. They are outside the
  final affected command. Direct native method checks use torch in py312; py312
  full pytest imports lack fasteners, so no full-runtime claim.
- Earlier bound fixture leaked manifest environment; monkeypatch tracks/restores
  environment keys. Combined final suites pass. Collector now gathers advisory
  provenance failures across ranks rather than trusting rank0 alone.

## Verification limits and prerequisites

No GPU/auth/platform/cloud/SSH/Docker/push/stop/deletion/delegation performed.
CORE's newer b39fc51/e255ede exact-query/layer/main interfaces are not imported
here; H04 tests use labeled CPU exact proofs, so integrated estimator/main is a
root/CORE acceptance dependency. Actual native local-copy/add observations,
frozen network/NIC/device inputs, effective protocol/TP parameters, actual
per-plan counts and independent parent receipt are absent. Production importer
rejects absent/unqualified/CPU evidence; no full-plan duration lookup exists.
Source/GPU/backend/graph/T2/G3 and four E2E<=10% remain parent runtime acceptance.
