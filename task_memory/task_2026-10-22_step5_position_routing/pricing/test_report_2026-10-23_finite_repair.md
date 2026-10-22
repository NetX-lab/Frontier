# Finite review repair verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record F3/F6 RED→GREEN; F4/F5 pending. |
| 2026-10-23 | Record F4/F5 actual seams, final JUnit counts and2944 scoped budget. |

TaskID: D29-PRICING-FINITE-REPAIR. Branch feat/step5-position-pricing.
Parent baseline8e61c2d; isolated submodule baseline1a16426.
Workdir `/data/ycfeng/Frontier/.worktrees/step5-position-pricing`.
Environment frontier-py310 Python3.10.6, CPU-only; no new dependency installation.

All parent invocations use:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_cost_export.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f36_green --junitxml=/data/ycfeng/tmp/d29_f36_green.xml
```
GREEN53 PASS,0 FAIL/ERROR/SKIP in10.85s. Log d29_f36_green.log under tmp.
RED same command using d29_f36_red names:49 PASS/4 FAIL (exporter partial/zero
DID NOT RAISE; nan/inf reached affine checks rather than completion seam).
F6 alone before edits (same flags, tests/unit/test_collective_sim_cc_backend.py,
d29_f6_red names):4 PASS/3 FAIL; exact network/context/count ModuleNotFoundError
reproduced root cold-process report. Constructor mock suppressed normal bootstrap.

Submodule cwd `frontier/cc_backend/backends/collective-sim`:
```sh
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-position-pricing:$PWD/python PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/test_grouped_plan_input.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f3_sub_green --junitxml=/data/ycfeng/tmp/d29_f3_sub_green.xml
```
GREEN12 PASS in1.69s; RED same d29_f3_sub_red2 names:7 PASS/5 FAIL, all truncated
partial/zero/nan/inf payloads accepted. Earlier d29_f3_sub_red omitted parent path
required by surrounding package pytest imports:12 setup errors, not bug evidence.
Shared completed_plan_network_ms rejects nonempty partial/zero/nonfinite HTSIM
before prediction or residual fitting. Valid actual HTSIM and TP-local no-network
tests pass; legacy predictor logic untouched. CPU local coefficients remain fixtures.

No GPU/platform/auth/cloud/SSH/Docker/delegation/push, peer edits or import.
Final F4/F5 counts/budget/full patch to be appended after affected validation.

## F4/F5 RED→GREEN

All commands run in the pricing worktree; same Python/env/thread flags above.
Exact F5 RED invocation:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_comm_transport.py -k bound_selected -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f5_red --junitxml=/data/ycfeng/tmp/d29_f5_red.xml
```
JUnit10 tests/4 FAIL/6 PASS/0 SKIP: stale op/group/cuda_graph/replay cases accepted.
GREEN same flags with no -k and d29_f5_green paths:37 PASS in3.42s.

F4 RED:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_comm_transport.py -k bound_selected tests/unit/test_collective_benchmark_helpers.py -k 'bound_selected or observed_policy' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f4_red --junitxml=/data/ycfeng/tmp/d29_f4_red.xml
```
JUnit13 tests/3 FAIL/10 PASS/0 SKIP. PyNccl policy and branch mismatch were accepted;
observation seam missing. GREEN affected four suites d29_f45_green3:261 PASS,
71 SKIP,0 FAIL/ERROR. Later focused exporter parent-policy sealing46 PASS.

## Final affected and independent cold checks

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_contract_reuse.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_package.py tests/unit/test_step5_comm_transport.py tests/unit/test_step5_dispatch_probe.py tests/unit/test_step5_moe_profiling.py tests/unit/test_step5_fixed_probe.py tests/unit/test_execution_time_op_times.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_finite_final04 --junitxml=/data/ycfeng/tmp/d29_finite_final04.xml

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f6_cold_final02 --junitxml=/data/ycfeng/tmp/d29_f6_cold_final02.xml
```
Final affected JUnit545:456 PASS/89 SKIP/0 FAIL/0 ERROR. Cold JUnit9:9 PASS,
0 FAIL/ERROR/SKIP. Logs are corresponding .log paths under /data/ycfeng/tmp.
No pytest loader mock or syspath injection is used by the backend test fixture.

Submodule final command identical to F3 GREEN above with d29_finite_sub_final02
paths:12 PASS/0 FAIL/ERROR/SKIP in1.70s, actual HTSIM completed flows. All new
partial/zero/nan/inf gates fail before local residual fitting. TP-only no-network
formula still passes. Native local coefficients are explicit cpu_fixture values.

Passing expectations: exact selectedop/group/cuda_graph; replay equals config;
all actual participating rank policies present; observed branches and actual
image defaults/version/runtime/registration match eager native probe; missing or
stale policies reject; exporter/importer/sealed parent receipt retains context;
actual preparation observes outside timed eager and inside capture (not replay);
auxiliary diagnostic fault never blocks native operation or replaces its exception.
All observed PASS via actual collector.prepare/DispatchObserver/model/exporter
seams. CPU substituted leaf operations are structural evidence, not GPU policy
proof, calibrated durations, native installed-runtime qualification or T2/G3.

## Failed iterations and verification limits

- d29_f45_green1:2 FAIL/258 PASS/71 SKIP. Fixture included all16 policy keys for
  one-rail scope; fixed keys. Generation-negative fixture dropped entire policy,
  changed only its intended shared identity. No production defaults introduced.
- d29_f4_green2:1 FAIL/11 PASS, branch negative mutated graph record excluded by
  proposed eager-only equality. Removed exclusion: all recorded scopes must have
  matching native probe branch evidence; graph mismatch remains UNVERIFIED.
- Exploratory env -u PYTHONPATH full14 (d29_finite_final01):1 FAIL/453 PASS/89 SKIP,
  old validate_result.py direct-script subprocess cannot import Frontier. Submodule
  no-path command:12 setup errors importing enclosing frontier package. These are
  invocation requirements outside requested F6 defect; no CLI/bootstrap fallback
  added. Independent cold backend9 PASS proves actual constructor bootstraps its
  collective_sim imports without ambient PYTHONPATH. Earlier final02/03 passing
  snapshots superseded by final04 after missing-policy checks were added.
- No published native GPU policy, local durations, network/device freeze, actual
  per-plan counts or parent qualification were collected. Auxiliary errors remain
  soft; no changed native exception semantics or global acceptance claim.

Budget:43/44 owned changed paths,+2849/-95=2944/3956 gross, remaining1012;
five isolated submodule files333 included. F3/F6 parent37ead7f/sub3dac53c; full
pinned d28fe917-to-tip patch retained at pricing/collective_sim.patch,
sha256c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d.
Root owns unstaged gitlink; no peer/import work. AST/JSON/whitespace checks PASS.
F4/F5 source/test/record commit181a534; F3/F6 early commit37ead7f, sub3dac53c.
Summary43 path totals checked against actual owned/submodule diff after commit;
isolated submodule is clean, parent status contains only root-owned gitlink.
