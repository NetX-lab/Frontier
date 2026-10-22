# Native19 integration verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record owned CPU, D30, source/image-context and static graph results. |

## Environment and exact commands
All commands ran in /data/ycfeng/Frontier/.worktrees/step5-position-native.
Prefix: `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Python /data/ycfeng/envs/asyncopd-py312/bin/python,3.12.13; Torch2.9.1+cu128.
No CUDA/GPU operation was run. Baseline251 PASS is in the interface test report.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_image.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_http_capture.py tests/unit/test_step5_native_source_contract.py tests/integration/step5_environment/test_role_capacity_observation.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-final04 > /data/ycfeng/tmp/d29-native-final04.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_finish_sync.py tests/unit/test_step5_native_forward_census.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-d30-final02 > /data/ycfeng/tmp/d29-native-d30-final02.log 2>&1
```

Actual:418 PASS/39.78s and48 PASS/67.40s. The latter used Python3.10.6.
Expected: retained old controls, actual client adapter propagation including
independent direct-D, HTTP fresh/reused/EOF semantics, conditional model/context/
factory/simulator wrong-root/spec/hash/missing module rejection, soft native
return/exception semantics, D30steps1/16 andengine/core source/counter coverage.
Actual derived staged observation retains13 canonical source hashes including
vllm.v1.engine.core. Static source pin agreement is not imported live evidence.

Earlier attempts: wrong capacity-suite path caused exit4/no tests;
asyncopd lacksplotly for Frontier scheduler imports(exit2). Existingfrontier-py310
was used instead, no dependency edits. One D30 test expects candidate parent root,
not candidate/vllm; final command uses its real convention, all48 PASS.
Logs are retained at /data/ycfeng/tmp/d29-native-final02.log,
d29-native-final03.log andd29-native-d30-final.log.

## Source/context/static checks
Builder directly executed `prepare(base, candidate, listing, output)` using:
- base /data/ycfeng/tmp/step5_profiling_20261020/image_src;
- candidate /data/ycfeng/tmp/d29-native-derived-20261023-candidate01;
- listing /data/ycfeng/tmp/d29-native-base-build-pins.txt;
- initial output /data/ycfeng/tmp/d29-native-build-context01;
- final output /data/ycfeng/tmp/d29-native-build-context02 (rerun after builder checks).
Result PASS: exact4 changed source files; all other candidate/base file bytes
agree; pin+manifest+patch hashes retained inbuild_context_receipt.json. Base
listing53 is50 existing sealed pins plus3 local base-export hashes, not actual
installed derived evidence. Container/image operations NOT_RUN.

Scoped follow-up `python -B -m pytest tests/unit/test_step5_native_image.py -q
-p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-image-final`:
1 PASS/0.15s after additional foreign-file/base-hash rejection controls.
Final builder iteration rejects all source symlinks before walking inventories;
same focused test1 PASS/0.15s atd29-native-image-final02 andactual context02 PASS.

AST parsed all owned Python files and4 actual native files. T2 static checks
require real native eager-break decorator, wrapper counters>0/>=2, stable tensor
addresses and capture counts, actual opaque-MoE seam, bound positions/mask,
FP8/BF16 existing layer loader and collective/expert observers. Result PASS
source-only; no runtime claim. Dockerfile's Python payload AST PASS,4 native
COPY paths only, checked-hash bytecode compilation, no forced ENV/pip/backend rebuild.
Receipt:t2_static_receipt.json. `git diff --check`:PASS.

Manifest byte checks: capture02all56 entries PASS;
capture03all57 entries PASS and original manifest
cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d.
capture01not present at inspected parent root; no historical capture writes.
Scope receipt counts actual19 paths and1017 gross diff includingnative99, ceiling1582.

## Limits
GPU NONE/PW/FULL, actual CUDA selected-score arithmetic, auto FP8/BF16 dispatch,
all-rank imported model/source evidence, derived image/dependencies/library bytes,
full serving/G3 state and transfer/op-gap/E2E remain UNVERIFIED. Mocked host-runtime
provenance controls above are not native execution or native qualification.
