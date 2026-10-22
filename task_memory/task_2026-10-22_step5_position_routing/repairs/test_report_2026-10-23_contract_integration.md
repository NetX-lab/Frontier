# D29 partial contract integration verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Freeze delivered interfaces and close all306 selected cases across their required environments. |

## Scope and source

Root verification worktree:
`/data/ycfeng/Frontier/.worktrees/verify-step5-position-contracts-01`.
Frozen commit: `bd97ac520fd0ba796ecd9d1905fdff352e859680`, combining CORE02f061d,
PRICING7b86614 and NATIVE2c4cc9e without editing source. These are development
integration imports. Native review N1–N4 remain open; selected trainers/schema/
complete model/fixture/main are still being implemented by their sole owners.
This is not full D29 source/runtime or numerical acceptance.

## Commands and results

All host commands run from the frozen worktree with
`PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp`.
Pytest uses `-B -m pytest`, `-q -p no:cacheprovider`, the indicated basetemp and
matching `--junitxml=<prefix>.xml`, stdout/stderr to `<prefix>.log`.

1. Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`, timeout240s,
   `MKL_NUM_THREADS=1`, prefix `/data/ycfeng/tmp/d29_parent_contracts01`:

```text
tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_ep_aggregate_admission.py tests/unit/test_step5_fixed_probe.py tests/unit/test_step5_attention_profiling.py tests/unit/test_step5_request_free_profiling.py tests/unit/test_step5_moe_profiling.py tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_step5_contract_reuse.py -k 'not native_forward and not table_extent and not variable_decode and not cache_mutation'
```

Observed **283 PASS,19 SKIP,4 deselected /18.36s**. Passing conditions are delivered
count/source/cache/constructor, selected operator/plan/cost contracts, legacy
scalar behavior, source identity/parser and existing consumer checks. Skips and
deselections require Torch, absent from this interpreter; they are not PASS.

2. Python3.12.13 `/data/ycfeng/envs/asyncopd-py312/bin/python`, timeout180s,
   `VLLM_TARGET_DEVICE=cpu`, prefix `/data/ycfeng/tmp/d29_parent_timer01`:

```text
tests/unit/test_step5_moe_profiling.py -k 'position_adapter or real_device_timer or masked_expert or oracle_callable or lazy_norm'
```

Observed **18 PASS,11 deselected /9.16s**, two SWIG deprecation warnings.
Actual CPU annotations/parser/singleton restoration, source tensor/router contract
and callable provenance tests pass. CUDA timer endpoints/tracer transport use
their declared dependency fixtures; no native CUDA timing is established.
Testcase-ID union of1+2 is292 PASS, with10 unresolved skips and4 deselections.

3. Same Python3.12 environment, timeout180s,
   prefix `/data/ycfeng/tmp/d29_parent_tensor01`, to address only those14 cases:

```text
tests/unit/test_moe_position_routing.py tests/unit/test_step5_attention_profiling.py tests/unit/test_step5_moe_profiling.py tests/unit/test_step5_request_free_profiling.py -k 'one_source_tensor_interface or native_metadata_preserves or builder_builds_swiglustep or sigmoid_bias_router or wrapper_rejects_layer_groups or kernel_times_bf16 or profile_row_records or native_forward or table_extent or variable_decode or cache_mutation'
```

Observed **11 PASS,3 FAIL,141 deselected /21.74s**. The new source tensor interface,
retained request-page metadata and four previously unavailable request-free
checks pass. The host vLLM lacks `fused_moe.activation` and `router.gate_linear`,
so the two native builder cases and legacy native gate case fail imports. No
production code is changed to accommodate this unrelated host vLLM installation.
Unique passing union is303; three actual native-dependency cases remain open.

## Correct native-image environment

The built local immutable image is
`step5-position-cyclic@sha256:10cdd7af56c77b244f70f280133464cce59573b0bcaa58adba6f39342ed1e94d`.
It contains the pinned native modules and Python3.12.3/Torch2.13.0+cu130.
First targeted container invocation reaches test collection but fails because
the serving image lacks pandas (`/data/ycfeng/tmp/d29_parent_image_legacy01.log`).
The production profiler already supplies pandas/dateutil/pytz/tzdata/yaml/tqdm
through an isolated no-deps extra-site in `run_on_worker.sh:52–81`.

Root reproduces that mechanism using only missing packages, downloaded wheels
under `/data/ycfeng/tmp/d29_profile_test_deps01`, isolated site and no native
library replacement. The host Python3.12 environment lacks pip; image pip
successfully downloads the six declared distributions from the same company
mirror. The isolated installation adds only pandas3.1.0,pytz2026.5,tzdata2026.5;
existing dateutil2.9.0.post0,PyYAML6.0.3,tqdm4.70.1 are retained. Actual native
Torch2.13.0+cu130/vllm0.1.dev22912+g8224bb82f/numpy2.2.6/Triton3.7.1 versions
are unchanged. Package versions and wheel hashes are in `evidence/profile_test_deps.json`.
No system/host environment or published image is modified.

The next collection exposes the image's conflicting `tests` package:
`ModuleNotFoundError: tests.unit`. Explicit pytest importlib mode and loading the
real fixture module first resolves that namespace collision without source edits:

```bash
docker run --rm --pull=never --network=none --memory=6g --cpus=2 --workdir /workspace --entrypoint python3 -e PYTHONPATH=/workspace:/deps/site -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR=/scratch -e XDG_CACHE_HOME=/scratch/cache -e TRITON_CACHE_DIR=/scratch/triton -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= -e VLLM_TARGET_DEVICE=cpu --mount type=bind,source=/data/ycfeng/Frontier/.worktrees/verify-step5-position-contracts-01,target=/workspace,readonly --mount type=bind,source=/data/ycfeng/tmp/d29_profile_test_deps01,target=/deps,readonly --mount type=bind,source=/data/ycfeng/tmp/d29_parent_image_legacy03,target=/scratch step5-position-cyclic@sha256:10cdd7af56c77b244f70f280133464cce59573b0bcaa58adba6f39342ed1e94d -B -m pytest --import-mode=importlib tests/unit/test_step5_fixed_probe.py tests/unit/test_step5_moe_profiling.py -k 'builder_builds_swiglustep or sigmoid_bias_router_runs' -q -p no:cacheprovider --basetemp=/scratch/temp --junitxml=/scratch/result.xml > /data/ycfeng/tmp/d29_parent_image_legacy03.log 2>&1
```

Observed **3 PASS,53 deselected /126.71s** with14 native torch.jit deprecation
warnings. The full selected testcase-ID union is **306 PASS**; each previously
skipped/deselected case is covered by an actual PASS in its required environment.
The builder and router tests retain their documented CPU dependency fixtures;
native GPU kernels/graphs are not exercised. JUnit receipts for all four runs
are retained in evidence/ alongside dependency versions/hashes.

## Remaining acceptance

Foundation review identifies F1 missing actual _select_experts provenance and
F2 mutable sequence inputs to frozen collective dataclasses. Owners are repairing
these within existing allocations;306 PASS is test evidence, not source acceptance
with open findings. Exact37-path diff review is being completed separately.

Receive complete peer implementations, close foundation findings and source review,
then full-unit failure/error comparison, strict77, actual-main and new GPU/native
evidence. Synthetic source/timer fixtures and these CPU checks provide no
collective-model, G3 state, CUDA graph or E2E numerical qualification.
