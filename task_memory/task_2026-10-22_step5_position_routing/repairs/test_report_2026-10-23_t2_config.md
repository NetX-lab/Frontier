# T2 explicit engine-limit integration

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record parent host/actual-image checks, independent No findings and full-file input rebinding. |

Task: D29-T2-ENGINE-LIMITS. Frozen verification source:662849e99223c98b9ea0d47d34783c3977cb3282.
Native a5e5189/4a7adb7 plus only the constructor/test hunks from mixed PRICING3f54bc6.
Both imported functions have identical ASTs to the author commit; other mixed
hunks were not imported. Evidence is under [t2_config/](t2_config/).

## Required result and observed values

Pass requires both existing config construction paths to receive the frozen
limits before construction and retain them after with_hf_config. The complete
official metadata bytes remain unchanged; its max_position_embeddings is1048576.
Source input: capture03 engine.json, SHA7947f58bf37ab0256756faf1ec0613f4dc92101e94ad4a7f596c50ae4a11655e.

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Module final model/scheduler config |7168/8192/128|7168/8192/128|PASS|
| Graph final model/scheduler config |7168/8192/128|7168/8192/128|PASS|
| Metadata SHA256 |e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63|unchanged|PASS|
| Host affected regression |no failures/errors|177 PASS,15 SKIP|PASS within available environment|
| Native incremental scope |two paths,<=100 gross|85 gross|PASS|
| PRICING selective integration |two paths,<=60 gross|46 gross|PASS|

Native cumulative scope is1350/1582; four native image source files are unchanged.
Host skips are the existing installed-vLLM-dependent cases, not passing runtime
evidence. Independent constructor/source review reports No findings. No
run/allocation is authorized.

## Exact host command

Working directory: `/data/ycfeng/Frontier/.worktrees/verify-step5-t2-limits-01`.
Environment: asyncopd-py312, Python3.12.13. Observed elapsed5.67s.

```bash
timeout 240s env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp \
  STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm \
  /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest \
  tests/unit/test_step5_native_position_cyclic.py tests/unit/test_step5_dispatch_probe.py \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_t2_parent_host01 \
  --junitxml=/data/ycfeng/tmp/d29_t2_parent_host01.xml \
  > /data/ycfeng/tmp/d29_t2_parent_host01.log 2>&1
```

## Actual installed-image check

Python3.12.3 in the published linux/amd64 image. The exact one-off payload is
`t2_config/image_check_payload.txt`, executed as
`/data/ycfeng/tmp/d29_t2_config_image01/check.py`. It imports the integrated helper,
executes the exact two graph-construction assignment AST nodes from T2 with real
installed EngineArgs, and uses the existing conftest CpuPlatform mechanism.
Both final configs expose the expected values and TP8/DP2/EP enabled. EngineArgs
origin/hash matches the published full installed-source receipt.

```bash
docker run --rm --pull=never --network=none --memory=6g --cpus=2 --workdir / \
  --entrypoint python3 -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
  -e TMPDIR=/scratch -e XDG_CACHE_HOME=/scratch/cache -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= -e RANK=0 \
  -e VLLM_MOE_ROUTING_SIMULATION_STRATEGY=position_cyclic -e VLLM_MOE_SKIP_PADDING=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount type=bind,source=/data/ycfeng/Frontier/.worktrees/verify-step5-t2-limits-01,target=/workspace,readonly \
  --mount type=bind,source=/data/ycfeng/tmp/d29_t2_config_image01,target=/scratch \
  hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee \
  -B /scratch/check.py > /data/ycfeng/tmp/d29_t2_config_image01.log 2>&1
```

Result: PASS_CPU_CONFIG_VALUES_ONLY. The command exposes no GPU devices and
executes no graph or expert workloads. CPU config-value verification does not
qualify GPU defaults, graphs, native backend, capacity fit, G3 or E2E.

## Remaining acceptance

Independent narrow review is complete; full script/helper/proposal provenance is
rebound to actual662849e files and its dedicated frozen source worktree. Original
prepared mixed-file identity is retained as history, not final-source evidence.
The16H800/1800s/8GPUh resource proposal remains unapproved. Full D29 consumers,
model/fixture/regression and runtime requirements retain their separate gates.
