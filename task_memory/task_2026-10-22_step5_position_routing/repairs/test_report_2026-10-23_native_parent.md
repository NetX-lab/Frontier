# D29 native parent verification and local image

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify native19 delivery, build and inspect local candidate, execute installed-source CPU tests; record four open review repairs. |

## Acceptance state

Decision D29-SCOPE-107-10000; native allocation19/1582. Source under check:
2c4cc9e050290a33430c2f931e9e2a58d9b5594c, worktree
`/data/ycfeng/Frontier/.worktrees/step5-position-native`. Parent tests/source/image
checks PASS below. **Native source acceptance remains open** for review.md N1–N4
(graph capture/replay, keyword capture, bound-forward and backend-object validation).
Original author is repairing these within the allocated paths/quota. No image
publication or GPU/platform operation has occurred.

## Parent source and affected checks — PASS

From the native worktree:

```bash
timeout 300s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_position_cyclic.py tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_reference.py tests/unit/test_step5_native_image.py tests/unit/test_step5_native_forward_census.py tests/unit/test_step5_http_capture.py tests/unit/test_step5_native_source_contract.py tests/integration/step5_environment/test_role_capacity_observation.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_native_parent01 --junitxml=/data/ycfeng/tmp/d29_native_parent01.xml > /data/ycfeng/tmp/d29_native_parent01.log 2>&1
```

Python3.12.13/Torch2.9.1+cu128; **418 PASS/35.45s**, zero failures/skips.
Parent independently compares Git numstat1aee0af→2c4cc9e for the15 repository
paths plus actual unified diffs for the4 native source files. Observed
**19 paths/1017 gross**, including99 native lines, within1582. All native
before hashes equal the actual base-image Python inventory; derived candidate,
retained native_source snapshots and context02 bytes match. Changed source AST,
Dockerfile bytes, patch/manifest hashes and exact9-file sparse context pass.
Receipt `evidence/d29_native_parent_source01.json` retains per-path numbers.

## Build dependency diagnosis and successful build

First `docker build --progress=plain ...` exits125: deprecated legacy builder and
`unknown flag: --progress`. Docker29.1.3 lacksBuildx (`docker info` lists only
trust). `apt-cache policy/show docker-buildx` identifies Ubuntu candidate
0.30.1-0ubuntu1~22.04.1. Initial `apt-get download` fails100 against unavailable
proxy127.0.0.1:51015. Per-command DIRECT access to the company mirror succeeds:

```bash
# Working directory: /data/ycfeng/tmp/d29_buildx_20261023_01
apt-get -o Acquire::http::Proxy=DIRECT -o Acquire::https::Proxy=DIRECT download docker-buildx=0.30.1-0ubuntu1~22.04.1
dpkg-deb -x 'docker-buildx_0.30.1-0ubuntu1~22.04.1_amd64.deb' extracted
sha256sum 'docker-buildx_0.30.1-0ubuntu1~22.04.1_amd64.deb'
/data/ycfeng/tmp/d29_buildx_20261023_01/extracted/usr/libexec/docker/cli-plugins/docker-buildx version
/data/ycfeng/tmp/d29_buildx_20261023_01/extracted/usr/libexec/docker/cli-plugins/docker-buildx build --builder default --load --pull=false --network=none --platform linux/amd64 --progress=plain --iidfile /data/ycfeng/tmp/d29_native_image02.iid -t step5-position-cyclic:20261023-native-2c4cc9e-02 /data/ycfeng/tmp/d29-native-build-context02 > /data/ycfeng/tmp/d29_native_image_build02.log 2>&1
```

Deb SHA256 matches the APT record:
`38389acc20a622d0cae4a1a4e603ec2a355334cc6739baae3a562cb9eaf1ec13`.
The extracted executable reportsBuildx0.30.1 and successfully builds with the
existing default Docker driver. No system package installation was performed.
Dockerfile uses the verifiedpython3 executable; checked-hash compilation passes.

Local image index:
`sha256:10cdd7af56c77b244f70f280133464cce59573b0bcaa58adba6f39342ed1e94d`.
Its linux/amd64 platform manifest is
`sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee`.
These are different identities (index includes an attestation); neither is
reported as a remotely published image. Build log is retained in evidence/.

## Installed local-image source and CPU checks — PASS

Root executes Python from neutralcwd `/`, no source mount/network/GPU, using the
local immutable image reference. Exact invocation and source payload are recorded
in `evidence/d29_derived_local_sources01.json`. It verifies all2907 installed
Python paths/hashes equal the actual base inventory with exactly4 approved
replacements; all40 original RootFS layers are a prefix of47 derived layers;
Docker Config equals the base; installed derivation-manifest SHA256 equals
`561c5a87518cd22b02e8ee47ec396860ad0af84548921874c8f3557e170c32ec`.

Observed interpreter/dependencies: Python3.12.3, Torch2.13.0+cu130,
pytest9.1.1, vllm0.1.dev22912+g8224bb82f. To check the installed source bodies with
that actual interpreter/Torch, mount only the existing standalone test file:

```bash
docker run --rm --pull=never --network=none --memory=4g --cpus=2 --workdir / --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR=/scratch -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= -e STEP5_NATIVE_SOURCE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm --mount type=bind,source=/data/ycfeng/Frontier/.worktrees/step5-position-native/tests/unit/test_step5_native_position_cyclic.py,target=/checks/test_step5_native_position_cyclic.py,readonly --mount type=bind,source=/data/ycfeng/tmp/d29_native_image_cpu01,target=/scratch step5-position-cyclic@sha256:10cdd7af56c77b244f70f280133464cce59573b0bcaa58adba6f39342ed1e94d -B -m pytest /checks/test_step5_native_position_cyclic.py -q -p no:cacheprovider --basetemp=/scratch/temp --junitxml=/scratch/result.xml > /data/ycfeng/tmp/d29_native_image_cpu01.log 2>&1
```

Observed **11 PASS/54.38s**, no skipped/failed cases. This executes actual installed
source bodies via the test's AST loader, with documented dependency fixtures;
it is not complete native-module import, CUDA arithmetic or serving qualification.
XML is retained as `evidence/d29_native_image_cpu01.xml`.

## Open acceptance

Close review N1–N4 and re-review affected changes; integrate accepted source,
publish/verify immutable image identity and emit final53 pins/full-source receipts.
Actual graph/auto backend/all-rank origins, G3 state/transfer, complete CLI and
clean E2E remain open. Additional collection resource/run decisions are made at
readiness. CPU-gap work remains last; baseline/unit/image checks above do not
provide native timing, state or numerical-model qualification.
