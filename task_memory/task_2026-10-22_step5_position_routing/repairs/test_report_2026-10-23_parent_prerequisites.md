# D29 parent image and regression prerequisites

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify locked image Python sources and retain the pre-D29 unit/fidelity baseline while three writers implement. |

## Scope and expectation

Decision D29-SCOPE-107-10000; source baseline1aee0af. These are root prerequisite
checks, not D29 implementation/runtime acceptance. No GPU/platform operation or
new derived-image build/publication occurred. The existing Docker handbook was
read before Docker commands. Dedicated source worktrees are active independently.

## Docker and locked image — PASS with corrected executable

Commands on the local host:

```bash
docker version --format '{{json .}}'
docker info --format 'ROOT={{.DockerRootDir}} MIRRORS={{json .RegistryConfig.Mirrors}}'
df -h /data/ycfeng/tmp /data/var/lib/docker
docker image ls --digests --no-trunc --format '{{.Repository}} {{.Digest}} {{.ID}} {{.Size}}'
docker pull --platform linux/amd64 a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20 > /data/ycfeng/tmp/d29_base_image_pull01.log 2>&1
docker image inspect a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20 --format 'ID={{.Id}} OS={{.Os}} ARCH={{.Architecture}} SIZE={{.Size}} DIGESTS={{json .RepoDigests}}'
```

Observed Docker client/server29.1.3, root `/data/var/lib/docker`, configured
current/legacy company mirrors and336GiB free on/data. Exact mirror reference
initially returned `No such image`; the identical digest existed as
`vllm/vllm-openai`. Pulling the prescribed immutable mirror reference succeeded
with that same digest, linux/amd64. Inspect size is9194367188 bytes; Docker's
human image listing reports32.9GB, so these two storage counters are not equated.

The proposed `--entrypoint python` returned127:

```text
OCI runtime create failed: ... exec: "python": executable file not found in $PATH
```

Use `--entrypoint python3`. Actual image interpreter is **Python3.12.3**,
installed root `/usr/local/lib/python3.12/dist-packages/vllm`. The post-build
emitter in `../profile_source_design.md` now uses this observed executable;
argument ordering and schema are unchanged.

Executed source command/payload and full inventory are in
`evidence/d29_base_python_sources01.json` (`command` and `payload` keys). The
host invokes Docker with `--rm --pull=never --network=none --memory=4g --cpus=2
--workdir / --entrypoint python3 -i IMAGE -B -`, supplies that payload on stdin,
and compares SHA256 per relative Python path against
`/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`. PASS requires no missing
or mismatched file. Observed **2907 exported and2907 installed Python files,
zero missing/mismatched**; sources match exactly. Receipt SHA256:
`1527cf60d9290132a1b9aad331e816fa218d91ad8284113bf3611d0942d84810`.

An earlier broader inventory including compiled libraries/package metadata
timed out after50s and produced no admitted receipt. Root confirmed no matching
container remained running. Narrowed Python inventory passed within the same
outer60s limit. This does not identify which earlier operation consumed the
timeout, attest compiled libraries, or prove the future derived image.

## Regression baseline reuse — PASS, baseline unit suite not green

Environment: `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6,
`PYTHONPATH=$PWD`, `PYTHONDONTWRITEBYTECODE=1`, OMP/OPENBLAS threads1. The complete
capture and affected-rerun commands are retained in the D30 reports:
`../../task_2026-10-22_step5_finish_sync/test_report_2026-10-22_parent_integration.md`
and `test_report_2026-10-23_canonical_source_assertion.md` in that same task.

Root reparsed `/data/ycfeng/tmp/d30_unit_baseline01.xml`,
`d30_unit_candidate01.xml` and `d30_canonical_count_green01.xml` using
`xml.etree.ElementTree`. It required unique classname/name IDs, proved the65
affected IDs are a subset, replaced only those outcomes, and checked failure
and error ID sets equal the original baseline. Observed latest evidence:
**7647 PASS,267 FAIL,31 ERROR,179 SKIP**. This is reuse, not a fresh full run.

Root reused read-only `load_side` from
`/data/ycfeng/tmp/step5_workflow_fidelity_20261022/audit_evidence.py` on the D30
candidate at `/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate`, revision
713e6a81708e87b86cf91fd89f3b1f954bdcbead. Expected77 unique clean cases,
unchanged definitions/cache/provenance/artifacts and dependency versions all
pass. It rehashed451 artifacts and checked current `_package_versions` against
the manifest. Git diff from713e6a8 to1aee0af is empty for `frontier`, `examples`,
`data`, `tests/e2e/refactor_fidelity`, `pyproject.toml` and `.gitmodules`.

Receipt with exact baseline failures/errors and audit:
`evidence/d29_regression_baseline01.json`, SHA256
`70a15ffd158e1971aa207e5017f2cf50a84e7b4ceb853f93965312d379d042b0`.
The later D29 candidate still needs full failure/error-set comparison, strict77,
both-side artifact checks and its own source provenance. Existing Step CLI and
native accuracy/graph/pressure qualification remain open.
