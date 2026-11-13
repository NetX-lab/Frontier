# D29 native image publication and source receipts

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Publish reviewed native image, verify remote identity and emit actual installed-source receipts. |

## Scope and environment

Decision D29-SCOPE-107-10000 retains internal image-publication authority.
Native source/CPU phase6827507 is integrated at9495cfb after independent N1–N4
closure and parent485 PASS. The four native image-source files are unchanged
from the locally tested build. Build/11 installed-source CPU checks are recorded
in `test_report_2026-10-23_native_parent.md`; this report covers publication.

Host Docker29.1.3; extracted Ubuntu Buildx0.30.1. Published image Python3.12.3,
installed vllm root `/usr/local/lib/python3.12/dist-packages/vllm`. Source export
uses Python standard-library functions from neutral cwd `/`, no source mounts,
no network and no GPU. Complete actual source identity is distinct from runtime
import/backend/graph or numerical qualification.

## Publication and verification

Target:
`hub.stepfun-inc.com/vllm-openai/vllm-openai:step5-position-cyclic-20261023-666081532040-01`.

The first manifest inspection failed because inherited proxy127.0.0.1:51015
refused connections. Per-command removal of proxy environment variables made
the same read succeed with `artifact ... not found`, proving the target absent.
No host/daemon proxy settings changed. Local target was also absent before tagging.

Exact action sequence (`SRC` and `TGT` expanded below):

```bash
SRC=step5-position-cyclic@sha256:10cdd7af56c77b244f70f280133464cce59573b0bcaa58adba6f39342ed1e94d
TGT=hub.stepfun-inc.com/vllm-openai/vllm-openai:step5-position-cyclic-20261023-666081532040-01
docker tag "$SRC" "$TGT"
docker image inspect "$SRC" "$TGT" --format '{{.Id}} {{json .RepoDigests}}'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy docker push "$TGT"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy docker manifest inspect --verbose "$TGT"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy docker pull --platform linux/amd64 hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee
```

Raw logs: `/data/ycfeng/tmp/d29_image_target_preflight0{1,2}.{json,err}`,
`d29_image_push01.log`, `d29_image_published_manifest01.{json,err}` and
`d29_image_published_pull01.log` under the same temporary root.

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Local retag | Same image ID | Both10cdd7af… | PASS |
| Push and remote OCI index | Equal locally tested immutable index | sha256:10cdd7af56c77b244f70f280133464cce59573b0bcaa58adba6f39342ed1e94d | PASS |
| Remote Linux/amd64 manifest | Exactly one platform; digest equals SHA256 of raw bytes | sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee | PASS |
| Published platform pull | Resolve exact published digest | Exit0; cached image contents reused | PASS |
| Actual full installed Python receipt | Equal tested local source inventory | 2907 equal entries | PASS |
| Serving pin subset | Recompute from actual installed hashes | 53 unique entries | PASS |

Buildx's first raw-index read rejected a mistyped digest locally (`invalid
checksum digest length`); the corrected request derives the digest from the
recorded local inventory. The remote raw index SHA256 equals the local descriptor.
Both returned raw manifest hashes are independently recalculated, and the
Linux/amd64 descriptor equals the index entry. The second descriptor is an
unknown-platform attestation, not another GPU runtime image.

Future bindings use the **platform** reference ending in f037d958… rather than
confusing it with the index10cdd7af…. Complete references, expanded inspection and
emitter commands, hashes and UTC timestamps are in
`../image_publication/publication_receipt.json`.

## Source artifacts

`../image_publication/native_installed_sources.json` is emitted using the exact
reviewed container payload from profile_source_design.md, against the published
platform. Actual package-root and all2907 hashes match the tested local image;
receipt SHA256 is `b99bc33b7e17f1256e970a93be5698544a2b52c11ab0683029b2c75e18f87a09`.
`installed_source_sha256.txt` recomputes the53 serving entries from this receipt;
it is not the earlier base-build pin file. `derivation_manifest.json` retains
the exact installed derivation identity. Commands/payloads and source hashes are
generated evidence, not a second maintained source registry.

The separate actual installed Python-tree export passes all2907 hash comparisons
and includes the12 backend schema modules required by N4. The archive contains
9,585,732 bytes; actual source root is `/data/ycfeng/tmp/d29_published_sources01/source`.
Its checksum, expanded command and timing are retained in
`../image_publication/installed_export_receipt.json`. This source tree is exported
from the published image, rather than relabeled from the candidate checkout.

## Remaining acceptance

Actual per-process imports/callables, effective settings, launched GPU image,
CUDA/NONE/PW/FULL, backend/MRv2 graphs, G3/state/pressure and E2E remain unverified.
Serving53 is only a subset of the full source inventory. No GPU job, new-stage
runtime qualification, numerical tolerance or extra resource approval is implied.
