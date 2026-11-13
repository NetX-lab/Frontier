# Build and review patch identity

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Resolve the apparent patch-hash discrepancy through byte-level path-prefix normalization. |

The derivation manifest correctly records the **build patch** SHA256:
`7a03c3de88329cd6123b00a9519849b4434b79411f7f14500a9bc02ee2d73df7`.
Its retained file is `/data/ycfeng/tmp/d29-native-build-context02/native.patch`.
The **review patch** is
`task_memory/task_2026-10-22_step5_d29_native_balanced/native_source.patch`, SHA256
`66608153204012927e8ef678378d95ca1ef93dab9e2f504a82a5318b9828bbab`.

Replacing `a/native-image/vllm/` with `a/vllm/` and the corresponding `b/` prefix
in the review patch produces bytes identical to the build patch. Eight header
lines account for the104-byte difference;203 non-header lines match. The build
patch uses the installed layout expected by the Docker build. The manifest does
not claim to hash the differently prefixed review copy.

Read-only auditor D29-FINAL-BUDGET-AUDIT additionally applied the build patch in
memory to the pinned base with no fuzz or offset and reproduced all four derived
hashes. Root independently checked normalized byte equality, manifest hash, all
four base pins at `/data/ycfeng/tmp/step5_profiling_20261020/image_src`, and all
four installed derived pins at `/data/ycfeng/tmp/d29_published_sources01/source`.
All pass. No image rebuild or manifest change was necessary; runtime/GPU
qualification is not implied.
