# Profile02 collection-gap diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Accepted the34-line CPU loader repair at0d745706; recovered prior graph-probe failures and kept capture mapping as a separate targeted question. |
| 2026-10-24 | Reproduced the CUPTI loader failure and tested existing-library alternatives in the locked image; source-backed window/lifecycle/capture design continues separately. |

## Scope and dependencies

The accepted diagnostic audit is `results/primary_acceptance.json`. Next work is `bounded source/environment diagnosis -> CPU repair and independent review -> fresh package/probe -> task-required H800 collection`. Preserve original packages/outputs and all existing calibration predicates. Native/C26 edits and unrelated CUPTI work are outside this scope.

## CUPTI library load: reproduced and diagnosed

The collector calls `ctypes.CDLL('libcupti.so')` in `native_serve_profile.py:237`. The locked2286e853 image, with the generated `/usr/local/cuda-13.0/compat` prefix and inherited library path, reproduces the exact OSError without a GPU. Minimal feedback command is the `command` array in `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_red_vqha8reg/receipt.json`; expected/actual exit1 and `libcupti.so: cannot open shared object file: No such file or directory` match the192 missing mappings.

Ranked hypotheses were missing search path, missing unversioned name, then missing library. Inventory found exactly one library under the relevant CUDA/NVIDIA/Torch roots: `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13`,4,152,208bytes, SHA256 `e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e`. It is outside the default loader path. Receipt `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_inventory_jlfd6ke8/receipt.json`.

One-variable controls in `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_controls_kqctupm4/receipt.json`: adding only the directory still fails; the versioned name plus directory succeeds; the absolute versioned path succeeds. Both successful loads export `cuptiGetStreamIdEx`. Thus the file exists, but both the assumed unversioned name and standalone search path are unsuitable.

The actual caller already imports Torch. A fresh-process Torch2.13.0+cu130 control preserves the original library path and leaves CUDA uninitialized. `/proc/self/maps` then contains the versioned CUPTI library; `ctypes.CDLL('libcupti.so.13')` succeeds without changing the search path, while `ctypes.CDLL(None).cuptiGetStreamIdEx` fails. Receipt `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_torch_tkgyzwx4/receipt.json`. Reusing Torch's already-loaded versioned library is the smallest supported repair candidate; a new path flag or compatibility symlink is unnecessary for this pinned runtime. Derive the CUDA major from the actual Torch build rather than hardcoding13 in the collector.

Source repair0d745706 is now accepted as a CPU component:34/80gross lines, actual `stream()` regression RED→GREEN, four separate suites33/19/28/45PASS, independent review No findings and locked-image reduced-method control. The old method still fails on `libcupti.so`; the repaired method reaches the expected no-CUDA-context boundary without initializing CUDA. [Acceptance](stream_identity/primary_acceptance.json) retains exact hashes/commands and the explicitly bound77-case matrix. An actual H800 stream-ID versus profiler-trace join remains required. Preserve the library hash in the new package/runtime evidence.

## Prior graph evidence recovered before probe implementation

The earlier H200 probe already demonstrated the target replay's kernel graph/node fields, but direct source-node to exec-node matching failed: full uint64 IDs differ between the two namespaces. See parent `test_report_2026-10-21_h200_graph_probe.md`. A subsequent custom clone-callback probe (`exp-1021-051603-575827`, `h200_graph_probe_attempt2_20261021/`) exited139 before producing a report/trace. Its failure and two-of-three loop history are unchanged.

The pinned CUDA13.0 compatibility layer cannot provide the tools-ID API required by Torch `get_graph_data`; cuda-bindings13.3.1 alone does not solve that requirement. The new DSV4 probe therefore neither repeats this API path nor uses custom callbacks, numeric ID conversion, image upgrades or the paused GLM investigation. It asks whether an ordinary native profiler active through capture and a subsequent replay cycle exposes direct trace correspondence. Missing correspondence stays UNKNOWN. Native wrapper counter resets support sequential start/stop by source; actual GPU restart behavior still needs the probe.

## Remaining design work

- PrefillDP1 has only2 useful steps in the first64 worker steps; native iteration count includes DP dummy forwards. Its observed first-formal window spans405 worker steps. A larger bound is a candidate, not a guarantee of future retained coverage or an equal per-lane request split.
- `results/shutdown_rca.md` records the marker-budget defect and the unproved cause of the delayed head. Test the actual lifecycle seam before changing it; reuse existing peer/time budgets and preserve auxiliary soft failure. A finite peer budget does not guarantee every delayed head will complete before it expires.
- Captured per-operator ownership and shared cross-DP iteration identity remain UNKNOWN. Inspect whether a separate native startup trace can observe capture without altering native sources; establish graph-node correspondence in a targeted probe before a full serving claim.
- Source-only design owner: `W128-PROFILE03-COLLECTION-DESIGN`, session `ses_e964f6295ffevDNygbwKXH9bu8`. The routing-ceiling, timeout-guarantee and mandatory-second-profiler claims were withdrawn. Existing records/bytes caps must be checked against actual drops; no guaranteed future volume or per-lane split is accepted. A pre-stop marker cannot replace the existing post-stop evidence without changing its meaning, so it is not selected.
- Targeted probe source owner: `W128-NATIVE-STREAM-PROBE`, session `ses_e963ba5bfffe1FTjmWbn6H8CF6`; one new non-core file/250lines. Source and CPU preparation are active; root retains package/image/SDK/resource checks and any1H800 submission. Native/C26/consumer changes remain outside this slice.
