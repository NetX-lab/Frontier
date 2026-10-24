# Profile02 collection-gap diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Linked accepted GPU stream/restart evidence, current DP/lifecycle work and authoritative graph-payload sample; corrected the callback budget to final3/3. |
| 2026-10-24 | Accepted the34-line CPU loader repair at0d745706; recovered prior graph-probe failures and kept capture mapping as a separate targeted question. |
| 2026-10-24 | Reproduced the CUPTI loader failure and tested existing-library alternatives in the locked image; source-backed window/lifecycle/capture design continues separately. |

## Scope and dependencies

The accepted diagnostic audit is `results/primary_acceptance.json`. Next work is `bounded source/environment diagnosis -> CPU repair and independent review -> fresh package/probe -> task-required H800 collection`. Preserve original packages/outputs and all existing calibration predicates. Native/C26 edits and unrelated CUPTI work are outside this scope.

## CUPTI library load: reproduced and diagnosed

The collector calls `ctypes.CDLL('libcupti.so')` in `native_serve_profile.py:237`. The locked2286e853 image, with the generated `/usr/local/cuda-13.0/compat` prefix and inherited library path, reproduces the exact OSError without a GPU. Minimal feedback command is the `command` array in `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_red_vqha8reg/receipt.json`; expected/actual exit1 and `libcupti.so: cannot open shared object file: No such file or directory` match the192 missing mappings.

Ranked hypotheses were missing search path, missing unversioned name, then missing library. Inventory found exactly one library under the relevant CUDA/NVIDIA/Torch roots: `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13`,4,152,208bytes, SHA256 `e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e`. It is outside the default loader path. Receipt `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_inventory_jlfd6ke8/receipt.json`.

One-variable controls in `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_controls_kqctupm4/receipt.json`: adding only the directory still fails; the versioned name plus directory succeeds; the absolute versioned path succeeds. Both successful loads export `cuptiGetStreamIdEx`. Thus the file exists, but both the assumed unversioned name and standalone search path are unsuitable.

The actual caller already imports Torch. A fresh-process Torch2.13.0+cu130 control preserves the original library path and leaves CUDA uninitialized. `/proc/self/maps` then contains the versioned CUPTI library; `ctypes.CDLL('libcupti.so.13')` succeeds without changing the search path, while `ctypes.CDLL(None).cuptiGetStreamIdEx` fails. Receipt `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_torch_tkgyzwx4/receipt.json`. Reusing Torch's already-loaded versioned library is the smallest supported repair candidate; a new path flag or compatibility symlink is unnecessary for this pinned runtime. Derive the CUDA major from the actual Torch build rather than hardcoding13 in the collector.

Source repair0d745706 is accepted at34/80gross lines with CPU RED/GREEN, independent review and exact-image controls. [Targeted GPU acceptance](stream_probe/results/primary_acceptance.json) now proves actual H800 default/auxiliary stream IDs7/13 and two native profiler cycles, with8eager/16priming joins and28unique kernels. Profile02's historical192null mappings remain unchanged. The probe captured/replayed successfully but retained no captured per-op node identity; graph ownership stays UNKNOWN.

## Prior graph evidence recovered before probe implementation

The earlier H200 probe demonstrated replay graph/node fields but source/exec full IDs differ. Attempt2 exited139 without a report/trace; final attempt3 (`exp-1021-120354-830178`) also exited139, localized to an in-callback `cuptiGetGraphNodeId` query. The final3/3 budget is exhausted; the earlier2/3 wording described only attempt2. Parent attempt3 execution/RCA records retain the unconfirmed cause.

Root's exact-image Torch source capture confirms arithmetic exec-ID recombination in the inspected tools pipeline; upgrading for that pipeline would not supply an independent witness. The completed ordinary-profiler probe proves restart but not captured ownership. [Official sample findings](graph_payload_sources.md) support the payload-indirection correction with existing event/DAG labels. YC approved the fresh bounded callback probe and similar diagnostic work under D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY. No numeric remapping, image upgrade or paused GLM resumption is selected.

## Remaining design work

- PrefillDP1 has only2 useful steps in the first64 worker steps; native iteration count includes DP dummy forwards. Its observed first-formal window spans405 worker steps. A larger bound is a candidate, not a guarantee of future retained coverage or an equal per-lane request split.
- `results/shutdown_rca.md` records the marker-budget defect and the unproved cause of the delayed head. Test the actual lifecycle seam before changing it; reuse existing peer/time budgets and preserve auxiliary soft failure. A finite peer budget does not guarantee every delayed head will complete before it expires.
- Captured per-operator ownership and actual cross-DP pairing remain UNKNOWN. DP source/review and restored-fixture verification are recorded in `../progress.md`; source integration and a fresh native collection remain necessary. The startup-profiler question was tested and did not close captured ownership.
- Source-only design owner: `W128-PROFILE03-COLLECTION-DESIGN`, session `ses_e964f6295ffevDNygbwKXH9bu8`. The routing-ceiling, timeout-guarantee and mandatory-second-profiler claims were withdrawn. Existing records/bytes caps must be checked against actual drops; no guaranteed future volume or per-lane split is accepted. A pre-stop marker cannot replace the existing post-stop evidence without changing its meaning, so it is not selected.
- Current coverage/lifecycle source allocation is two existing non-core files/260gross lines, detailed in [profile03_design.md](profile03_design.md). The earlier stream-probe owner/source allocation is completed and accepted, not active preparation. Native/C26/consumer changes remain outside this slice.
