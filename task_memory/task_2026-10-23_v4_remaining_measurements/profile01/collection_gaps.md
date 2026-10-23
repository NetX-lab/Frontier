# Profile02 collection-gap diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Reproduced the CUPTI loader failure and tested existing-library alternatives in the locked image; source-backed window/lifecycle/capture design continues separately. |

## Scope and dependencies

The accepted diagnostic audit is `results/primary_acceptance.json`. Next work is `bounded source/environment diagnosis -> CPU repair and independent review -> fresh package/probe -> task-required H800 collection`. Preserve original packages/outputs and all existing calibration predicates. Native/C26 edits and unrelated CUPTI work are outside this scope.

## CUPTI library load: reproduced and diagnosed

The collector calls `ctypes.CDLL('libcupti.so')` in `native_serve_profile.py:237`. The locked2286e853 image, with the generated `/usr/local/cuda-13.0/compat` prefix and inherited library path, reproduces the exact OSError without a GPU. Minimal feedback command is the `command` array in `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_red_vqha8reg/receipt.json`; expected/actual exit1 and `libcupti.so: cannot open shared object file: No such file or directory` match the192 missing mappings.

Ranked hypotheses were missing search path, missing unversioned name, then missing library. Inventory found exactly one library under the relevant CUDA/NVIDIA/Torch roots: `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13`,4,152,208bytes, SHA256 `e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e`. It is outside the default loader path. Receipt `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_inventory_jlfd6ke8/receipt.json`.

One-variable controls in `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_controls_kqctupm4/receipt.json`: adding only the directory still fails; the versioned name plus directory succeeds; the absolute versioned path succeeds. Both successful loads export `cuptiGetStreamIdEx`. Thus the file exists, but both the assumed unversioned name and standalone search path are unsuitable.

The actual caller already imports Torch. A fresh-process Torch2.13.0+cu130 control preserves the original library path and leaves CUDA uninitialized. `/proc/self/maps` then contains the versioned CUPTI library; `ctypes.CDLL('libcupti.so.13')` succeeds without changing the search path, while `ctypes.CDLL(None).cuptiGetStreamIdEx` fails. Receipt `/data/ycfeng/tmp/glm53_dsv4/profile02_cupti_torch_tkgyzwx4/receipt.json`. Reusing Torch's already-loaded versioned library is the smallest supported repair candidate; a new path flag or compatibility symlink is unnecessary for this pinned runtime. Derive the CUDA major from the actual Torch build rather than hardcoding13 in the collector.

These are CPU loader/symbol checks only. Source is still unchanged. A repaired collector must pass its existing CPU failure-isolation tests and an actual H800 stream-ID versus profiler-trace join before claiming stream mapping recovery. Preserve the library hash in new package/runtime evidence.

## Remaining design work

- PrefillDP1 has only2 useful steps in the first64 worker steps; native iteration count includes DP dummy forwards. Its observed first-formal window spans405 worker steps. A larger bound is a candidate, not a guarantee of future retained coverage or an equal per-lane request split.
- `results/shutdown_rca.md` records the marker-budget defect and the unproved cause of the delayed head. Test the actual lifecycle seam before changing it; reuse existing peer/time budgets and preserve auxiliary soft failure. A finite peer budget does not guarantee every delayed head will complete before it expires.
- Captured per-operator ownership and shared cross-DP iteration identity remain UNKNOWN. Inspect whether a separate native startup trace can observe capture without altering native sources; establish graph-node correspondence in a targeted probe before a full serving claim.
- Source-only design owner: `W128-PROFILE03-COLLECTION-DESIGN`, session `ses_e964f6295ffevDNygbwKXH9bu8`. Root requested corrections to the initial routing-ceiling, timeout-guarantee and profiler-reuse claims. No implementation, new package or GPU submission follows from this diagnosis record.
