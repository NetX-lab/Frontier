# Stream probe implementation plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Added bounded review-repair RED/GREEN phase; original image checks passed, final bindings remain root-owned. |
| 2026-10-24 | Defined the targeted implementation, verification and root-owned dependencies. |

Dependency: `source/native/parser inspection -> probe implementation -> CPU checks/source-budget/records -> root review and actual locked-image import check -> root package/1H800 execution -> retained-data acceptance`.

1. Inspect actual collector stream-ID API, native profiler lifecycle/config and existing parser primitives. Historical H200 source/exec namespace mismatch and CUDA13.1-only graph API are constraints, not viable premises.
2. Implement one at-most250-line probe: standard native trace handler, exactly two cycles on one wrapper, default+aux eager targets, separately retained four priming kernels per stream/cycle, capture of two operations under cycle1, same graph replay in both cycles. Direct exact trace graph/node identity transfer only.
3. Retain original traces, raw CPU/runtime/kernel rows, snapshots, native logs, errors and source/library/runtime/GPU identities. Missing/capture/restart failures do not erase eager evidence; script exit never substitutes for mapping status.
4. Verify AST/CLI/import without CUDA using the available local environment, document exact commands and limitations, re-read parent requirements, and deliver source hash/budget/task ID. Stop writing after delivery.

Root-owned prerequisites: accepted collector repair; pinned native package/source/image/driver configuration; launch output-dir and existing collector limits; platform submission; all GPU correctness/attribution checks. Probe import must not enable native serving inventory/routing hooks; root should leave their environment variables absent.

## Review repair follow-up

Dependency: `retain original249-line bytes -> RED real-function CPU fault controls -> two minimal P2 repairs within275gross lines -> identical GREEN controls/AST/help -> owner records/delivery -> root re-review and final source/package binding`.

1. Retain original SHA256 `237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7` and existing CPU errors. Temporary controls use actual imported `graph_correspondence`/`joined` and unmodified extracted `analyze`/`attempt`/final eager-verdict AST bodies, with synthetic trace fields only.
2. Require successful capture and both replay returns, which are set only after their native operations complete (replay includes synchronization). No evidence-only graph PASS after a failing completion.
3. Store eager status immediately after eager joins, before optional analysis. Isolate priming and graph analysis through the existing `attempt` mechanism; preserve exact exceptions and raw traces. Add no API, configuration or tracer mechanism.
4. Keep valid graph/eager positive controls and failed capture/replay/malformed graph negatives. Record exact commands, raw RED/GREEN receipts, hashes and source line budget. Stop writing on delivery.

The root collector is accepted at `0d745706`/`6ef05dd8`. Root original-image receipt is already PASS and remains immutable. No image or GPU rerun is part of this repair owner scope.
