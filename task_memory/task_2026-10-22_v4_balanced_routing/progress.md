# Balanced prefill progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded initial clean status, root approval, and builder baseline. |
| 2026-10-22 | Completed five-file189-line source commit and CPU handoff; packages unbuilt. |

- completed: inspected clean `v4-h800-prefill` at `9b8494c`; read local and root requirements/plan and native design61-105.
- completed: captured builder pre-edit bytes in `builder_before_routing.txt`:9086bytes, SHA256 `3e2f462748490b4e53c88f506a1db0faf697a9223a462d9152cc1ca1d1de3471`.
- completed: approved shared merge, five-file source wiring and focused CPU checks.
- completed: source commit and handoff artifacts. Package construction, sealing and all hardware operations remain root-only and unperformed.

Rationale before edits: current adapter only checks installed bytes, current inventory omits actual routing, CSV discards any added inventory, and the isolated consumer admits layouts without routing identity. Reuse those existing seams and existing config tool/source hashes; preserve absent-selector legacy behavior and do not relabel historical raw rows.

- completed: fast-forward to shared `410f062`, retaining all committed current dependencies; five-file source implementation drafted within200 lines.
- completed: existing focused `test_dsv4_attention_profiling.py` and `test_dsv4_graph_measured.py`:184PASS in76.78s. CPU-only Python3.10 environment. This settles legacy/capacity behavior, not new balanced identity controls or hardware qualification.
- completed: direct source-backed balanced startup5PASS, model identity9PASS, CSV/consumer16PASS; receipt shape matches native owner's current runners/native_sources keys.9 native files independently bound.
- completed: final affected regression22PASS2.12s; full existing184PASS95.81s before final receipt-shape adjustment. Source AST and diff whitespace PASS.
- completed: old h800_prefill and h800_prefill_capacity128 each527bound files verified unchanged. No fresh package writes.
- completed: audit baseline commit `a71fcd3`; source commit `ac21f88`, exactly5files/182added+7deleted=189routing lines against baseline. `integration_cases.md` gives native-owned tests/interfaces; root alone owns generation, final source revision, strict77, joint review/sealing/image admission/GPU.

## Handoff

Use source commit `ac21f88` together with baseline audit `a71fcd3` (ordinary branch merge preserves both). Do not interpret the unchanged141-line builder baseline adoption as routing additions; raw pre-edit bytes/hash and effective five-file diff are provided. The builder retains historical generation/source constants intentionally; root must select `h800_prefill_balanced` and finalized source identity before execution. Never execute the builder at its old9b8494c source identity: the existing committed-source guard prevents this.

Unfinished root/native tasks: persistent integration negatives in native-owned test file; combined23/900 review/count; fresh generation/source selection; strict77; image/origin/native runtime qualification; GPU balanced counts/backend/timings and retained evidence. No CPU-overhead work is authorized in this phase.
