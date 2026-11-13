# Position source wrapper validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify real source wrapper with explicit CPU leaf seams before authorized PRICING import. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Thread-capped, no bytecode/cacheprovider; Python3.10.6 frontier-py310 and Python3.12.13 asyncopd-py312.

`env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-core-wrapper-legacy01 tests/unit/test_step5_moe_profiling.py tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_step5_measured_ffn.py > /data/ycfeng/tmp/d29-core-wrapper-legacy01.log 2>&1`: **35 PASS,20 SKIP**. Legacy rows/typed contracts and selected pairs preserved; native/torch and not-yet-imported schema cases skipped.

New persistent test `tests/integration/test_step5_measured_main.py:test_position_source_wrapper_uses_actual_local_widths_and_selected_prepare`. Normal py312 collection fails before execution: missing fasteners; sklearn also absent. No dependency install or production fallback. Exact failed command retained in `/data/ycfeng/tmp/d29-core-position-source-wrapper01.log`.

AST-isolated only that existing function under py312: parse file with ast, remove parametrize decorator, compile actual FunctionDef unchanged, inject pytest/json/BaseModelConfig, invoke for `(None,False)`, `(float8_e4m3fn,False)`, `(float8_e4m3fn,True)` with fresh MonkeyPatch.context. Same env and `python -B`; logs `/data/ycfeng/tmp/d29-core-position-source-wrapper02.log` and03.log. First isolated attempt failed because test-only all-padding source used a zero-length interval; descriptor correctly rejected it. Fixed fixture to empty query/context arrays, no production relaxation. **03:3 PASS**.

Observed actual wrapper routes source16/live0/masked16 while gathering17/IDs17x8 outside timing; source-specific hot/legacy gating retained. Only reached nondeferred FP8 invokes source prepare during measured scope; BF16/deferred omit target entirely. No gathered expert or legacy shuffle callable is timed. Typed targets exactly equal selected timed targets; JSON source/context/provenance retained.

CPU tensors and actual MoEWrapper are used; NativeRoutedExperts and timing leaves are narrow CPU spies. This is plumbing, not H17 source/native/GPU qualification. Full gathered schema and production CSV/preflight still await authorized PRICING import at this checkpoint. Root delivered import authority8e61c2d after this slice.
