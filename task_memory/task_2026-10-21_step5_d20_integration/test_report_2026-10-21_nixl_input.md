# NIXL input and combined foundation validation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record loader/lifecycle checks, fixture correction and combined160 PASS. |

## Environment

Repository: `/data/ycfeng/Frontier/.worktrees/step5-d20-integration`. Python `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6. Commands use `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp`.

## Commands and results

All commands use `python -m pytest`, `-q -p no:cacheprovider --tb=short`.

1. `tests/unit/test_nixl_read_cost.py tests/unit/test_step5_measured_pd.py tests/unit/test_pd_transfer_prompt_recompute.py` →80 PASS,2 FAIL,101.92s. Raw `/data/ycfeng/tmp/d20_nixl_read_cost01.log`.
2. `tests/unit/test_nixl_read_cost.py -k file_backed` →2 PASS,25 deselected,20.94s. Raw `/data/ycfeng/tmp/d20_nixl_read_cost02.log`.
3. `tests/unit/test_nixl_read_cost.py tests/unit/test_step5_measured_pd.py tests/unit/test_pd_transfer_prompt_recompute.py tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_replay.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_model_bundle.py` →160 PASS,118.35s. Raw `/data/ycfeng/tmp/d20_integration_foundation01.log`.

## Expected and observed

- PASS: explicit JSON input validates full static cache geometry, devices, TP, model profile, exact prompt prefix, per-rank bytes and complete readiness duration. Missing/duplicate records, unknown scope, invalid values and cross-profile lookup reject.
- PASS: file-backed6144-input/1024-output controlled lifecycle in offline and online modes. Transfer uses the supplied3.5ms fixture, first D batch has16 tokens at cached6128, then1023 one-token steps at contexts6144..7166. Request finishes at7168 total tokens and1024 emitted outputs; source/target allocation maps empty.
- PASS: attention fit/cache/replay and existing PD foundations together with the new input,160 checks total. These remain CPU contract evidence.

## Initial failure and cause

The first long-output test stopped on the33rd scheduling call after32 completed outputs. Suspected causes were absent idle DP passes, missing output callbacks, or a KV-page boundary. The existing scheduler's `_end_iteration` waits for all DP lanes every32 iterations; the short inherited9-output fixture had never reached that boundary. Advancing idle lanes via their normal `on_schedule`/`on_dummy_forward_end` callbacks made both full1024-output cases pass. No scheduler/product behavior changed for this failure. All negative loader checks had already passed.

## Limits

Baseline prerequisite completed: the exact cleanb65609b worktree ran the existing77-case matrix, all return codes0. All451 artifacts match their recorded sizes/SHA256, verified against paths relative to the baseline results directory (`baseline_integrity02.json`). An initial inline integrity check mistakenly used shellcwd for those relative paths and reported missing files; only that path resolution was corrected. No baseline artifacts were modified or regenerated. The final candidate comparison remains pending.

The file-backed execution provider is a test adapter, not yet the normal sklearn lazy-delegate path. Full FFN/manager integration, whole `frontier.main`, integrated77-case fidelity, immutable full bundle and actual canonical/native/profile admission are still pending. No GPU work or timing correction occurred in this step.
