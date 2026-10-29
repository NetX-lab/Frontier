# Test report: B16, the V4 operator consumer on the H200 tables

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: B16 part 1 (Codex `54e2e32d`, partial): table promotion, gap precision fix, the three admission prerequisites that stopped the runtime work, and the lead checks. |

## Part 1: Codex `54e2e32d` (outcome partial)

Assignment: [h3o_operator_costs/codex_assignment_B16_v4_operator_consumer.md](h3o_operator_costs/codex_assignment_B16_v4_operator_consumer.md). Codex report: `h3o_operator_costs/codex-implement-54e2e32d-92b5-49b8-b578-4119b950ab6b/implementation.md`.

### Delivered

| Path | Change |
| --- | --- |
| `h3o_operator_costs/h200_tables/promote_tables.py` | New. Builds the four H200 V4 tables with the existing reducers (`dsv4_attention_main aggregate`, `dsv4_replay_gaps`) on scratch copies of the G1h-02 raw records, and the row unions of the G2h-01 operator + EP16 shared-expert rows and the G3h EP16 + EP32 MoE rows. Writes `manifest.json` (12 inputs with SHA-256, 4 outputs with SHA-256 and row counts, revisions). |
| `h3o_operator_costs/h200_tables/manifest.json` | New provenance manifest. |
| `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/` | New: `attention_graph_replay_kernel_only.csv` (4,945 rows), `kernel_gap.csv` (4), `linear_op_kernel_only.csv` (41), `moe_kernel_only.csv` (33). |
| `frontier/execution_time_predictor/kernel_gap.py` | `read_dsv4_kernel_gap_ms` parses with `float_precision="round_trip"`, so every recorded digit survives; the generic reader is unchanged. |
| `tests/unit/test_dsv4_h200_tables.py` | New, 13 cases: union, revision reader, hashes and counts against the manifest, KERNEL_ONLY only, exact gaps, invalid gap rows. |

Lead correction: the promotion script's scratch directory was the Codex request scratch by default. `--scratch-dir` is now required, because the scratch location belongs to the executor.

### Not delivered, and why

Codex stopped the runtime part (training function, trained pricing, composition, segment gaps, host time, runner coverage error). Three admission guards block a non-dummy V4 predictor, and their files were outside the assignment's writable paths. The lead missed them when writing B16. Observed by Codex and re-checked by the lead:

1. `frontier/operators/families.py:314-318`: `DEEPSEEK_V4_MHC_FAMILY` has `execution_enabled=False` ("source inventory only"). `bind_operator_query("dsv4_mhc_first_pre")` and `validate_typed_operator_contracts(...)` raise `NotImplementedError` for every row of the promoted linear-op table (lead probe: 0 of 41 rows pass).
2. The assignment told the loader to validate every operator contract with `profile.resolve_layer_contract`. That is wrong for the mHC and endpoint operators: the producer writes them with the non-layer contract of `frontier/profiling/linear_op/profiling_plan.py:153` (`layer_kind: null`), and the `deepseek_v4` profile has no layer contract for that family (`get_layer_contract_for_family` finds 0). The shared expert and the MoE rows do resolve through layer contracts (15 and 132 contracts passed).
3. `frontier/attention/families.py:652`: `DEEPSEEK_V4_ATTENTION_FAMILY` has `execution_enabled=False`. `base_execution_time_predictor.py:44-47` admits it only in dummy mode or with a bound step-cost table, so a non-dummy TP8/DP2/EP16 predictor fails at construction.

These go to the follow-up assignment B16b, with the corrected loader rule for non-layer contracts.

### Lead checks (HEAD `1fe9603d` plus the B16 changes)

Environment: Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `TMPDIR=/data/ycfeng/tmp/b16_accept/tmp`, `ulimit -v 31457280`.

| # | Check | Command | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Promotion reproduces after the lead correction | `promote_tables.py --scratch-dir /data/ycfeng/tmp/b16_accept/promotion` | Exit 0; the four tables and `manifest.json` byte-identical to the Codex outputs | Exit 0; `sha256sum` of the four tables equal; `cmp` of the manifest equal | PASS |
| 2 | Focused and import regression | `pytest tests/unit/test_dsv4_h200_tables.py test_dsv4_runner_operators.py test_kernel_gap_pricing.py test_dsv4_replay_rows.py test_dsv4_replay_gaps.py test_dsa_predictor.py -k 'not test_every_collective_launches_one_kernel_by_default'` | All pass | 259 passed, 1 deselected (22.27 s) | PASS |
| 3 | Whitespace | `git diff --check` | Clean | Clean | PASS |
| 4 | mHC registry validation (prerequisite 1) | Probe: `validate_typed_operator_contracts` on each row of the promoted `linear_op_kernel_only.csv` | Documents the blocker | `NotImplementedError: Operator family deepseek_v4_mhc is not enabled for execution` on the first row; 0 of 41 pass | Blocker confirmed |

Codex's own checks (not rerun by the lead): two promotions byte-identical; 4,945 attention identities equal to `/data/ycfeng/tmp/b13r_accept/after/rows.json` over 95 fields; the four gap values equal to G1h-02 to every digit; placement suite 18 passed / 1 failed before and after (the known late-join failure); dummy smoke matrix 12 of 12; cold imports load no torch, vLLM or CUDA.

### Observations reported by Codex (not gated)

- EP16 and EP32 shared-expert medians agree within 1 % at every L (largest gap: `share_expert_up_proj` L = 1, 8.8485 against 8.7685 µs).
- The existing trainer reduces duplicate feature rows by the arithmetic mean (`sklearn_execution_time_predictor.py:191`). The G3h gate rows have 12, 4, 4, 3 and 10 duplicates at L = 1, 2, 3, 4 and 768.
- All seven V4 collective specs report `kernel_count=1`.

## Limits

- No trained V4 forward exists yet, so gates A1, A3-A9 and the composition and gap checks are not run. They move to B16b.
