# Test report: P1h package 03 on 1 H200 (rest of P1h after B4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: package 03 run, lead CPU analysis of the copied download, kernel identity coverage finding. |

## Scope

- Package: `h3o_operator_costs/p1h_bridge/package_03/` (manifest SHA-256 `d3dd2220…a9d3`, 538 files), plan pin `56e06be2`, source revision `230239c83cf44cfb2e3d117cb8424d763b64d314` (B4).
- Purpose ([stage1a_replay_route.md](h3o_operator_costs/stage1a_replay_route.md) §4): confirm on a real engine that the realized KV-cache groups equal the 11-layer prediction and that the tile-schedule planner runs in layers 0-3 only; then the bridge check, the A4 own-gap gate, the gap reducer and the kernel identity check against the native H200 traces.
- Execution: Pi label `p1h-graph-bridge-1027a`, turn 4 (prompt `/data/ycfeng/tmp/ds4_runs/prompts/p1h-graph-bridge-1027a-turn4.md`, exit 0, `[settled] aborted=false`). Analysis: lead, CPU only.

## Run

| Item | Value |
| --- | --- |
| Job | `exp-1028-132243-707076`, tag `v4-h200-p1h-graph-bridge-20261028-03`, StepMind Python RJobBackend, `i-fengyicheng`, step_main, 1 H200, non-preemptible, worker timeout 3600 s |
| Quota at submit | step_main total 512, non-preemptible used 472, free 40 |
| Time | 05:22:44Z → 05:37:23Z, 879 s, about 0.244 GPU-hours; P1h total about 0.575 of the 2 GPU-hour cap |
| Worker runtime | NVIDIA H200 143771 MiB, driver 570.124.06, Python 3.12.3, torch 2.13.0+cu130, vLLM 0.28.0, FlashInfer 0.6.16.post3 |
| Runs | `FULL_AND_PIECEWISE/e0` exit 0, 576.1 s (21 cases); `FULL_AND_PIECEWISE/e1` exit 0, 90.3 s (4 first-tail cases) |
| Transfer | remote listing 38 files, 614,604,339 bytes, equal to the local download; `trace_bytes_archived` 479,054,697 equals the trace sum; 0 bytes left on the worker |
| Download | `/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_03/download/` (unchanged) |
| Analysis copy | `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge/`, 35 files, SHA-256 list `bridge_copy_sha256.txt`, all equal to the download (the report tools write into `output_root`, so they run on the copy) |

Engine record (both engines): backend `FLASHMLA_SPARSE_DSV4`, KV dtype `fp8_ds_mla`, 346,705 GPU blocks, chunk budget 8192, `step_hold_us` 200000, V2 model runner, 3 auxiliary streams, `max_model_len` 7168. Capture: `FULL_AND_PIECEWISE`, 35 capture sizes 1-256, `{graph: 19, breakable_segment: 420}`, 25,985 nodes, 385 eager breaks, owner scope `dsv4_attention_layer` only.

## Environment and commands

- Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vllm); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`; `ulimit -v 31457280`; timeouts per step.
- Driver: `bash /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/run_analysis.sh` (`STEPS` selects steps). The script and the analysis helpers are kept in `h3o_operator_costs/evidence/p1h_bridge/package_03/`.
- Native references: P2-32 `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces` (32 traces), P2-64 `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces` (64 traces).

## Checks

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `realized_groups.py`: plan rebuilt from `plan_args` and `graph_mode_flags('FULL_AND_PIECEWISE', sizes)`, then `require_realized_kv_cache_groups(plan, engine)` per raw file | the recorded groups equal the 11-layer prediction on both engines | e0 and e1 groups [14, 6, 5, 10, 4], equal; planned layers 0-3 plan, 4-10 reuse | PASS |
| 2 | `planner_layers.py`: `smxx::decode::get_mla_metadata_kernel` per (case, step) per layer | exactly one in each of layers 0-3 in decode-path steps, none elsewhere | e0: decode FULL 64, decode PIECEWISE 16, dummy FULL 40, dummy NONE 8, dummy PIECEWISE (1 row) 8 steps as expected; dummy PIECEWISE (2, 4, 8 rows) 24 and prefill NONE 3 steps without a planner (prefill path, see "Dummy layouts"); e1: first_tail PIECEWISE 12 as expected; no unexpected step | PASS |
| 3 | `dsv4_graph_bridge_check.py --report` | PASS | `PASS: 2 runs, FULL=104, PIECEWISE=60`, no reason | PASS |
| 4 | `dsv4_composition_check.py` (A4 own-gap gate) | every group composes from its own measured signed gaps | `PASS: 125 groups, {'pass': 125}` | PASS |
| 5 | host-ahead hold (`gap_diagnostics.json`, `hold_by_step_mode_and_case`) | every recorded step host-ahead, no launch after the hold | 175 steps, minimum slack 151.8 ms, 0 attention-owned and 0 unowned post-hold launches | PASS |
| 6 | `dsv4_replay_gaps` reducer | writes `kernel_gap.csv`, no reason | written, `reasons []`; values below | PASS |
| 7 | `dsv4_kernel_identity_check.py` against P2-32 (margin cases `e0-decode-b24-p6144`, `e0-dummy-full16`) | PASS | `FAIL: 175 steps, {'PASS': 46, 'FAIL': 129}` | FAIL (tool, see below) |
| 8 | same against P2-64 | PASS | `FAIL: 175 steps, {'PASS': 54, 'FAIL': 121}` | FAIL (tool, see below) |
| 9 | `shape_matched.py` (lead diagnostic): identity against native forwards of the same class, mode and padded rows | no FAIL where a native reference exists | P2-32: 46 PASS, 129 NO_REFERENCE, 0 FAIL; P2-64: 54 PASS, 110 NO_REFERENCE, 11 FAIL (real-row mismatches, below) | PASS where referenced |

Gap reducer values (µs, all host-ahead samples): `cuda_graph` 0.45752 (51,008 gaps), `cuda_graph_piecewise` 0.60760 (26,520), `eager` 3.54581 (5,709), `eager_preparation` 1.14984 (14,687). The B2 bridge data gave 0.4029, 0.9618 and 1.9345 µs and no `eager_preparation` value. Under remedy (b) these class values price only non-attention operators and pre- and post-forward work; the attention scopes carry their own measured gaps, which check 4 verifies. The differences from B2 are an observation, not a gate: package 03 has the host-ahead hold, 11 layers and the dummy and first-tail phases.

## Kernel identity: the misses come from the shape, not from the kernels

The identity tool (`match_kernels`) selects native candidates by step class and mode only and takes the closest multiset. When the native traces have no forward of the pilot's shape, it compares against another shape and reports the shape-dependent kernels as missing.

Native forward shapes in the retained traces (class, mode: padded rows with forward counts):

| Class, mode | P2-32 | P2-64 |
| --- | --- | --- |
| decode FULL | 1 (68), 2 (48) | 1 (162), 16 (72) |
| decode PIECEWISE | 1 | 1, 16 |
| dummy FULL | 1 (257) | 1 (655) |
| dummy NONE | 1 | 1 |
| dummy PIECEWISE | 1 | 1 |
| first_tail_alone PIECEWISE | 1 | 1, 16 |
| first_tail_with_decodes PIECEWISE | 2 | 16 |
| prefill NONE | 6144 (26) | 6144 (35) |

Matching by (class, mode, padded rows), with padded rows read from the grid of `_fused_q_kv_rmsnorm_kernel`:

- Every pilot step that has a native forward of its shape passes: prefill 6144; decode FULL 1 and 2 (P2-32), 1 and 16 (P2-64, including `e0-decode-b8-p6144-full16`); dummy FULL 1, PIECEWISE 1 and NONE 1; first-tail alone 1.
- The 11 P2-64 failures are `e0-decode-b8-p6144-piecewise9` (8 steps) and `e1-first_tail-b10-p6144` (3 steps), both PIECEWISE at 16 padded rows. Only kernels sized by real rows are missing: in PIECEWISE the eager-break attention kernels (`_compute_global_topk_indices_and_lens_kernel`, `flash_fwd_mla_combine_kernel`, `cooperative_topk_cs8`/`cs4`) run on the real rows, not the padded rows. Pilot grids are 8 and 11 rows; the native forwards at 16 padded rows have 9 and 13 real rows (`native_piecewise16.py`). With real rows in the key both cases have no reference either.
- No reference at all: decode FULL 4, 8, 24 (and 2 on P2-64); decode PIECEWISE 4 and (16 padded, 8 real); dummy FULL 2, 4, 8, 16; dummy PIECEWISE 2, 4, 8; first-tail with decodes 4, 8 and (16 padded, 11 real).

Observation: no kernel identity difference at any shape with a native reference. Inference: the raw FAIL is a coverage limit of the retained native traces, not the "identity differs" stop condition of design §5. The shapes without a reference stay unqualified by this check; their qualification comes from the stage 4 end-to-end replays.

Lead decision (D-NO-INTERRUPT): correct the identity tool to key native candidates by (class, mode, padded rows, real rows of the eager breaks), report `NO_REFERENCE` for steps without a candidate, give PASS when every referenced step passes, and list the covered and uncovered shapes. Codex assignment B5 ([h3o_operator_costs/codex_assignment_B5_identity_shape_keys.md](h3o_operator_costs/codex_assignment_B5_identity_shape_keys.md), request `a1c6212e`).

## Dummy layouts (input to entry 4)

The attention metadata of the first dummy step of each case (`dummy_scopes.py`, `dummy_planners.py`):

- FULL dummy N: `num_reqs` N, `max_query_len` 1, `max_seq_len` 7168 (= `max_model_len`, the re-staged capture metadata), decode path with the planner.
- PIECEWISE dummy N > 1: `num_reqs` 1, `max_query_len` N, `max_seq_len` N; sparse prefill path (`sm90::fwd::sparse_attn_fwd_kernel`, `DequantGatherK`, `topKPerRowPrefill`, `cp_gather_indexer_k_quant_cache_kernel`), no decode planner.
- PIECEWISE dummy 1 and the NONE dummy beside a 6144-token peer: 1 request, 1 token, decode path.

This agrees with design §3.7 item 4 (FULL pads requests to the descriptor, PIECEWISE and NONE keep one request, FULL re-stages its capture metadata). Entry 4 must reproduce the consequence: a PIECEWISE dummy wider than 1 row is priced on the prefill attention path, and a FULL dummy on the decode path at the capture sequence length.

## Pi's questions (lead answers)

1. `plans_tile_schedule` is not in the engine record; it is a plan field. Keep the plan as its only source: check 1 requires the realized groups to equal the plan, and check 2 verifies the planner positions from the trace.
2. Three idle periods before the first recorded step (5.8 s in `e0-prefill-p6144`, 9.2 s in `e0-decode-b2-p6144` with two TileLang compilations, 2.7 s in `e1-first_tail-b3-p6144` with a new cutlass kernel variant) lie outside every recorded step and do not affect the per-step values. No change; the trace span is not comparable across cases.
3. Nothing more to collect.

## Limits

- The identity check qualifies only the shapes the native traces contain (list above).
- The class gap values are diagnostics under remedy (b); per-scope gap columns land with entries 1, 3 and 10.
- One H200, reduced 11-layer engine, TP 1 engine with the target TP 8 head layout; full-model counts come from the per-role expansion.

## Evidence

- `h3o_operator_costs/evidence/p1h_bridge/package_03/`: analysis scripts, logs of checks 1-9, `bridge_report.json`, `gap_diagnostics.json`, `kernel_gap.csv`, `shape_p32.json`, `shape_p64.json`, `bridge_copy_sha256.txt`.
- Not copied (size): `composition_report.json` (738 KB) and `identity_p32.json` / `identity_p64.json` (1.5 / 1.3 MB) under `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/`.
