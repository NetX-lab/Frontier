D-G1H-WRITER-EXTREMES CPU supplement plan

This directory plans the difference between the amended shared G1h grid and the
G1h-02 raw measurements. It contains 432 added windows and 18 fixed repeats,
using 450 cases and 450 measured steps. Every case has zero warm-up steps and
one measured step. The lead built `package_01` at `a689ed05` in
`.worktrees/g1h-supplement-build`; H200 job `exp-1030-021110-248027` measured
all 450 steps, and the 2,160 non-repeat rows are promoted into the committed
attention table (see `../../test_report_2026-10-29_b22_g1h_writer_supplement.md`).

| Category | Added windows | Repeats | Cases / measured steps | Engine |
| --- | ---: | ---: | ---: | ---: |
| FULL | 144 | 6 | 150 | 0 |
| PIECEWISE | 144 | 6 | 150 | 0 |
| First tail | 144 | 6 | 150 | 1 |

The original grid had 1,041 windows: FULL 351, PIECEWISE 351, first tail 318,
multi-tail 7, dummy 13, prefill 1. The amended grid has 1,473: FULL 495,
PIECEWISE 495, first tail 462, with the other categories unchanged. Of 1,776
candidate windows, 303 duplicates are removed in grid order. The original
1,074 candidates removed 33 duplicates. Added windows by width are W8 72,
W16 216, W24 144; own counts at most four gain none.

For each of FULL, PIECEWISE and first tail, repeat W1/own1, W2/own2,
W4/own4, W8/own8, W16/own16 and W24/own24 at the high profile, with the native
(k4, 0) mix after the tail floor. Their mixes are (1,0), (2,0), (4,0), (7,0),
(7,0), (7,0) for the two tail-free categories and (1,1), (2,1), (4,1), (7,1),
(7,1), (7,1) for first tail. `coverage_grid.json` records the exact windows,
case IDs and purposes. `placement_examples.json` lists every request's cached
and query tokens, writer indices at both ratios, padding and assignment for
one case of each category and width. The existing FCFS prediction reaches all
450 requested windows without changing construction or native engine limits.

Both engines retain the original 8,192-token budget; engine 1 retains the
6,143-token chunk threshold. Engine 2 is unnecessary because multi-tail
windows are already covered by G1h-02.

| Engine | Cases | Estimated engine steps | GPU seconds | GPU hours | Envelope hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 300 | 3,628 | 1,289.843642 | 0.358290 | 0.391318 |
| 1 | 150 | 1,875 | 900.478642 | 0.250133 | 0.263800 |
| Total | 450 | 5,503 | 2,190.322284 | 0.608423 | 0.655117 |

These estimates reuse `../g1h/build_cases.py` without copying its timing model
or limits. The seven-hour G1h-G4h allowance has approximately 3.93 hours used
and 3.07 hours left. Even the envelope leaves approximately 2.415 hours. The
estimate is below 1.5 hours, so no reduction is proposed. Profiler overhead,
autotuning and GPU kernel identity remain to be measured.

From the working root, use the supplied Python 3.10 environment and a writable
scratch directory. Generation updates only this directory's planning files:

```bash
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
B22_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python
B22_H3O=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs
# Set B22_SCRATCH to a writable private directory.
timeout 180s "$B22_PYTHON" "$B22_H3O/g1h_supplement/build_cases.py" \
  --source-revision a689ed05f462eb269b9dffe640da151e6614a5ef --evidence-dir "$B22_SCRATCH/plan_evidence"
timeout 180s "$B22_PYTHON" tests/analysis/v4_attention_coverage.py check \
  --package-plan "$B22_H3O/g1h/package_plan.json" \
  --raw /data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw \
  --predict-plan "$B22_H3O/g1h_supplement/package_plan.json" --require-complete-grid \
  --c1-json "$B22_H3O/evidence/coverage/p2_32.json" "$B22_H3O/evidence/coverage/p2_64.json" \
            "$B22_H3O/evidence/coverage/p0.json" --output "$B22_SCRATCH/union_coverage.json"
```

`package_plan.json` follows G1h's package fields, selects `package_01`, and uses
name `G1H_SUPPLEMENT`, job tag `v4-h200-g1h-supplement-20261029-01` and output
`/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_s01`.
Its source revision is that of G1h-02 (see below). `prepare_package.py` loads G1h's existing P1h snapshot
builder and binds its directory here. Package creation belongs to the lead.

After measuring, run the ordinary attention reducer on the supplement raw
records **including repeats** into scratch, then compare its table against
G1h-02 with `tests/analysis/v4_attention_package_difference.py`. Arguments are
`--base-table`, `--supplement-table`, `--base-grid ../g1h/coverage_grid.json`,
`--supplement-grid coverage_grid.json`, and `--output`. The output reports each
repeat window, variant, planner role, scope and time/gap target. Relative
median difference is `(supplement - base) / base`; a zero baseline retains the
absolute difference and reports a null relative difference.

`../h200_tables/promote_tables.py --supplement-raw-dir RAW` merges the two
attention reductions and excludes repeat cases using the shared repeat-window
geometry. It reduces G4h gaps from G1h-02 alone. Use `--scratch-dir`,
`--output-root` and `--manifest-path` pointing into scratch for validation;
`--package-dir` can identify the accepted G1h-02 package stored in the original
worktree. No-supplement promotion reproduces all four current tables byte for
byte. A synthetic merge retained ten added rows, excluded ten repeat rows,
and kept the kernel-gap, linear and MoE tables identical.

The attention loader requires one source revision and one runtime fingerprint
per table (`validate_measured_identity`). The package is therefore built from
the G1h-02 source revision `a689ed05f462eb269b9dffe640da151e6614a5ef`, recorded
in `package_plan.json`, so both packages run the same profiling code and carry
one identity. The P1h builder requires the snapshot paths of its working tree
to equal that revision, so the lead builds the package from a worktree at
`a689ed05` with this directory copied in. A synthetic merge with this plan
loads through `load_dsv4_attention_dataframe` with one fingerprint. If the
measured runtime versions differ from G1h-02, the loader rejects the merged
table and the difference is reported before any change.

Replay checks use `tests/analysis/v4_attention_replay_coverage.py` with `--table`,
`--supplement-plan`, read-only **copies** supplied via `--census`, and `--output`.
The tool computes the ranges by calling the trainer itself, projects
admitted requests with the measured scope inventory, excludes repeats, and
lists every outside request with all features and model/column reasons. These
bounds are predictions until the GPU measurements qualify them. Live census
copies only establish closure for records present at the snapshot; completed
replays must be checked again by the lead.
