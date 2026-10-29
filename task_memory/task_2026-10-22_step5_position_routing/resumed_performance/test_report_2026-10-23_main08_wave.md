# Parent main08 wave verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Reproduce layer61 and verify all208 operators of the complete adjacent layer60 wave. |

## Environment and commands

Frozen checkout: `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`,
source `79f1a3fd`, submodule `3dac53c`. Interpreter:
`/data/ycfeng/envs/frontier-py310/bin/python` (Python3.10.6).
All commands below use `timeout 60s`, `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, and
`FRONTIER_TMP_ROOT=/data/ycfeng/tmp`. Scripts set RLIMIT_AS30GiB; the owner's
failed systemd scope is not represented as a resident-memory cgroup guarantee.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/wave_loop.py --output /data/ycfeng/tmp/d29_main08_residual_repro/parent-red01
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_main08_residual_repro/compare_actual_wave.py --replay parent-red01
/data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_main08_residual_repro/parent_complete_wave.py
```

Logs: `parent-red01.log`, `parent-red01-comparison.log`, `parent-layer60.log`
under `/data/ycfeng/tmp/d29_main08_residual_repro/`.

## Expected and observed results

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Original layer61 replay | Stable count-based RED and unchanged owner payload | Exit1;3.304/2.653s pricing;28004/28003 decodes;80 reads each;b2593e payload | PASS diagnostic reproduction |
| Original artifact comparison | All captured pairs agree | Exit0;165/208 observed pairs;0 mismatches | PASS within sample |
| Complete layer60 replay | Real complete wave, stable RED | Internal exit1;3.292/2.837s;28004/28003 decodes;80 reads each | PASS diagnostic reproduction |
| Complete artifact comparison | All208 actual operators agree | Wrapper exit0;208/208 pairs;0 mismatches | PASS |

The complete-wave wrapper selects layer60 from the same bounded8MiB tail. It
changes exactly one source-sample path literal in the original loop and one
layer literal in the comparator, in memory. Their files remain unchanged.
The resulting `parent_binding.json` records both script hashes and substitutions.
Compared fields are duration, operator proof, native context/source and complete
collective plan/cost. Diagnostic batch/request IDs are not claimed identical.

Layer60 and61 have identical durations, native contexts and collective plans/costs,
but144 operator proofs differ in `global_layer_id`. Their full payload hashes
correctly differ; reuse must continue deriving the current layer's proof.
Hashes and exact results are in `main08_wave_verification.json`.

## Decision and limitations

The measured single-variable plan-key improvement alone is insufficient evidence
for the1024-token/2400-second workflow. Three disjoint investigations now cover
prepared row/plan selection, common collective preparation lifetime, and trace
serialization. Only the trace's single-path replacement receives a20-gross source
allocation under `D29-REPAIR-PATH-AUTHORITY`; other proposals await evidence.
No full CLI is restarted from an isolated point result. Initialization remains
partially attributed, and workflow/native/G3/E2E acceptance remains open.

One summary-print command failed with `zsh: command not found: python`; repeating
it with the absolute interpreter succeeded. It did not affect either replay.
