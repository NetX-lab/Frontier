# Parent input-validation comparison

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Independently run the corrected minimal family-only control on3000 retained rows. |

Source:e41abc073500d61f224eda4fca1123e74e40544d. Cwd:
/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05.
Python:/data/ycfeng/envs/frontier-py310/bin/python,3.10.6.
Script SHA256:be8973b4778694555d5626e9e6b4f570f65cd1a61fcf9e8656fdd414592203ea.

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B -u \
  /data/ycfeng/tmp/d29_init_validation_scope/family_only_loop.py \
  --output /data/ycfeng/tmp/d29_init_validation_scope/parent-family01 --rows 3000 \
  > /data/ycfeng/tmp/d29_init_validation_scope/parent-family01.log 2>&1
```

The script uses RLIMIT_AS30GiB and the real retained CSV head, production validator
and Step model config. Expected:one config derivation and exact payload equality.
Observed:exit0;one derivation;2.837165373s baseline versus0.274640171s hoisted;
equal contracts and equal SHA256
f460359008814440c80a40fd2e97b557e7bcbb802c72a4076c6ebe3c8a2cdd3f. PASS for this
bounded equivalence/efficiency comparison. Result:parent-family01/result.json.

The original probe double-counted the once-only call in its report. Its author
corrected that diagnostic, preserved old outputs and documented the earlier
unauthorized full scans/75s outer bound. This parent run uses the corrected
counter and original60s limit. It is not a full loader, complete initialization,
whole-CLI, or native qualification result; the actual loader regression belongs
to D29-INIT-FAMILY-RESOLUTION.
