# Startup census progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Begin the source-bound CPU regression. |
| 2026-10-22 | Complete lifecycle-bound exclusion and verify 181 affected checks plus 19 final census checks. |

- completed: inspect capture01 judgment and locked `v1/worker/gpu_worker.py:940–989`; native compilation calls `kernel_warmup(self)` and `warmup_kernels(..., self.execute_model, ...)`.
- completed: RED02 reproduces two contaminated start/end pairs through actual observation wrappers and the native `StepTimingCollector.record_batch` body. Both normal and exceptional compilation fail the zero-serving-forward assertion; source callback check passes. The first RED assertion also included a hook-install record; RED02 narrows it to actual forward events.
- completed: wrap the existing native `compile_or_warm_up_model` lifetime. Startup execution skips worker business capture and runtime census; `finally` clears the marker even on native failure. Native calls/results/exceptions are unchanged, and no request-prefix filter is introduced.
- completed: 181 affected census/observer/canonical/stage checks PASS in 2.67s; final explicit warmup-name preservation in the source fixture passes all 19 census checks in 0.56s. `git diff --check` PASS.
- completed: evidence and limitations recorded in `test_report_2026-10-22_startup_census.md` and `summary.md`; scoped commit follows.

Ranked causes were lifecycle boundary omission, thread-local leakage and incorrect `is_profile` classification. The fresh-observer reproduction starts with no worker context, excluding leakage. Startup enters the normal `execute_model` callback rather than the profiling path. Binding its actual native compilation lifetime removes the false serving records while preserving later normal and dummy calls.

Original census FAIL and G3 not-qualified verdict remain unchanged.
