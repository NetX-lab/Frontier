# Parent check of immutable plan-key diagnosis

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Verify retained evidence and the existing immutable-plan test seam; leave implementation allocation pending snapshot accounting. |

## Result

PASS_DIAGNOSTIC_ONLY. Root independently verified24 artifact/input hashes and
six source hashes against code2f2d9339. Owner records7d1cdc06 are archived on
the primary branch as a3525856. The208-event wave timings remain owner evidence;
root did not rerun that wave in this check.

The existing `test_step5_moe_pricing.py` plan fixture supplies the real immutable
plan and tensors. An in-process control retains only the canonical key string.
Two calls to the existing interface invoke `to_dict()` twice on baseline and
once with retention. Both return full-key SHA256
`c72f1a0a6bd3d585ad32a90b7e4d1ebe16af134ddadd6f0848939c82cf739a28`.
Mutating returned context/buffer/tensor dictionaries does not change the plan,
its next public dictionary, or the key. Six valid `dataclasses.replace()` cases
(sizes, context, buffers, phase, graph modality, tensor geometry) each obtain a
new key identical to the original uncached encoder's result. The existing
caller-owned-sequence isolation test passes under both implementations.

Source inspection found no production/test `vars(plan)`, `plan.__dict__`,
CollectivePlan subclass, or post-construction mutation consumer. The matched
production `vars()` calls concern tensors, which this proposal does not change.
Real typed tensors, normalized tuples and JSON strings establish immutability;
the frozen decorator alone is not the argument. `to_dict()` must stay fresh.

## Command and environment

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview:/data/ycfeng/Frontier/.worktrees/step5-preview/tests/unit TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_immutable_plan_parent01/verify.py > /data/ycfeng/tmp/d29_immutable_plan_parent01/run.log 2>&1
```

Exit0. Existing frontier-py310 executable, the same executable recorded as
Python3.10.6 by the owner; no conda invocation. Exact checks and observed values
are in `/data/ycfeng/tmp/d29_immutable_plan_parent01/{verify.py,result.json,run.log}`.
The script modifies the imported class only within its own Python process.
Production, tests and retained inputs are unchanged.

## Next action and limits

H1 is closed: existing preparation already retains complete plans. H2 has a
plausible narrow repair: approximately6 gross/4 retained source lines, and an
extension of the existing identity/isolation test rather than a separate35–55
line test suite. A provisional20-gross step (source8/test12) would need actual
retained allowance of roughly14 lines; these are estimates, not an allocation.
Wait for the in-progress snapshot's final PRICING consumption before reserving
remaining allowance. Do not reduce its active caps or raise owner totals.

The owner is separately attributing H3 from the existing key-control profiles;
trace encoding remains with its existing owner. Snapshot implementation,
combined review, new-source wave/negative/native checks, full CLI and final
regressions are not established by this diagnostic check.
