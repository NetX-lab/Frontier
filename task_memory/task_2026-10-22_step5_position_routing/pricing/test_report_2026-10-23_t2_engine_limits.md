# T2 optional engine-limit seam validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record root-decision constructor binding and focused affected checks. |

TaskID: D29-PRICING-IMPLEMENTATION. Root decision D29-T2-ENGINE-LIMITS.
Workdir: /data/ycfeng/Frontier/.worktrees/step5-position-pricing.
Environment: frontier-py310 Python3.10.6; no GPU/native-platform operation.

```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_step5_comm_package.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_t2_limits_delivery
```

PASS187/SKIP71 in2.36s. Log:/data/ycfeng/tmp/d29_t2_limits_delivery.log.
Expected: supplied7168/8192/128 reaches final model/scheduler config after native
text-submodel seam; unknown key excluded; omitted limits retain exact legacy
kwargs; HF max_position_embeddings1048576 unchanged. All PASS in helper tests.
Fixture EngineArgs uses the existing seam to expose final outputs; no actual-image
or graph qualification is inferred. Parent owns actual-image config validation.

Also verified source-context lifetime/materializer-once and original exception
identity. Collector identity faults remain UNVERIFIED soft errors, while native
collective failure propagates the original RuntimeError. Bound worker executes
both phases after binding failure; job evidence is persisted. Prior package-test
failure was a missing master_addr in a two-node fixture, not production behavior.

T2-specific delta:5 production +43 test gross=48, within root's60 and total3956.
Commit also includes previously verified bound-probe schema/source-context tests;
those are existing producer work, not charged a second time to the T2 delta.
