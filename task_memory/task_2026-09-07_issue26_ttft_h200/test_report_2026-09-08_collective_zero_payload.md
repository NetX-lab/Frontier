## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Captured the real collective runner regression before the scoped repair. |

# Zero-payload Collective Runner Verification

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
Python: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, Python3.13.13, conda environment `dev-vidur-v03-hopper-e2e`. The check calls the actual Scenario serializer, runner subprocess, built htsim binary, and nvlink_analytic model. This CPU boundary check is not the H200 full-case result.

```bash
TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_collective_sim_zero_payload.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/issue26-zero-payload-red-01
```

Full regression script: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/unit/test_collective_sim_zero_payload.py`.
Raw log: `/data/ycfeng/tmp/issue26-h200-network/zero-payload-red-01.log`.

## Criteria

- Explicit zero payload must reach the real runner and preserve the existing nvlink_analytic step latency; expected EP8 all-to-all latency is0.0035ms with zero transferred bytes.
- Positive32768-byte payload must retain the existing prediction formula and zero effective inter-server time.
- Missing payload remains an input error; negative payload must be rejected as invalid rather than accepted as an empty transfer.
- Explicit CLI zero must override a positive value in the JSON specification.

## Before repair: FAIL

Observed `3 failed, 2 passed in1.17s`:
1. Scenario retains `tensor_bytes=0`, but runner returns2 and `missing required fields: [tensor_bytes]`.
2. Negative payload is incorrectly accepted with exit0.
3. CLI zero is overwritten by the32768-byte specification.
Positive-payload prediction and missing-field rejection passed. No complete Frontier TTFT mean or simulation error can be computed from generation02 because it stopped in the first decode EP wave.

## After repair

PASS:26 tests in27.14s, including5 real runner/predictor cases and21 communication registry/policy checks. Exact command:

```bash
TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_collective_sim_zero_payload.py tests/unit/test_comm_operator_families.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/issue26-zero-payload-green-01
```

Observed zero-payload prediction0.0035ms, no transferred bytes and0 effective network time. Positive32768bytes matches the existing formula; missing/negative inputs and CLIzero precedence meet the criteria. Two unused private conversion helpers were removed after whole-submodule call-site inspection; the remaining module split analysis is recorded separately.

Commits:Frontier69764e50;collective-sim e564935. H200 full-case completion and TTFT comparison remain pending fresh generation03. No empirical communication calibration is claimed by this boundary check.
