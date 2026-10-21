# Step canonical client metrics verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record executable Step/GLM CLI evidence, regressions and native-input limits. |

## Result

**CPU implementation PASS. Native accuracy UNVERIFIED.** The existing normalize and
gate CLI works on both source formats. The checked-in Step matrix still returns
`INSUFFICIENT_EVIDENCE` without real native admission and routing evidence.
Fixture receipt assertions are synthetic test oracles, not actual source/image/
runtime qualification. No measured native or trained Frontier timing was collected.

## Environment and commands

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-client-metrics`.
No conda activation or dependency installation. Python 3.10.6 from
`/data/ycfeng/envs/frontier-py310/bin/python`; Python 3.12.13 from
`/data/ycfeng/envs/asyncopd-py312/bin/python` (aiohttp 3.13.3, pandas 3.0.1).
Simulator fixture exports use the Python 3.10 environment's real `Request` and
`MetricsStore`; they are controlled CPU event sequences, not measured predictions.

```bash
export PYTHONPATH=$PWD
export TMPDIR=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1
export FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python
PY310=/data/ycfeng/envs/frontier-py310/bin/python
PY312=/data/ycfeng/envs/asyncopd-py312/bin/python

$PY312 -m pytest tests/unit/test_step5_client_metrics.py tests/unit/test_canonical_client_metrics.py tests/unit/test_e2e_gate.py tests/unit/test_step5_canonical_native.py -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/step5_client_metrics_verify_20261021_02
# 106 passed in 47.19s

$PY310 -m pytest tests/unit/test_calibration_e2e_metrics_gap.py tests/unit/test_e2e_gate.py tests/unit/test_pd_client_ttft.py tests/unit/test_transfer_metrics_contract.py tests/unit/test_request_preempted_step.py tests/unit/test_vllm_v1_recompute_pricing.py -q -p no:cacheprovider --tb=short
# 148 passed in 6.24s

$PY312 -m pytest tests/unit/test_calibration_pd_harness.py tests/unit/test_native_pd_worker.py -q -p no:cacheprovider --tb=short
# 77 passed in 11.38s
```

These are overlapping sets, not 331 distinct tests. No complete simulator fidelity
matrix was rerun: no simulator/core/public-example code changed. Existing legacy
gate fixture tests and client/internal-TTFT contracts remain passing.

Initial failures and corrections are preserved in `progress.md`: missing import,
source-variable shadowing, an unavailable shared test helper, Python 3.10's absent
aiohttp and a wrong test filename. No test failure was suppressed to qualify data.
The shared HTTP-only helper test was replaced by a direct compressed raw-arrival
rejection test because its helper and GLM replay implementation are not in this
branch. Local-HTTP coverage comes from the unchanged P/D harness suite.

## Actual CLI calls and outputs

The following variables abbreviate the exact retained test paths used by the
manual shell calls. All calls used Python 3.10 and `PYTHONPATH=$PWD`,
`PYTHONDONTWRITEBYTECODE=1`. Existing CLI help was inspected first.

```bash
E=/data/ycfeng/tmp/step5_client_metrics_verify_20261021_02
S=$E/test_step_cli_computes_four_me0
G=$E/test_canonical_run_gate_cli_pr0
CLI=tests/comparison/e2e_gate/run_gate.py
STEP=step5_canonical/fixed6144_1024_req128_qps1

$PY310 $CLI normalize --matrix "$S/matrix.json" --cell "$STEP" --artifacts-root "$S" --vllm-run-dir "$S/canonical/runs/groundtruth_clean/native" --frontier-run "sim=$E/simulator0" --output-dir "$E/cli_step_rows"
$PY310 $CLI gate --matrix "$S/matrix.json" --cell "$STEP" --rows "$E/cli_step_rows" --output "$E/cli_step_reading.json"

$PY310 $CLI normalize --matrix "$G/matrix.json" --cell canonical/fixed --artifacts-root "$G" --vllm-run-dir "$G/canonical/runs/groundtruth_clean/native" --frontier-run "sim=$E/simulator1" --output-dir "$E/cli_glm_rows"
$PY310 $CLI gate --matrix "$G/matrix.json" --cell canonical/fixed --rows "$E/cli_glm_rows" --output "$E/cli_glm_reading.json"

$PY310 $CLI normalize --matrix tests/comparison/e2e_gate/cases/step5_canonical.json --cell "$STEP" --artifacts-root "$PWD" --vllm-run-dir "$S/canonical/runs/groundtruth_clean/native" --frontier-run "sim=$E/simulator0" --output-dir "$E/cli_unverified_step_rows"
$PY310 $CLI gate --matrix tests/comparison/e2e_gate/cases/step5_canonical.json --cell "$STEP" --rows "$E/cli_unverified_step_rows" --output "$E/cli_unverified_step_reading.json"
```

Both synthetic arithmetic-oracle matrices print:

```text
ttft_ms                  vllm 2000.0000 frontier 2000.0000 rel 0.0000 PASS
tpot_ms                  vllm 2.0000 frontier 2.0000 rel 0.0000 PASS
request_e2e_time_ms      vllm 4046.0000 frontier 4046.0000 rel 0.0000 PASS
token_throughput_tps     vllm 7001.3888 frontier 7001.3888 rel 0.0000 PASS
```

The shipped Step matrix prints the same numeric values with all four statuses
`INSUFFICIENT_EVIDENCE`, ending:

```text
step5_canonical/fixed6144_1024_req128_qps1: INSUFFICIENT_EVIDENCE (routing UNSET)
```

Every normalize/gate call exited 0. As in the existing CLI, exit 0 means a reading
was generated; the JSON verdict, not the process exit, is the acceptance result.

## Metric oracle

| Metric | Expected native and Frontier | Observed | Absolute / relative error | Result |
| --- | --- | --- | --- | --- |
| Client TTFT | 2000 ms | 2000 ms | 0 / 0 | PASS |
| TPOT | 2 ms | 2 ms | 1.0236256287e-13 / 5.1181281435e-14 | PASS |
| Request E2E | 4046 ms | 4046 ms | 1.0368239600e-10 / 2.5625901137e-14 | PASS |
| Total THPT | `917504 / 131.046` = 7001.388825297988 token/s | 7001.3888 token/s printed | 4.5474735089e-12 / 6.4951020752e-16 | PASS |

Each native repeat has 128 formal requests, 917504 total tokens and a separate
131.046-second window starting at 1100, 2100 and 3100 monotonic seconds.
Warmups do not enter the metrics. Internal TTFT remains 500 ms while client TTFT
is 2000 ms. Output-only and request throughput remain nongating diagnostics.

An AST comparison against `git show 9f58dbe:tests/comparison/calibration/pd_replay.py`
verified that the reused `canonical_workload` function is identical. A diff from
`6db2250` confirmed no changes to `frontier/`, `data/processed_traces/` or the
R2 observation/verdict implementation. Selected complete CLI JSON outputs are
retained beside this report as `cli_{step,glm,unverified_step}_reading.json`.

Each native repetition is gated independently against the deterministic Frontier
replay. Pooled latency means and `total_tokens/sum(window_s)` THPT remain diagnostics;
they are not the arithmetic mean of per-repeat throughput. The shared bad-round
test proves a pooled passing mean cannot conceal a failed repeat. Four limits
remain 10%; the frozen workload and historical/legacy TTFT are unchanged.

## Negative cases

- Missing/wrong native output counts, prompt hash mismatch, wrong client clock,
  wrong phase origin, missing/duplicate request, missing repeat and unstable summary reject.
- Missing/mismatched run check, same-name foreign origin, runtime package/image/source
  mismatch, missing/changed artifact pin, missing native node and missing actual
  runtime receipt yield unverified admission and `INSUFFICIENT_EVIDENCE`.
- Even with refreshed file hashes, wrong actual node version, duplicate role/lane
  population and config engine-image mismatch do not pass admission.
- Shared normalized missing/duplicate/count/semantic/clock checks and individual
  bad-round rejection remain covered. Missing simulator first-output truth never
  falls back to a later token or prefill completion.

## Remaining real inputs

1. Original clean canonical H200 P32/D32 execution artifacts: all 432 client rows,
   full three-repeat summary, idle/phase boundaries and eight original node receipts.
   One-repeat qualification output is not a formal reference.
2. Byte-identical original staged `config.json` alongside the returned evidence,
   source/hash declaration, and actual observed platform-image, installed-source
   and native qualification receipts.
3. A real passing case-bound `groundtruth-run-check` receipt, from the existing
   admission owner, pinning all inputs and actual runtime receipts. Its required
   Step fields are documented in `tests/comparison/e2e_gate/client_metrics.md`.
   This CPU task implements the consumer, not a fabricated post-run checker result.
4. A genuine measured Frontier replay with request-completion truth and original
   config on the frozen 6144/1024 request schedule; actual trained H200 profiles,
   communication and native transfer prerequisites remain the parent's work.
5. Accepted routing/coverage disposition and a frozen task-local matrix pinning
   those actual inputs; no copied GLM routing claim or fake fixture may qualify it.

No GPU/authentication/external network operation, publication, frozen-stage change,
or scheduler/predictor/profiling/R2 edit occurred. HTTP tests used local loopback only.
