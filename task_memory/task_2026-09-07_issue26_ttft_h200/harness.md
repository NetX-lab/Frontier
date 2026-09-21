## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Recorded approved single-call AR bypass and standard-scale gate. |
| 2026-09-09 | Added the mandatory warmup/formal-data gate and fixed the standard H200 batch test chain for all subsequent tests and analyses. |
| 2026-09-10 | Added the authoritative H200/H800 cluster parameter matrix and restored the H800 normal/scalar_sync/skip post-MoE AR diagnostic contract. |
| 2026-09-12 | Raised the active warmup contract to ten complete drained replays and bound standard workers through `ISSUE26_WARMUPS`. |
| 2026-09-13 | Added clean PPLX boundary-only and Kineto profiling gates; formal artifacts require 10 warmup drains plus 100 requests (1100 rows). |
| 2026-09-13 | Recorded two completed-workload NVTX capture failures; NVTX report absence cannot be treated as a timing result. |

# H200 Batch Harness Contract

## Cluster Parameter Matrix

Every run report MUST record the cluster row used for that run. The two rows
are independent and their timing values MUST NOT be combined.

| Cluster | Quota group | GPU selector | GPUs/CPU/memory | `num_gpu_blocks_override` | Purpose |
| --- | --- | --- | --- | ---: | --- |
| H200 | `step_main` | `h200` | 8 / 64 / 400 GiB | `310809` | H200 historical formal baseline |
| H800 | `codesign` | `h800` | 8 / 64 / 400 GiB | `176000` | H800 reference-scaled formal run |

The H800 value `4096` is probe-only and MUST NOT be used for formal capacity
comparison. It produced a valid bounded diagnostic run but caused later formal
prefill chunking. The H800 `176000` value is derived from the measured memory
ratio `310809 * 81559 / 143771 = 176316.9988`, rounded down to preserve
initialization headroom. H200 remains `310809`.

## Post-MoE AR Diagnostic Modes

The restored H800 A/B diagnostic uses the same first 4096-prefill case and
ten fully drained warmup replays before the bounded formal request. Run the
three modes with the H800 row above and separate output directories:

- `normal`: full post-MoE TP AR payload;
- `scalar_sync`: one-element TP all-reduce preserving participant rendezvous;
- `skip`: bypasses the post-MoE TP AR and is timing-only, semantically invalid.

`normal` and `scalar_sync` support communication timing diagnosis. `skip` MUST
NOT support accuracy, production latency, or Frontier calibration claims. The
mode, source commit, H800 capacity setting, warmup count, request count, and
first-formal identity MUST be recorded for every run.

## Warmup and Formal Data Gate

Every subsequent test, trace, and analysis in this task MUST use data collected
after the complete client warmup phase. Startup, initialization, JIT, CUDA
initialization, FlashInfer initialization, and client warmup rows are diagnostic
only and MUST NOT be reported as formal first-forward measurements.

The standard execution chain is fixed as follows:

```text
tests/e2e/issue26_h200_replay_worker.sh
  -> tests/e2e/issue26_h200_uniform_groundtruth_worker.sh
  -> tests/e2e/issue26_h200_diagnostics_worker.sh batch
  -> tests/e2e/issue26_token_id_client.py
  -> tests/e2e/issue26_diagnostic_identity_analysis.py --mode batch
```

Parameter changes are allowed when they are documented in the run manifest and
test report. The worker chain, complete-drain semantics, client identity
validation, and formal-boundary predicates remain mandatory. A test that uses a
different chain or bypasses the identity validator is exploratory evidence and
cannot support a task conclusion.

## Required Client Warmup Contract

The standard client contract is ten fully drained warmup replays followed by
100 formal requests. A run is admissible only when all of the following hold:

- ten warmup replays complete with 100 completed requests each;
- every replay drains before the next replay starts;
- the formal phase contains 100 unique completed requests;
- the combined client artifact contains exactly 1100 rows (10 warmup replays × 100 plus 100 formal requests);
- the identity validator reports `PASS`.

Reports MUST state the warmup replay count, completion count per replay, drain
status, formal row count, and validator result. A missing completion, partial
drain, duplicate identity, or absent formal row invalidates the run for timing
analysis.

## First Formal 4096-Prefill Boundary

The first formal batch MUST be selected by request identity and all of these
batch predicates:

```text
request_id = cmpl-pf4096_dc1024:0-0
batch_size = 1
request_num_tokens = [4096]
batch_num_prefill_tokens = 4096
batch_num_decode_tokens = 0
```

The analysis MUST use the same DP lane and DP0 TP0--TP3 ranks. It MUST report
the batch identity, DP/TP lane, batch ID, and the exact predicates used for
selection. It MUST compute the rank median, P90, rank maximum, and rank spread
from those four rank rows when all rows are present.

The analysis MUST NOT select a first batch by minimum `batch_id`. It MUST
exclude request IDs beginning with `warmup:`, initialization rows, mixed
prefill/decode batches when a single-request formal boundary is required,
operator-only spans, record-function spans, and client TTFT values.

## Scope and Provenance Reporting

Every test report MUST distinguish these scopes:

- engine/internal warmup;
- client warmup;
- formal prefill;
- mixed prefill/decode batch;
- decode batch;
- clean batch-only span;
- operator, communication, CUDA-event, or record-function instrumentation.

Reports MUST state whether operator, communication, or AR instrumentation was
enabled. Instrumented spans MUST remain diagnostic and MUST NOT replace the
clean batch-only first formal boundary. Historical values MAY establish context,
but a new test report MUST label them as historical and MUST keep them separate
from newly collected measurements.

## Acceptance Rule

The H200 batch baseline is valid only after the warmup/formal gate, identity
predicates, lane selection, and provenance fields above are all satisfied. Any
failure MUST be surfaced explicitly and recorded in `progress.md` or
`issues.md`; fallback selection of a startup row or minimum batch ID is
prohibited.

## Standard scale and minimal AR ablation — 2026-09-10

H800 codesign + h800 + 176000 blocks has valid normal medians 80.654880524 ms and 77.483665466 ms (two runs, not a variability interval). Treat about 80 ms as the observed reference scale. A same-config 5x discrepancy triggers immediate harness/config/instrumentation RCA. H200 remains step_main + h200 + 310809 blocks and separate evidence.

All follow-up AR ablations use the standard replay suite: 10 drained 100-request warmups plus 100 formal requests in each server mode. Select cmpl-pf4096_dc1024:0-0, DP0 TP0-3, one request, 4096 prefill, zero decode. Boundary file logging OFF. Minimal bypass comments only the post-MoE TP AR call, preserves DP combine and returns existing states, with no clone or scalar replacement. Record actual source and commit for both server modes. Normal/skip delta is an ablation effect, not automatically pure collective duration.

Verified minimal bypass (2026-09-10): full standard run median77.935920715ms, max78.142173767ms; first-formal identity and all warmup/drain gates PASS. The expected scale is tens of milliseconds (~80ms) for this exact configuration, including this single-call bypass. Select a request's prefill by its own scheduled-token entry; batch-level prefill>0 also includes unrelated requests' prefills during target decode.

## H200 all2all backend comparison — 2026-09-12

The communication backend experiment is restricted to the H200 row above and
the same Qwen3 4096-prefill/1024-output workload. Each backend is an isolated
fresh RJob and uses the standard replay chain, with
`ISSUE26_WARMUPS=10`, ten drained 100-request warmup replays, and 100 formal
requests. The backend value is passed as `ISSUE26_ALL2ALL_BACKEND` and is
recorded in the diagnostics mode manifest.

The candidate set is `naive`, `pplx`, `deepep_high_throughput`, and
`deepep_low_latency`, matching the vLLM source selector. `naive` is the current
reference. `pplx` and the two DeepEP modes are capability-gated: an import,
initialization, or model-start failure is a recorded unsupported result and
does not produce a timing value. A successful candidate must pass the complete
clean and batch identity/drain checks and the first-formal predicates.

Only the clean batch-only first-formal span from DP0 TP0--TP3 is compared
across backend runs. Full diagnostic outer spans, operator rows, and rank-local
inclusive collective scopes are excluded from the clean comparison. Backend
results are reported independently; no values are averaged, subtracted, or
combined across different backend implementations. A backend that changes the
MoE kernel or post-MoE reduction path is treated as a semantic variant whose
closeness to Frontier's ideal EP abstraction is assessed from source behavior
and clean span, not as a production correction.

## Profiler capture gate

Profiler runs may use Nsight or Kineto with `VLLM_FRONTIER_INSTRUMENTATION=0`,
but the profiler window is a diagnostic scope. A profiler result is admissible
only when the standard ten-warmup/1100-row and first-formal identity gates pass,
the trace artifact is nonempty and parseable, and the trace boundary is mapped
to the selected DP0 TP0--TP3 processes. A completed replay with no `.nsys-rep`
or equivalent trace is a profiler/harness failure and has no latency or
kernel-composition value. The accepted clean native reference remains the
standard uninstrumented `78.118782043--79.307357788 ms` boundary; profiler
windows above that scale must be reported separately and cannot authorize
Frontier correction or clean/diagnostic reconciliation. Nsight category values
are per-device interval unions and must not be summed across TP ranks.
