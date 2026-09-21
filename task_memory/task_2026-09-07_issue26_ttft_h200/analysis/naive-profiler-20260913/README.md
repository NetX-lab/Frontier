# H200 naive Kineto profiler

This directory contains the external profiling plan for the clean H200
`VLLM_ALL2ALL_BACKEND=naive` path.  The launch is intentionally deferred until
the coordinator confirms that the PPLX allocation has released the H200 slot.

## Contract

- H200 `step_main`, eight GPUs, `num_gpu_blocks_override=310809`.
- Qwen3 30B A3B dummy model, TP4/DP2/EP8, BF16, FLASHINFER, eager execution,
  uniform routing, prefix caching and chunked prefill disabled.
- Ten complete 100-request warmup replays, then 100 formal requests with the
  same `pf4096_dc1024` request schedule and token IDs used by the standard
  client.
- Frontier instrumentation and all Frontier operator/batch loggers are OFF.
  The only profiler is vLLM's existing `torch.profiler` (Kineto) endpoint.
- `/start_profile` is called only after the tenth warmup drain.  `/stop_profile`
  is called from the first formal request's first-token callback; remaining
  formal requests continue so identity and drain checks still cover all 100.

The profile captures a small amount of the first decode step after prefill.  It
is therefore a diagnostic activity window, not a replacement for the clean
78--79 ms batch logger reference.  The report must show the profiler window
span beside the accepted clean references and state profiler perturbation.

## Breakdown parser

`parse_kineto_trace.py` classifies GPU kernel names into `comm`, `mem`, and
`comp`, then reports inclusive kernel duration, per-category interval union,
all-kernel union, and idle/non-kernel gap for the formal 48-layer GPU window.
Invoke it with `--phase-records` and `--client-jsonl`; it converts the recorded
profile-start and first-formal first-token wall timestamps into Kineto time and
selects the complete 48-marker `unified_attention_with_output` /
`moe_forward` group. This explicitly excludes the partially queued forward
that was already running when `/start_profile` was called. A bounded contiguous
GPU-segment fallback remains available for traces that lack CPU markers and is
labelled in the JSON output. Intervals from different CUDA streams may overlap;
category unions are consequently not summed into the span.

## Static verification

```text
bash -n tests/e2e/issue26_h200_naive_profiler_worker.sh
PYTHONPATH=tests/e2e python -m py_compile tests/e2e/issue26_naive_profiler_client.py
python -m py_compile analysis/naive-profiler-20260913/parse_kineto_trace.py
git diff --check
```

The worker and client were committed as `e8ff0d79`; task-memory documents are
ignored by the repository and remain coordinator-owned evidence.

## Deferred launch command

After a capacity check and coordinator approval, run `launch.sh` (which submits
`yc26-h200-naive-kineto-20260913-01`) from the repository root.  The command
uses `/kubebrain/rlaunch --charged-group=step_main --private-machine=group
--positive-tags=h200 --gpu=8 --cpu=64 --memory=409600`, the fixed image digest,
and mounts `/data`.  If predict-only reports no immediate resource, submit the
same detached RJob and monitor it FIFO as required by the H200 handbook; do not
submit a duplicate name.
