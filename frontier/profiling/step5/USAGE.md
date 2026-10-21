# Native Step attention profiling

This producer measures the locked Step image's complete `Step5Attention` forward.
It owns QKV/indexer/gate projections, QK normalization/RoPE/cache writes, SWA or
sparse attention, sparse summaries/scoring/top-k/staging, headwise gating and
the output projection. The output TP all-reduce is excluded.

One physical GPU measures one isolated TP-local rank. The native model and
linear constructors' TP-size/rank getters select the requested local shapes;
the actual process groups stay at world size one. All native kernel and forward
implementations remain unchanged. This is compute profiling, not a real TP8
serving or critical-path qualification. For TP8 the sparse indexer has **four**
local query heads (one replicated provider), not two.

## CPU planning

```bash
python -m frontier.profiling.step5.main \
  --model-path /path/to/locked/metadata --max-model-len 8192 \
  --prefixes 0 4104 --cudagraph-capture-sizes 1 2 \
  --min-timing-window-ms 0.5 \
  --output-dir /path/to/output --dry-run
```

The smoke grid contains 12 native cells: initial and continuation prefill at
16/64 query tokens, and FULL decode at 512/4104 prefix tokens, for both layer
variants. It emits 12 event-latency rows and eight separate GPU-active rows.
The long prefix exercises selection beyond 512 regions
of eight tokens. Prefill must contain at least two query tokens because the
native one-token branch is decode. Batches currently use equal prefix lengths;
heterogeneous-prefix and mixed prefill/decode batches are not implemented.

## Native graph policy

The locked Step image auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for the Step
architecture before model decorators bind (`vllm/config/vllm.py:814-842`). Its O2
policy is `FULL_AND_PIECEWISE`, with compilation mode NONE, not torch.compile
piecewise. This does **not** mean every prefill uses a graph. The observed native
configuration has `max_num_seqs=1`, giving capture sizes `[1,2]`
(`config/vllm.py:2389-2392,2473`). Native dispatch chooses NONE for 16/64-token
prefill and FULL for single-token decode (`v1/worker/gpu/cudagraph_utils.py:518-563`).
The parent chose to retain that configuration; this package does not enlarge its
capture ceiling just to obtain faster-looking prefill numbers.

A two-token prefill fits the ceiling and uses the image's actual
`BreakableCUDAGraphWrapper`. FULL capture mirrors the native graph context,
pool and offloader synchronization and captures the forward with runtime NONE
(`cudagraph_utils.py:435-502`). Input and metadata buffers remain address-stable.
Padding to an intermediate capture bucket is not implemented; such inputs fail
clearly rather than silently measuring an unpadded graph.

## Stage; do not submit automatically

```bash
python -m frontier.profiling.step5.job \
  --stage /data/ycfeng/tmp/step5_attention_20261020/smoke_stage04 \
  --dependency-site /path/to/cpu-verified/cp312/dependency-site \
  --run-id attention_smoke_20261020_03
```

`manifest.json` records the locked image digest, one-H800/codesign resources,
local NFS source mount, personal cloud root, every staged source SHA-256, worker
command and exact submission command. The parent must run `submit_with_env.sh`,
which sources `/data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh`, checks that
personal credentials are set and replaces itself with the existing launcher
path in `submit.sh`. Running `submit.sh` alone requires an already initialized
credential environment. Keep the launcher alive through terminal job status;
do not close it when the job is merely queued. Neither staging nor verification
executes the real submission wrapper.
The stage includes CPU-prepared pandas, dateutil, pytz, tzdata, PyYAML, tqdm and
six. The worker must import them successfully before entering the existing
wrapper, so its normal path performs no dependency installation. The wrapper
sets CUDA compatibility before importing torch, without changing torch/vLLM/Triton.

## Outputs and judgment

Below `compute/h800/stepfun-ai/Step-5-Preview/` in the requested output root:

- `step5_attention.csv` has 12 `CUDA_EVENT` rows. Eager rows include native host
  launch gaps and are diagnostic latency, **not** GPU-compute calibration data.
- `step5_attention_kernel_only.csv` has eight `KERNEL_ONLY` eager rows, containing
  correlated kernel **and device-copy** duration sums, without host gaps. This
  sum is not an elapsed interval or a union of overlapping device intervals.

`summary.json` reports grid/event counts, GPU-active counts and stable calibration
targets (FULL event rows plus eager GPU-active rows). Raw Chrome traces remain in
`traces/`. Completed rows are written incrementally; the two methods never share
one dataframe or pretend to be the same timing quantity.

- Require 12 event rows and eight GPU-active rows, finite nonnegative timings and
  positive whole-layer times. All 12 calibration targets must pass quality checks.
- Native uninstrumented versus instrumented output and KV/summary state must
  match at `rtol=atol=0.02`. Continuations also compare against a full-prefix
  native forward. Graph rows compare actual replay output/state against eager.
- Only contract-inactive or fused targets receive zeros. Missing active scopes
  fail rather than being filled. Outermost observed scopes own fused work once.
- Capture and warmup stay outside event windows. Each accepted clean window uses
  one event pair around N consecutive replays and lasts at least the configured
  0.5 ms. Eager GPU-active windows independently reach 0.5 ms of actual device
  activity, not 0.5 ms of host waiting. Per-replay statistics use a continuous
  cohort with N+1 shared event boundaries and one final synchronization, avoiding
  a stopped-queue launch gap before every diagnostic sample.
- Scope cohorts own distinct event handles per replay. All N matching spans are
  summed before normalization; a last-overwritten graph event is never divided
  by N. `operator_sum_to_layer_ratio` uses the matching instrumented root, not
  the clean graph. `instrumentation_overhead_fraction` reports the clean/scoped
  outer-window difference separately, without clipping or rescaling.
- The 5% CV bound applies to the layer and every active target. Window means must
  also agree with per-replay medians within that bound. One explicit retry keeps
  the initial outcome; failure remains unqualified evidence.
- Repeated DSA timing requires region-aligned contexts because native partial
  summary tails accumulate state (`ops/summary_update.py:35-58,140-148`). Output
  and cache are checked immediately after the final measured replay, without an
  extra forward masking the measured state.
- Compare GPU-active prefill costs at matching context, not the eager event
  latency. Missing activity/correlation or unowned measured work fails rather
  than silently dropping a kernel or copy. Restore/priming work is excluded.
- Inspect native kernel names: BF16 GEMMs, FlashAttention SWA,
  `_headwise_sigmoid_gate`, sparse prefill/indexer/summary/top-k kernels and
  `_sparse_decode_*`. Keep `memory_copy_count` separate from kernel counts;
  graph decode should have zero device copies. The raw trace settles exact
  spelling/branch identity; substring expectations alone do not admit a row.

Estimated smoke runtime is 15–45 minutes (0.25–0.75 H800 GPU-hours), with a
one-hour worker timeout. This is an estimate, not a measurement.

## Qualification limits

No Step simulator profiling/execution flag is enabled and no runtime path is
changed. The shared attention dataframe validator is reused through a private
schema view; the registered family remains disabled. Typed operator contracts
are emitted, but downstream admission still needs the option-A linear-profile
ownership cleanup, full-grid/reproducibility measurements, physical-scope
reconciliation thresholds and native GPU verification. A successful smoke is
not full ticket-10 admission, trained prediction, or full-role qualification.

## Fixed-workload H200 small probe

This is a **small operator probe**, not the canonical 64-H200 serving run or a
full grid. It pins the unchanged D18 workload from shared commit `68f891b`
(local `958d81a`) under
`data/processed_traces/canonical_fixed_6144_1024_req128_qps1/`.

```bash
python -m frontier.profiling.step5.job --fixed-workload-probe \
  --stage /data/ycfeng/tmp/d18_profile_20261021/h200_small_probe_stage01 \
  --dependency-site /data/ycfeng/tmp/step5_attention_20261020/offline_site01
```

The existing source packager creates a new stage only. The manifest requests
one H200/step_main, eight CPUs, 32 GiB memory, non-preemptible, with the pinned
Step image and a one-hour timeout. Only the parent submits `submit_with_env.sh`.
The existing worker checks its host driver before Python, using CUDA 13 forward
compatibility only below R580. Estimated runtime is 20–45 minutes, or
0.33–0.75 GPU-hours, excluding queueing; this is not a measured cost.

The existing attention producer consumes explicit cells from
`fixed_probe_inputs.json`, rather than an infeasible Cartesian product:

- Both SWA and DSA: 6144-token initial prefill; 16-token continuation after 6128
  prefix tokens; one-token decode at contexts 6144, 6656 and 7160, batch 1/2.
- SWA only: one-token decode at context 7167, batch 1/2, reaching sequence length
  7168. Context excludes the current query token. Repeated DSA partial-region
  contexts remain unsupported; they are not silently rounded or qualified.
- Explicit isolated policy: `max_num_seqs=2`, `max_model_len=7168`,
  `max_num_batched_tokens=14336`, prefix caching/chunked prefill off,
  `FULL_AND_PIECEWISE`, exact capture buckets `[1,2]`, breakable graphs on.
  Prefill/tail use NONE and decode uses FULL. These widths follow that probe's
  configured capacity, not QPS or a claim about canonical DP4 batches.

Expected attention output: 18 event rows and four eager GPU-active rows under
`attention/compute/h200/stepfun-ai/Step-5-Preview/`. The existing linear producer
measures tokens 1, 2, 16 and 6144, with eight rows per method under
`linear/compute/h200/stepfun-ai/Step-5-Preview/`. Native command logs are retained
in the output root; `pipefail` preserves failures. The two timing methods remain
distinct. No new operator, endpoint or LM-head coverage is implied.

The failed H800 linear smoke selected generic FP8 MLP defaults despite the
native module exclusions. Profiling now derives the quantized-operation selector
from the existing Step layer descriptions, preserving dense/shared BF16 and
routed FP8. It does not pad TP8 widths 1728/192 or relax the native block guard.
The precision correction and attention event priming are CPU-verified only until
fresh native execution. All existing output/state, 5% CV/median-agreement,
activity and phase-policy gates remain unchanged. Canonical P32/D32 runtime,
capacity, transfer-tail behavior, reached batches and actual H200 timing still
need their separate qualifications.
