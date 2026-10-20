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
  --prefixes 0 4104 --graph-modes none full \
  --output-dir /path/to/output --dry-run
```

The smoke grid contains 16 rows: eight per layer variant, including initial and
continuation prefill at 16/64 query tokens, and eager/full-graph decode at
512/4104 prefix tokens. The long prefix exercises selection beyond 512 regions
of eight tokens. Prefill must contain at least two query tokens because the
native one-token branch is decode. Batches currently use equal prefix lengths;
heterogeneous-prefix and mixed prefill/decode batches are not implemented.

## Stage; do not submit automatically

```bash
python -m frontier.profiling.step5.job \
  --stage /data/ycfeng/tmp/step5_attention_smoke_stage \
  --run-id attention_smoke_20261020_01
```

`manifest.json` records the locked image digest, one-H800/codesign resources,
local NFS source mount, personal cloud root, every staged source SHA-256, worker
command and exact submission command. `submit.sh` only executes that existing
StepMind launcher when a human/parent explicitly runs it with personal auth and
`STEPMIND_BACKEND=rjob`. Keep that launcher alive through terminal job status.
The worker wrapper sets CUDA compatibility before starting Python and installs
only missing lightweight dependencies, without changing torch/vLLM/Triton.

## Outputs and judgment

The CSV is `compute/h800/stepfun-ai/Step-5-Preview/step5_attention.csv` below the
requested output root. `summary.json` reports expected/actual/stable row counts;
raw Chrome traces remain in `traces/`. Completed rows are written incrementally.

- Require 16 rows, finite nonnegative timings and positive whole-layer times.
- Native uninstrumented versus instrumented output and KV/summary state must
  match at `rtol=atol=0.02`. Continuations also compare against a full-prefix
  native forward. Graph rows compare actual replay output/state against eager.
- Only contract-inactive or fused targets receive zeros. Missing active scopes
  fail rather than being filled. Outermost observed scopes own fused work once.
- CUDA events time the complete layer, including DSA prefill device copies.
  Per-operator scopes use a separate observed native pass; full-graph operator
  measurements use external CUDA-event nodes in a separate captured graph.
- The default stability threshold is 5% coefficient of variation, with one
  explicit retry. Unstable rows remain raw evidence, not admitted measurements.
- Inspect `operator_sum_to_layer_ratio`; host launch gaps and instrumentation
  make it a diagnostic, not an automatic reconciliation PASS. Compare 16/64
  prefill timings at matching context and graph policy; tiny-token launch-bound
  cases can be flat/noisy, so do not rescale a nonmonotone result.
- Inspect native kernel names: BF16 GEMMs, FlashAttention SWA,
  `_headwise_sigmoid_gate`, sparse prefill/indexer/summary/top-k kernels and
  `_sparse_decode_*`. Keep `memory_copy_count` separate from kernel counts;
  graph decode should have zero device copies. The raw trace settles exact
  spelling/branch identity; substring expectations alone do not admit a row.

Estimated smoke runtime is 15–40 minutes (0.25–0.67 H800 GPU-hours), with a
one-hour worker timeout. This is an estimate, not a measurement.

## Qualification limits

No Step simulator profiling/execution flag is enabled and no runtime path is
changed. The shared attention dataframe validator is reused through a private
schema view; the registered family remains disabled. Typed operator contracts
are emitted, but downstream admission still needs the option-A linear-profile
ownership cleanup, full-grid/reproducibility measurements, physical-scope
reconciliation thresholds and native GPU verification. A successful smoke is
not full ticket-10 admission, trained prediction, or full-role qualification.
