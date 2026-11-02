# Standard vLLM GDN profiler

`frontier.profiling.gdn.main` is the standard vLLM producer for Qwen3.5
gated-delta-network layers. It supports fresh prefill, continuation/chunked
prefill, and ordinary decode. A same-batch prefill/decode input is rejected
before any CSV row is written.

The producer uses synthetic BF16 weights with the real model dimensions and
records `weight_source=synthetic_bf16`. This is a profiling-path smoke and
schema source; it is not checkpoint-exact timing or AMD benchmark evidence.
The producer runs on ROCm only and accepts only `--profile-method device_event`
(the default), which writes `DEVICE_EVENT` rows; the model dtype must be
`bfloat16`. Rank 0 writes the rows to
`<output-dir>/compute/<device>/<model>/gdn.csv`, the simulator's default GDN
input path.

```bash
python -m frontier.profiling.gdn.main \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --model-path /path/to/hf_model_dir \
  --device mi355x \
  --output-dir data/profiling \
  --tensor-parallel-size 1
```

With `--tensor-parallel-size N` for `N > 1`, launch one process per rank with
`torchrun --standalone --nproc-per-node=N -m frontier.profiling.gdn.main ...`.
The workload flags are `--prefill-seq-lens`, `--prefill-batch-sizes`,
`--decode-batch-sizes`, `--decode-context-len`, `--include-continuation-prefill`
with `--continuation-context-len`, `--warmup-iterations`,
`--profile-iterations`, `--max-model-len` and `--max-batch-size`.
`--include-mixed` adds a same-batch prefill/decode input, which is rejected.
See `docs/profiling/ROCM_MI355X.md` for the MI355X workflow.

For a carried continuation or decode state, the wrapper primes the state
outside the timed region and restores the same snapshot before each warmup and
timed iteration. vLLM state page zero remains reserved as the null page.

The normal CPU environment can import the input and CLI planning modules. The
vLLM, Triton, and PyTorch GPU dependencies are imported only when the wrapper
is constructed. Without MI355X hardware, runtime timing and decomposition
checks remain `SKIP: AMD/MI355X hardware unavailable`.
