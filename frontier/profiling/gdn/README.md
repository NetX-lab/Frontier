# Standard vLLM GDN profiler

`frontier.profiling.gdn.main` is the standard vLLM producer for Qwen3.5
gated-delta-network layers. It supports fresh prefill, continuation/chunked
prefill, and ordinary decode. A same-batch prefill/decode input is rejected
before any CSV row is written.

The producer uses synthetic BF16 weights with the real model dimensions and
records `weight_source=synthetic_bf16`. This is a profiling-path smoke and
schema source; it is not checkpoint-exact timing or AMD benchmark evidence.
ROCm output uses `DEVICE_EVENT`; `CUDA_EVENT` is rejected for the standard
ROCm producer. The output path is selected by the existing measurement-family
path builder, so event families remain separate.

For a carried continuation or decode state, the wrapper primes the state
outside the timed region and restores the same snapshot before each warmup and
timed iteration. vLLM state page zero remains reserved as the null page.

The normal CPU environment can import the input and CLI planning modules. The
vLLM, Triton, and PyTorch GPU dependencies are imported only when the wrapper
is constructed. Without MI355X hardware, runtime timing and decomposition
checks remain `SKIP: AMD/MI355X hardware unavailable`.
