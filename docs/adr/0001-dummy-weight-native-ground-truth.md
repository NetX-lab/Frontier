# Dummy-weight native runs are the ground truth for GLM-5.3 and DeepSeek-V4-Flash

Frontier's calibration of GLM-5.3 and DeepSeek-V4-Flash against vLLM v0.28.0 uses dummy-weight native runs (`--load-format dummy`) as its ground truth: the full model structure, quantization, kernels, communication, and memory allocation execute on real GPUs, but the weights are random and no checkpoint is downloaded. We chose this because it removes checkpoint acquisition and qualification from every native run, profiling pass, and re-measurement. The cost is that expert routing follows the random weights rather than a trained model, so the calibration covers the serving system under that initialization and claims nothing about routing distributions or output quality of a production checkpoint.

## Consequences

- Native MoE routing on dummy weights is kept as measured; it is not forced to round-robin and not assumed to be balanced.
- Equal seeds do not prove that prefill and decode Replicas hold identical parameters or auxiliary buffers, so token-for-token output equality is asserted only where that identity is shown.
- Frontier's dummy timing mode remains a structural test tool and never counts as calibration evidence.
