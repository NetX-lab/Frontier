# Frontier Roadmap

## Overview

This roadmap lists the work planned for the next phases of Frontier. It is
organized into three horizons:

- **Near-term** — finish the disaggregation release and harden the public
  surface (docs, CLI, config, examples).
- **Mid-term** — broaden model coverage, accelerate the disaggregated
  simulation workflow, and ship a dedicated use-cases module.
- **Long-term** — deepen the core models for KV-cache scheduling, cross-cluster
  transfer, and analytical compute hardware.

Items within a horizon are not strictly ordered. Scope and timing may shift as
the release stabilizes and community feedback arrives.

## Near-term

We have released the initial version of Frontier, targeting the `co-location`
serving architecture. Disaggregated serving is next on deck.

Goal: make disaggregated serving a first-class, documented part of the public
release and smooth the on-ramp for new users.

- **PDD and AFD release.** Over the next two weeks, we will release the
  Prefill-Decode Disaggregation (PDD) and Attention-FFN Disaggregation (AFD)
  code paths: finalizing, cleaning up, and documenting them, then lifting the
  release guard so these architectures are supported on the public branch
  alongside `co-location`. Stay tuned!
- **Documentation pass.** Expand the docs set (CLI, config, profiling,
  architecture) so every supported architecture and runtime option is covered,
  with consistent terminology across the `Workload/Config`, `Fidelity Plane`,
  `Control Plane`, and `Execution Plane` concepts.
- **CLI and config conventions.** Normalize the CLI surface and config schema:
  align flag naming across architectures, document disaggregated cluster and
  transfer fields, and remove the temporary release guards once the paths are
  validated.
- **More examples.** Add runnable example scripts that cover dense and MoE
  models across co-location, PDD, and AFD, with predictable metrics output and
  the runtime optimizations (CUDA Graph, speculative decoding / MTP, prefix
  caching, chunked prefill) enabled.

## Mid-term

Goal: widen the modeling envelope and turn Frontier into a platform for
repeatable what-if studies.

- **Expanded model support.** Add calibrated support for additional
  state-of-the-art model families, including DeepSeek, Kimi, and MiniMax,
  covering their attention, MoE, and runtime characteristics.
- **Faster disaggregated simulation.** Optimize the simulation workflow for
  disaggregated architectures to reduce wall-clock time per run, so large
  design-space sweeps over PDD and AFD configurations become practical.
- **Use-cases module.** Introduce a dedicated `use-cases` module that packages
  end-to-end study scripts and analysis, including:
  - SLA-aware Pareto frontier search,
  - scheduling-policy validation,
  - heterogeneous disaggregation scenario exploration,
  - dynamic configuration / parallelism switching,
  - and additional reference cases as they mature.

## Long-term

Goal: deepen the core fidelity models so Frontier can study serving
architectures and hardware that do not exist in the current stack.

- **Serving Engines Integration**: Support for SGLang and TensorRT-LLM frameworks.
- **Advanced KV-cache modeling.** Strengthen the `kv_cache` module so Frontier
  can simulate richer KV-cache scheduling and management policies, including
  architectures such as Dynamo and Mooncake (hierarchical caching, KV-cache
  pooling, and cross-node reuse).
- **Transfer modeling.** Improve the simulation and modeling of PDD and AFD
  transfers (KV cache and activations), capturing topology, contention, and
  scheduling effects on cross-cluster movement with higher fidelity.
- **Analytical compute simulator.** Add an analytical computation simulator
  module that estimates operator runtime from hardware specifications, enabling
  studies of future or hypothetical compute hardware without measured
  profiles.
