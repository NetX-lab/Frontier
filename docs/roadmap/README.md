# Frontier Roadmap

## Overview

This roadmap starts from the current `pre-release-v0.3` surface: co-location,
sequential PDD / `pd-disaggregation`, and sequential PD-AF /
`pd-af-disaggregation`. It organizes remaining work into three horizons:

- **Near-term:** harden the released sequential architecture paths and close
  documented gaps with measured evidence.
- **Mid-term:** broaden model coverage, accelerate the disaggregated
  simulation workflow, and ship a dedicated use-cases module.
- **Long-term:** deepen the core models for KV-cache scheduling, cross-cluster
  transfer, and analytical compute hardware.

Items within a horizon are not strictly ordered. Scope and timing may shift as
the release stabilizes and community feedback arrives.

## Near-term

Goal: make the current three-architecture release predictable to configure,
profile, simulate, and compare.

- **Sequential release hardening.** Keep co-location, PDD, and PD-AF examples,
  CLI contracts, profiling inputs, and metrics output aligned as the v0.3
  release stabilizes.
- **Parallel PDD performance.** Preserve internal correctness coverage while
  investigating the measured coordination overhead. Public PDD remains
  sequential until evidence supports a faster parallel path.
- **PD-AF deferred features.** Validate parallel cluster processing, Thinking
  Mode, Speculative Decoding / MTP, Prefix Caching, and trace replay before
  adding any of them to the PD-AF release surface.
- **Profiling contract coverage.** Extend the existing operator and model
  architecture registries when new model families require new typed profiling
  ownership or TP/EP semantics.

## Mid-term

Goal: widen the modeling envelope and turn Frontier into a platform for
repeatable what-if studies.

- **Expanded model support.** Add calibrated support for additional
  state-of-the-art model families, including DeepSeek, Kimi, and MiniMax,
  covering their attention, MoE, and runtime characteristics.
- **Faster disaggregated simulation.** Optimize the simulation workflow for
  disaggregated architectures to reduce wall-clock time per run, so large
  design-space sweeps over PDD and PD-AF configurations become practical.
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

- **Serving engine integration.** Extend SGLang beyond its current co-location
  scope and add validated TensorRT-LLM workflows.
- **Advanced KV-cache modeling.** Strengthen the `kv_cache` module so Frontier
  can simulate richer KV-cache scheduling and management policies, including
  architectures such as Dynamo and Mooncake (hierarchical caching, KV-cache
  pooling, and cross-node reuse).
- **Transfer modeling.** Improve the simulation and modeling of PDD and PD-AF
  transfers (KV cache and activations), capturing topology, contention, and
  scheduling effects on cross-cluster movement with higher fidelity.
- **Analytical compute simulator.** Add an analytical computation simulator
  module that estimates operator runtime from hardware specifications, enabling
  studies of future or hypothetical compute hardware without measured
  profiles.
