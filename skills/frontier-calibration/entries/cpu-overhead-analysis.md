# CPU Overhead Analysis

**When to use:** Use when the error budget attributes part of a failing gated metric to host time.

Read `../references/contract.md` (Error-Source Classes),
`../references/evidence-tiers.md`, and `../references/pitfalls.md` (Admission
and Tables). Do not enter for a passing E2E cell. Require the error budget, a
fresh clean E2E status, CPU-probe runs admitted by the rules of
`cpu-overhead-supplement`, and the CPU-overhead tables and settings the gated
Frontier label used.

Map admitted engine-native CPU spans to Frontier's host terms, preserving
launch/preparation overlap and keeping simulator-owned waits out of tables.
A missing source span cannot be replaced by an unrelated engine's host cost.

For the compatible vLLM profile, read engine and per-stage records through
the terms of the repository producer
(`frontier/profiling/cpu_overhead/vllm_cpu_probe.py`, defined in
`frontier/profiling/README.md`, "From vLLM CPU-probe logs"): `schedule`,
`prepare_inputs_e2e`, `forward_launch`, `forward_drain`, `sampler_e2e`, and
`process_model_outputs`, keyed by step shape, pipeline stage, and engine-idle
bucket. The DP, PP, and KV-load waits are Frontier's own simulated intervals,
not table terms. Compare per step class against the Frontier label's host
terms, and report the host share of the step period. A FULL CUDA Graph step
publishes kernel-only rows only, and a forward that overlaps another DP
engine's prefill leaves the table.

Classify each part: a missing or misordered host term in Frontier's step
composition is `simulation_logic` (route to `cpu-overhead-repair` under its
predicate); a table that the admitted runs no longer represent is
`measurement_input` (route to `cpu-overhead-supplement`, which needs the user's
approval); a slow probe host is `reference_instability`. Introduce no
independent CPU threshold.

Write `analysis/cpu_overhead_table.csv`, summary, and status. This entry is
read-only and never changes code or tables.
