# Ground-Truth Run

**When to use:** Use when a passed preflight permits a run of the selected reference engine in one declared mode.

Read `../references/contract.md` (Ground-Truth Engine and Build, Run Modes),
`../references/engine-integration.md`, `../references/tool-boundaries.md`, and
`../references/pitfalls.md` (Ground-Truth Collection). Require the passed
preflight of the same run tag.

Run the collector recorded in `engine_binding` with the pinned engine/build,
native configuration, workload, and mode. Verify its support before allocation.
For compatible vLLM builds, use the repository workers described in
`tool-boundaries.md`, Existing vLLM Collection Profile. Other engines use their
verified collector; vLLM worker flags, probes, and routing switches do not apply
by default. Write evidence to `runs/groundtruth_<mode>/<run_tag>/` and retain
the native outputs alongside any converted records.

Submit GPU jobs through the operator's platform rules and existing resource
authorization. Group paired clean and diagnostic runs on the same node group,
with warmup before the formal window, and keep the launcher alive through the
worker's terminal status. Choose independent clean repetition counts by
`contract.md`, Calibration Objective. Collect only the diagnostics needed by
the selected analyses and supported by the engine binding.

Retain command receipts, stdout/stderr, exit code, node identity, and output
hashes; hand every run tag to `groundtruth-run-check`. Clean output supports the
E2E reference; diagnostic output supports only its declared analysis. This entry
changes no Frontier or reference-engine source. A new probe, serving change,
or collector change follows the existing change rules and starts a new run
generation with its build and tool identities recorded.
