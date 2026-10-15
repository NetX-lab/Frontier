# Ground-Truth Run

**When to use:** Use when a passed preflight permits a vLLM-BS ground-truth run in one declared mode.

Read `../references/contract.md` (Ground-Truth Checkout, Run Modes),
`../references/tool-boundaries.md`, and `../references/pitfalls.md`
(Ground-Truth Collection). Require the passed preflight of the same run tag.

Use the repository workers: `tests/comparison/dp_placement_pp/run_vllm_worker.sh`
with `vllm_replay.py` for co-location and DP x PP cells, and
`tests/comparison/calibration/run_pd_worker.sh` with `pd_replay.py`,
`pd_proxy.py`, and `pd_metrics.py` for PDD cells. The engine file selects the
model, layout, scheduler limits, graph mode, attention backend, and routing
(`moe_uniform_routing` for MoE references). The worker writes evidence to
`runs/groundtruth_<mode>/<run_tag>/`.

Submit GPU jobs through the operator's platform rules: the user's standing
authorization covers jobs already planned in the current tickets, and any new
job shape, resource size, or hardware needs a question first. Group the cell's
clean run with the runs that feed tables or analysis in one job, chained in
order (warm pass, clean, then the diagnostic modes), and keep the launcher
alive until the worker's terminal status.

After the run, retain the launcher receipt, stdout and stderr, exit code, node
identity, and output hashes, then hand every run tag to the post-run
`groundtruth-run-check`. Clean output serves only the E2E gate; each diagnostic
mode serves only its analysis. This entry changes no Frontier or vLLM source. A new
probe, a fork fix, or a harness change needs its own decision and lands as a new
ground-truth or harness commit, which starts a new run generation.
