# Review request (round 2): revised H3o-(a) proposal for operator-level DeepSeek-V4-Flash costs

Report language: Chinese, with English technical terms and identifiers kept as is.

## Working root and scope

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`. The revised files are uncommitted working-tree changes on top of HEAD; read the working tree.
- Primary target (design review, not a diff review): `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md` (revision 2).
- Your round-1 report: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/codex-review-9c8b2435-c3f3-468e-8c41-6a504ff54fda/review.md`.
- Author's positions on each round-1 finding: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/review.md`. Revision 1 is `git show HEAD:task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md`.
- Governing records as in round 1: `requirements.md` items 20-27, `plan.md` row H3o, `docs/adr/0002-operator-costs-not-measured-step-tables.md`, `AGENTS.md`.

## What changed

The author agreed with F001-F008 and revised the design. The author also added three items of their own:

1. The V4 attention component accepts `MONOLITHIC` only (`frontier/attention/deepseek_v4.py:101`).
2. Pre-forward input and metadata preparation kernels are missing from the inventory.
3. A first CPU look at a retained native H800 P0 trace (decode dp1 rank 0) shows the decode-only steps as FULL graph replays, with eager pre-forward kernels on a separate stream.
   - Script: `/data/ycfeng/tmp/glm53_dsv4/h3o/census_peek.py` and `census_peek2.py`.
   - Trace: `/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04/torch_traces/decode/dp1/dp1_pp0_tp0_dcp0_ep8_rank0.1792842735368033988.pt.trace.json.gz`, 12.8 MB compressed.
   - Observed: 142,567 kernels, 137,252 inside 150 `cudaGraphLaunch` calls; 5,315 eager kernels, all on stream 23; annotation `model_dispatch_run_fullgraph`.
   - If you rerun it, use a memory limit (`ulimit -v 31457280`) and a timeout.

## Questions to answer

1. **Closure.** Does revision 2 resolve each of F001-F008? For each one, say resolved, partly resolved (what remains) or not resolved.
2. **New defects.** Are there new errors or omissions introduced by revision 2, including wrong code references?
3. **Composition under FULL graph.** Given that decode-only steps may be FULL graph replays, where auxiliary-stream branches become graph branches, are the following sound and sufficient answers to F001/F002?
   - the critical path over a source-defined stream graph (§3.2);
   - the graph-internal gap charged per kernel on the critical path (§3.3).
4. **Pilot P1 and gate A4.** Is P1 (§5, stage 1) designed so that it can falsify the composition, and is the median 5% / p90 10% tolerance appropriate there?
5. **Decision D2.** The existing CPU-overhead `forward_launch` rows are a measured host launch time per forward, keyed by batch features, and are used only for the host stream of the two-stream model. Is this consistent with ADR 0002 / D-NO-STEP-COST-SHORTCUT? Or should the design treat it as the same kind of shortcut and prefer D2(b) or D2(c)?
6. **Training CSV columns.** The author departs from one part of your F001 correction: union and span are kept in pilot P1 and in the raw interval records for audit, but not as training-CSV columns, since no consumer reads them. Do you accept this?
7. **Smallest complete change.** Is anything in §6 (19 files, about 2,500 lines) unnecessary, or still missing?
8. **Order and resources.** Are the H800-first order, the stage gates and the GPU caps in §7 reasonable? Are the H200 rules (non-preemptible, running plus queued ≤ 64, soft gates for runs of 32 or more GPUs) stated correctly?

If you have no remaining material findings, give the verdict `no_findings`. Otherwise give concrete corrections. Do not edit any file.
