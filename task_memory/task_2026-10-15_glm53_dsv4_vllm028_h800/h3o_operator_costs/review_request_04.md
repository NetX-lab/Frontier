# Review request (round 3): H3o-(a) revision 2c for operator-level DeepSeek-V4-Flash costs

Report language: Chinese, with English technical terms and identifiers kept as is.

## Working root and scope

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`, HEAD `4b4b3ab0`.
- Primary target (design review, not a diff review): `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md` (revision 2c).
- Your round-2 report: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/codex-review-4678dfdc-a4d3-47a2-8c00-de2e7803a915/review.md`.
- Author's spot checks and positions on F009-F015: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/review.md`, section "Round 2, resubmission".
- Revision 2b is `git show 40299c0a:<design path>`; `git diff 40299c0a 4b4b3ab0 -- <design path>` shows the revision 2c change.
- Evidence added in this round: `h3o_operator_costs/evidence/routing/mode_census.py` / `.txt` (native graph mode by step class, first node of each role, rank 0) and `review_counter_checks.py` / `.txt` (your round-2 counter script, copied unchanged). Both were rerun on 2026-10-26 with byte-identical output.
- Governing records: `requirements.md` items 26-33 (items 32-33 are new: the lead's working decisions for revision 2c, and YC's decisions D1 (a), growth confirmed, D7 (a), round 3 approved), `docs/adr/0002-operator-costs-not-measured-step-tables.md`, `AGENTS.md`.
- Pinned vLLM source: `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm`.

## What changed since your round-2 report

The author accepted F009-F015. Revision 2c:

1. §3.7: the source rule of `v1/worker/gpu/dp_utils.py:16-95` with the uniform-count and max-query redispatch; the native `FULL_AND_PIECEWISE` local dispatch as a new value `full_and_piecewise` of `decode_cuda_graph_mode` (V4 only, D7); the corrected census (first tails run PIECEWISE on every lane; revision 2b's statements on the uniform-disagreement branch were wrong); the DP rendezvous before post-DP preparation and first-layer device work; the V2 dummy workload.
2. §3.2: a conditional stream graph (multi-stream threshold from the recorded runtime identity, Breakable serialization, indexer state) and a consumer: the composition returns the composed elapsed and per-scope offsets; the V4 attention getter returns the composed elapsed with the gap charged once.
3. §3.3: a per-stream V4 gap, one value per (mode, TP) in the existing `kernel_gap.csv` schema, declared as an approximation and checked per group by A4.
4. §3.4: the CPU-probe `prepare_inputs_e2e` interval is split at the recorded DP all-reduce if R2h / R2 run.
5. §5 stage 1a: a FULL / PIECEWISE replay route before pilot P1h (signature ownership from an eager hook pass, one-to-one interval match, joined boundary).
6. Gates: A4 per group on FULL, PIECEWISE and eager replays; A11 adds a mode and width census; A13 (c) from Frontier batches through the classifier and pricing consumer; A13 (d) timing with your 110 ms sentinel.
7. §6 entries 24-28 (28 entries, about +3,310 / −330 lines); §7 caps H200 45 and H800 21 GPU-hours; §8 about 15-16.5 days.

## Questions to answer

1. **Closure.** Does revision 2c resolve each of F009-F015, and the remaining parts of F001, F002 and F006? For each: resolved, partly resolved (what remains) or not resolved.
2. **New defects.** Are there new errors or omissions in revision 2c, including wrong source or code references, wrong counts, or claims the evidence does not support?
3. **DP rendezvous (§3.7 item 3, entries 20-22).** Is the stated timeline faithful to the V2 runner for real and dummy lanes, PREFILL and DECODE, FULL, PIECEWISE and eager? Does replacing the launch-start alignment of `sync_entry.py:113-131` for V4 forwards break anything else (the stage components, pipeline stages with PP = 1 only, KV-transfer or NIXL timing, the dummy-pass lifecycle)?
4. **Local classifier (§3.7 item 1, entries 24-25).** Is the `full_and_piecewise` rule as stated (FULL bound, PIECEWISE token-only padding, NONE above the largest capture size, PREFILL metadata under this value) equal to the native dispatch for the three cases? Is a V4-only value of an existing field the smallest correct change?
5. **V2 dummy workload (§3.7 item 4, §4).** Is the dummy workload as stated equal to the native one under FULL, PIECEWISE and NONE, including the request count?
6. **Per-stream gap (§3.3).** Is the definition measurable from replay intervals and sound as the declared approximation? Does A4 per group guard it enough?
7. **Replay route (§5 stage 1a).** Is the route feasible on the reduced engine, and does the one-to-one signature match settle kernel ownership under FULL and PIECEWISE replay?
8. **Smallest complete change and resources.** Is anything in §6 unnecessary or still missing? Are the caps, the stage gates and the H200 rules of §7 still right?

If you rerun any script on the native traces or counters, use a memory limit (`ulimit -v 31457280`) and a timeout.

If you have no remaining material findings, give the verdict `no_findings`. Otherwise give concrete corrections. Do not edit any file.
