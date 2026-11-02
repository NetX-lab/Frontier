# Review request (round 2, resubmission): H3o-(a) revisions 2, 2a and 2b for operator-level DeepSeek-V4-Flash costs

Report language: Chinese, with English technical terms and identifiers kept as is.

## Working root and scope

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`, HEAD `40299c0a`.
- Primary target (design review, not a diff review): `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/design.md` (revision 2b).
- Your round-1 report: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/codex-review-9c8b2435-c3f3-468e-8c41-6a504ff54fda/review.md`.
- Author's positions on each round-1 finding: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/review.md`.
- Earlier revisions: `git show 0ce31851:<design path>` (revision 1), `git show 886290a1:<design path>` (revision 2), `git show e936fe70:<design path>` (revision 2a). `git diff e936fe70 40299c0a` shows the revision 2b change and the routing test change.
- Governing records: `requirements.md` items 20-31 (items 28-31 are new: the review request, YC's decisions, the routing identity rule, the round-2 approval), `plan.md` row H3o, `docs/adr/0002-operator-costs-not-measured-step-tables.md`, `AGENTS.md`.
- An earlier round-2 request (`55d630bb-…`, on revision 2) failed with HTTP 429 before any report. This request replaces it.

## What changed since your round-1 report

1. **Revision 2** (commit `886290a1`): the author accepted F001-F008 and revised the design. The author also added: the V4 attention component accepts `MONOLITHIC` only (`frontier/attention/deepseek_v4.py:101`); pre-forward preparation kernels were missing from the inventory; a retained native H800 decode trace shows FULL graph replays plus eager pre-forward kernels on stream 23.
2. **Revision 2a** (commit `e936fe70`): the retained H200 native profiled runs (`/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/`, 32 ranks; `.../dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces/`, 64 ranks) show the same decode structure, so the runtime census covers H200 on CPU, and the 32-GPU job R1h is removed.
3. **YC's decisions** (requirements item 29): D2-D5 adopted as recommended; D6 is H200 first (P2-32 / P2-64 mainline), H800 after; D1 stays open with (a) as the working choice.
4. **Revision 2b** (commit `40299c0a`), routing identity (requirements item 30). YC asked whether F004 is a routing question and ordered Frontier's routing to equal native `balanced_topk`. The author's findings (§3.7, evidence in `h3o_operator_costs/evidence/routing/`):
   - Algorithm, per-expert counts and per-EP-rank counts already match. `tests/unit/test_balanced_routing.py` executes native `balanced_topk`, `DPMetadata` / `_compute_sp_num_tokens` and `determine_expert_map` from the pinned vLLM source; DP2 / EP16 and DP4 / EP32 widths were added; 102 passed. To rerun: `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_balanced_routing.py -q -p no:cacheprovider`.
   - The routed row count differs in 1.7-5.4% of decode forwards, because Frontier applies vLLM's DP synchronization (`vllm/v1/worker/gpu/dp_utils.py:16-75` under the pinned source) only to dummy forwards, not to real V4 lanes. A rule check on the retained native counters reproduces every recorded descriptor (3,588 rows, 245,088 forwards, 0 mismatches).
   - Proposed change: one DP-synchronized descriptor applied to every V4 lane at the stage's first pre_moe room, reusing the dummy deferral (files 20-23); gate A13; the non-V4 gap deferred (§10).

## Questions to answer

1. **Closure.** Does the current design resolve each of F001-F008? For each one: resolved, partly resolved (what remains) or not resolved.
2. **New defects.** Are there new errors or omissions in revisions 2, 2a or 2b, including wrong code or source references?
3. **Composition under FULL graph.** Decode-only steps are FULL graph replays, where auxiliary-stream branches become graph branches. Are the critical path over a source-defined stream graph (§3.2) and the graph-internal gap charged per kernel on the critical path (§3.3) sound and sufficient answers to F001 / F002?
4. **Pilot and gate A4.** Can pilot P1h (now H200, first) falsify the composition? Is the median 5% / p90 10% tolerance appropriate there?
5. **Decision D2 (decided by YC as recommended).** Is using the existing CPU-overhead `forward_launch` rows for the host stream only consistent with ADR 0002 / D-NO-STEP-COST-SHORTCUT, if R1 shows eager host time is material? If you think it is not, say what the design should do instead.
6. **Training CSV columns.** Union and span are kept in pilot P1h and in the raw interval records for audit, but not as training-CSV columns, since no consumer reads them. Do you accept this departure from your F001 correction?
7. **Routing identity (§3.7, A13, files 20-23).**
   - Is the rule correct as stated, including the branch that is not reached (lanes that disagree on the uniform token count)?
   - Is deferring the first-layer device pricing of real V4 lanes to the first pre_moe room, while their host preparation stays lane-local, faithful to the native order? Does it break timing elsewhere, for example the launch-start alignment in `frontier/scheduler/utils/sync_entry.py:113-131`, pipeline stages, or the PREFILL path?
   - Is anything else in the V4 workload still lane-local where native vLLM uses the synchronized descriptor?
   - Is A13 sufficient to show identity with native balanced routing?
8. **Smallest complete change.** Is anything in §6 (23 files, about 2,740 lines) unnecessary, or still missing?
9. **Order and resources.** Are the H200-first order, the stage gates and the GPU caps in §7 reasonable? Are the H200 rules (step_main, non-preemptible, running plus queued at most 64 GPUs, soft gates for runs of 32 or more GPUs) stated correctly?

If you rerun any script on the native traces or counters, use a memory limit (`ulimit -v 31457280`) and a timeout.

If you have no remaining material findings, give the verdict `no_findings`. Otherwise give concrete corrections. Do not edit any file.
