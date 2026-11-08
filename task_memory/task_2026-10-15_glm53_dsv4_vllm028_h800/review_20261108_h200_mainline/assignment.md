# Task: audit whether the H200 mainline (DSV4 simulation support + accuracy calibration) is actually complete

## Goal

The lead claims the H200 mainline of this task is finished: Frontier can simulate DeepSeek-V4-Flash (DSV4) on H200,
and the accuracy calibration against a native vLLM 0.28 reference passes the four-metric gate at both 32 and 64 GPUs.

Decide independently whether that claim holds. Produce a review that answers three questions:

1. **Complete?** For each acceptance criterion that is actually in force, is it satisfied by artifacts that exist in
   this repository, or only asserted in prose?
2. **Errors?** Are there factual, numerical, or logical errors in the recorded results, their arithmetic, their units,
   their provenance bindings, or the code that produced them?
3. **Omissions?** Was any critical execution skipped, quietly substituted, or silently relaxed — an acceptance
   criterion weakened, a threshold widened, a required repeat dropped, a required pre-condition supplied after the
   fact, a diagnostic result presented as a formal one?

Audit the artifacts. Do not restate the records' own summaries back as findings. Where a record states a number, find
the file that number came from and check it. Where a record states a check passed, find the check's output. Where a
record states code was changed, read the committed diff. A claim you could not verify is itself a reportable finding;
say so instead of accepting it.

Do not assume the lead is right, and do not assume the lead is wrong. Both a confirmed PASS and a defect are useful
outcomes; a vague "looks reasonable" is not.

## Background you need

- Subject: branch `feat/glm53-dsv4-vllm028` in this worktree. Model DSV4 (DeepSeek-V4-Flash), hardware H200, native
  baseline vLLM 0.28, PD-disaggregated serving. `<TASK>` = `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`.
- Active scope is DSV4 only. GLM-5.3 work is **paused by explicit user decision** (`D-DSV4-ONLY-GLM-PAUSED`) and is
  **not** part of completion. Do not report paused GLM items as omissions.
- Two case sizes are in scope on H200: **P2-32** (32 GPUs: prefill 16 / decode 16, TP8/DP2/EP16/PP1 per role) and
  **P2-64** (64 GPUs: prefill 32 / decode 32, TP8/DP4/EP32/PP1 per role). Both are recorded as formal PASS.
- The workload is frozen: 128 requests, 6,144 prompt tokens, 1,024 decode tokens, QPS 1.0 with fixed one-second
  arrivals, three admitted repeats per case.
- The gate: `abs(frontier - vllm) / abs(vllm) <= 0.10`, independently for mean client TTFT, mean per-request TPOT,
  throughput and mean per-request E2E, **in every admitted repeat**, under the `D-WARM-TARGET` warm reading, with the
  unchanged all-request reading retained alongside.
- H800 is explicitly a later scope: "H200 results never qualify H800." Its absence is not an H200 omission, but check
  that no H200 evidence is used to qualify H800 anywhere.

## Authoritative entry points

Start here; these define what "complete" means and what is claimed:

- `<TASK>/tickets/19-comparison-and-acceptance.md` — the authoritative acceptance definition. `## Milestones`
  (line ~142) lists `canonical_metric_binding`, `h4a_diagnostic_comparison`, `frontier_reference_run`,
  `comparison_and_calibration`, `promotion_32_to_64`, `final_acceptance`; `## Acceptance` is at line ~209. Note the
  ticket's own `**Status:**` line (~127) says `PARTIAL`. Reconcile that with the completion claim: either the open
  checkboxes are genuinely open, or the status is stale. Say which.
- `<TASK>/tickets/17-dsv4-canonical-case.md` and `<TASK>/tickets/18-native-ground-truth-h200.md` — case binding and
  native ground truth.
- `<TASK>/plan.md` section `## Current execution route — Codex CLI, H200 operator path (2026-10-31)` (line 196) and
  its `Current dependency:` chain (line ~217). Everything below `## Historical execution route` (line 265) is
  superseded history — read it only to resolve a specific question, not as current state.
- `<TASK>/requirements.md` — the `D-*` decision records (139 of them). These are the only source of authority.
- `<TASK>/test_report_2026-11-07_p2_64_generation08_formal.md` — the P2-64 formal PASS report, i.e. the final result.
- `<TASK>/issues.md` — `I-001` … `I-015`; the headings are indexed at the top of the file.
- `<TASK>/progress.md` and `<TASK>/summary.md` — chronological record and English archive.

**These files are large.** `progress.md` is 820 KB / 3,210 lines, `requirements.md` 280 KB / 1,809 lines,
`summary.md` 169 KB. The task directory holds 206 `test_report_*.md` files and 119 subdirectories. Read targeted
sections by line range or search; do not try to read everything, and do not let breadth of reading substitute for
depth of checking. Prioritise: the acceptance definition, the final results and their raw inputs, and the code that
produced them.

## The final numbers to check

`<TASK>/test_report_2026-11-07_p2_64_generation08_formal.md` publishes, for P2-64 generation08
(RJob `exp-1108-002857-059466`, run tag `dsv4-h200-p2-64-formal-20261107-08`):

| | TTFT ms | TPOT ms | E2E ms | THPT tok/s |
| --- | ---: | ---: | ---: | ---: |
| Frontier | 684.31 | 25.30 | 26,561.44 | 5,975.25 |
| vLLM pooled | 705.20 | 24.75 | 26,026.14 | 6,001.12 |

Worst relative error 3.16 %, at repeat02 TTFT; all 16 errors within 10 % (12 per-repeat plus 4 pooled).

> Correction recorded 2026-11-08, after the review: this line originally said "all 20 errors", repeating the
> overcount that the review raised as finding F3. The actual quantities are 4 Frontier values, 12 per-repeat vLLM
> values, 4 pooled vLLM values and 16 relative errors. The brief's wrong count was not a review premise; the
> executor identified and reported it independently.

Per-repeat row CSVs are under
`<TASK>/h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2/gate/rows/`.
The lead already recomputed these from the row CSVs alone with
`<TASK>/h200_mainline/p2_64_formal_retry08/headline_recompute.py` (output
`headline_recompute.json`, `VERDICT=PASS WORST=3.16% AT=repeat02/ttft`). The pooling convention it uses, and that the
comparison harness uses, is: **latency metrics pool as the mean of the per-repeat means; throughput pools as total
tokens divided by total span summed across repeats** (not the mean of per-repeat throughputs; and *not* one span
concatenating the repeats, which would wrongly include the idle gaps between them). TPOT excludes rows with
`num_decode_tokens <= 1`.

Treat that recomputation as a claim to audit, not as evidence. Independent angles worth taking: does the row
population match the frozen workload (counts, exact prompt/decode lengths, arrival schedule)? Are the Frontier and
vLLM sides measuring the same interval with the same definition? Does the warm reading's paired-exclusion rule get
applied symmetrically to both sides and uniformly across repeats, with the exclusion IDs and their mechanism evidence
retained? Is the all-request reading still recorded unchanged, including the historical repeat01 TTFT failures that
earlier generations preserved? Does the native side actually come from the Succeeded job whose receipt is bound, with
432/432 requests?

Also check the P2-32 formal acceptance and the 32→64 promotion receipt
(`<TASK>/h3o_operator_costs/formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json`, issued
2026-11-02T07:38:49Z). Ticket 19 requires the pre-launch 30 % receipt to **exist before** the formal 64-GPU native
submission and forbids supplying it retrospectively; generation08 was submitted 2026-11-08T00:28:57Z. Verify the
ordering and the receipt's own predicate (every finite error <= 0.30 in every required repeat, no averaging across
metrics or repeats, a zero native value gives an undefined error and cannot qualify).

## The simulation-support side, not only the numbers

The task has two halves and the second is easy to skip. Audit the DSV4 support in `frontier/` as committed code, not
through its reports: whether the DSV4 model/architecture profile, MLA attention, MoE routing with EP, the NIXL KV
transfer path, the memory planner, and the CUDA-graph and quantization handling are actually implemented and reachable
on the paths the formal runs used; whether anything the reports credit was left a stub, a hard-coded constant, a
benchmark-specific special case, or a default that silently differs from the native effective configuration; and
whether any arbitrary multiplier or tuned offset entered a priced path. Ticket 19 forbids exactly that:
"Fix causes, not symptoms. No arbitrary multipliers, no benchmark-specific numerical patches, no duplicated predictor
logic, no widened threshold."

Named code anchors the records credit, worth reading as diffs: `0fb8955d` (NIXL wire pricing — selected pages times
`physical_block_size_bytes`), `a69a5213` (ingress option A / B29), `57caf2f7` (native-harness peer heartbeat),
`f33306d4` (the scoped startup-harness verification behind generation02). Note that the `a06a72a4…` the records cite
beside `f33306d4` is a **native vLLM source SHA** bound into the harness `runtime_version`
(`a06a72a4326cba02234df6cfbdc21b78d3a15d26`, per `<TASK>/test_report_2026-11-02_startup_harness.md`), not a commit in
this repository — it does not resolve here. Check that every such cross-repo identity is bound to evidence that exists
and is distinguishable from a local commit. Also check the project's own
development gates in `AGENTS.md` (the 2,000-line module limit and its recorded exceptions in
`docs/general/module-size-exceptions.md`; preserved behaviour and simulator fidelity for refactors) against what the
calibration changes actually did.

A known, recorded limitation to verify rather than rediscover: the EP dispatch kernel count is **deferred**
(`D-WARM-EP-DEFER`) — native traces show three dispatch kernels per MoE layer only with unequal DP lanes, the registry
prices one, and the undercount is at most 0.0468 ms per step. Judge whether that deferral is honestly scoped and
whether it can affect either formal verdict.

## What the lead already believes is open

State your own verdict on each of these, and look for others the lead has not named:

1. **GPU-hour accounting.** Ticket 19's `final_acceptance` requires the GPU-hours actually spent. 35 earlier jobs are
   exact at 158.271667 container GPUh, but generation01/02 are missing seven finish clocks, generation03/04 have null
   exact-clock receipts, and one generation has an observation estimate (34.0188… GPUh) rather than measured clocks.
   The lead calls this an accounting limitation with no numerical consequence for the metrics. Check that framing: is
   any *metric* result dependent on these clocks, and does the checkbox's own wording permit closing the case with
   them open?
2. **H800 P0/P1.** Retained for later, needs a concrete package/resource approval, and H200 evidence never qualifies
   it. Confirm it is genuinely out of H200 scope and not load-bearing for any H200 claim.

Beyond these, the places defects most plausibly hide: a criterion whose wording was edited after the result was
known; a "diagnostic" input used in a formal gate (`D-NO-STEP-COST-SHORTCUT` forbids whole-step tables in the formal
gates — check it held); a coverage miss or `NOT_EVALUABLE` class that a later summary rounded up to PASS; a
provenance binding that points at a different generation's artifacts than the text claims; a hash or manifest recorded
but never actually compared; a repeat or request population that differs between the two sides.

## Scope and authority

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. The whole repository is readable.
- **Writable: only `<TASK>/review_20261108_h200_mainline/`.** Put the report, any scripts you write, and all scratch
  there. This is deliberate: a review must not be able to modify the evidence it audits. If a check would require
  writing elsewhere, do not work around it — report the check as not run and say what it needed.
- Read-only Git inspection (`git log`, `git show`, `git diff`, `git cat-file`) is expected and encouraged. **No Git
  mutation**: no commit, add, checkout, reset, stash, branch, tag, or worktree change. The lead owns commits.
- **No platform, RJob, StepMind, or GPU operation of any kind, and no new GPU run.** Do not submit, query, stop, or
  inspect any cloud job, and do not read or write the cloud volume. Local CPU-only checks only. If a finding would
  need fresh GPU evidence, state it as a recommendation for the lead.
- No package installation, no network fetches, no external writes, no deletions or moves. Leave `~/.pi`, `.cortexkit/`,
  `.pi/`, `launcher.pid.txt` and `remaining_measurements/` untouched. Never print credentials or proxy values.
- Environment: system `python3`. Run it with `-B` so no bytecode is written. If you run a focused pytest to settle a
  specific claim, use `-p no:cacheprovider` and keep any output under your writable directory; whole-suite runs are
  not the point of this review and several pre-existing failures are already recorded in `issues.md`.
- Routine decisions about which artifacts to open and which checks to run are yours. Return scope questions, design
  judgements, and anything needing new user authority to the lead.

## Expected return

Report in English, written to your record directory. Include:

- **A verdict on the completion claim**: `COMPLETE`, `COMPLETE_WITH_LIMITATIONS`, or `INCOMPLETE`, with the reasoning
  that decides it. If the honest answer is that some criteria cannot be settled from the artifacts, say
  `INSUFFICIENT_EVIDENCE` for those specific criteria rather than averaging into a global verdict.
- **A per-criterion table** over ticket 19's milestones and `## Acceptance`: criterion, claimed state, your verdict,
  and the artifact that decides it.
- **Findings, most severe first.** Each finding needs: what is wrong, the exact evidence (`path:line`, artifact path,
  command and its actual output, or commit and hunk), the concrete consequence for the published result, and whether
  it is a blocking defect, a mislabelled limitation, or a record-keeping error. Separate "the number is wrong" from
  "the number is right but unsupported by the cited evidence" — they need different fixes.
- **Omitted critical executions**: anything required that was never run, with what it would take to run it. Exclude
  paused GLM work and later H800 scope.
- **Verification log**: every check you ran, with the exact command, the expectation, the actual result, and
  PASS / FAIL / NOT RUN. State plainly what you did not check and why, and separate your observations from your
  inferences.
- If you find nothing wrong with a major claim, say so explicitly and name the evidence that convinced you. A clean
  verdict with its evidence is a useful result.
