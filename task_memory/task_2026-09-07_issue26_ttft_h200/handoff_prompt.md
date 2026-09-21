## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Added concise English cross-session continuation prompt. |

# Continue the Frontier TTFT Calibration Task

Resume the existing task; do not restart the investigation or create another worktree.

- Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`
- Branch: `task/issue26-ttft-h200-20260907`
- Task directory: `task_memory/task_2026-09-07_issue26_ttft_h200/` within that repository.

## Read first

Read these task documents in order:
1. `handoff.md` — authoritative resume snapshot, environment, completed candidate, commands, module map and remaining execution order.
2. `requirements.md`, the active section of `plan.md`, and the latest `progress.md` checkpoint — user decisions and acceptance gates. Historical pending-approval/status text is superseded by the handoff.
3. `analysis/d019-linear-context-candidate/report.md` — the completed candidate experiment and its validation limits.
4. `analysis/d019-first-forward-integrated/report.md` and `analysis/d019-compute-results/report.md` — actual integrated result and complete operator mapping.
5. `analysis/d019-linear-timing-context.md`, `analysis/d019-linear-context-proposal.md`, `analysis/d019-communication-sweep.md`, and `analysis/d019-communication-fit-review.md` — evidence, limits and supported next choices.

## Resume from actual state

Check `git status`, current HEAD and the recorded artifacts before launching work. At final handoff, HEAD was `fb3ed797`, all CPU/GPU jobs had ended, and only the two pre-existing D005 scripts remained untracked.

The last candidate query completed with actual boundary **54.914898417ms**,17RF fits, three verified changed ops and eight unchanged compute models. Its actual receipt/log/validation are persisted in `analysis/d019-linear-context-candidate/`. Candidate ingestion passed; CUDA accuracy failed. The exporter is already committed. Do not rerun this finished experiment or wait for its former PID616810.

Use `tests/e2e/issue26_predictor_query_audit.py` for the actual bounded first-stage query. Consult the handoff's module map before editing production profiling, predictor or communication code.

## Continue the remaining plan

1. Review the completed candidate evidence and decide the next scoped profiling correction; distinguish diagnostic input from a production profiling correction.
2. Resolve remaining first-forward operator/context/communication residuals, implement only reviewed scoped corrections, and verify a fresh CUDA forward result.
3. Obtain a fresh clean E2E checkpoint after CUDA attribution/validation. Qualify corrected GG shape coverage and small custom-AR behavior before claiming full-case accuracy.
4. Only then integrate independently measured outside-forward CPU/workflow costs and finish official TTFT/TPOT/E2E/throughput validation.

The verified GG+AR result is59.190354384ms versus vLLM batch-only79.307357788/78.118782043ms: the CUDA gate FAILS. Earlier near-parity involved error cancellation. Do not claim completion from a passing isolated check or an expected value.

## Preserve these decisions

- Only the existing4096-prefill/1024-output case; H200 `step_main`, fixed image, uniform routing, both prefix caches OFF, eager execution.
- Keep `collective_sim`/htsim, `nvlink_analytic`, and the current ideal EP abstraction. D020 explicitly defers naive-protocol modeling; do not reopen it implicitly.
- No historical numerical reuse, fitted residual constants, or CPU add-on before CUDA closure. CUDA events can include host-induced device gaps; instrumented spans are not pure kernel time.
- General profiling-context selection is a new shared contract. Present a concrete minimal design and consult the user before implementing it; isolated candidate validation is already authorized.
- Existing parallel A/B/C work is authorized. Assign disjoint ownership, preserve others' edits, and centralize GPU scheduling. Follow the handbooks and company proxy recipe in `handoff.md`.

Keep task docs current, report failures and limitations, and commit verified code sub-steps. Continue until the remaining acceptance criteria are met or a genuine unresolved decision requires the user. Start user-facing replies with `boss YC`; communicate in Chinese and retain English technical terms.
