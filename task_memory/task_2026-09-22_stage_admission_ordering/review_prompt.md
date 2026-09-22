# Resume prompt for the reviewing agent

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Round 2: the prompt now asks the reviewer to verify the round-1 dispositions in `review.md` and the corrected plan. |
| 2026-09-23 | Created for round 1. |

Copy everything below the line into the review agent's first message.

---

You are continuing the review of draft PR https://github.com/NetX-lab/Frontier/pull/36 on `NetX-lab/Frontier`, branch `fix/stage-admission-ordering`, base `main` at `1f694f7`. This is round 2. In round 1 you reviewed the plan at `a6ec6a6` and gave a conditional GO for option B, with ten findings. The owner had every finding verified against the source and the records corrected. Nothing was executed: no P0 run, no source change.

Start with the repository's `AGENTS.md`. Then read, under `task_memory/task_2026-09-22_stage_admission_ordering/`, in this order: `review.md` (your findings, the source re-check, the disposition of each, and three new facts found during the re-check), `design.md`, `plan.md`, `requirements.md` (R-5), `progress.md`.

Background, briefly. One `StageExecutionContext` (`frontier/scheduler/replica_stage_scheduler/stage_execution_context.py`) owns each physical `(replica, stage)` and is shared by that stage's attention-DP lanes. Today `try_acquire` admits only the strict FIFO head. At `PP > 1` a lane holds several queued tickets while it consumes one, so a busy lane's queued ticket at the head can block an idle lane that its own sync room is waiting for. Option B, adopted as D-1: a full-stage ticket is refused only by an EP wave queued ahead of it; EP waves keep the strict head rule; the admitted ticket leaves the FIFO by `remove(ticket)`.

What to check, in priority order:

1. For each of the ten findings in `review.md`: is the disposition faithful to what you asked, and is it applied where the table says? Flag anything weakened, misread or missing.
2. The three new facts in `review.md`. Check each against the source:
   - Queued EP waves exist only on `DECODE_FFN`, because `enqueue_ep_wave`'s sole caller is `round_robin_cluster_scheduler.py:1052`.
   - Sibling wake-ups follow lane-key order (`stage_wakeup.py:30-32`), which makes `PP=1, attn_dp=4` an expected-unchanged class rather than a guaranteed one.
   - The stage ledger is written only with `write_metrics=True`.

   Also check one correction made during the re-check: the first draft's dense "lanes serialized" label is withdrawn, because from source the base already overlaps lanes after the first release. P2(c)'s dense assertion was changed to a same-start condition for that reason.
3. `design.md` "Where behaviour is expected to stay unchanged". Is the caller-level condition correct and sufficient for `DECODE_FFN` and `DECODE_ATTN`, and is it stated with the right strength?
4. `plan.md` §4: fixture, case list, outcome classes and signature, acceptance paths U/L/T, the ledger metric, and the test-identity comparison. Can P0 run from this text alone? Are the base hypotheses and paths consistent with C1–C4? Is any stop condition missing?
5. P2(a), (a′), (b) and (c). Does each test fail on the base for the stated reason and pass after P1? Is any of them redundant with an existing test?
6. Fit with the owner's core-module gates: readability, no hard-coding, no temporary patches, no over-defensive branches, no redundant mechanisms, plain domain names. This covers the planned harness `tests/e2e/stage_admission_matrix.py` as well as the one-file rule change.

Report findings as a numbered list with a source or record anchor (`path:line`) and a verdict per item (agree / disagree / needs evidence). End with a one-paragraph recommendation on whether P0 may start as written. Do not change source or push; the owner decides.
