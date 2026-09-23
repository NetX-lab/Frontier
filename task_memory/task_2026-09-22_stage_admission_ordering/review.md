# Stage admission ordering under pipeline parallelism — Review record

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Round 2 dispositions recorded (R-10): 14 findings applied and verified, R2-06 recorded as pre-merge step P6. |
| 2026-09-23 | Round 2: code review of the implementation at `ecff89a` recorded and posted to PR 36; findings verified, fixes deferred by the owner. |
| 2026-09-23 | Created. First external plan review of PR 36 at `a6ec6a6` recorded; each finding re-checked against `1f694f7` source, with a disposition and the place it was applied. |

## Round 1: plan review of PR 36 at `a6ec6a6`

| Item | Value |
| --- | --- |
| Component / phase | Plan and design, before P0; no source change on the branch |
| Reviewer | External review agent, started from `review_prompt.md` |
| Inspected by the reviewer | `stage_execution_context.py`, `replica_stage_schduler.py`, `sync_entry.py`, `base_replica_scheduler.py`, `stage_contexts.py`, `stage_wakeup.py`, `batch_stage_end_event.py`, `replica_stage_schedule_event.py`, `base_event.py`, `round_robin_cluster_scheduler.py`, `metrics_store.py`, `batch_stage.py`, the three admission test files, `AGENTS.md`, and the task records |
| Reviewer's recommendation | Conditional GO for option B. P0 may proceed. Before P1: fix C4 and the matrix outcome classes, name the stage ledger for C3, publish the reproduction inputs, and qualify the capacity-1 and option-A claims. |
| Re-check | Every source anchor below re-read on `1f694f7` in this worktree on 2026-09-23. No simulator was run. |
| Owner instruction | "核实每个comments，采纳高价值和必要决策，修复完善docs，暂不执行" (requirements R-5) |

### Findings and dispositions

| # | Reviewer verdict | Re-check against source | Disposition | Applied in |
| --- | --- | --- | --- | --- |
| 1 | Agree with the diagnosis; the cited admission loop is the wrong branch | Confirmed. `base_replica_scheduler.py:893` is the unified `DECODE` loop. The co-location reproduction runs the `MONOLITHIC`/`PREFILL` `else` branch at `:1037-1054`, with the loop at `:1039`. Both loops use the same `num_running_batches < num_stages` bound. | Adopted. Anchor corrected. The shape table is labelled author-reported until P0 republishes it from the case inputs. | `design.md` "The defect" and shape table |
| 2 | Agree with B; make `remove(ticket)` explicit; test both sides of the EP boundary | Confirmed. `try_acquire` ends with `popleft()` at `stage_execution_context.py:337`, which would remove the wrong ticket once a non-head ticket can be admitted. `cancel` already uses `self._ready_fifo.remove(ticket)` (`:456`). | Adopted. The sketch removes the admitted ticket with `remove(ticket)`. P2(a) adds a `full0, full1, wave0, full2` contract test. | `design.md` "Recommended rule"; `plan.md` P1, P2(a) |
| 3 | Needs evidence. At capacity 1 the context API does change. The unchanged-behaviour claim belongs to the callers. | Confirmed on four points:<br>(a) On an idle capacity-1 context with FIFO `[full0, full1]`, the current rule refuses `full1` and B admits it.<br>(b) `DenseFFNBatchGroup` takes `global_id = _batch_group_creation_counter` (`round_robin_cluster_scheduler.py:1097,1118`). It gets one full-stage ticket (`:1138`) and is queued on the one full-stage scheduler per replica (`:1100`).<br>(c) EP child batches share one `EP_WAVE` ticket (`:1052-1057`).<br>(d) New fact: `enqueue_ep_wave` has no other caller. Queued EP waves therefore exist only on `DECODE_FFN` contexts. On `MONOLITHIC`/`PREFILL`/`DECODE`, `EP_WAVE` is only an active-scope transition of owners that were already admitted. | Adopted. The section is retitled and restated as a caller-level condition, with the API-level change stated explicitly. P2(a′) adds the control with two successive dense FFN groups and a neighbouring EP group. No capacity-1 special case is added. | `design.md` "Where behaviour is expected to stay unchanged"; `plan.md` P2(a′) |
| 4 | Needs evidence. Acquisition emits no wake, but the exact stall trace for option A is not established. | Confirmed on four points:<br>(a) `BatchStageEndEvent` emits the releasing lane's own retry (`batch_stage_end_event.py:139-146`) before its sibling retries (`:148-158`).<br>(b) `build_stage_wakeup_events` orders siblings by lane key, not by FIFO position (`stage_wakeup.py:30-32`).<br>(c) The queue key is `_priority_number = (time, id, event_type)` (`base_event.py:63-64`, `simulator.py:1268`). `BaseEvent.__lt__` (`:66-70`) compares type before id.<br>(d) A refused attempt returns `[]` (`replica_stage_schedule_event.py` "No batch to schedule" branch). | Adopted. A stays rejected on design grounds. The first-draft trace is now labelled an unverified hypothesis and is not pursued, because B does not depend on it. No acquisition wake-up is added. | `design.md` Options table, row A |
| 5 | Disagree with C4 as written | Confirmed. C4 required that all unit tests pass, while G2 expected a baseline failure set of 84 imported from another checkpoint. | Adopted. C4 is replaced with the reviewer's wording, verbatim. G2 now compares node id → outcome per suite against a fresh run of the base source in the same environment. | `plan.md` C4, §4.6 |
| 6 | Needs evidence. The liveness claims are too broad, and a mixed-phase scope boundary is missing. | Confirmed on three points:<br>(a) `attn_dp=2, PP=2` with 3 requests completes on main (author-run log), so the shape alone does not imply a drain.<br>(b) The shared forward across mixed prefill and decode source lanes is PR 35 W3 (`65ed8a7`), which is not on main.<br>(c) New fact: a `MONOLITHIC` request with `decode_tokens=1` completes at the prefill boundary (`request.py:1286-1293,1379-1384`), so it gives a prefill-only witness. | Adopted. The drain condition is restated as a queued-ticket arrangement. Mixed-phase failures become a stop-and-report boundary. C1 witnesses come from a phase-controlled prefill-only group. Composition with PR 35 is checked in the parent task before Step 9 resumes. | `design.md` "The defect", "Scope boundary"; `plan.md` C1, C6, G3a/G3b, §6 |
| 7 | Disagree with the matrix as written | Confirmed on three points:<br>(a) G3 labelled every `PP>1` cell "drain → complete".<br>(b) P3 allowed differences only in C3.<br>(c) DP2/PP3 was replaced by DP4/PP3.<br>New fact: the node-size rule (`parallel_semantics.py:236-262`) is applied only when materializing `collective_sim` (`cluster.py:203`) and `astra_sim_analytical` (`cluster.py:267`). `analytical` has no such rule, so `attn_dp=2, PP=3` on 6 devices is constructible there. The earlier probe used the default `astra_sim_analytical` (`config.py:2494`), and there it was rejected (parent W9-02). | Adopted. P0 now classifies outcomes into four classes. Acceptance runs on three separate paths: U unchanged, L repaired liveness, T timing. Synthetic cases set `AnalyticalCCBackendConfig` explicitly. A concrete case list is published. | `plan.md` §4.1-§4.4, P3 |
| 8 | Agree; name the ledger and measure overlap duration | Confirmed. `frontier_stage_batch_ledger.jsonl` rows carry `cluster_type`, `replica_id`, `stage_id`, `execution_scope`, `replica_local_id`, `stage_start_ts` and `stage_end_ts` (`metrics_store.py:4401-4422`). `execution_scope` is `ATTN_DP_LANE` for lane rows outside `DECODE_FFN` (`:1510-1521`). Capture defaults to on (`config.py:1202-1205`). The file is written by `plot()`, which runs only with `write_metrics=True` (`metrics_store.py:62-67,2200-2218`). | Adopted, with the reviewer's metric definition. The fixture sets `write_metrics=True`, because the earlier probe's `write_metrics=False` would have produced no ledger. | `plan.md` C3, §4.5 |
| 9 | Needs evidence. P0 must make the evidence reproducible. P2(b) and P2(c) need correcting. | Confirmed. The reproduction scripts exist only in the session scratchpad. P2(b) covered only the first blocked group. P2(c) said "drains on main" for the dense fixture too. | Adopted. P0 now has an artifact list per case, and DRAINED cases get a state report instead of a hash. P2(b) runs through release and restore into the next group, in both lane orders. P2(c) gives per-fixture base expectations. | `plan.md` P0, P2(b), P2(c), §4.3 |
| 10 | Agree; narrow the queue bound | Confirmed. The per-lane bound comes from the admission loops (`:893`, `:1039`). `DECODE_FFN` queues are fed by M2N groups and have no such bound. | Adopted. The bound is restated for the shared-lane contexts only. | `design.md` "Recommended rule" |

### Items not adopted, and why

- **A finite event trace for option A (finding 4, optional).** Not produced. It would require implementing A in order to reject it, and the reviewer states that B does not depend on it. The trace stays labelled unverified.
- **None of the required corrections was declined.**

### New facts found during the re-check (not in the review)

1. Queued EP waves exist only on `DECODE_FFN` (see finding 3(d)). On the shared-lane contexts where the defect lives, B's "EP wave queued ahead" clause never fires, and B reduces to "admit any queued full-stage ticket within capacity and seal".
2. Sibling wake-ups follow lane-key order (`stage_wakeup.py:30-32`), not FIFO order. At `PP=1` with `attn_dp ≥ 3`, one release can wake two idle siblings whose tickets are queued in the opposite order. The current rule refuses the first sibling woken. So `PP=1` cells with `attn_dp=4` are expected, not guaranteed, to stay unchanged. A difference there stops the work for diagnosis (plan P3). It is not accepted automatically.
3. The ledger needs `write_metrics=True` (see finding 8).

### Status after round 1

Docs corrected. P0 has not started, per the owner's "暂不执行". The next step is the owner's decision to start P0.

## Round 2: code review of PR 36 at `ecff89a`

| Item | Value |
| --- | --- |
| Component / phase | Implementation after P4: the rule, P2 tests, case matrix, vLLM comparison tools, and task evidence scripts; diff `1f694f7..ecff89a` |
| Reviewer | `/code-review` skill, run as a forked review agent |
| Inspected by the reviewer | The full PR diff; three touched unit files run on the branch (180 passed); a short script reproduced R2-01 on both trees |
| Re-check | Each finding re-read against the cited source on `ecff89a`, 2026-09-23. R2-01 was reproduced again here. R2-02 numbers were read from `analysis/co_execution_decomposition_sa-pp-20260923b.json`. R2-09 ordering was checked in Python. |
| Owner instruction | "review pr36，将review comments提交到该remote repo的pr36上，暂不执行修复。" (requirements R-9) |
| Posted | https://github.com/NetX-lab/Frontier/pull/36#pullrequestreview-5286523149 (event `COMMENT`, 15 inline comments on `ecff89a`) |

### Findings

| Id | Anchor | Verdict | Finding |
| --- | --- | --- | --- |
| R2-01 | `stage_execution_context.py:351` | confirmed, reproduced | `try_acquire` on an already-active full-stage ticket runs off the scan and `remove` raises `ValueError` ("not in deque"); base returned `False`. The only production caller checks `owns()` first. |
| R2-02 | `compare_lanes.py:43` | confirmed | The D-9 rationale ("duration variance, not admission") is not fully supported. Start offsets exceed end offsets in 3 of 6 dense rounds, and `M5_equal_durations` is 0.667–0.911. The union-minus-overlap identity in `decompose_co_execution.py` overcounts disjoint pairs. |
| R2-03 | `stage_admission_matrix.py:255` | plausible | Before/after cases are offline co-location only. There are no PDD or online cells at `attn_dp>1, PP>1`. |
| R2-04 | `compare_lanes.py:99` | confirmed | `vllm_placement` `ok` ignores `unseen`, so missing placement logs still pass. |
| R2-05 | `compare_lanes.py:182` | confirmed | V1 and dense V4 fold the base negative control into the vLLM MATCH status, so a rerun against a fixed base reports MISMATCH. |
| R2-06 | `.gitignore:173` | confirmed | The task-directory exception publishes records into `main` on merge, reversing `26b490a`. D-5 has no pre-merge removal step. |
| R2-07 | `stage_admission_matrix.py:581` | confirmed | No `subprocess.run` timeout in `_run_one` / `_run_recipe_case`. |
| R2-08 | `stage_admission_matrix.py:445` | confirmed | `work/<case_id>` is shared across sets, so concurrent sets delete each other's outputs. |
| R2-09 | `vllm_burst_driver.py:71` | confirmed | `differing` (sorted as `Path`) is compared with `expected` (sorted as `str`), so some identical sets are rejected. |
| R2-10 | `vllm_burst_driver.py:95` | confirmed | `apply_patch` skips unknown-tag lines without counting them, and `hunks[target] = []` drops a repeated file's earlier hunks. |
| R2-11 | `stage_execution_context.py:342` | confirmed | The scope branch is repeated and the FIFO is scanned twice (simplification). |
| R2-12 | `test_stage_execution_context.py:136` | plausible | Capacity-1 contexts lose the context-level full-stage insertion order. This should be stated as a contract change, not as "unaffected". |
| R2-13 | `stage_execution_context.py:345` | design note | On shared-lane contexts the FIFO no longer orders admission. `queued_tickets` / `admission_seq` still read as an ordered queue there. |
| R2-14 | `evidence/step9_probe/probe_main.py:15` | confirmed | A hard-coded worktree `ROOT` is put first on `sys.path`, so a #35 rerun would import this tree. |
| R2-15 | `calibration/.../analysis/synthetic_check.py:7` | confirmed | A hard-coded scratch `BASE` bypasses `matrix_root()`, and reusable probes live under `task_memory/` rather than `tests/`. |

The owner first deferred fixes ("暂不执行修复"), then decided (R-10):
"确认，执行上上述修复； R2-02 采纳你的推荐；R2-03需要补充  PDD+online（如果你认为pd-af+online有必要，请一并补充）".

### Dispositions (R-10)

Evidence for each row is in the test report §8.

| Id | Disposition | Where | Verification |
| --- | --- | --- | --- |
| R2-01 | Fixed: an active ticket is refused, as on the base | `1661bf1` | unit test; `after` vs `after-r2` 98/98 byte-identical |
| R2-02 | Rationale restated with both sources; the script's identity is limited to overlapping pairs; a derived barrier-aligned M5 added. No GPU job. | `e35242f`, `plan.md` D-9 (b), test report §5.3 | reran on runs a and b: no disjoint pair; aligned M5 dense 0.66–0.98, MoE 0.988–0.994 |
| R2-03 | Fixed: G8 PDD offline, G9 PDD online, G10 co-location online, G11 PD-AF `PREFILL_PP=2` (50 cases). Online burst cells added because MONOLITHIC/PREFILL place incremental arrivals on lane 0 on `main` (PR 35 W2). | `a8e8d8a` | 12 more base deadlocks repaired; 0 STOP |
| R2-04 | Fixed | `e35242f` | unit test |
| R2-05 | Fixed: rows N1 and N4, `negative_control_holds` | `e35242f` | unit tests; run b rerun PASS, controls hold |
| R2-06 | Recorded as pre-merge step P6; not executed | `plan.md` §7 | waits for the owner |
| R2-07 | Fixed | `a8e8d8a` | a 2 s timeout recorded as `other_failure` |
| R2-08 | Fixed by a set lock; the shared `work/` path stays for byte identity | `a8e8d8a` | a concurrent run fails at once |
| R2-09 | Fixed | `e35242f` | unit test |
| R2-10 | Fixed | `e35242f` | unit tests |
| R2-11 | Fixed with R2-01 | `1661bf1` | as R2-01 |
| R2-12 | Stated as a contract change | `design.md` round-2 notes, PR body | G1 PD-AF and G11 byte-identical |
| R2-13 | Docstring states the FIFO meaning; the EP-only queue variant deferred | `1661bf1`, `design.md` | review |
| R2-14 | Fixed | `e35242f` | C6 probe 6/6 with `PYTHONPATH` only |
| R2-15 | Replaced by `tests/unit/test_stage_admission_pp_tools.py`; `synthetic_check.py` removed | `e35242f` | 9 passed; 7 fail on the `ecff89a` tools |
