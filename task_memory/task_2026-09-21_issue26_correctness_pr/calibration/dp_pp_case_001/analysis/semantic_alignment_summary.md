# Semantic alignment summary: dp_pp_case_001

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Post-review corrections from workflow `wf_7606e14e-f10`: S10 scope, S39 impact, steady segment unclassified. No status change. |

Table: `semantic_alignment_table.csv` (43 rows: 37 MATCH, 5 MISMATCH, 1 UNSET).
Status: `analysis_state=COMPLETE`, `status=PASS`,
`correction_state=not_applicable` (`semantic_alignment_status.json`).

The Frontier values come from the configuration Frontier actually built:
`runs/frontier_g4/*/effective_settings.json` for the G5 runs, and the earlier
pre-check directories for the rows fixed before G4. Both sides read the same
`inputs/engine_g4.json`, so a setting cannot drift between them. The
`vllm_anchor` column cites the vLLM default that applies when a setting is
not passed.

- Parallel domains (S01-S07). DP=2 is the request-owner lane count. Attention
  TP is 1. With expert parallelism on, vLLM sets the MoE TP to 1 and the EP size
  to TP x DP = 2. The case is one pod with one API server.
- Scheduler (S14-S24) and coordinator/frontend constants (S25-S29) match.
  `max_num_batched_tokens` is 20448, set from the G3 forward times by plan
  §18.19 rule 1 (S14).
- S17 is now MATCH. There are 284775 KV blocks per engine, the smaller of the
  two PP workers' counts in the G4 startup log. Frontier uses the same value.
- MISMATCH:
  - S30: IPC latency is not modeled.
  - S31: Frontier uses dummy timing, per D-e. The value is 0.8943 ms. It
    matches the mean G4 first-chunk duration, 111.00 ms, with 111.14 ms in
    Frontier. The dummy time does not depend on token count, so a 32-token
    forward also takes 111 ms in Frontier. Natively it takes 19.7 to 33.8 ms.
  - S33: routing followed arrival and delivery order. In G4 bursts a-c, the
    40896-token body routed last, about 1 ms after the first short request.
    Burst d, spaced 6 ms apart, routed in trace order. Frontier routes in
    trace order.
  - S42: the DP wave start and dummy forwards are not modeled. G4 is
    consistent with both:
    - in burst d, engine 1 has no step-0 record;
    - engine 0 has a record gap after its empty-schedule iteration.
  - S43: after an iteration that scheduled nothing, vLLM blocks on its oldest
    batch, and requests that arrive meanwhile wait. G4 shows these delays, for
    example burst a's b4 and b2, which were admitted 47 ms after routing.
    Frontier admits into any free pipeline slot.

  S42 and S43 do not enter the T1 replay, which feeds native receipts and keys
  to Frontier's coordinator and frontend. `analysis/workflow_gap_table.csv`
  labels every natural-history difference in the bursts by its first cause.
  The steady segment is not classified (post-review erratum 2026-09-24,
  `workflow_gap_summary.md` "Post-review errata").
- UNSET: S39. There are no vLLM MoE routing records. That blocks only an E2E
  numeric gate, and this case has none.

Post-review corrections (2026-09-24, workflow `wf_7606e14e-f10`), with no
status change:

- S10 compares weight loading only. vLLM's `load_format=dummy` replaces the
  weight values and still measures GPU time. Frontier loads no weights. The
  Frontier dummy predictor replaces timing instead, and that is row S31
  (MISMATCH), not S10.
- S39's impact "none on placement" holds only under dummy timing. With trained
  predictors, `balanced` routing fails predictor construction on the checked-in
  `h800/Qwen3-30B-A3B-tiny` profiles, which hold only `uniform_topk` rows.
