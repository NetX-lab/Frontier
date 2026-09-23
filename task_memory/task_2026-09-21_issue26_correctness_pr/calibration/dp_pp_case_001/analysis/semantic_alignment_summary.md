# Semantic alignment summary: dp_pp_case_001

Table: `semantic_alignment_table.csv` (43 rows). Status: `analysis_state=COMPLETE`,
`status=PASS`, `correction_state=not_applicable` (`semantic_alignment_status.json`).

The Frontier values come from the configuration Frontier actually built
(`runs/frontier_precheck_2ms/*/effective_settings.json`, and
`runs/frontier_precheck_g4_2ms/*/effective_settings.json` for the G4 chunk
budget). Both sides read the same `inputs/engine_g4.json`, so a setting cannot
drift between them. The `vllm_anchor` column cites the vLLM default that
applies when a setting is not passed.

- Parallel domains (S01-S07). DP=2 is the request-owner lane count. Attention
  TP is 1. With expert parallelism on, vLLM sets the MoE TP to 1 and the EP size
  to TP x DP = 2. The case is one pod with one API server.
- Scheduler (S14-S24) and coordinator/frontend constants (S25-S29) match.
  `max_num_batched_tokens` is 20448, set from the G3 forward times by plan
  §18.19 rule 1 (S14).
- Declared MISMATCH:
  - S30: IPC latency is not modeled.
  - S31: Frontier uses dummy timing, per D-e.
  - S42: the DP wave start and the idle engine's dummy forwards are not
    modeled. A dummy pass advances vLLM's step counter, so peer report keys
    can differ in the natural history.
  - S43: under the batch queue, vLLM blocks after an iteration that scheduled
    nothing, even with room in the queue, and requests that arrive meanwhile
    wait. Frontier admits into any free pipeline slot.

  S42 and S43 do not enter the T1 replay, which feeds native receipts and keys
  to Frontier's coordinator and frontend. G5 labels any natural-history
  difference by its first cause.
- UNSET, each resolved by a run:
  - S17: the KV block count comes from the G4 startup log.
  - S33: G3 showed that the client's dispatch order decides the route order.
    G4 dispatches every burst in trace order, and G5 qualifies each burst.
  - S39: there are no vLLM MoE routing records. That blocks only an E2E
    numeric gate, and this case has none.
