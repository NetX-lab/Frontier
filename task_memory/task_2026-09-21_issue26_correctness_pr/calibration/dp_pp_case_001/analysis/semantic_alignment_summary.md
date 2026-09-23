# Semantic alignment summary: dp_pp_case_001

Table: `semantic_alignment_table.csv` (41 rows). Status: `analysis_state=COMPLETE`,
`status=PASS`, `correction_state=not_applicable` (`semantic_alignment_status.json`).

The Frontier values come from the configuration Frontier actually built
(`runs/frontier_precheck_2ms/*/effective_settings.json`). Both sides read the
same `inputs/engine_g4.json`, so a setting cannot drift between them. The
`vllm_anchor` column cites the vLLM default that applies when a setting is not
passed.

- Parallel domains (S01-S07). DP=2 is the request-owner lane count. Attention
  TP is 1. With expert parallelism on, vLLM sets the MoE TP to 1 and the EP size
  to TP x DP = 2. The case is one pod with one API server.
- Scheduler (S14-S24) and coordinator/frontend constants (S25-S29) match.
  `max_num_batched_tokens` is provisional: G3 measures forward time and the G4
  value is set from that measurement.
- Declared MISMATCH:
  - S30: IPC latency is not modeled.
  - S31: Frontier uses dummy timing, per D-e.

  The T2 window keeps its edges tens of ms away from the probe. G5 labels any
  reordering by its first cause.
- UNSET, each resolved by a run:
  - S17: the KV block count comes from the G4 startup log.
  - S33: the burst's route order is measured in G3 and qualified in G4.
  - S39: there are no vLLM MoE routing records. That blocks only an E2E
    numeric gate, and this case has none.
