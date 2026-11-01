# Residual Lane Classification

**When to use:** Use when the error budget leaves a part in PP, collective, KV transfer, or M2N transfer time that needs a source classification.

Read `../references/contract.md` (Error-Source Classes) and
`../references/evidence-tiers.md`. Require the latest clean E2E status, the
error budget, the workflow and operator statuses, and native evidence for each
lane under test. Use engine-native observations with verified boundaries; the
following file names describe the existing vLLM profile:

- PP: stage hand-off stamps from the `cpu` probe stage records and the
  `op_timing` run's `pp_boundary.jsonl`;
- collective: all-reduce and all-to-all scope times, `all_reduce_us` from
  `forward_device_timeline.py`, and fits from `vllm_tp_allreduce_bench.py`;
- KV or M2N transfer: the `kv_save_timing` run, connector logs, and fits from
  `nccl_p2p_bench.py` at the case's per-layer message size.

Test each lane with its own identity, topology, scope, overlap, and timing
fields. Remove PP rank copies, EP sub-batches, and duplicate transfer events
without inventing batches. Separate a collective's wire time from its wait for
peer ranks; the wait belongs to the peers' work.

Assign one evidence-backed primary class: `pp_overhead`,
`collective_communication`, `kv_or_m2n_transfer`, `mixed`, or `still_unclear`.
For the first three, resolve the repair lane and receipt through the
`lane_registry` in `../references/registry.json`; `mixed` and `still_unclear`
have no repair lane and stop without a code change. Record both sides' timings,
the numeric gaps, the source type, the ruled-out lanes, the missing coverage,
and the error-source class (a fit is `measurement_input`; a missing or wrong
transfer or overlap rule is `simulation_logic`).

Write `analysis/residual_gap_table.csv`, summary, and status. This entry is
read-only; a classified lane never grants permission for a code change.
