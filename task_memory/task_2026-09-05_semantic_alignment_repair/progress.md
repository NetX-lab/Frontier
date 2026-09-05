## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Logged implementation, validation, and publication results. |
| 2026-09-05 | Recorded user-confirmed continuation for the reviewed semantic-alignment fixes. |

# Progress

- Completed: created `.worktrees/semantic-alignment-repair-20260905` from `main`.
- Completed: restored co-location Replica-local DP lanes and PDD admission.
- Completed: repaired PD-AF scheduler and metrics lane contracts after two deterministic smoke failures.
- Completed: co-location tests `45 passed`; PDD tests `27 passed`; PD-AF tests `86 passed`.
- Completed: co-location, PDD, and PD-AF one-request dummy smokes exited `0` and wrote metrics.
- Completed: opened issue `https://github.com/NetX-lab/Frontier/issues/27`.
- Completed: opened PR `https://github.com/NetX-lab/Frontier/pull/28`.
- Pending outside this task: groundtruth rerun and formal timing parity.

## 2026-09-05 Reviewed-Fix Continuation

- Authorization: user confirmed continuation after PR #28 review.
- Code-change marker: `SEMANTIC_ALIGNMENT_REPAIR_START: reviewed attn_dp CLI, resolver, layout, scheduler, and sync identity fixes`.
- Scope: expose and propagate attention DP, preserve one Replica as one GPU pod, keep logical TP abstraction, route request ownership by `(replica_id, dp_id)`, and isolate DP-lane batch/synchronization identity.
- Out of scope: groundtruth rerun, timing comparison, legacy H800 CSV reuse, per-rank schedulers, and unrelated documentation failures.

### Sub-step 1: CLI and parallel-domain materialization

- Motivation: the reviewed PR admitted `attn_dp > 1` in Python objects but hid the field from the supported CLI, hard-coded the canonical resolver to `attn_dp=1`, and materialized attention collective DP from outer Replica capacity.
- Expectation: formal CLI/config construction and the canonical resolver produce `TP4 x DP2 x EP8`, while collective layout keeps attention DP local to one complete Replica pod.
- Method: exposed `ReplicaConfig.attn_dp`, added resolver propagation and local EP derivation, changed attention layout DP to `mapping.attn_dp`, and added parser/resolver/layout tests.
- Result: `python -m pytest tests/unit/test_parallel_semantics.py tests/unit/test_canonical_parallel_names.py -q -p no:cacheprovider` -> `8 passed`; `frontier.main --help` lists `--replica_config_attn_dp`.
