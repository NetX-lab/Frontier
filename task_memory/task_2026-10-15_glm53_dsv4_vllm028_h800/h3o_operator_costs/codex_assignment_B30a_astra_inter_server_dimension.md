# Task: price cross-server ring steps on the inter-server link in the astra_sim_analytical CC backend (stage 4 assignment B30a, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Created from the F1 per-call check (decision 46, Track 1). |

## Goal

`AstraSimAnalyticalCCBackend` (`frontier/cc_backend/backends/astra_sim_analytical_cc_backend.py`) prices a ring collective as rounds of point-to-point sends between linearized participant ranks. Each send is priced on one physical dimension:

- `_rank_to_physical_address(rank)` returns `(rank % gpus_per_server, rank // gpus_per_server)`, that is `(local index, server index)`.
- `_get_first_differing_dimension` returns the first component that differs.
- `_send_ms` then uses that dimension's topology, bandwidth and latency.

For a ring neighbour pair on different servers, such as 7 -> 8 in a 2 x 8 pod (addresses `(7, 0)` and `(0, 1)`), the local index differs first, so the send is priced on the intra-server link. A ring over the ranks of a multi-server pod therefore never uses the inter-server parameters. Only pairs with the same local index on different servers (0 -> 8) reach them.

Evidence (lead, on the current code, H200 A10 cases with 2 x 8 and 4 x 8 decode pods):

- With `inter_server_latency_us=50` and `intra_server_latency_us=1`, `_send_ms(0 -> 1)` = 0.001 ms, `_send_ms(7 -> 8)` = 0.001 ms, `_send_ms(0 -> 8)` = 0.05 ms.
- A sweep of the inter-server latency over 5, 6, 7 and 8 us gave byte-identical predictions for every EP16 and EP32 collective (`f1_astra_per_call.py`, outputs `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/f1/astra_fc_lat{5,6,7,8}/per_call.csv`).

Expected behavior after the fix:

1. A send between two ranks on different servers is priced on the inter-server dimension: its topology, its size (`cluster_servers`), its bandwidth and its latency, with hops computed from the two server indices.
2. A send between two ranks on the same server is priced exactly as today.
3. A pod that fits on one server (`cluster_servers == 1`) gives byte-identical predictions to today for every collective.

## Approach (lead decisions)

- **Dimension rule.** Price a send on the outermost (highest) dimension in which the two addresses differ. Upstream ASTRA-Sim's `MultiDimTopology::send` picks the first differing dimension because its system layer splits collectives per dimension, so each send there differs in exactly one dimension. Frontier's port runs a flat ring over linearized ranks, so a send can differ in both dimensions; the outermost one is the link the data must cross. Keep the per-send cost form `hops * latency + bytes / bandwidth` on that one dimension. Do not add an intra-server hop to cross-server sends, and do not add a rail or PXN model; those are outside this fix.
- **No new configuration.** The existing fields (`intra_server_*`, `inter_server_*`, `cluster_servers`, `cluster_gpus_per_server`, `placement_order`) already describe the topology. Add no flag, no compatibility switch and no fallback to the old rule; the old rule is the defect.
- **Smallest change.** The fix belongs in the dimension selection of this backend. Rename `_get_first_differing_dimension` to a name that states the new rule, and update its callers. Keep the rest of the backend unchanged.
- **Fidelity.** This is a fidelity fix to a core module, and `astra_sim_analytical` is the direct CLI default backend, so multi-server results under that backend change. The lead keeps it on this branch until YC approves the merge. In the report, list every existing test, golden value or example whose result changes, with old and new values and why the fix explains the difference. Do not change an expected value whose change the fix does not explain; report it instead.

## Scope and context

- Working root: `/data/ycfeng/Frontier/.worktrees/b30a-astra-inter-server`, branch `b30a-astra-inter-server`, HEAD `b0e568f5`.
- Task records: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/requirements.md` (decision 46) and `plan.md`; the F1 check `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_variants/f1_astra_per_call.py` (read only).
- Writable paths:
  - `frontier/cc_backend/backends/astra_sim_analytical_cc_backend.py`
  - `tests/unit`
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30a` (report evidence only)
- Exclusions: no change to other CC backends, to `Cluster` materialization, to the config schema, or to the measured backend. Do not commit; the lead reviews and commits.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `PYTHONPATH=$PWD`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Run commands from the working root with `ulimit -v 31457280` and a `timeout`. Use `/usr/bin/grep`; `grep` is a shell function in the interactive shell.

## Tests

Add focused unit tests for the backend (a new file under `tests/unit` or the existing astra-related test file, as fits):

1. In a 2 x 8 pod, `_send_ms(7 -> 8)` and `_send_ms(0 -> 8)` use the inter-server latency and bandwidth; `_send_ms(0 -> 1)` and `_send_ms(6 -> 7)` use the intra-server ones.
2. A ring all-gather and a reduce-scatter over the 16 ranks of a 2 x 8 pod equal `(N - 1)` rounds of the slowest send of a round, and that slowest send is a cross-server one when the inter-server link is slower.
3. Changing only `inter_server_latency_us` changes the EP16 prediction of a 2 x 8 pod and leaves the TP8 prediction unchanged.
4. A one-server pod (1 x 8) gives the same predictions as before the fix for all-reduce, all-gather, reduce-scatter and all-to-all; pin the values computed on HEAD `b0e568f5`.
5. With the inter-server topology `Ring` and 4 servers, the hop count of a cross-server send follows the server-index distance.

Run the new tests and the existing related suites:

```bash
pytest tests/unit/test_cc_backend_replica_local_layout.py tests/unit/test_cc_backend_default_collective_sim.py tests/unit/test_examples_profiling_contracts.py tests/unit/test_module_split_boundaries.py <new tests> -q -p no:cacheprovider
```

Also run the astra refactor-fidelity case (`tests/e2e/refactor_fidelity/cases.py`, case `pdd_dense_offline_astra_sim_analytical`) with its existing runner, and state whether its pod spans more than one server and whether its outputs changed.

## Expected return

- Report language: English.
- Changed paths and behavior; the new dimension rule in one sentence.
- Each check with its command, expected result and actual result.
- The list of changed results required by the Fidelity decision.
- Remaining limitations, for example that a cross-server send carries no intra-server hop.
