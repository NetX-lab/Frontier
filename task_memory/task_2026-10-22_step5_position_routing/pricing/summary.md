# Pricing implementation completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive scoped delivery, exact interfaces/budget and acceptance limits. |
| 2026-10-23 | Archive finite review F3-F6, new submodule tip and2944-line budget. |
| 2026-10-23 | Archive F4a/F4b frozen identity/call-time repairs and3067-line budget. |

TaskID: D29-PRICING-IMPLEMENTATION. Branch: feat/step5-position-pricing.
Baseline:1aee0af. Working directory:
`/data/ycfeng/Frontier/.worktrees/step5-position-pricing`.

## Outcome

Owned implementation delivered within44 approved paths and3956 gross lines.
Selected ROUTED prepare registry, immutable collective workload/cost, ordered
joint-eight-rail HTSIM, independently measured-local/affine-remainder contracts,
native prepared inclusive scope, H04/H17 and metrics/trace/ledger consumers are
implemented. Complete selected producer/evidence validation and separate parent
qualified import are implemented. Ordinary scalar and no-context defaults remain.
Actual production parameters and GPU qualification were not manufactured.

Early interface commits:7b86614,409093c,511a3db. Review/seam repairs:f03286f,
06bae09. T2 existing constructor seam:3f54bc6 (48/60 additional gross).
Final selected producer/qualified import/legacy trace repair:940cc84.
Finite repair F3/F6:37ead7f; F4/F5:181a534, verified scoped code/records.
Independent re-review closed F3/F5/F6 and reopened F4a/F4b; boundary repair uses
frozen runtime for fit grouping and full call-time policy for raw/count/cache/probe
equality. Same frozen policy can choose different implementations by size. Native
lazy-disable/fallback changes startup enabled-state without changing frozen policy.
F4a/F4b scoped source/record commit506b43ac; root independent re-review remains open.
Authorized CORE02f061d/Native2c4cc9e and published67d5b5e dependencies imported;
437b312 records published dependency integration. No peer file edited.
Isolated collective-sim branch d29-position-pricing commits:
ae75af0,1a16426,3dac53c; final tip3dac53c. Retained diff: `collective_sim.patch` in this
directory. Parent owns the gitlink integration; unstaged gitlink is intentional.
The retained submodule diff is byte-identical to pinned baseline-to-tip diff:
sha256c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d.
Isolated checkout is clean; no gitlink or peer changes were staged.

## Interface and validation artifacts

- `task_memory/task_2026-10-22_step5_position_routing/pricing/interface_receipt.md`:
  exact selected/operator/native/model/producer/CORE APIs and CPU payload.
- `task_memory/task_2026-10-22_step5_position_routing/pricing/test_report_2026-10-23_delivery.md`:
  exact final commands, errors, results and limits. Earlier test reports remain.
- `task_memory/task_2026-10-22_step5_position_routing/pricing/test_report_2026-10-23_finite_repair.md`:
   F3-F6 exact RED/GREEN commands, JUnit counts and invocation limits.
- `task_memory/task_2026-10-22_step5_position_routing/pricing/test_report_2026-10-23_f4_boundaries.md`:
  F4a/F4b exact RED/GREEN,552-case JUnit, failed iterations and3067-line budget.
- Final affected CPU checks:463 PASS/89 SKIP; real HTSIM12 PASS; independent
  cold constructor/importer without PYTHONPATH10 PASS. Published native original-method
  CPU execution remains the earlier recorded PASS, not a GPU qualification.
- AST/JSON parse and whitespace:PASS. No GPU/calibration claim.

## Exact owned code/test/config diff

43 changed paths; +2972/-95=3067 gross versus3956, remaining889. The five
submodule files total333 gross and are included. Peer imports/task records/root
gitlink are excluded. The sole unchanged approved path is
`tests/unit/test_step5_contract_reuse.py`; its existing tests pass unchanged.

| Path | Added | Deleted | Gross |
| --- | ---: | ---: | ---: |
| frontier/attention/step5.py |3|0|3|
| frontier/cc_backend/backends/collective-sim/htsim_runner.py |52|2|54|
| frontier/cc_backend/backends/collective-sim/python/collective_sim_core/intra_server_model.py |77|0|77|
| frontier/cc_backend/backends/collective-sim/python/collective_sim_core/predictor.py |30|0|30|
| frontier/cc_backend/backends/collective-sim/python/collective_sim_core/schema.py |53|0|53|
| frontier/cc_backend/backends/collective-sim/tests/test_grouped_plan_input.py |119|0|119|
| frontier/cc_backend/backends/collective_sim_cc_backend.py |64|0|64|
| frontier/cc_backend/base_cc_backend.py |6|0|6|
| frontier/cc_backend/collective_plan.py |136|0|136|
| frontier/config/quantization_manager.py |1|1|2|
| frontier/entities/execution_time.py |40|10|50|
| frontier/entities/time_components.py |14|0|14|
| frontier/execution_time_predictor/moe_operator_times.py |50|0|50|
| frontier/execution_time_predictor/sklearn_execution_time_predictor.py |6|0|6|
| frontier/execution_time_predictor/step5_cost.py |37|2|39|
| frontier/metrics/constants.py |3|0|3|
| frontier/metrics/ep_wave_metrics.py |20|0|20|
| frontier/metrics/op_trace_utils.py |44|0|44|
| frontier/operators/families.py |41|1|42|
| frontier/operators/spec.py |15|5|20|
| frontier/profiling/collectives/step5_dispatch_probe.py |101|9|110|
| frontier/profiling/collectives/vllm_collective_benchmark.py |277|39|316|
| frontier/profiling/collectives/vllm_forward_collectives.py |18|7|25|
| frontier/profiling/moe/fp8_block_experts.py |98|1|99|
| frontier/profiling/moe/moe_vllm_kernel.py |12|0|12|
| frontier/scheduler/utils/batch_builders.py |4|0|4|
| frontier/scheduler/utils/expert_parallel.py |6|0|6|
| tests/integration/step5_comm/analyze_result.py |1|1|2|
| tests/integration/step5_comm/export_costs.py |239|0|239|
| tests/integration/step5_comm/prepare.py |68|2|70|
| tests/integration/step5_comm/step5_position_model_run.json |68|0|68|
| tests/integration/step5_comm/validate_result.py |78|5|83|
| tests/integration/step5_comm/worker.py |34|4|38|
| tests/unit/test_collective_benchmark_helpers.py |151|1|152|
| tests/unit/test_collective_sim_cc_backend.py |102|0|102|
| tests/unit/test_comm_operator_families.py |26|0|26|
| tests/unit/test_ep_wave_trace_context.py |46|0|46|
| tests/unit/test_op_trace_utils.py |25|0|25|
| tests/unit/test_step5_comm_cost_export.py |169|0|169|
| tests/unit/test_step5_comm_package.py |84|0|84|
| tests/unit/test_step5_comm_transport.py |188|4|192|
| tests/unit/test_step5_dispatch_probe.py |113|1|114|
| tests/unit/test_step5_moe_pricing.py |253|0|253|

## Remaining acceptance dependencies

1. Root imports tested pricing/CORE tips and gitlink3dac53c; CORE connects its
   context-aware paired estimator, actual native-context schema/ranges, wrapper
   source/gathered scopes and layer/main H04 calls. No duplicate fallback is built.
   Peer b39fc51/e255ede exists but is not authorized/imported here.
   CORE also retains actual full per-rank execution_policy in plan context; guessed
   communicator defaults/version/buffer branch cannot satisfy the importer. The
   exporter/importer derive collective_model_context only for frozen fit selection;
   complete observed branches/buffers/runtime remain in raw cells and exact counts.
2. Root prepares actual bound input vectors/layers/frozen device/network inputs
   and independently measured local/protocol/TP/count evidence, then separate
   parent qualification. The bounded template is
   `tests/integration/step5_comm/step5_position_model_run.json` and uses published
   linux/amd64 image digest f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee.
3. Parent owns integrated full-unit/strict77/review, CPU actual-main6144/1024,
   Native T2/source/backend/G3, native CPU-gap campaigns and four E2E<=10%.

No additional agents, platform/GPU/auth/cloud/SSH/Docker/push/stop/deletion or
scope borrowing. This archive records scoped implementation, not final D29
production acceptance or peer qualification.
