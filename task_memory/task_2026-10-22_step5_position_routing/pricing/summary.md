# Pricing implementation completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive scoped delivery, exact interfaces/budget and acceptance limits. |

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
Authorized CORE02f061d/Native2c4cc9e and published67d5b5e dependencies imported;
437b312 records published dependency integration. No peer file edited.
Isolated collective-sim branch d29-position-pricing commits:
ae75af0,1a16426; final tip1a16426. Retained diff: `collective_sim.patch` in this
directory. Parent owns the gitlink integration; unstaged gitlink is intentional.

## Interface and validation artifacts

- `task_memory/task_2026-10-22_step5_position_routing/pricing/interface_receipt.md`:
  exact selected/operator/native/model/producer/CORE APIs and CPU payload.
- `task_memory/task_2026-10-22_step5_position_routing/pricing/test_report_2026-10-23_delivery.md`:
  exact final commands, errors, results and limits. Earlier test reports remain.
- Final affected CPU checks:438 PASS/89 SKIP; real HTSIM7 PASS;
  published native original methods executed through actual adapter:PASS.
- AST/JSON parse and whitespace:PASS. No GPU/calibration claim.

## Exact owned code/test/config diff

43 changed paths; +2555/-71=2626 gross versus3956, remaining1330. The five
submodule files total312 gross and are included. Peer imports/task records/root
gitlink are excluded. The sole unchanged approved path is
`tests/unit/test_step5_contract_reuse.py`; its existing tests pass unchanged.

| Path | Added | Deleted | Gross |
| --- | ---: | ---: | ---: |
| frontier/attention/step5.py |3|0|3|
| frontier/cc_backend/backends/collective-sim/htsim_runner.py |52|2|54|
| frontier/cc_backend/backends/collective-sim/python/collective_sim_core/intra_server_model.py |77|0|77|
| frontier/cc_backend/backends/collective-sim/python/collective_sim_core/predictor.py |23|0|23|
| frontier/cc_backend/backends/collective-sim/python/collective_sim_core/schema.py |53|0|53|
| frontier/cc_backend/backends/collective-sim/tests/test_grouped_plan_input.py |105|0|105|
| frontier/cc_backend/backends/collective_sim_cc_backend.py |59|0|59|
| frontier/cc_backend/base_cc_backend.py |6|0|6|
| frontier/cc_backend/collective_plan.py |122|0|122|
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
| frontier/profiling/collectives/step5_dispatch_probe.py |80|8|88|
| frontier/profiling/collectives/vllm_collective_benchmark.py |171|16|187|
| frontier/profiling/collectives/vllm_forward_collectives.py |18|7|25|
| frontier/profiling/moe/fp8_block_experts.py |98|1|99|
| frontier/profiling/moe/moe_vllm_kernel.py |12|0|12|
| frontier/scheduler/utils/batch_builders.py |4|0|4|
| frontier/scheduler/utils/expert_parallel.py |6|0|6|
| tests/integration/step5_comm/analyze_result.py |1|1|2|
| tests/integration/step5_comm/export_costs.py |233|0|233|
| tests/integration/step5_comm/prepare.py |68|2|70|
| tests/integration/step5_comm/step5_position_model_run.json |68|0|68|
| tests/integration/step5_comm/validate_result.py |61|5|66|
| tests/integration/step5_comm/worker.py |34|4|38|
| tests/unit/test_collective_benchmark_helpers.py |93|1|94|
| tests/unit/test_collective_sim_cc_backend.py |74|0|74|
| tests/unit/test_comm_operator_families.py |26|0|26|
| tests/unit/test_ep_wave_trace_context.py |46|0|46|
| tests/unit/test_op_trace_utils.py |25|0|25|
| tests/unit/test_step5_comm_cost_export.py |125|0|125|
| tests/unit/test_step5_comm_package.py |84|0|84|
| tests/unit/test_step5_comm_transport.py |118|4|122|
| tests/unit/test_step5_dispatch_probe.py |86|1|87|
| tests/unit/test_step5_moe_pricing.py |253|0|253|

## Remaining acceptance dependencies

1. Root imports tested pricing/CORE tips and gitlink1a16426; CORE connects its
   context-aware paired estimator, actual native-context schema/ranges, wrapper
   source/gathered scopes and layer/main H04 calls. No duplicate fallback is built.
   Peer b39fc51/e255ede exists but is not authorized/imported here.
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
