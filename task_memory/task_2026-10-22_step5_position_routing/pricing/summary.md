# Pricing implementation completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive scoped delivery, exact interfaces/budget and acceptance limits. |
| 2026-10-23 | Archive finite review F3-F6, new submodule tip and2944-line budget. |
| 2026-10-23 | Archive F4a/F4b frozen identity/call-time repairs and3067-line budget. |
| 2026-10-23 | Archive C1 observed routing/C2 compute binding and3305-line budget. |
| 2026-10-23 | Close observed point producer/count binding and3606-line budget. |
| 2026-10-23 | Archive explicit CPU plan consumer, truthful trace and3713-line budget. |

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
C1/C2 owned repairs observe original native routing once and validate exact IDs,
weights, padding and actual policy after execution; H04 validates all point/plan
contexts and tensor layouts before compute or collective pricing. Complete observed
compute_context belongs to the per-cell buffer/count key, not frozen affine fit
identity. The owned probe now emits the actual observed point; exporter binds it
to exact collective counts through source, call-time policy and layout evidence.
Missing binding fails instead of guessing or inventing evidence.
Verified C1/C2 source/record commit989a883d. Root independent re-review and actual
runtime GPU observation/integration remain open, not owner-approved runtime proof.
Observed compute-context producer closure: original router/dispatch/expert tensor
references, existing prepared_context/verifier, validator and exact buffer/count
association implemented. Actual selected validator->exporter fixture passes;
stale compute, source, policy and geometry negatives reject. No peer API required.
Verified source/tests/records commit9dbca5b3; root independent re-review pending.
Approved D29-CPU-PLAN-CONSUMER is implemented: explicitly unadmitted CPU artifact
and plan.shared identity gate, exact cost/count/context/buffer proofs before native
qualification, with native counts.cost still ignored. Trace labels CPU fixture
composition without claiming joint HTSIM/native measured input. Actual CORE writer
four-phase costs pass the real constructor; stale and native-label negatives fail.
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
- `task_memory/task_2026-10-22_step5_position_routing/pricing/test_report_2026-10-23_compute_repair.md`:
   C1/C2 RED/GREEN,682-case JUnit, exact commands, native64 PASS and3305-line budget.
- `task_memory/task_2026-10-22_step5_position_routing/pricing/test_report_2026-10-23_compute_context_producer.md`:
   observed writer RED/GREEN, actual validator/exporter positive, final430 distinct
   affected cases350 PASS/80 SKIP, exact commands and3606-line budget.
- Latest changed-path CPU checks:350 PASS/80 SKIP; published native original-method
   observer64 PASS. Previous unchanged real HTSIM12/cold10 results remain historical,
   not rerun or claimed as new tests. No GPU numerical qualification.
- AST/JSON parse and whitespace:PASS. No GPU/calibration claim.
- `test_report_2026-10-23_cpu_plan.md`: actual writer RED/GREEN, three affected
  suites68 PASS/0 SKIP/FAIL/ERROR without PYTHONPATH, exact commands and budget.
  Current CPU-consumer patch+109/-2=111 versus120 reserve; native model behavior
  and completed observed producer unchanged. No broad suite rerun claimed.

## Exact owned code/test/config diff

43 changed paths; +3617/-96=3713 gross versus3956, remaining243. The five
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
| frontier/cc_backend/backends/collective_sim_cc_backend.py |78|0|78|
| frontier/cc_backend/base_cc_backend.py |6|0|6|
| frontier/cc_backend/collective_plan.py |136|0|136|
| frontier/config/quantization_manager.py |1|1|2|
| frontier/entities/execution_time.py |40|10|50|
| frontier/entities/time_components.py |14|0|14|
| frontier/execution_time_predictor/moe_operator_times.py |79|0|79|
| frontier/execution_time_predictor/sklearn_execution_time_predictor.py |6|0|6|
| frontier/execution_time_predictor/step5_cost.py |37|2|39|
| frontier/metrics/constants.py |3|0|3|
| frontier/metrics/ep_wave_metrics.py |20|0|20|
| frontier/metrics/op_trace_utils.py |45|0|45|
| frontier/operators/families.py |41|1|42|
| frontier/operators/spec.py |15|5|20|
| frontier/profiling/collectives/step5_dispatch_probe.py |223|10|233|
| frontier/profiling/collectives/vllm_collective_benchmark.py |277|39|316|
| frontier/profiling/collectives/vllm_forward_collectives.py |18|7|25|
| frontier/profiling/moe/fp8_block_experts.py |98|1|99|
| frontier/profiling/moe/moe_vllm_kernel.py |12|0|12|
| frontier/scheduler/utils/batch_builders.py |4|0|4|
| frontier/scheduler/utils/expert_parallel.py |6|0|6|
| tests/integration/step5_comm/analyze_result.py |1|1|2|
| tests/integration/step5_comm/export_costs.py |297|0|297|
| tests/integration/step5_comm/prepare.py |68|2|70|
| tests/integration/step5_comm/step5_position_model_run.json |68|0|68|
| tests/integration/step5_comm/validate_result.py |88|5|93|
| tests/integration/step5_comm/worker.py |34|4|38|
| tests/unit/test_collective_benchmark_helpers.py |151|1|152|
| tests/unit/test_collective_sim_cc_backend.py |172|0|172|
| tests/unit/test_comm_operator_families.py |26|0|26|
| tests/unit/test_ep_wave_trace_context.py |46|0|46|
| tests/unit/test_op_trace_utils.py |47|0|47|
| tests/unit/test_step5_comm_cost_export.py |271|0|271|
| tests/unit/test_step5_comm_package.py |84|0|84|
| tests/unit/test_step5_comm_transport.py |200|4|204|
| tests/unit/test_step5_dispatch_probe.py |271|1|272|
| tests/unit/test_step5_moe_pricing.py |300|0|300|

## Remaining acceptance dependencies

1. Root imports tested pricing/CORE tips and gitlink3dac53c; CORE connects its
   context-aware paired estimator, actual native-context schema/ranges, wrapper
   source/gathered scopes and layer/main H04 calls. No duplicate fallback is built.
   Peer b39fc51/e255ede exists but is not authorized/imported here.
   CORE also retains actual full per-rank execution_policy in plan context; guessed
   communicator defaults/version/buffer branch cannot satisfy the importer. The
   exporter/importer derive collective_model_context only for frozen fit selection;
    complete observed branches/buffers/runtime remain in raw cells and exact counts.
    C2 additionally requires observed complete native compute_context in each
     exact plan's buffer_json/count receipt. Owned writer/exporter now supplies
     it; inspected CORE chooser already consumes the field and checks the buffer
     receipt. Root integrates these existing interfaces without inferred context.
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
