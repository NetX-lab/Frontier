# Module Size Exceptions

AGENTS.md keeps each Python source file at or below 2,000 lines as a soft
limit. A critical module above the limit needs a cleanup pass, a concrete
reason for its current boundary, and a planned functional split. This record
covers the four modules that the calibration changes of PR 44
(`calibration/t43-combined`) touch while they are above the limit.

## Inventory

"Before" is PR 42's head (`17f8088`). "After" is PR 44 with the review fixes.

| Module | Before | After | What PR 44 adds |
| --- | ---: | ---: | --- |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8,273 | 8,441 | Per-step measurement-family binding, stage-keyed CPU-overhead terms, the `forward_launch` and `forward_drain` host terms, the attention KV-cache extract term |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | 2,985 | 3,006 | Passes the pipeline stage and the new terms into its timing records; prices layers through the base predictor's kernel-gap wrapper and step binding |
| `frontier/metrics/metrics_store.py` | 5,586 | 5,593 | Two fields in the stage batch component ledger |
| `frontier/profiling/attention/main.py` | 2,157 | 2,161 | The memory filter counts KV blocks instead of tokens |

## Cleanup done in PR 44

- New table logic lives in its own modules rather than in the predictor:
  `kernel_gap.py` loads and validates the kernel-gap table, and
  `measurement_input_paths.py` decides two-stream eager pricing, including
  the model, TP and pipeline-stage coverage check.
- The CPU-overhead model names and feature columns come from
  `frontier/profiling/cpu_overhead/validation.py`, the same functions that
  validate the table. The predictor's hard-coded name and feature lists and
  its second exact-lookup builder are removed.
- Kernel-only attention training reuses the event-family training block. The
  duplicated decode and decode-in-mixed training copy (52 lines) is removed.

## Why each boundary remains

**Main predictor.** The added code reads and rebinds the predictor's
measurement-family state: `_active_measurement_type`, the per-family model and
prediction maps, and the active input files. Every operator-time method of the
class, and of `SklearnDisaggregationExecutionTimePredictor` and
`SklearnMoEExecutionTimePredictor`, prices through that state on `self`.
Moving only the new lookups out would leave two owners of one state, or pass
the state into every call. The subclasses also override and call the touched
private methods, and unit tests bind them directly, so a split has to move the
state with its methods in one step.

**Disaggregation predictor.** It gains no responsibility. The added lines pass
`stage_id` and the new timing fields through records that the base predictor
defines. They shrink to one call each once the CPU-overhead terms move out
(step 1 below).

**Metrics store.** The two ledger fields extend the existing stage batch
component ledger, which already owns every per-component time of a stage batch.
A separate owner for two fields would split one ledger across two modules.

**Attention profiler.** The change corrects an existing memory filter: vLLM
allocates KV cache in blocks, so the capacity check now counts blocks. No new
responsibility is added.

## Planned split

Each step is a movement-only change. It passes when the 77-case fidelity
matrix (`tests/e2e/refactor_fidelity/run_matrix.py`) is identical, the
predictor cache file names are unchanged, and the class keeps its public and
subclass-facing method names as delegates until their callers move.

1. **CPU-overhead pricing** to `execution_time_predictor/cpu_overhead_terms.py`.
   One object owns the per-family CPU-overhead models and predictions,
   `_lookup_cpu_overhead_prediction`, the required and optional term policy,
   and the stage-keyed term getters (`schedule`, `sampler_e2e`,
   `prepare_inputs_e2e`, `process_model_outputs`, `ray_comm_time`,
   `forward_launch`, `forward_drain`). Rows stay keyed by `model_name`,
   `tensor_parallel_degree`, `pipeline_stage_id` and batch shape.
2. **Measurement-family binding** to `execution_time_predictor/measurement_family.py`.
   One table keyed by `MeasurementType` replaces the `_activate_measurement_type`
   branch ladder and holds each family's input files, models and predictions.
   The step selector (`_select_measurement_type_for_batch`,
   `_activate_measurement_type_for_batch`) moves with it. This depends on step 1,
   which removes the CPU-overhead state from the families.
3. **Metrics store**: move the stage batch component and diagnostic ledgers
   (`_build_frontier_stage_batch_component_ledger` and its neighbors) to
   `frontier/metrics/stage_batch_ledger.py`. Independent of steps 1 and 2.
4. **Attention profiler**: move argument parsing (`parse_args`) and the
   GPU worker tasks to `frontier/profiling/attention/cli.py` and
   `frontier/profiling/attention/workers.py`. Independent of the other steps.

Steps 1 and 2 follow PR 44's merge, so the split does not mix with the
calibration review. Steps 3 and 4 can run at any time.
