# Module Size Exceptions

AGENTS.md keeps each Python source file at or below 2,000 lines as a soft
limit. A critical module above the limit needs a cleanup pass, a concrete
reason for its current boundary, and a planned functional split. This record
covers the four modules that the calibration changes of PR 44
(`calibration/t43-combined`) touch while they are above the limit. They are
also the only Python files under `frontier/` above 2,000 lines on `main` at
`7039060` (the merge of PR 47), where each still has its "After" line count.

## Inventory

D26-PREFILL-TARGET (2026-10-22) changes `frontier/entities/request.py` from
2,020 lines at `ebd5baf` to 2,049. Cleanup inspection found no duplicate or dead
target/completion implementation in the touched lifecycle methods. The added
scalar remains beside semantic prompt length and the existing D recompute cursor:
moving it alone would split shared P/D request mutation across two owners. Only
the registered measured Step NIXL/PP1 path selects a shorter P target; client
lengths, legacy structural timing and other attention contracts remain unchanged.

A later movement-only split can move thinking-round affinity, hidden/final round
timing and requeue methods together into a request thinking-round module, retaining
public delegates and the shared state. Derived request timing accessors can move
separately after the lifecycle boundary is stable. Neither split is required for
this approved correction; it would expand unrelated regression scope. The parent
owns independent review and the unchanged strict 77-case fidelity gate before
integration acceptance. This exception authorizes no further Request growth.

D5 T1 (2026-10-20) extends the existing request metric exporter in
`frontier/metrics/metrics_store.py` with PD-only `client_ttft`. The CSV is in
milliseconds, from request arrival to `first_decode_token_completed_at` (the
first D output); optional ground-truth rows expose `client_ttft_s` and
`client_ttft_ms`. Historical `ttft`, `ttft_statistics` and trace `ttft_ms` remain
arrival to P-prefill completion for existing comparison/gate consumers. No
proxy/HTTP delivery or host overhead is invented; this is the simulator's
client endpoint, not evidence of native client accuracy. Non-PD exports are
unchanged. The existing request distributions own its storage and unit
conversion; moving this field alone would introduce a second metrics owner.
The planned metrics-ledger split below remains independent. `request.py` stays
below 2,000 lines by replacing the old TTFT docstring rather than growing its
lifecycle logic. There is no execution, token-accounting or event-order change.

"Before" is PR 42's head (`17f8088`). "After" is PR 44 with the review fixes.

| Module | Before | After | What PR 44 adds |
| --- | ---: | ---: | --- |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8,273 | 8,464 | Per-step measurement-family binding, stage-keyed CPU-overhead terms, the `forward_launch` and `forward_drain` host terms, the attention KV-cache extract term, the measured KV-send slowdown term |
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

Steps 1 and 2 were scheduled after PR 44's merge, so the split would not mix
with the calibration review. PR 44 is merged (`9a650fb`), but no step has
started: the target modules named above do not exist yet. Steps 3 and 4 can
run at any time.


## Step controlled request path on `feat/step5-preview`

The Step structural slice extends two modules covered above. Counts use baseline
`1bde1999af07a41651bb025d92e95bf14ad9e1ca` and the current source on
2026-10-19.

| Module | Baseline | Current | Added responsibility |
| --- | ---: | ---: | --- |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8,491 | 8,536 | Select existing Step attention scopes for the actual phase, context, graph policy and padded rows in controlled timing mode |
| `frontier/metrics/metrics_store.py` | 5,658 | 5,763 | Preserve source layer/tensor identity and the existing stage and EP-wave cost owners; include model input and terminal scopes once |

The cleanup inspection found no dead copy of the new Step pricing or ledger
paths. Layer geometry uses the existing metadata caches in the predictor and
metrics store; stage and EP-wave traces share the store cache instead of
rebuilding each layer for every operator. FFN merge trace enumeration uses the
existing operator-family contract. The new once-only scopes use the existing
operator-to-time attribute map rather than a second timing registry. No second
scheduler, event path or model graph is introduced.

Step pricing reads the predictor's resolved batch, block, parallel and dummy
policy state and returns the existing attention component. Moving this small
extension alone would split that state from all adjacent family pricing. The
metrics changes extend the store's existing trace emitter, catalogs and stage
component ledger; a separate Step ledger would create two cost owners. The
functional split above remains the appropriate later sequence: measurement
binding moves with its pricing state, and the stage ledger moves with its
consumers. This capability does not require that core refactor. Any such future
refactor follows the task's material-architecture approval policy and must
preserve legacy numeric/intermediate fidelity and the same operator ownership.

## Scoped V4 cache-zero metric admission, 2026-10-19

At `b5120e6`, `frontier/metrics/metrics_store.py` has 5,703 lines. The
cache-zero change adds two lines to the existing operation-metric admission
filter, bringing it to 5,705 lines. The corresponding enum entry is required
by actual structural execution; excluding that entry for models without the
V4 structural attention family preserves their existing output files instead
of creating an empty cache-zero table.

The cleanup inspection found this filter already owns the exclusion of absent
structural operations. Extending it avoids a second metrics owner or an
additional model classifier. Cache allocation metadata and timing remain in
their existing scheduler, Batch and V4 predictor modules; no cache algorithm
or new ledger responsibility is added to the metrics store. Moving unrelated
ledger code during this correction would expand the regression surface
without improving cache-zero representation.

The functional split remains step 3 above, as a separate movement-only change.
This exception covers only the two-line filter extension, with focused metric
checks and the unchanged 77-case artifact and predictor-cache filename
comparison required before acceptance. It does not authorize further growth
of the metrics store or claim that the planned split has been performed.

## Step structural PD handoff on `feat/step5-pd-structural`, 2026-10-20

The Step PD slice touches three modules above the limit. Counts use the merge
base `c14e642` and the current source on 2026-10-20.

| Module | Baseline | Current | Added responsibility |
| --- | ---: | ---: | --- |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8,628 | 8,639 | Price the request-free attention-DP dummy forward of Step layers from existing scopes |
| `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | 3,006 | 3,019 | Route Step PREFILL and unified DECODE stages to the existing co-location structural pricing |
| `frontier/metrics/metrics_store.py` | 5,802 | 5,803 | Pass the resolved transfer payload to the existing transfer trace builder |

The cleanup inspection found no duplicate Step role pricing to remove: PD
roles run complete Step layers, so the disaggregation predictor delegates to
the base pricing instead of copying it. The page payload itself lives in the
attention family modules (`memory.py`, `step5_memory.py`, `families.py`),
all below the limit, so the large modules only route to it. Moving the role
routing alone would separate it from the stage pricing it selects. The
functional splits planned above are unchanged and are not performed here.

## Measured inclusive attention root projection, 2026-10-22

D23-ROOT-METRICS touches the existing metrics/trace projection at baseline
`61cbac9`. `frontier/metrics/metrics_store.py` initially grew from 5,817 to
5,825 lines; the P1 owner correction below reduces it to 5,803 lines.
The cleanup moves the per-family ledger projection from the stage loop to the
existing layer ledger, removing the duplicate projection and allowing both
layer and stage consumers to retain a scalar-free measured root. Stage-owned
work keeps its existing owner filter; the scalar-attribute map is not a total
operator registry.

The remaining wiring serializes the existing per-layer OperatorCost into trace
metadata and excludes absent structural operators through the family registry.
Root classification, validation, and source tensor boundaries live in the
existing attention modules below 2,000 lines. No separate Step ledger, scalar,
predictor, or child cost is introduced. Moving this wiring alone would split
the same ledger/trace owner, while a broad store split would expand the fix's
regression surface. The planned movement-only ledger split (step 3 above)
remains deferred; this exception authorizes only this root projection fix.

The P1 follow-up removes recursive layer projection of the stage-only owner.
Measured first/last boundary prediction retains the owner's attention duration
but intentionally does not preserve a layer OperatorCost there. The owner's
layer payload is not authoritative and must not be projected or validated as
one. Existing scalar/canonical stage fields are now projected once before
layer aggregation, using the same names and stage-owned classification as
before; their old later projection is removed. Authoritative layers still use
the strict measured root projection. No new flag, module, owner proof, or
predictor replacement behavior is added. The correction is net minus 22 lines
and stays within the existing ledger split exception.

## Step measured fixed-binding reuse, 2026-10-22

D23-MRV2-CONTRACT-REUSE replaces repeated physical-layer metadata and operator
precision/contract construction at baseline `74980bb`. The implementation stays
in the existing `frontier/execution_time_predictor/step5_cost.py` (220 to 263
lines), shared by measured attention batch features and the active measured FFN
layer context. `sklearn_moe_execution_time_predictor.py` only delegates fixed
binding construction and shrinks from 1,814 to 1,791 lines. Neither touched
module exceeds 2,000 lines; the large base predictor, disaggregation predictor,
metrics store and event modules are unchanged.

The cleanup removes the former per-query precision/contract block and duplicate
layer description at the two consumers, rather than retaining a second binding
path. Predictor-instance tables hold physical-layer metadata and fixed
`(physical layer, operator)` precision/serialized contract identity. The existing
measurement-family registry still chooses the current estimator on every query;
dynamic features, duration/count lookups, bounds, source proofs and costs stay
with the existing measured query. Nested active-layer context and cost records
retain their original owner and restoration behavior. A new global/model/output
cache or a broad predictor split would add lifetime ownership without removing
the measured fixed construction. The functional splits above remain deferred
and are not prerequisites for this bounded correction.
