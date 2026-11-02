# Execution costs are composed from operator profiling; measured whole-step tables are diagnostic only

Frontier predicts the time of a step by composing the costs of the operators that the batch executes, from operator profiling inputs under `data/profiling/` and the predictors trained from them, together with the communication backend. A table that stores whole steps measured in real serving runs and looks them up by batch composition is not a cost model. It holds the answer for the workload, topology and hardware it was measured on, so it has no value for other batch sizes, context lengths, topologies or hardware, and a comparison priced by it does not show that Frontier can predict. We chose this rule after the DeepSeek-V4-Flash calibration used such a table (`dsv4_step_cost.json`) to obtain its first error attribution while no operator-level V4 serving costs existed. That trade gave quick diagnostics of the scheduling and transfer paths but no evidence for the cost model.

## Consequences

- A measured whole-step table is never placed in Frontier's default profiling input path. A diagnostic driver selects it explicitly, and its results are labelled DIAGNOSTIC.
- Results priced by such a table never count toward accuracy gates, including gate readings that pass.
- When operator-level inputs are missing for a step, the run stops with a typed coverage error; the step is not priced from a measured step instead.
- A measured step time may be used to cross-check the step time composed from operator costs.
