"""V4 segment gaps and attention composition on the trained operator route."""

from types import MappingProxyType

from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase, segment_class
from frontier.operators.deepseek_v4_axes import DEEPSEEK_V4_OPERATOR_AXES
from frontier.operators.families import COMM_FAMILY
from frontier.profiling.kernel_gap.schema import EAGER_PREPARATION
from frontier.execution_time_predictor.dsv4_runner import ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE

# All attention scopes use segment_class (including PIECEWISE eager breaks).
# The remaining standalone, routed and collective scopes run inside the forward,
# except its selected logits endpoints and runner activities.
V4_SEGMENT_OWNERS = MappingProxyType({
    **{name: "forward" for name in DEEPSEEK_V4_OPERATOR_AXES},
    **{op.name: "forward" for op in DEEPSEEK_V4_ATTENTION_FAMILY.operators},
    **{op.name: "forward" for op in COMM_FAMILY.operators if op.name.startswith("dsv4_")},
    "moe_gating_linear": "forward", "moe_gating_routing_topk": "forward",
    "moe_grouped_gemm": "forward",
    **{name: EAGER_PREPARATION for name in (ZERO_SCOPE, WRITE_SCOPE, INPUT_SCOPE, SAMPLING_SCOPE)},
    "dsv4_lm_head": EAGER_PREPARATION, "dsv4_logits_gather": EAGER_PREPARATION,
})


def operator_segment_class(mode, name):
    owner = V4_SEGMENT_OWNERS[name]
    return segment_class(mode, name) if owner == "forward" else owner


def _operator_gaps(predictor, layer, mode):
    communication = {op.name: op for op in COMM_FAMILY.operators}
    gaps = {}
    for name, cost in layer.op_times.items():
        if name in layer._dsv4_scope_gap_ms:
            continue
        count = (communication[name].kernel_count if cost > 0 else 0.0) if name in communication else layer._dsv4_kernel_counts[name]
        gaps[name] = count * predictor._kernel_gap_ms[operator_segment_class(mode, name)]
    return gaps


def compose_trained_attention(predictor, layer, batch):
    from frontier.execution_time_predictor.deepseek_v4 import timing_operator_times

    operator_gaps = _operator_gaps(predictor, layer, batch.get_decode_cuda_graph_runtime_mode())
    costs = {name: layer.op_times[name] for _, name, _ in timing_operator_times(layer, "attention")}
    gaps = {name: (layer._dsv4_scope_gap_ms[name] if name in layer._dsv4_scope_gap_ms
                   else operator_gaps[name]) for name in costs}
    layer._dsv4_attention_schedule = compose_attention_phase(
        costs, gaps, **layer._dsv4_composition_inputs,
    )


def add_dsv4_operator_gaps(predictor, layer, batch):
    """Publish operator gaps once; attention elapsed already includes its gaps."""
    layer._dsv4_operator_gap_ms = MappingProxyType(
        _operator_gaps(predictor, layer, batch.get_decode_cuda_graph_runtime_mode()),
    )


def add_runner_scopes(predictor, batch, layer_id, *, include_attention, include_ffn,
                      selected_logits_rows, times, counts, workloads):
    """Price each runner invocation at its owning boundary, without host time."""
    from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
    from frontier.entities.batch import DummyForwardBatch
    from frontier.execution_time_predictor.dsv4_runner import preparation_coordinate
    from frontier.types import ClusterType

    dummy = isinstance(batch, DummyForwardBatch)

    def price(name, coordinate, prediction=None):
        if predictor._enable_dummy_mode:
            times[name] = predictor._dummy_execution_time
        else:
            result = prediction or predictor._predictions["dsv4_operators"].runner(name, coordinate)
            times[name], counts[name] = result.time_ms, result.kernel_count
        workloads[name] = coordinate

    if include_attention and layer_id == 0:
        if not dummy:
            zeroing = batch.cache_block_zeroing
            if zeroing is not None and zeroing.block_ids:
                price(ZERO_SCOPE, {"n": len(zeroing.block_ids)})
                workloads[ZERO_SCOPE].update({
                    "block_ids": zeroing.block_ids,
                    "block_assignment_count": len(zeroing.block_ids),
                    "unique_block_count": len(set(zeroing.block_ids)),
                    "physical_block_size_bytes": zeroing.physical_block_size_bytes,
                    "physical_reservation_extent_bytes": zeroing.physical_reservation_extent_bytes,
                    "extent_scope": "distinct physical slab reservations; not native store traffic",
                    "native_store_traffic_bytes": "UNKNOWN",
                })
            writes = batch.kv_block_table_writes
            if writes is not None and (writes.w_c, writes.r) != (0, 0):
                prediction = (None if predictor._enable_dummy_mode else
                              predictor._predictions["dsv4_operators"].block_table_writes(writes))
                price(WRITE_SCOPE, {"w_c": writes.w_c, "r": writes.r,
                                    "continuing_entries": writes.continuing_entries,
                                    "new_rows": writes.new_rows}, prediction)
        price(INPUT_SCOPE, preparation_coordinate(batch))
    if (include_ffn and layer_id == predictor._model_config.num_layers - 1
            and not dummy and selected_logits_rows):
        if not predictor._enable_dummy_mode and predictor._cluster_type is ClusterType.PREFILL:
            raise DeepSeekV4CoverageError(
                "V4 dsv4_sampling: PREFILL sampling is unmeasured; the native P2 proxy "
                "sets max_tokens=1 and removes min_tokens, while sampling rows require "
                "min_tokens=max_tokens and the generation-config EOS stop-token bias"
            )
        price(SAMPLING_SCOPE, {"b": len(batch.requests)})
