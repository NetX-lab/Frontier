"""V4 segment gaps and attention composition on the trained operator route."""

from types import MappingProxyType

from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase, segment_class
from frontier.operators.deepseek_v4_axes import DEEPSEEK_V4_OPERATOR_AXES
from frontier.operators.families import COMM_FAMILY
from frontier.profiling.kernel_gap.schema import EAGER_PREPARATION

# All attention scopes use segment_class (including PIECEWISE eager breaks).
# The remaining standalone, routed and collective scopes run inside the forward,
# except its selected logits endpoints. B17 adds the runner scopes here.
V4_SEGMENT_OWNERS = MappingProxyType({
    **{name: "forward" for name in DEEPSEEK_V4_OPERATOR_AXES},
    **{op.name: "forward" for op in DEEPSEEK_V4_ATTENTION_FAMILY.operators},
    **{op.name: "forward" for op in COMM_FAMILY.operators if op.name.startswith("dsv4_")},
    "moe_gating_linear": "forward", "moe_gating_routing_topk": "forward",
    "moe_grouped_gemm": "forward",
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
