"""Replay GLM source/EP collective contracts without synthetic compute latency.

The historical all-to-all path has no measured curve. Compare old/new geometry
with the same analytical backend, and report measured new geometry separately.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from frontier.cc_backend import MeasuredCCBackend, MeasuredCCBackendConfig, MeasuredCCOutOfRangeError
from frontier.cc_backend.backends.analytical_cc_backend import AnalyticalCCBackend
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.config.model_config import BaseModelConfig
from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
from frontier.entities import Batch, Request
from frontier.moe_ep_workload import build_contiguous_expert_ownership, materialize_layer_ep_workload
from frontier.operators.families import get_comm_operator
from frontier.operators.sequence_parallel_moe import source_tp_operator_names, validate_combine_coverage
from frontier.operators.spec import CommPayloadContext
from frontier.types import ClusterType


class BF16Activations:
    def adjust_tensor_size(self, operation, size, cluster_type):
        return size


def replay(role, lane_tokens, logits_rows):
    model = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    replica = SimpleNamespace(attn_tensor_parallel_size=8, attn_dp=4,
        num_pipeline_stages=1, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=32, router_topk=model.num_experts_per_tok)
    widths = sequence_parallel_moe_token_widths(lane_tokens, replica.attn_tensor_parallel_size)
    batch = Batch(0, [Request(0., max(lane_tokens), 1) for _ in range(logits_rows)],
                  [1] * logits_rows, is_moe=True)
    batch.get_effective_total_tokens_rounded = lambda _: max(lane_tokens)
    batch.moe_sequence_parallel_token_widths = widths
    routing = materialize_layer_ep_workload(
        routing_ratios={expert: 1 / model.num_experts for expert in range(model.num_experts)},
        target_replica_id=0, global_layer_id=3, routing_token_count=widths.gathered_tokens,
        router_topk=replica.router_topk, total_expert_num=model.num_experts,
        moe_expert_parallel_size=replica.moe_expert_parallel_size,
        expert_to_ep=build_contiguous_expert_ownership(model.num_experts, replica.moe_expert_parallel_size),
    )
    ctx = CommPayloadContext(batch, model, replica, role, BF16Activations(), routing.lane(0))
    analytical = AnalyticalCCBackend(AnalyticalCCBackendConfig(), role, "h800", "h800_dgx", 32)
    measured = MeasuredCCBackend(MeasuredCCBackendConfig(), role, "h800", "h800_dgx", 32)
    before, after = Counter(), Counter()
    for layer in range(model.num_layers):
        before.update(("attn_tensor_parallel_allreduce",))
        names = source_tp_operator_names(model, replica, layer)
        after.update(names)
        if model.is_moe_layer(layer):
            before.update({"expert_parallel_alltoall": 2})
            after.update(("expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine"))
        else:
            before.update(("mlp_tensor_parallel_allreduce",))
            after.update(("attn_tensor_parallel_allreduce", "mlp_tensor_parallel_allreduce"))

    def prices(operators, backend):
        rows = []
        for name, calls in operators.items():
            operator = get_comm_operator(name)
            size = operator.build_payload_bytes(ctx)
            try:
                validate_combine_coverage(operator, ctx, backend, size)
                alias = operator.resolve_collective_alias(ctx)
                time = backend.predict_comm_cost(
                    "all_to_all" if alias == "alltoall" else alias, data_size_bytes=size,
                    num_devices=operator.num_devices(ctx), cluster_type=role,
                    comm_domain=operator.resolve_comm_domain(ctx),
                )
                rows.append(dict(operator=name, calls=calls, bytes=size, per_call_ms=time,
                                 step_ms=calls * time, coverage="covered"))
            except MeasuredCCOutOfRangeError as error:
                rows.append(dict(operator=name, calls=calls, bytes=size, coverage="uncovered",
                                 reason=error.reason, key=error.curve_key,
                                 covered_range=error.covered_range))
        return dict(operators=rows, step_ms=(sum(row["step_ms"] for row in rows)
                    if all(row["coverage"] == "covered" for row in rows) else None))

    return dict(role=role.name, lane_tokens=lane_tokens, logits_rows=logits_rows,
        before_analytical=prices(before, analytical), after_analytical=prices(after, analytical),
        after_measured=prices(after, measured))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    results = [replay(role, lanes, rows)
        for role, lanes, rows in (
            (ClusterType.PREFILL, [8192] * 4, 1),
            (ClusterType.DECODE, [64] * 4, 64),
            (ClusterType.PREFILL, [64, 1, 1, 1], 1),
            (ClusterType.PREFILL, [8193] * 4, 1),
        )]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in ("role", "lane_tokens", "logits_rows")}
        | {key: row[key]["step_ms"] for key in ("before_analytical", "after_analytical", "after_measured")}
        for row in results], indent=2))


if __name__ == "__main__":
    main()
