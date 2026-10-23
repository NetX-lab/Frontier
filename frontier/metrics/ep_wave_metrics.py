"""Project actual EP lane operators without converting barrier maxima to work."""

from frontier.metrics.constants import OperationMetrics
from frontier.metrics.op_trace_utils import (
    OpTraceContext, build_parallel_context, compute_op_trace_meta,
)
from frontier.metrics.trace_store import TraceEvent


def record_ep_wave(store, plan, *, time, replica_id, stage_id, cluster_type):
    """Keep per-lane work and shared phase barriers as separate quantities."""
    phases = plan.phase_times
    starts = {
        "pre_dispatch": time,
        "dispatch": time + max(phases.pre_dispatch_times_ms) * 1e-3,
        "routed_compute": plan.timing.dispatch_barrier_end_time_s,
        "combine": plan.timing.dispatch_barrier_end_time_s
        + max(phases.routed_compute_times_ms) * 1e-3,
        "post_combine": plan.timing.combine_barrier_end_time_s,
    }
    config = store._config
    trace_enabled = bool(config.enable_op_level_tracing and store.trace_store)
    operation_enabled = config.write_metrics and config.store_operation_metrics
    ledger_enabled = config.write_metrics and config.store_frontier_stage_batch_ledger
    replica = store._cluster_configs[cluster_type].replica_config
    for record in phases.lane_records:
        batch = record.batch
        layer = record.execution_time.layer_execution_times[0]
        moe=layer.moe_or_mlp_time_component.operator_times
        step5_frame=moe.physical_frame if moe is not None and moe.physical_frame else None
        communication=layer.communication_time_component.operator_times
        destination_request_ids = [str(request.id) for request in batch.requests]
        source_rows = getattr(layer, "_dsv4_source_rows", None)
        request_ids = (source_rows["source_request_ids"] if source_rows is not None
                       else destination_request_ids)
        expand = trace_enabled and store._should_expand_layers(
            cluster_type, request_ids, batch_ids=tuple(batch.source_batch_ids)
        )
        common = {
            "ep_id": record.ep_id, "stage_id": stage_id,
            "global_layer_ids": [layer.global_layer_id],
            "attention_family_id": layer.attention_family_id,
            "attention_variant_id": layer.attention_variant_id,
            "request_ids": request_ids, "source_batch_ids": list(batch.source_batch_ids),
        }
        if source_rows is not None:
            common.update(source_rows)
            common["synthetic_destination_request_ids"] = destination_request_ids
        if step5_frame is not None:
            from dataclasses import asdict
            common["routing_source"]=asdict(step5_frame["routing_source"])
        layer_metadata = None
        if trace_enabled:
            total_tokens = sum(batch.num_tokens)
            if getattr(replica.model_config, "model_type", None) == "step5":
                layer_metadata = store._get_step5_layer_metadata(replica, layer.global_layer_id)
        ledger_phases = {} if ledger_enabled else None
        for phase, start in starts.items():
            operators = layer.moe_phase_operator_times(phase)
            cursor = start
            if ledger_enabled:
                ledger_phases[phase] = {
                    "start_time_s": start,
                    "duration_ms": sum(duration for _, _, duration in operators),
                    "operators_ms": {name: duration for _, name, duration in operators},
                }
                if layer.has_kernel_gap:
                    kernel_gap_ms = layer.moe_phase_kernel_gap_time(phase)
                    ledger_phases[phase]["kernel_gap_ms"] = kernel_gap_ms
                    ledger_phases[phase]["duration_ms"] += kernel_gap_ms
            if trace_enabled and any(duration > 0 for _, _, duration in operators):
                routed = phase == "routed_compute"
                tokens = (batch.lane_workload.routed_token_count if routed
                          else batch.get_effective_total_tokens_for_compute(cluster_type))
                if source_rows is not None and not routed:
                    tokens = source_rows["physical_local_rows"]
                if step5_frame is not None:
                    source=step5_frame["routing_source"]
                    source_lane=source.source_lanes[record.ep_id//replica.attn_tensor_parallel_size]
                    tokens=source.gathered_physical_rows if routed else source_lane.physical_rows
                context = OpTraceContext(
                    cluster_type, replica.model_config, replica,
                    total_tokens, tokens, tokens, tokens, routed,
                    global_layer_id=layer.global_layer_id, layer_metadata=layer_metadata,
                    pre_routing_tokens=batch.get_effective_total_tokens_for_compute(cluster_type),
                    step5_moe_frame=step5_frame,
                    collective_plans=communication.plans if communication is not None else None,
                    collective_costs=communication.costs if communication is not None else None,
                )
                parallel_context = build_parallel_context(context)
            for kind, name, duration in operators:
                if duration <= 0:
                    continue
                if operation_enabled:
                    store._push_metric(OperationMetrics(name), batch.id, duration, cluster_type)
                if trace_enabled:
                    workload_meta = {}
                    if source_rows is not None:
                        from frontier.execution_time_predictor.deepseek_v4 import operator_workload_metadata

                        workload_meta = operator_workload_metadata(layer, name)
                    meta = {
                        **compute_op_trace_meta(name, kind, context), **common, **workload_meta,
                        "phase": phase, "phase_start_time_s": start,
                        "wave_start_time_s": time,
                        "wave_end_time_s": plan.timing.wave_end_time_s,
                        "model_name": replica.model_name,
                        "parallel_context": parallel_context.copy(),
                        "effective_total_tokens_compute": tokens,
                        "tokens_are_post_routing": routed, "num_layers": 1,
                    }
                    store.trace_store.log_event(TraceEvent(
                        type=kind, name=name, ts_start=cursor, duration_ms=duration,
                        cluster=cluster_type.name, replica_id=replica_id,
                        batch_id=batch.id, layer_id=layer.global_layer_id if expand else -1,
                        meta=meta,
                    ))
                cursor += duration * 1e-3
            if ledger_enabled and communication is not None:
                from dataclasses import asdict
                ledger_phases[phase]["collective_proofs"]={name:dict(plan=communication.plans[name].to_dict(),
                    cost=asdict(communication.costs[name])) for _,name,_ in operators if name in communication.plans}
            if ledger_enabled and step5_frame is not None:
                ledger_phases[phase]["compute_proofs"]={name:step5_frame["operator_costs"][name]
                    for kind,name,_ in operators if kind == "COMPUTE" and name in step5_frame["operator_costs"]}
        if ledger_enabled:
            store._frontier_ep_wave_lane_ledger_rows.append({
                **common, "record_type": "ep_wave_lane", "cluster": cluster_type.name,
                "replica_id": replica_id, "batch_id": batch.id,
                "wave_start_time_s": time, "wave_end_time_s": plan.timing.wave_end_time_s,
                "phases": ledger_phases,
            })
