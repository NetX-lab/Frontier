def _step5_position_queries(self, lane):
    """Recover actual point facts and complete measured plans, never numerical defaults."""
    from frontier.profiling.moe.inputs import MoESourceFrame
    from frontier.cc_backend.collective_plan import CollectivePlan, CollectiveTensor
    source = lane.routing_source
    group = self._step5_routed_layer_group(self._step5_layer_metadata.attention_spec.global_layer_id)
    family = self._measurement_family_name(self._active_measurement_type)
    models = self._model_manager._models_view_for_family(family, self._cluster_type)
    frames = tuple(MoESourceFrame(s.lane_id,tuple(i.length for i in s.intervals),
        tuple(i.start for i in s.intervals),s.physical_rows,s.gating_runtime_context) for s in source.source_lanes)
    contexts = {}
    for model in models.values():
        if getattr(model, "_frontier_cache_model_name", None) != "moe_grouped_gemm":
            continue
        binding = self._model_manager._native_measurement_binding(
            self._model_manager._measurement_identity(model))
        if binding["selected_native_metadata"]["moe_layer_ids"] == group.row_metadata()["moe_layer_ids"]:
            contexts.update(self._step5_model_points(model).get((lane.ep_id, lane.routed_token_count, frames), {}))
    if len(contexts) != 1: raise ValueError("Position expert query lacks a unique actual source/native point receipt")
    context = json.loads(next(iter(contexts)))
    features = self._build_grouped_gemm_features(lane,native_context=context)
    local = source.source_lanes[lane.ep_id//self._replica_config.attn_tensor_parallel_size]
    from frontier.operators.families import get_moe_profiling_operators
    queries = {op.name:dict(num_tokens=local.physical_rows,live_routing_token_count=local.live_rows)
               if op.ep_agnostic else features for op in get_moe_profiling_operators(context)}
    retained = self._step5_collective_points()
    plans = {}
    phases = {"step5_moe_dispatch":"dispatch","step5_moe_combine":"combine",
        "moe_tensor_parallel_allreduce":"routed_tp"}
    if self._model_config.supports_share_expert(): phases["share_expert_tensor_parallel_allreduce"] = "shared_tp"
    mode = "full" if local.graph_mode == "FULL" else "none"
    for name,phase in phases.items():
        matches = {}
        tensors = context["collective_tensors"]["dispatch" if phase == "dispatch" else "output"]
        artifact, prepared, candidates = self._step5_plan_candidates(retained,phase,
            tuple(s.physical_rows for s in source.source_lanes),mode,tensors,context)
        for key,value,buffer in candidates:
            receipt = artifact["counts"][key]
            if (value["phase"] != phase or tuple(value["sizes"]) != tuple(s.physical_rows for s in source.source_lanes)
                    or value["graph_mode"] != mode or value["tensors"] != tensors): continue
            measured = receipt["context"]
            if measured["shared"] != context["native_facts"]["shared"] or not measured.get("execution_policy"): continue
            from frontier.profiling.common.runtime_identity import canonical_json
            if canonical_json(buffer.get("compute_context")) != canonical_json(context): continue
            if receipt["buffer_receipt"] != buffer: raise ValueError("Measured collective buffer/count key mismatch")
            if key not in prepared:
                plan = CollectivePlan(phase,tuple(value["sizes"]),tuple(CollectiveTensor(**t) for t in tensors),
                    tuple(tuple(g) for g in value["groups"]),mode,value["context_json"],value["buffer_json"])
                decoded, rebuilt = json.loads(plan.context_json), plan.cache_key()
                if decoded != measured or rebuilt != key: raise ValueError("Measured collective context/count key mismatch")
                prepared[key] = (plan, decoded, rebuilt)
            plan, decoded, rebuilt = prepared[key]
            if decoded != measured or rebuilt != key: raise ValueError("Measured collective context/count key mismatch")
            matches[key] = plan
        if len(matches) != 1: raise ValueError("Position collective lacks unique measured policy/layout/buffer coverage")
        plans[name] = next(iter(matches.values()))
    self._step5_collective_receipts = retained
    return context,queries,plans
