"""Generate explicitly synthetic CPU inputs and fit the real shared cache/bundles.

This is a single-request integration fixture, never native profiling or admission.
Run with the repository and tests/unit on PYTHONPATH; no GPU or network is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pytest import MonkeyPatch

from frontier.config import global_vars
from frontier.config.config import RandomForrestExecutionTimePredictorConfig
from frontier.config.model_config import BaseModelConfig
from frontier.attention.step5 import Step5AttentionWorkload, describe_step5_layer
from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.execution_time_predictor.step5 import STEP5_SCOPE_NAMES, STEP5_OPERATOR_NAMES, step5_measurement_type
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.profiling.step5.inputs import Step5ProfileInput
from frontier.profiling.step5.rows import build_row, rows_to_dataframe
from frontier.moe_ep_workload import (
    build_contiguous_expert_ownership, generate_moe_routing_ratios, materialize_layer_ep_workload,
)
from frontier.moe_load_imbalance import MoELoadImbalanceInput
from frontier.types import CCBackendType, ClusterType, MeasurementType
from tests.unit import test_step5_measured_ffn as ffn
from tests.unit.test_step5_measured_training import _scheduler


CAPTURE_SIZES = (1, 2, 4, *range(8, 256, 8), 256)
MAX_MODEL_LEN = 7168
PREFILL_CASES = ((6143, 6112, 31, 32), (6144, 6128, 16, 16), (6145, 6128, 17, 24))
LOCAL_COMPUTE_WIDTHS = (1, 16, 24, 32, 6112, 6128)
AGGREGATE_COMPUTE_WIDTHS = (2, 32, 48, 64, 6113, 6129)


def attention_row(model, cell, *, physical_query_rows=None, device="h800"):
    """Build labelled CPU costs while preserving real queries and physical padding."""
    workload = Step5AttentionWorkload.from_lengths(
        model, phase=cell.phase, query_lengths=cell.query_lens, sequence_lengths=cell.sequence_lens,
        physical_query_rows=physical_query_rows or cell.num_tokens, block_size_tokens=16,
        block_table_columns=448, max_model_len_tokens=MAX_MODEL_LEN, graph_mode=cell.graph_mode,
    )
    metadata = describe_step5_layer(model, cell.global_layer_id, attention_tp_size=8,
                                   moe_tp_size=1, expert_parallel_size=1)
    scopes = metadata.attention_scopes(workload)
    active = {name for name, _ in scopes}
    stats = {}
    for name in (*STEP5_SCOPE_NAMES, *STEP5_OPERATOR_NAMES):
        value = .01 + workload.physical_query_rows / 10000 if name in active or name == "step5_layer_e2e" else 0.
        stats[name] = dict(min=value, max=value, mean=value, median=value, std=0., count=10)
        if name == "step5_layer_e2e":
            stats[name]["kernel_count"] = 17 if cell.dummy_descriptor else 23
    measurement = step5_measurement_type(cell.graph_mode)
    row = build_row(
        model, cell, tp=8, block_size=16, max_model_len=MAX_MODEL_LEN,
        runtime_signature="cpu_fixture_unadmitted", device=device, time_stats=stats,
        measurement_type=measurement.value,
        evidence=dict(attention_backend="cpu_fixture_unadmitted", evidence_kind="cpu_fixture",
                      timing_basis="cuda_event_native_runtime_latency" if measurement is MeasurementType.CUDA_EVENT
                      else "correlated_kernel_plus_memcpy_duration_sum", active_operators=json.dumps(sorted(active))),
    )
    contracts = json.loads(row["typed_operator_contracts"])
    row.update(typed_operator_contracts=json.dumps({name: contracts[name] for name in STEP5_SCOPE_NAMES}),
               padded_tokens=workload.padded_tokens, fusion=json.dumps(scopes),
               block_table_columns=448, score_regions=workload.score_regions)
    return row


def reached_rows(model, *, device):
    """Exact1023 context rows reuse one layer/shape contract, never serving credit."""
    for layer in range(model.num_layers):
        for prefix in sorted({prefix for _, prefix, _, _ in PREFILL_CASES}):
            yield attention_row(model, Step5ProfileInput(layer, (prefix,), (0,), "prefill", "none",
                                                       block_table_columns=448), device=device)
        for _, prefix, tail, physical in PREFILL_CASES:
            yield attention_row(model, Step5ProfileInput(layer, (tail,), (prefix,), "prefill", "piecewise",
                                                       block_table_columns=448), physical_query_rows=physical,
                                device=device)
        # The final output does not execute a subsequent forward at context7167.
        template = attention_row(model, Step5ProfileInput(layer, (1,), (6144,), "decode", "full",
                                                        block_table_columns=448), device=device)
        # Uniform FULL scopes/score allocation depend on physical capture/capacity,
        # not the live context. Every row still retains its own actual context.
        for context in range(6143, 7167):
            yield {**template, "context_lens": json.dumps((context,))}
        for descriptor in (
            MRv2DummyDescriptor(1, 1, "none", None, "CPU fixture of MRv2 InputBatch"),
            MRv2DummyDescriptor(16, None, "piecewise", None, "CPU fixture of MRv2 InputBatch"),
            MRv2DummyDescriptor(24, None, "piecewise", None, "CPU fixture of MRv2 InputBatch"),
            MRv2DummyDescriptor(32, None, "piecewise", None, "CPU fixture of MRv2 InputBatch"),
            MRv2DummyDescriptor(1, 1, "full", None, "CPU fixture of MRv2 InputBatch", 1),
        ):
            yield attention_row(model, Step5ProfileInput.from_mrv2_dummy(layer, descriptor, block_table_columns=448),
                                device=device)


def compute_tables(directory, model, *, device, expert_parallel_size, measurement):
    """Cover local operators and exact balanced physical EP loads, including dummy work."""
    with MonkeyPatch.context() as patch:
        metadata = ffn._metadata
        patch.setattr(ffn, "_metadata", lambda config: {
            **metadata(config), "device": device, "measurement_type": measurement.value,
        })
        paths = ffn._tables(
            directory, model, token_counts=tuple(sorted(set(LOCAL_COMPUTE_WIDTHS + AGGREGATE_COMPUTE_WIDTHS))),
            expert_parallel_size=expert_parallel_size, include_stage_boundaries=True,
        )
    templates = pd.read_csv(paths[1], float_precision="round_trip")
    rows = []
    for _, template in templates[templates.num_tokens.isin(AGGREGATE_COMPUTE_WIDTHS)].iterrows():
        layer = int(template.moe_layer_ids.split("-")[0])
        tokens = int(template.num_tokens)
        workload = materialize_layer_ep_workload(
            routing_ratios=generate_moe_routing_ratios(
                total_expert_num=model.num_experts, distribution_type="balanced", seed=42, layer_id=layer),
            target_replica_id=0, global_layer_id=layer, routing_token_count=tokens,
            router_topk=model.num_experts_per_tok, total_expert_num=model.num_experts,
            moe_expert_parallel_size=expert_parallel_size,
            expert_to_ep=build_contiguous_expert_ownership(model.num_experts, expert_parallel_size),
        )
        for ep_id in workload.participant_ep_ids:
            lane = workload.lane(ep_id)
            if lane.routed_token_count == 0:
                continue
            counts = list(lane.local_token_counts)
            features = MoELoadImbalanceInput(
                num_tokens=tokens, num_experts_per_device=lane.local_expert_width,
                hidden_dim=model.embedding_dim, expert_hidden_dim=model.mlp_hidden_dim,
                router_topk=lane.router_topk, expert_token_counts=counts,
            ).to_features_dict()
            rows.append({**template.to_dict(), **features, "expert_token_counts": json.dumps(counts)})
    pd.DataFrame(rows).drop_duplicates().to_csv(paths[1], index=False)
    return paths


def position_compute_tables(directory, model, *, device, measurement):
    """Fresh CPU-only source/count rows; synthetic Triton receipts are never admission."""
    from frontier.entities import Batch, Request
    from frontier.entities.batch import DummyForwardBatch
    from frontier.scheduler.utils.forward_start_agreement import bind_forward_start
    from frontier.moe_routing_policy import build_wave_routing_source
    from frontier.profiling.moe.inputs import MoESourceFrame, MoERoutingProfileInput, MoERoutingProfileCell
    from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
    from frontier.operators.families import get_moe_profiling_operators
    from frontier.moe_load_imbalance import moe_native_binding_context
    from frontier.execution_time_predictor.step5_cost import step5_native_expert_features
    with MonkeyPatch.context() as patch:
        metadata = ffn._metadata
        patch.setattr(ffn, "_metadata", lambda config: {**metadata(config),
            "device":device,"measurement_type":measurement.value})
        paths = ffn._tables(directory,model,token_counts=LOCAL_COMPUTE_WIDTHS,
            expert_parallel_size=16,include_stage_boundaries=True)
    rows, inputs = [], []
    groups = resolve_routed_layer_groups(model)
    profile = model.get_model_architecture_profile()
    metadata = ffn._metadata(model)
    contracts_by_group = {group.layer_range:{op.name:profile.resolve_layer_contract(model,
        layer_id=group.layer_ids[0],operator_name=op.name,moe_tp_size=1,expert_parallel_size=16).typed_metadata_identity()
        for op in get_moe_profiling_operators(dict(model_architecture_profile="step5",routing_runtime_path="position_cyclic",
            quant_dtype="float8_e4m3fn" if group.precision.name=="FP8" else None,defer_input_quant=False))} for group in groups}
    for query, context in [(prefix,0) for prefix in (6112,6128)] + [(tail,prefix) for _,prefix,tail,_ in PREFILL_CASES] + [(1,c) for c in range(6143,7167)]:
        request = Request(0,query+context,1024)
        request._num_processed_tokens = context
        batch, dummy = Batch(0,[request],[query],is_moe=True), DummyForwardBatch(0,0,0)
        bind_forward_start([batch,dummy],group_id=len(inputs),request_capacity=128)
        source = build_wave_routing_source({0:batch,1:dummy},policy_id="position_cyclic",
            cluster_type=ClusterType.DECODE,gathered_physical_rows=sum(b.forward_start_descriptor.num_tokens for b in (batch,dummy)),model_config=model)
        frames = tuple(MoESourceFrame(lane.lane_id,tuple(i.length for i in lane.intervals),
            tuple(i.start for i in lane.intervals),lane.physical_rows,lane.gating_runtime_context) for lane in source.source_lanes)
        routing = MoERoutingProfileInput(f"cpu_fixture_q{query}_c{context}",frames,source.gathered_physical_rows)
        inputs.append(routing)
        for group in groups:
            workload = materialize_layer_ep_workload(target_replica_id=0,global_layer_id=group.layer_ids[0],
                routing_token_count=source.gathered_physical_rows,router_topk=model.num_experts_per_tok,
                total_expert_num=model.num_experts,moe_expert_parallel_size=16,
                expert_to_ep=build_contiguous_expert_ownership(model.num_experts,16),policy_id="position_cyclic",routing_source=source)
            for scope,source_id,ep_id in [("source_local",i,None) for i in range(2)] + [("dp_gathered",i//8,i) for i in range(16)]:
                frame = frames[source_id]
                physical = frame.physical_rows if ep_id is None else source.gathered_physical_rows
                counts = None if ep_id is None else workload.lane(ep_id).local_token_counts
                native = dict(model_architecture_profile="step5",routing_runtime_path="position_cyclic",
                    quant_dtype="float8_e4m3fn" if group.precision.name == "FP8" else None,
                    defer_input_quant=False,backend="cpu_fixture_TRITON",experts_class="TritonExperts",
                    physical_scope="native_experts_inclusive",activation_layout=dict(shape=[source.gathered_physical_rows,model.embedding_dim],
                        stride=[model.embedding_dim,1],dtype="bfloat16",element_size=2),
                    native_padding=dict(scheme="triton_expert_assignment",alignment=16,
                        num_tokens_post_padded=0 if counts is None else sum((n+15)//16*16 for n in counts)))
                def layout(rows,columns,dtype,element_size):
                    return dict(shape=[rows,columns],stride=[columns,1],dtype=dtype,element_size=element_size)
                fp8 = native["quant_dtype"] is not None
                dtype, element_size = (native["quant_dtype"],1) if fp8 else ("bfloat16",2)
                native.update(native_facts=dict(shared=dict(evidence_kind="cpu_fixture",source_receipt="CPU_CONTRACT_FIXTURE_NOT_NATIVE")),
                    activation_layout=layout(source.gathered_physical_rows,model.embedding_dim,dtype,element_size),
                    source_activation_layout=layout(frame.physical_rows,model.embedding_dim,dtype,element_size),
                    scale_layout=None,source_scale_layout=None)
                if fp8:
                    columns = (model.embedding_dim+model.quantization_config.weight_block_size[1]-1)//model.quantization_config.weight_block_size[1]
                    native.update(scale_layout=layout(source.gathered_physical_rows,columns,"float32",4),
                                  source_scale_layout=layout(frame.physical_rows,columns,"float32",4))
                def tensor(name,dtype,element_size,columns):
                    return dict(name=name,dtype=dtype,element_size=element_size,columns=columns,row_stride=columns)
                native["collective_tensors"] = dict(dispatch=[tensor("activation",dtype,element_size,model.embedding_dim),
                    tensor("weights","float32",4,model.num_experts_per_tok),tensor("ids","int32",4,model.num_experts_per_tok),
                    *([] if not fp8 else [tensor("scales","float32",4,columns)])],output=[tensor("output","bfloat16",2,model.embedding_dim)])
                specs = [op for op in get_moe_profiling_operators(native) if op.ep_agnostic == (ep_id is None)]
                value = .01+physical/10000
                contracts = {op.name:contracts_by_group[group.layer_range][op.name] for op in specs}
                row = dict(num_tokens=physical,num_tensor_parallel_workers=1,expert_parallel_size=16,
                    num_experts=model.num_experts,router_topk=model.num_experts_per_tok,hidden_dim=model.embedding_dim,
                    expert_hidden_dim=contracts[specs[0].name]["effective_ffn_width"],
                    gating_runtime_context=frame.gating_runtime_context,model_architecture_profile="step5",
                    routing_runtime_path="position_cyclic",typed_operator_contracts=contracts,
                    time_stats={op.name:dict(median=value,kernel_count=3) for op in specs},**group.row_metadata())
                cell = MoERoutingProfileCell(routing,scope,source_id,ep_id)
                live = frame.live_rows if ep_id is None else source.live_rows
                row.update(metadata,device=device,measurement_type=measurement.value,moe_frame_scope=scope,
                    moe_input_id=routing.moe_input_id,moe_source_lane_id=source_id,moe_ep_rank=ep_id,
                    live_routing_token_count=live,masked_routing_token_count=physical-live,
                    graph_mode=source.source_lanes[source_id].graph_mode.lower(),native_facts=json.dumps(native["native_facts"]),
                    routing_input=json.dumps(cell.to_json()["routing_input"]),native_context=json.dumps(native),
                    native_binding=json.dumps(moe_native_binding_context(native),sort_keys=True,separators=(",",":")),moe_quantization_mode=group.precision.value)
                if ep_id is not None:
                    row.update(step5_native_expert_features(native,physical_rows=physical,live_assignments=sum(counts)),expert_token_counts=json.dumps(counts))
                rows.append(row)
    frame = pd.json_normalize(rows)
    frame["typed_operator_contracts"] = [json.dumps(row["typed_operator_contracts"]) for row in rows]
    frame = frame.drop(columns=[name for name in frame if name.startswith("typed_operator_contracts.")])
    frame.to_csv(paths[1],index=False)
    return paths, inputs


def position_collective_costs(output, table, *, device):
    """Write explicit CPU plan costs, not fitted or native-qualified parameters."""
    from dataclasses import asdict
    from frontier.cc_backend.collective_plan import CollectivePlan, CollectiveTensor, CollectivePlanCost
    counts = {}
    frame = pd.read_csv(table)
    for row in frame.loc[frame.moe_frame_scope.eq("dp_gathered"),["routing_input","native_context","graph_mode"]].drop_duplicates().itertuples(index=False):
        routing,native = json.loads(row.routing_input),json.loads(row.native_context)
        sources = routing["source_frames"]
        sizes = tuple(s["physical_rows"] for s in sources)
        mode = "full" if row.graph_mode == "full" else "none"
        measured = dict(shared=native["native_facts"]["shared"],execution_policy={str(rank):dict(
            runtime=dict(evidence_kind="cpu_fixture"),branches=["cpu_fixture"],buffers=dict(evidence_kind="cpu_fixture")) for rank in range(16)})
        buffers = dict(evidence_kind="cpu_fixture",compute_context=native)
        for phase in ("dispatch","combine","routed_tp","shared_tp"):
            tensors = tuple(CollectiveTensor(**t) for t in native["collective_tensors"]["dispatch" if phase=="dispatch" else "output"])
            groups = tuple((i,8+i) for i in range(8)) if phase in ("dispatch","combine") else (tuple(range(8)),tuple(range(8,16)))
            plan = CollectivePlan(phase,sizes,tensors,groups,mode,json.dumps(measured,sort_keys=True),json.dumps(buffers,sort_keys=True))
            receipt = dict(evidence_kind="cpu_fixture",kernel_counts=[3]*16,per_rank_events=[["cpu_fixture"]]*16,
                source_receipt="CPU_CONTRACT_FIXTURE_NOT_NATIVE",context=measured,buffer_receipt=buffers,plan_key=plan.cache_key())
            cost = CollectivePlanCost(.03,.01,(.01,)*16,.01,(3,)*16,json.dumps(receipt,sort_keys=True),plan.cache_key(),"CPU_CONTRACT_FIXTURE_NOT_NATIVE")
            counts[plan.cache_key()] = dict(receipt,cost=asdict(cost))
    path = output/"fixture_collectives"/device/"collective_local.json"
    path.parent.mkdir(parents=True,exist_ok=False)
    path.write_text(json.dumps(dict(schema_version="step5.collective_local.v1",device_type=device,
        evidence_kind="cpu_fixture",native_admitted=False,candidate_only=True,counts=counts),sort_keys=True)+"\n")
    return path


def refresh_compute(output: Path):
    """Repair only synthetic compute inputs; retain attention output and the shared cache."""
    path = output / "fixture_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["evidence_kind"] != "cpu_fixture" or manifest["native_admitted"]:
        raise ValueError("Only unadmitted CPU fixture compute inputs may be refreshed")
    if manifest["schema"] != "step5_measured_compute_cpu_fixture.v2":
        raise ValueError("Historical full-prompt fixtures are immutable; generate a new P-target fixture")
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        directory = output / f"fixture_compute_{measurement.value.lower()}"
        for name in ("linear_op.csv", "moe.csv"):
            source = directory / name
            backup = directory / f"{name}.before_mrv2_compute"
            with backup.open("xb") as stream:
                stream.write(source.read_bytes())
        compute_tables(directory, model, device=manifest["device"],
                       expert_parallel_size=manifest["moe_ep"], measurement=measurement)
    manifest["compute_coverage"] = {
        "local_widths": LOCAL_COMPUTE_WIDTHS, "aggregate_widths": AGGREGATE_COMPUTE_WIDTHS,
        "routing": "existing balanced Hamilton materializer, exact nonempty EP lanes",
    }
    manifest["bundles"] = {}
    manifest["limitations"].append("Compute changed; actual CLI must rebuild shared compute bundles")
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def generate(output: Path, *, device: str, gpus_per_role: int, nixl_read_profile: str | None = None, position_cyclic=False):
    if gpus_per_role <= 0 or gpus_per_role % 8:
        raise ValueError("Fixture GPUs per role must be divisible by attention TP8")
    output.mkdir(parents=True, exist_ok=False)
    if nixl_read_profile is None:
        from tests.integration.run_step5_measured_main import write_read_cost

        read_cost = output / "synthetic_read_cost.json"
        write_read_cost(read_cost)
        nixl_read_profile = str(read_cost)
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    inputs = {}
    # Explicit fixture topology and reached single-request cases, not native evidence.
    ep, capacity, block = gpus_per_role, MAX_MODEL_LEN, 16
    cases = [(6128, 0, "prefill", "none"), (16, 6128, "prefill", "piecewise"),
             (1, "6144..7166 inclusive", "decode", "full")]
    rows_by_measurement = {measurement: [] for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT)}
    for row in reached_rows(model, device=device):
        rows_by_measurement[step5_measurement_type(row["graph_mode"])].append(row)
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        path = output / f"fixture_attention_{measurement.value.lower()}.csv"
        rows_to_dataframe(rows_by_measurement.pop(measurement), measurement_type=measurement).to_csv(path, index=False)
        inputs[f"attention_{measurement.value}"] = str(path)
    compute = {}
    for measurement in (MeasurementType.KERNEL_ONLY, MeasurementType.CUDA_EVENT):
        directory = output / f"fixture_compute_{measurement.value.lower()}"
        directory.mkdir()
        if position_cyclic:
            with MonkeyPatch.context() as patch:
                patch.setattr(global_vars,"DECODE_CUDA_GRAPH_MODE","full_and_piecewise")
                patch.setattr(global_vars,"CUDAGRAPH_CAPTURE_SIZES",list(CAPTURE_SIZES))
                compute[measurement], routing_inputs = position_compute_tables(directory,model,device=device,measurement=measurement)
        else:
            compute[measurement] = compute_tables(directory, model, device=device,
                                                 expert_parallel_size=ep, measurement=measurement)
    collective_path = position_collective_costs(output,compute[MeasurementType.CUDA_EVENT][1],device=device) if position_cyclic else None
    gap = output / "fixture_kernel_gap.csv"
    pd.DataFrame({"execution_mode": ["eager", "cuda_graph"],
                  "num_tensor_parallel_workers": [8, 8],
                  "kernel_gap_us": [1., 2.]}).to_csv(gap, index=False)
    options = dict(
        enable_dummy_mode=False, skip_cpu_overhead_modeling=True,
        linear_op_input_file=str(compute[MeasurementType.CUDA_EVENT][0]),
        moe_input_file=str(compute[MeasurementType.CUDA_EVENT][1]),
        linear_op_kernel_only_input_file=str(compute[MeasurementType.KERNEL_ONLY][0]),
        moe_kernel_only_input_file=str(compute[MeasurementType.KERNEL_ONLY][1]),
        atten_input_file=inputs["attention_CUDA_EVENT"],
        atten_kernel_only_input_file=inputs["attention_KERNEL_ONLY"],
        kernel_gap_input_file=str(gap), num_estimators=[2], max_depth=[3],
        min_samples_split=[2], k_fold_cv_splits=2, num_training_job_threads=1,
    )
    config = RandomForrestExecutionTimePredictorConfig(**options)
    config.nixl_read_profile = nixl_read_profile
    replica = ffn._replica(model)
    replica.device, replica.moe_expert_parallel_size = device, ep
    replica.moe_routing_distribution_type, replica.moe_routing_seed = "position_cyclic" if position_cyclic else "balanced", 42
    replica.attn_dp, replica.num_pipeline_stages = gpus_per_role // 8, 1
    replica.kv_cache_dtype = "auto"
    replica.network_device = "cpu_fixture_network"
    replica.speculative_decoding_config = SimpleNamespace(enabled=False)
    scheduler = _scheduler()
    scheduler.max_model_len = capacity
    scheduler.max_tokens_in_batch, scheduler.batch_size_cap = 8192, 128
    scheduler.long_prefill_token_threshold, scheduler.enable_chunked_prefill = 0, False
    scheduler.get_type = lambda: "cpu_fixture_scheduler"
    backend = SimpleNamespace(get_type=lambda: CCBackendType.COLLECTIVE_SIM,
                              get_name=lambda: "collective_sim")
    cluster = SimpleNamespace(replica_config=replica, replica_scheduler_config=scheduler,
                              execution_time_predictor_config=config, cc_backend_config=backend)
    with MonkeyPatch.context() as patch:
        patch.setattr(global_vars, "get_sys_arch", lambda: "pd-disaggregation")
        patch.setattr(global_vars, "get_decode_cuda_graph_mode", lambda: "full_and_piecewise")
        manager = ExecutionTimePredictionModelManager(
            {ClusterType.PREFILL: cluster, ClusterType.DECODE: cluster},
            SimpleNamespace(cache_dir=str(output / "cache")),
        )
    manifest = {
        "schema": "step5_measured_compute_cpu_fixture.v2", "evidence_kind": "cpu_fixture",
        "native_admitted": False, "device": device, "attention_tp": 8, "attention_dp": gpus_per_role // 8,
        "moe_tp": 1, "moe_ep": ep, "global_layer_ids": list(range(model.num_layers)),
        "max_model_len": capacity, "block_size": block, "batch_width": 1,
        "decode_cuda_graph_mode": "full_and_piecewise", "cudagraph_capture_sizes": CAPTURE_SIZES,
        "cases": cases, "predictor_config": {**options, "nixl_read_profile": nixl_read_profile},
        "prefill_cases": [dict(num_prefill_tokens=prompt, prefill_compute_target=prefix,
                               decode_tail_tokens=tail, decode_physical_width=physical)
                          for prompt, prefix, tail, physical in PREFILL_CASES],
        "compute_coverage": {"local_widths": LOCAL_COMPUTE_WIDTHS,
                             "aggregate_widths": AGGREGATE_COMPUTE_WIDTHS,
                             "routing": "existing balanced Hamilton materializer, exact nonempty EP lanes"},
        "routing_distribution_type": "balanced", "routing_seed": 42,
        "cache_dir": str(output / "cache"),
        "bundles": {str(role.value): receipts for role, receipts in manager._step5_model_bundles.items()},
        "limitations": ["Synthetic timings/counts; no native admission or accuracy claim",
                        "CPU and communication isolation are explicit",
                        "Unlisted physical shapes remain coverage errors"],
    }
    if position_cyclic:
        from dataclasses import asdict
        manifest.update(schema="step5_position_compute_cpu_fixture.v1",routing_distribution_type="position_cyclic",
            routing_inputs=[asdict(item) for item in routing_inputs],collective_plan_input_file=str(collective_path),
            compute_coverage=dict(local_widths=LOCAL_COMPUTE_WIDTHS,routing="position_cyclic supplied sources; all physical EP cells including live0"))
    (output / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("h800", "h200"))
    parser.add_argument("--gpus-per-role", type=int, choices=(16,))
    parser.add_argument("--refresh-compute", action="store_true")
    parser.add_argument("--nixl-read-profile")
    parser.add_argument("--position-cyclic",action="store_true")
    args = parser.parse_args()
    if args.refresh_compute:
        result = refresh_compute(args.output_dir.resolve())
    else:
        if args.device is None or args.gpus_per_role is None:
            parser.error("Generation requires --device and --gpus-per-role")
        result = generate(args.output_dir.resolve(), device=args.device, gpus_per_role=args.gpus_per_role,
                          nixl_read_profile=args.nixl_read_profile,position_cyclic=args.position_cyclic)
    print(json.dumps({"manifest": str(args.output_dir.resolve() / "fixture_manifest.json"),
                      "native_admitted": False, "roles": list(result["bundles"])}))
