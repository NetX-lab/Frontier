"""Typed synthetic CPU fixtures; full subprocess acceptance is the paired runner."""

import json
import pytest

from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.step5 import step5_row_features
from frontier.profiling.step5.inputs import Step5ProfileInput
from tests.integration.step5_measured_compute_fixture import (
    attention_row, CAPTURE_SIZES, LOCAL_COMPUTE_WIDTHS, AGGREGATE_COMPUTE_WIDTHS,
)
from tests.integration.run_step5_measured_main import command, write_read_cost


@pytest.mark.parametrize("quant,deferred", [(None, False), ("float8_e4m3fn", False), ("float8_e4m3fn", True)])
def test_position_source_wrapper_uses_actual_local_widths_and_selected_prepare(tmp_path, monkeypatch, quant, deferred):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from frontier.profiling.moe import fp8_block_experts, moe_wrapper
    from frontier.profiling.moe.inputs import MoESourceFrame, MoERoutingProfileInput, MoERoutingProfileCell, source_frame_tensors
    from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
    from frontier.profiling.common.runtime_identity import canonical_json
    wrapper = object.__new__(moe_wrapper.MoEWrapper)
    wrapper.model_config = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    wrapper.routed_layer_group = resolve_routed_layer_groups(wrapper.model_config)[0]
    wrapper.routing_runtime_path = "position_cyclic"
    wrapper.routing_runtime_metadata = dict(routing_runtime_path="position_cyclic")
    wrapper.gating_runtime_context_metadata = dict(gating_runtime_context="standalone_legacy")
    wrapper.num_tensor_parallel_workers, wrapper.expert_parallel_size = 1, 16
    wrapper.num_experts, wrapper.num_experts_per_device, wrapper.router_topk = 352, 22, 8
    wrapper.hidden_dim, wrapper.expert_hidden_dim, wrapper.use_gated = 4, 4, True
    wrapper.quantization_mode, wrapper.activation, wrapper.block_shape = "fp8", "swiglustep", (128,128)
    wrapper._dtype = torch.bfloat16
    facts = dict(shared=dict(evidence_kind="cpu_fixture"), imports={}, callables={})
    observed, prepared = [], []
    def bind(frame, *, device):
        wrapper.positions, wrapper.mask = source_frame_tensors(frame, device=device)
    wrapper.gating = SimpleNamespace(native_facts=facts, linear_backend="cpu_fixture", bind_source_frame=bind)
    wrapper._materialize_gating_hidden_states = lambda rows: torch.zeros((rows,4), dtype=torch.bfloat16)
    def gate(hidden, *, gating_runtime_context):
        observed.append((hidden.shape[0],gating_runtime_context))
        ids = (wrapper.positions[:,None]+torch.arange(8)) % 352
        ids[wrapper.mask] = -1
        return torch.ones_like(ids,dtype=torch.float32), ids.to(torch.int32)
    wrapper._run_gating_with_runtime_context = gate
    def prepare(value):
        prepared.append(value.shape[0])
        return value, None
    def context(value, scale, *, topk_ids, expert_map, source_tensors):
        assert value.shape == (17,4) and topk_ids.shape == (17,8)
        assert source_tensors[0].shape == (16,4)
        return dict(quant_dtype=quant,defer_input_quant=deferred,backend="cpu_fixture",
            experts_class="TritonExperts",physical_scope="native_experts_inclusive",native_facts=facts)
    native = SimpleNamespace(prepare_source=prepare,prepared_context=context,provenance=facts,
                             apply_prepared=lambda *a,**k: pytest.fail("source cell cannot time gathered experts"))
    monkeypatch.setattr(fp8_block_experts,"get_native_routed_experts",lambda **k:native)
    class Timer:
        def __init__(self,name): self.name=name
        def __enter__(self): return self
        def __exit__(self,*exc): pass
    monkeypatch.setattr(moe_wrapper,"CudaTimer",Timer)
    def profile(run,names):
        before=len(prepared)
        run()
        assert prepared[before:] == ([16] if quant is not None and not deferred else [])
        return {name:dict(median=1.,kernel_count=2) for name in names}
    wrapper._profile_cell_callable = profile
    frames = (MoESourceFrame(0,(),(),16,"prefill_hot"), MoESourceFrame(1,(1,),(6128,),1,"standalone_legacy"))
    cell = MoERoutingProfileCell(MoERoutingProfileInput("cpu_empty_local",frames,17),"source_local",0)
    row = wrapper.profile(routing_cell=cell.to_json())
    expected={"moe_gating_linear","moe_gating_routing_topk"}
    if quant is not None and not deferred: expected.add("moe_input_quantize")
    assert set(row["time_stats"]) == set(row["typed_operator_contracts"]) == expected
    assert row["num_tokens"] == 16 and row["live_routing_token_count"] == 0 and row["masked_routing_token_count"] == 16
    assert row["moe_source_lane_id"] == 0 and row["moe_ep_rank"] is None
    assert observed == [(16,"prefill_hot"),(1,"standalone_legacy"),(16,"prefill_hot")]
    assert row["native_facts"] == canonical_json(facts)
    assert json.loads(row["native_context"])["defer_input_quant"] is deferred
    assert json.loads(row["routing_input"])["gathered_physical_rows"] == 17


def test_real_graph_padding_does_not_duplicate_requests_or_context():
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    cell = Step5ProfileInput(3, (17,), (6128,), "prefill", "piecewise", block_table_columns=448)
    row = attention_row(model, cell, physical_query_rows=24)
    features = step5_row_features(row)
    assert json.loads(row["query_lens"]) == [17]
    assert json.loads(row["context_lens"]) == [6128]
    assert (features["batch_size"], features["valid_tokens"], features["padded_tokens"]) == (1., 17., 7.)
    assert row["native_workload"] == "attention_workload"
    assert row["evidence_kind"] == "cpu_fixture"
    assert 24 in CAPTURE_SIZES and max(CAPTURE_SIZES) == 256


@pytest.mark.parametrize("tokens,requests,mode,uniform", [
    (1, 1, "none", None), (16, None, "piecewise", None), (1, 1, "full", 1),
])
def test_mrv2_dummy_rows_preserve_inputbatch_descriptor(tokens, requests, mode, uniform):
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    descriptor = MRv2DummyDescriptor(tokens, requests, mode, None, "CPU fixture", uniform)
    row = attention_row(model, Step5ProfileInput.from_mrv2_dummy(3, descriptor, block_table_columns=448))
    step5_row_features(row)
    source = json.loads(row["dummy_descriptor"])
    assert source["num_reqs"] == requests and source["max_query_len"] is None
    assert json.loads(row["query_lens"]) == list(descriptor.query_lens)
    assert row["native_workload"] == "mrv2_dp_dummy"
    assert row["dummy_owned_requests"] == 0 and row["has_attention_metadata"] == 1
    assert row["block_table_columns"] == 448


def test_actual_main_command_retains_capacity_graph_topology_and_measured_read(tmp_path):
    fixture = dict(decode_cuda_graph_mode="full_and_piecewise", cudagraph_capture_sizes=CAPTURE_SIZES,
                   max_model_len=7168, cache_dir=str(tmp_path / "cache"), predictor_config={})
    path = tmp_path / "read.json"
    document = write_read_cost(path)
    argv = command("online", tmp_path, fixture, path)
    assert argv[:4] == [argv[0], "-B", "-m", "frontier.main"]
    assert argv[argv.index("--vllm_v1_scheduler_config_max_model_len") + 1] == "7168"
    assert argv[argv.index("--vllm_v1_scheduler_config_batch_size_cap") + 1] == "128"
    assert argv[argv.index("--vllm_v1_scheduler_config_max_tokens_in_batch") + 1] == "8192"
    assert argv[argv.index("--decode_cuda_graph_mode") + 1] == "full_and_piecewise"
    assert "--no-use_cuda_graph" in argv and "--use_cuda_graph" not in argv
    assert argv[argv.index("--replica_config_moe_routing_distribution_type") + 1] == "balanced"
    assert argv[argv.index("--replica_config_moe_routing_seed") + 1] == "42"
    for role in ("prefill", "decode"):
        for key, value in (("num_pipeline_stages", "1"), ("attn_tensor_parallel_size", "8"),
                           ("moe_expert_parallel_size", "16")):
            assert argv[argv.index(f"--cluster_config_{role}_replica_config_{key}") + 1] == value
    assert document["measurements"][0]["num_computed_tokens"] == 6128
    assert {(row["num_prefill_tokens"], row["num_computed_tokens"]) for row in document["measurements"]} == {
        (6143, 6112), (6144, 6128), (6145, 6128)}
    assert document["measurements"][0]["transfer_bytes_per_rank"] > 0


def test_context_row_contract_reuse_matches_independent_workload_build():
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    for layer in (0, 3):
        template = attention_row(model, Step5ProfileInput(layer, (1,), (6144,), "decode", "full", block_table_columns=448))
        for context in (6145, 6500, 7166):
            reused = {**template, "context_lens": json.dumps((context,))}
            independent = attention_row(model, Step5ProfileInput(layer, (1,), (context,), "decode", "full", block_table_columns=448))
            assert reused == independent
            assert step5_row_features(reused)["max_context_len"] == context


def test_compute_fixture_covers_exact_native_bound_aggregate_and_ep_loads(tmp_path):
    import pandas as pd
    from frontier.moe_ep_workload import (
        build_contiguous_expert_ownership, generate_moe_routing_ratios,
        materialize_layer_ep_workload,
    )
    from frontier.moe_load_imbalance import MOE_LOAD_IMBALANCE_FEATURES, MoELoadImbalanceInput
    from frontier.types import MeasurementType
    from tests.integration.step5_measured_compute_fixture import compute_tables

    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    linear_path, moe_path = compute_tables(tmp_path, model, device="h800", expert_parallel_size=16,
                                         measurement=MeasurementType.CUDA_EVENT)
    linear = pd.read_csv(linear_path, float_precision="round_trip")
    moe = pd.read_csv(moe_path, float_precision="round_trip")
    assert set(linear.num_tokens) == set(LOCAL_COMPUTE_WIDTHS + AGGREGATE_COMPUTE_WIDTHS)
    assert set(moe.num_tokens) == set(AGGREGATE_COMPUTE_WIDTHS)
    assert set(moe.measurement_type) == {"CUDA_EVENT"}
    assert set(moe.evidence_kind) == {"cpu_fixture"}
    for layer, group in ((3, "3-87"), (88, "88-90")):
        for tokens in AGGREGATE_COMPUTE_WIDTHS:
            rows = moe[(moe.moe_layer_ids == group) & (moe.num_tokens == tokens)]
            workload = materialize_layer_ep_workload(
                routing_ratios=generate_moe_routing_ratios(
                    total_expert_num=352, distribution_type="balanced", seed=42, layer_id=layer),
                target_replica_id=0, global_layer_id=layer, routing_token_count=tokens,
                router_topk=8, total_expert_num=352, moe_expert_parallel_size=16,
                expert_to_ep=build_contiguous_expert_ownership(352, 16),
            )
            expected = set()
            for ep_id in workload.participant_ep_ids:
                lane = workload.lane(ep_id)
                if lane.routed_token_count == 0:
                    continue
                features = MoELoadImbalanceInput(
                    num_tokens=tokens, num_experts_per_device=lane.local_expert_width,
                    hidden_dim=model.embedding_dim, expert_hidden_dim=model.mlp_hidden_dim,
                    router_topk=8, expert_token_counts=list(lane.local_token_counts),
                ).to_features_dict()
                expected.add(tuple(features[name] for name in MOE_LOAD_IMBALANCE_FEATURES))
            actual = {tuple(row[name] for name in MOE_LOAD_IMBALANCE_FEATURES)
                      for _, row in rows.iterrows()}
            assert actual == expected
            assert all(sum(json.loads(row.expert_token_counts)) == row.total_routed_tokens
                       for _, row in rows.iterrows())
    assert moe["time_stats.moe_grouped_gemm.kernel_count"].eq(7).all()


def test_exact_ep_fixture_replays_shared_cache_duration_count_and_missing_coverage(tmp_path):
    from types import SimpleNamespace
    from frontier.moe_ep_workload import (
        build_contiguous_expert_ownership, generate_moe_routing_ratios, materialize_layer_ep_workload,
    )
    from frontier.types import ClusterType, MeasurementType
    from tests.integration.step5_measured_compute_fixture import compute_tables
    from tests.unit.test_step5_measured_ffn import _config, _replica
    from tests.unit.test_step5_measured_ffn_replay import _predictor
    from tests.unit.test_step5_measured_training import _manager

    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    paths = compute_tables(tmp_path, model, device="h800", expert_parallel_size=16,
                           measurement=MeasurementType.KERNEL_ONLY)
    replica = _replica(model)
    replica.moe_expert_parallel_size = 16
    manager = _manager(tmp_path / "cache")
    manager._cluster_configs = {ClusterType.MONOLITHIC: SimpleNamespace(replica_config=replica)}
    manager._train_ffn_models_for_cluster(
        ClusterType.MONOLITHIC, replica, _config(), str(paths[0]), str(paths[1]), True, set())
    predictor = _predictor(model, manager)
    predictor._replica_config.moe_expert_parallel_size = predictor._moe_ep_size = 16
    for layer, precision in ((3, "FP8"), (88, "BF16")):
        for tokens in AGGREGATE_COMPUTE_WIDTHS:
            workload = materialize_layer_ep_workload(
                routing_ratios=generate_moe_routing_ratios(
                    total_expert_num=352, distribution_type="balanced", seed=42, layer_id=layer),
                target_replica_id=0, global_layer_id=layer, routing_token_count=tokens,
                router_topk=8, total_expert_num=352, moe_expert_parallel_size=16,
                expert_to_ep=build_contiguous_expert_ownership(352, 16))
            batch = SimpleNamespace(get_effective_total_tokens_rounded=lambda _: tokens)
            with predictor._step5_measured_layer(layer):
                predictor._get_share_expert_up_proj_execution_time(batch)
                assert predictor._step5_ffn_costs["share_expert_up_proj"]["exact_lookup"]
                for ep_id in workload.participant_ep_ids:
                    lane = workload.lane(ep_id)
                    grouped = predictor._get_grouped_gemm_time(lane, batch=batch)
                    shuffling = predictor._get_moe_shuffling_time(batch, moe_tokens_input=lane)
                    if not lane.routed_token_count:
                        assert grouped == shuffling == 0.
                        continue
                    for name in ("moe_grouped_gemm", "moe_shuffling"):
                        proof = predictor._step5_ffn_costs[name]
                        assert proof["exact_lookup"] and proof["kernel_count"] == 7.
                        assert proof["evidence_kind"] == "cpu_fixture" and proof["duration_ms"] > 0.
                    assert predictor._step5_ffn_costs["moe_grouped_gemm"]["precision"] == precision
    with predictor._step5_measured_layer(3), pytest.raises(ValueError, match="above declared maximum"):
        predictor._get_share_expert_up_proj_execution_time(
            SimpleNamespace(get_effective_total_tokens_rounded=lambda _: 7168))


def test_corrected_attention_fixture_has_exact_canonical_boundary_shapes_and_no_old_prefill():
    from tests.integration.step5_measured_compute_fixture import reached_rows

    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    rows = []
    for row in reached_rows(model, device="h800"):
        if row["global_layer_id"] != 0:
            break
        step5_row_features(row)
        rows.append(row)
    real = [row for row in rows if row["native_workload"] == "attention_workload"]
    prefill = {(tuple(json.loads(row["query_lens"])), tuple(json.loads(row["context_lens"])),
                row["graph_mode"], row["valid_tokens"] + row["padded_tokens"])
               for row in real if row["phase"] == "prefill"}
    assert prefill == {((6112,), (0,), "none", 6112), ((6128,), (0,), "none", 6128),
                       ((31,), (6112,), "piecewise", 32), ((16,), (6128,), "piecewise", 16),
                       ((17,), (6128,), "piecewise", 24)}
    assert all(row["evidence_kind"] == "cpu_fixture" for row in rows)
    dummy = [row for row in rows if row["native_workload"] == "mrv2_dp_dummy"]
    assert {(row["graph_mode"], row["valid_tokens"] + row["padded_tokens"]) for row in dummy} == {
        ("none", 1), ("piecewise", 16), ("piecewise", 24), ("piecewise", 32), ("full", 1)}
    assert all(row["dummy_owned_requests"] == 0 for row in dummy)


def test_historical_fixture_cannot_be_read_or_refreshed(tmp_path):
    from tests.integration.run_step5_measured_main import read_profile
    from tests.integration.step5_measured_compute_fixture import refresh_compute

    path = tmp_path / "fixture_manifest.json"
    original = json.dumps(dict(schema="step5_measured_compute_cpu_fixture.v1",
                               evidence_kind="cpu_fixture", native_admitted=False))
    path.write_text(original)
    with pytest.raises(AssertionError, match="Historical full-prompt"):
        read_profile(path)
    with pytest.raises(ValueError, match="immutable"):
        refresh_compute(tmp_path)
    assert path.read_text() == original


@pytest.mark.parametrize("method_name", ["KERNEL_ONLY", "CUDA_EVENT"])
def test_corrected_shapes_fit_and_match_actual_agreed_batch_keys(tmp_path, monkeypatch, method_name):
    from frontier.config import global_vars, RandomForrestExecutionTimePredictorConfig
    from frontier.entities import Batch, Request
    from frontier.entities.batch import DummyForwardBatch
    from frontier.execution_time_predictor.step5 import STEP5_FEATURE_COLUMNS
    from frontier.execution_time_predictor.step5_cost import step5_batch_features, predict_step5_attention_time
    from frontier.scheduler.utils.forward_start_agreement import bind_forward_start
    from frontier.types import ClusterType, MeasurementType
    from tests.unit.test_step5_measured_replay import _predictor
    from tests.unit.test_step5_measured_training import _manager, _replica, _scheduler, _write

    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", list(CAPTURE_SIZES))
    model = BaseModelConfig.create_from_name("stepfun-ai/Step-5-Preview")
    method = MeasurementType[method_name]
    shapes = ([(6112, 0), (6128, 0)] if method is MeasurementType.KERNEL_ONLY
              else [(31, 6112), (16, 6128), (17, 6128), (1, 6144), (1, 7166)])
    batches, rows = [], []
    for query, context in shapes:
        request = Request(0, query + context, 1)
        request._num_processed_tokens = context
        batch = Batch(0, [request], [query], is_moe=True)
        dummy = DummyForwardBatch(0, 0, 0)
        bind_forward_start([batch, dummy], group_id=0, request_capacity=128)
        descriptor = dummy.mrv2_dummy_descriptor
        for layer in (0, 3):
            real = Step5ProfileInput(layer, (query,), (context,),
                "decode" if query == 1 else "prefill", descriptor.graph_mode, block_table_columns=448)
            rows.append(attention_row(model, real, physical_query_rows=batch.forward_start_descriptor.num_tokens))
            rows.append(attention_row(model, Step5ProfileInput.from_mrv2_dummy(
                layer, descriptor, block_table_columns=448)))
        batches.append((batch, dummy))
    path = _write(tmp_path, rows, method)
    manager = _manager(tmp_path / "cache", method)
    replica, scheduler = _replica(model), _scheduler()
    scheduler.max_model_len = 7168
    config = RandomForrestExecutionTimePredictorConfig(num_estimators=[2], max_depth=[3],
                min_samples_split=[2], k_fold_cv_splits=2, num_training_job_threads=1)
    models = manager._train_attn_models_for_cluster(ClusterType.PREFILL, replica, config, scheduler,
                str(tmp_path / "absent_linear.csv"), str(path), set())
    predictor = _predictor(model, models, method)
    predictor._step5_max_model_len = 7168
    predictor.requires_forward_start_agreement = True
    source = {tuple(step5_row_features(row)[name] for name in STEP5_FEATURE_COLUMNS): row for row in rows}
    for pair in batches:
        for batch in pair:
            for layer in (0, 3):
                features, _ = step5_batch_features(predictor, batch, layer)
                row = source[tuple(features[name] for name in STEP5_FEATURE_COLUMNS)]
                cost = predict_step5_attention_time(predictor, batch, layer)
                statistic = "median" if method is MeasurementType.KERNEL_ONLY else "mean"
                assert cost.total_time() == pytest.approx(row["time_stats"]["step5_layer_e2e"][statistic])
                assert cost.kernel_count == (17 if isinstance(batch, DummyForwardBatch) else 23)
                proof = cost.operator_costs["step5_layer_e2e"]
                assert proof.exact_lookup and proof.evidence_kind == "cpu_fixture"
                assert set(cost.operator_times.op_times) == {"step5_layer_e2e"}
