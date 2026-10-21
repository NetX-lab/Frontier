"""Typed synthetic CPU fixtures; full subprocess acceptance is the paired runner."""

import json
import pytest

from frontier.attention.step5_inputs import MRv2DummyDescriptor
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.step5 import step5_row_features
from frontier.profiling.step5.inputs import Step5ProfileInput
from tests.integration.step5_measured_compute_fixture import attention_row, CAPTURE_SIZES
from tests.integration.run_step5_measured_main import command, write_read_cost


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
    for role in ("prefill", "decode"):
        for key, value in (("num_pipeline_stages", "1"), ("attn_tensor_parallel_size", "8"),
                           ("moe_expert_parallel_size", "16")):
            assert argv[argv.index(f"--cluster_config_{role}_replica_config_{key}") + 1] == value
    assert document["measurements"][0]["num_computed_tokens"] == 6128
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
    assert set(linear.num_tokens) == {1, 2, 16, 32, 6144, 6145}
    assert set(moe.num_tokens) == {2, 32, 6145}
    assert set(moe.measurement_type) == {"CUDA_EVENT"}
    assert set(moe.evidence_kind) == {"cpu_fixture"}
    for layer, group in ((3, "3-87"), (88, "88-90")):
        for tokens in (2, 32, 6145):
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
        for tokens in (2, 32, 6145):
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
