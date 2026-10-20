"""Measured expert backends own permutation exactly once."""

from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import SklearnMoEExecutionTimePredictor
from frontier.moe_routing_runtime import moe_permutation_is_fused
from frontier.moe_load_imbalance import MoELoadImbalanceInput
from frontier.types import ClusterType


class Predictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


@pytest.mark.parametrize("backend,fused", [("FLASHINFER_CUTLASS", True), ("MARLIN", False), (None, False)])
def test_training_skips_only_fused_shuffle(tmp_path, monkeypatch, backend, fused):
    model = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    rows = []
    for tokens in [1, 8]:
        row = dict(
            num_tokens=tokens, num_experts=model.num_experts,
            router_topk=model.num_experts_per_tok, hidden_dim=model.embedding_dim,
            expert_hidden_dim=model.mlp_hidden_dim, num_tensor_parallel_workers=1,
            expert_parallel_size=32, routing_runtime_path="grouped_topk_sigmoid",
            gating_runtime_context="standalone_legacy", gating_runtime_context_impl="none",
        )
        row.update(MoELoadImbalanceInput(
            num_tokens=tokens, num_experts_per_device=model.num_experts // 32,
            hidden_dim=model.embedding_dim, expert_hidden_dim=model.mlp_hidden_dim,
            router_topk=model.num_experts_per_tok,
            expert_token_counts=[tokens] * (model.num_experts // 32),
            load_distribution="runtime",
        ).to_features_dict())
        for name in ["moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling", "moe_grouped_gemm"]:
            row[f"time_stats.{name}.median"] = float(tokens)
        if backend is not None:
            row["moe_native_backend"] = backend
        rows.append(row)
    path = tmp_path / "moe.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    predictor = Predictor.__new__(Predictor)
    predictor._model_config = model
    predictor._cluster_type = ClusterType.PREFILL
    predictor._replica_config = SimpleNamespace(
        moe_tensor_parallel_size=1, moe_expert_parallel_size=32,
        attn_tensor_parallel_size=8, model_name="zai-org/GLM-5.3",
    )
    predictor._moe_input_file = str(path)
    predictor._moe_routing_distribution_type = "balanced"
    predictor._get_profiling_metadata = lambda *_: None
    predictor._validate_active_measurement_type = lambda *_: None
    predictor._register_profiling_metadata_for_ops = lambda *_: None
    predictor._train_model = lambda **kw: LinearRegression().fit(kw["df"][kw["feature_cols"]], kw["df"][kw["target_col"]])
    trained = predictor._train_moe_models()
    assert ("moe_shuffling" not in trained) is fused
    assert getattr(trained["moe_grouped_gemm"], "_frontier_moe_permutation_fused", False) is fused
    predictor._models = trained
    predictor._max_tokens = 8
    monkeypatch.setattr(SklearnExecutionTimePredictor, "_predict_for_compute_models", lambda _: {})
    predictor._predictions = predictor._predict_for_compute_models()
    if fused:
        assert predictor._get_moe_shuffling_time(SimpleNamespace()) == 0.0
    else:
        assert "moe_shuffling" not in predictor._predictions


def test_mixed_permutation_ownership_is_not_one_model():
    with pytest.raises(ValueError, match="mix fused and standalone"):
        moe_permutation_is_fused(pd.DataFrame({"moe_native_backend": ["FLASHINFER_CUTLASS", "MARLIN"]}))


def test_unclassified_backend_is_explicit():
    with pytest.raises(ValueError, match="Unclassified"):
        moe_permutation_is_fused(pd.DataFrame({"moe_native_backend": ["unclassified"]}))
