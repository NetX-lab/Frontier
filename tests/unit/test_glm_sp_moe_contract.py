"""Sequence-parallel GLM shared-expert row selection and token widths."""

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.operators.spec import TensorParallelMode
from frontier.operators.typed_contracts import serialize_typed_operator_contracts
from frontier.types import ClusterType


class Predictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def glm_predictor(*, dp=4, pp=1):
    predictor = Predictor.__new__(Predictor)
    predictor._model_config = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    predictor._cluster_type = ClusterType.PREFILL
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8, attn_dp=dp, num_pipeline_stages=pp,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=8 * dp,
    )
    return predictor


@pytest.mark.parametrize("dp,pp,expected_tp", [(4, 1, 1), (1, 1, 8), (4, 2, 8)])
def test_glm_shared_expert_contract(dp, pp, expected_tp):
    predictor = glm_predictor(dp=dp, pp=pp)
    contract = predictor._resolve_typed_linear_contract("share_expert_up_proj")
    assert contract.tensor_parallel_size == expected_tp
    assert contract.tensor_parallel_mode is (
        TensorParallelMode.REPLICATED if expected_tp == 1
        else TensorParallelMode.ATTENTION_TP
    )
    assert predictor._get_linear_op_tp_key("share_expert_up_proj") == expected_tp


def test_glm_shared_expert_csv_selects_replicated_sp_row(tmp_path):
    predictor = glm_predictor()
    selected = predictor._resolve_typed_linear_contract("share_expert_up_proj")
    rows = []
    for dp, time_ms in [(1, 8.0), (4, 1.0)]:
        contract = glm_predictor(dp=dp)._resolve_typed_linear_contract("share_expert_up_proj")
        rows.append({
            "num_tensor_parallel_workers": contract.tensor_parallel_size,
            "n_expanded_embd": contract.effective_ffn_width,
            "typed_operator_contracts": serialize_typed_operator_contracts({
                "share_expert_up_proj": contract.typed_metadata_identity(),
            }),
            "num_tokens": 4,
            "time_stats.share_expert_up_proj.median": time_ms,
        })
    path = tmp_path / "linear_op.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    manager = object.__new__(ExecutionTimePredictionModelManager)
    loaded = manager._load_linear_op_df(
        str(path), selected.tensor_parallel_size,
        layer_contract=selected, operator_name="share_expert_up_proj",
    )
    assert loaded["time_stats.share_expert_up_proj.median"].tolist() == [1.0]
