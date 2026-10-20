"""Measured FP8 kernels must not receive a second BF16-to-FP8 speedup."""

import json
import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.config.quantization_manager import QuantizationManager
from frontier.config.precision_type import PrecisionType
from frontier.execution_time_predictor.profiling_metadata import operation_profiling_precisions
from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.types import ClusterType


class Predictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


@pytest.mark.parametrize("kind,operations", [
    ("linear", ["share_expert_up_proj", "share_expert_down_proj", "mlp_up_proj"]),
    ("moe", ["moe_grouped_gemm"]),
])
def test_measured_fp8_keeps_measured_times(kind, operations):
    config = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    row = dict(profiling_precision="BF16", quant_signature=config.get_quant_signature(),
               model_arch=config.model_arch, model_architecture_profile="glm_moe_dsa",
               measurement_type="cuda_event")
    if kind == "linear":
        row["linear_backends"] = json.dumps({op: ["FlashInferFp8DeepGEMMDynamicBlockScaledKernel"]
                                             for op in operations})
    else:
        row["moe_quantization_mode"] = "fp8"
    predictor = Predictor.__new__(Predictor)
    predictor._model_config = config
    metadata = predictor._get_profiling_metadata(pd.DataFrame([row]), "fixture.csv")
    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        manager.configure_from_model_config(config)
        predictor._register_profiling_metadata_for_ops(operations, metadata, "fixture.csv")
        for op in operations:
            assert manager.get_precision(op) is PrecisionType.FP8
            assert manager.adjust_compute_time(op, 0.131, ClusterType.PREFILL) == 0.131
    finally:
        QuantizationManager.reset()


def test_legacy_file_precision_is_not_reinterpreted():
    assert operation_profiling_precisions(pd.DataFrame([{"profiling_precision": "BF16"}])) == {}
