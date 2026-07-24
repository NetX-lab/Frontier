"""Unit tests for mixed-layer MoE/dense detection via BaseModelConfig.is_moe_layer()."""

import pytest

from frontier.config.model_config import BaseModelConfig


class TestStepMoeNoquantLayerDetection:
    """Validate is_moe_layer() for step-moe-noquant (61 layers, mixed MoE/dense)."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        self.model_config = BaseModelConfig.create_from_name("step-moe-noquant")

    def test_total_layers(self):
        assert self.model_config.num_layers == 61

    def test_dense_layers_0_to_3(self):
        for layer_id in range(4):
            assert not self.model_config.is_moe_layer(layer_id), (
                f"Layer {layer_id} should be dense, got MoE"
            )

    def test_moe_layers_4_to_59(self):
        for layer_id in range(4, 60):
            assert self.model_config.is_moe_layer(layer_id), (
                f"Layer {layer_id} should be MoE, got dense"
            )

    def test_dense_layer_60(self):
        assert not self.model_config.is_moe_layer(60), (
            "Layer 60 should be dense, got MoE"
        )

    def test_total_moe_layers_count(self):
        assert self.model_config.get_num_moe_layers() == 56

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            self.model_config.is_moe_layer(61)
        with pytest.raises(ValueError):
            self.model_config.is_moe_layer(-1)
