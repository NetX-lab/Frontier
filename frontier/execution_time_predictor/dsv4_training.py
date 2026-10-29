"""One CPU training entry for all DeepSeek-V4 operator tables."""

import functools
from pathlib import Path

from frontier.execution_time_predictor.dsv4_measurement import validate_dsv4_host_time
from frontier.execution_time_predictor.dsv4_operator_predictor import (
    DeepSeekV4OperatorPredictor, load_dsv4_operator_dataframe,
    load_dsv4_moe_dataframe, train_dsv4_operator_models,
    load_dsv4_runner_dataframe, train_dsv4_runner_models,
)
from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, load_dsv4_attention_dataframe, train_dsv4_attention_models,
)
from frontier.types import MeasurementType
from frontier.execution_time_predictor.dsv4_runner import BLOCK_TABLE, REPLAY_TABLE


def train_dsv4_models(model, replica, block_size, linear_file, attention_file, moe_file, train):
    """The caller fits and caches costs and their paired kernel-count models."""
    attention = load_dsv4_attention_dataframe(attention_file, model, replica, block_size,
                                             MeasurementType.KERNEL_ONLY)
    operators = load_dsv4_operator_dataframe(linear_file, model, replica)
    moe = load_dsv4_moe_dataframe(moe_file, model, replica)
    runner_dir = Path(attention_file).parent
    block = load_dsv4_runner_dataframe(runner_dir / BLOCK_TABLE)
    replay = load_dsv4_runner_dataframe(runner_dir / REPLAY_TABLE)
    models = train_dsv4_attention_models(attention, train)
    models.update(train_dsv4_operator_models(operators, moe, train))
    models.update(train_dsv4_runner_models(block, replay, model, train))
    return models


def train_dsv4_for_predictor(predictor):
    return train_dsv4_models(
        predictor._model_config, predictor._replica_config, predictor._block_size,
        predictor._compute_input_file, predictor._attention_input_file, predictor._moe_input_file,
        predictor._train_model,
    )


def train_dsv4_for_manager(manager, cluster_type, cluster_config, linear, attention, moe):
    replica = cluster_config.replica_config
    config = cluster_config.execution_time_predictor_config
    validate_dsv4_host_time(config, replica.model_config)
    fit = functools.partial(
        manager._train_single_model, execution_time_predictor_config=config,
        training_context={"cluster_type": str(cluster_type), "device": replica.device,
                          "model_name": replica.model_name,
                          "model_architecture_profile": replica.model_config.get_model_architecture_profile().profile_id},
    )
    return train_dsv4_models(replica.model_config, replica, cluster_config.replica_scheduler_config.block_size,
                             linear, attention, moe, fit)


def dsv4_predictions(models):
    return {"dsv4_attention": DeepSeekV4AttentionPredictor(models),
            "dsv4_operators": DeepSeekV4OperatorPredictor(models)}
