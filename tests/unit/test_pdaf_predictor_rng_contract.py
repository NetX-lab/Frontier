from types import SimpleNamespace

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.types import ExecutionTimePredictorType


def test_shared_manager_random_forest_uses_fixed_predictor_seed() -> None:
    manager = object.__new__(ExecutionTimePredictionModelManager)
    config = SimpleNamespace(
        get_type=lambda: ExecutionTimePredictorType.RANDOM_FORREST,
        num_estimators=[250, 500, 750],
        max_depth=[8, 16, 32],
        min_samples_split=[2, 5, 10],
    )

    estimator, _ = manager._create_estimator_and_params(config)

    assert estimator.random_state == 0
