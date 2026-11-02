from typing import Dict, Optional, TYPE_CHECKING

from frontier.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    ClusterConfig,
    MetricsConfig,
    ReplicaConfig,
)
from frontier.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.linear_regression_execution_time_predictor import (
    LinearRegressionExecutionTimePredictor,
)
from frontier.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from frontier.types import ClusterType, ExecutionTimePredictorType
from frontier.utils.base_registry import BaseRegistry

from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)

if TYPE_CHECKING:
    from frontier.cc_backend import BaseCCBackend


class ExecutionTimePredictorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ExecutionTimePredictorType:
        if isinstance(key_str, ExecutionTimePredictorType):
            return key_str
        return ExecutionTimePredictorType.from_str(key_str)

    @classmethod
    def get(
        cls,
        predictor_type: str,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        cluster_config: ClusterConfig = None,
        model_manager: ExecutionTimePredictionModelManager = None,
        cluster_type: ClusterType = None,
        training_file_paths: Dict[str, str] = None,
        actual_replica_ids: Optional[list] = None,
        cc_backend: Optional["BaseCCBackend"] = None,
    ) -> BaseExecutionTimePredictor:
        # Handle legacy sklearn type
        if predictor_type == "sklearn":
            return SklearnExecutionTimePredictor(
                predictor_config,
                replica_config,
                replica_scheduler_config,
                metrics_config,
                model_manager,
                cluster_type,
                training_file_paths,
                cc_backend,
            )

        # Use registry for registered types
        try:
            predictor_enum = cls.get_key_from_str(predictor_type)
            if predictor_enum not in cls._registry:
                raise ValueError(f"{predictor_type} is not registered")

            predictor_class = cls._registry[predictor_enum]

            # The factory classes (like RandomForrestExecutionTimePredictor) handle parameter filtering
            # They will create the appropriate underlying predictor class based on configuration
            return predictor_class(
                predictor_config=predictor_config,
                replica_config=replica_config,
                replica_scheduler_config=replica_scheduler_config,
                metrics_config=metrics_config,
                cluster_config=cluster_config,
                model_manager=model_manager,
                cluster_type=cluster_type,
                training_file_paths=training_file_paths,
                actual_replica_ids=actual_replica_ids,
                cc_backend=cc_backend,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to create predictor of type '{predictor_type}': {e}"
            )


# Register available predictors
ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.RANDOM_FORREST, RandomForrestExecutionTimePredictor
)
ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.LINEAR_REGRESSION, LinearRegressionExecutionTimePredictor
)
