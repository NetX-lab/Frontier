from typing import Dict, Optional, TYPE_CHECKING

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from frontier.config import (
    BaseReplicaSchedulerConfig,
    LinearRegressionExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
    ClusterConfig,
)
from frontier.config import global_vars
from frontier.types import ClusterType
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)

if TYPE_CHECKING:
    from frontier.cc_backend import BaseCCBackend


def _get_base_class(replica_config: ReplicaConfig):
    if global_vars.is_disaggregated_mode():
        from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
            SklearnDisaggregationExecutionTimePredictor,
        )

        return SklearnDisaggregationExecutionTimePredictor
    # Check if model is MoE based on model_config, NOT parallelism settings
    # (Consistent with RandomForrestExecutionTimePredictor)
    elif replica_config.model_config is not None and replica_config.model_config.is_moe:
        from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
            SklearnMoEExecutionTimePredictor,
        )

        return SklearnMoEExecutionTimePredictor
    else:
        from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
            SklearnExecutionTimePredictor,
        )

        return SklearnExecutionTimePredictor


class LinearRegressionExecutionTimePredictor:
    def __new__(
        cls,
        predictor_config: LinearRegressionExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        cluster_config: ClusterConfig = None,
        model_manager: ExecutionTimePredictionModelManager = None,
        cluster_type: ClusterType = None,
        training_file_paths: Dict[str, str] = None,
        actual_replica_ids: Optional[list] = None,
        cc_backend: Optional["BaseCCBackend"] = None,
    ):
        base_class = _get_base_class(replica_config)

        class _LinearRegressionExecutionTimePredictor(base_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _get_grid_search_params(self):
                return {
                    "polynomialfeatures__degree": self._config.polynomial_degree,
                    "polynomialfeatures__include_bias": self._config.polynomial_include_bias,
                    "polynomialfeatures__interaction_only": self._config.polynomial_interaction_only,
                    "linearregression__fit_intercept": self._config.fit_intercept,
                }

            def _get_estimator(self):
                return make_pipeline(PolynomialFeatures(), LinearRegression())

        # Build kwargs based on what the base class accepts
        # SklearnDisaggregationExecutionTimePredictor accepts cluster_config and actual_replica_ids
        # SklearnExecutionTimePredictor and SklearnMoEExecutionTimePredictor do not
        kwargs = {
            "predictor_config": predictor_config,
            "replica_config": replica_config,
            "replica_scheduler_config": replica_scheduler_config,
            "metrics_config": metrics_config,
            "model_manager": model_manager,
            "cluster_type": cluster_type,
            "training_file_paths": training_file_paths,
            "cc_backend": cc_backend,
        }

        # Only pass cluster_config and actual_replica_ids for disaggregated mode
        if global_vars.is_disaggregated_mode():
            kwargs["cluster_config"] = cluster_config
            kwargs["actual_replica_ids"] = actual_replica_ids

        return _LinearRegressionExecutionTimePredictor(**kwargs)
