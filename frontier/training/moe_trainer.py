"""
MoE (Mixture of Experts) trainer for execution time prediction models.

This module provides standalone training for MoE-specific execution time predictors,
allowing pre-training and saving of model weights for later use in simulations.
"""

import os
from typing import Dict, List

import pandas as pd

from frontier.training.base_trainer import BaseTrainer
from frontier.logger import init_logger
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
    filter_moe_gating_rows_by_runtime_context,
    get_moe_gating_base_model_name,
    has_prefill_hot_moe_gating_rows,
    validate_moe_gating_runtime_context,
)
from frontier.moe_routing_runtime import (
    STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
    filter_moe_gating_routing_topk_rows,
    validate_moe_gating_routing_runtime_path,
)
from frontier.operators.families import MOE_FAMILY, get_family_profiling_names
from frontier.operators.spec import TensorParallelMode

logger = init_logger(__name__)


def _get_moe_family_model_names() -> List[str]:
    return list(get_family_profiling_names(MOE_FAMILY))


def _get_moe_family_operator_by_model_name(model_name: str):
    base_model_name = get_moe_gating_base_model_name(model_name)
    by_model_name = {
        operator.profiling_name(): operator
        for operator in MOE_FAMILY.profiling_ops()
    }
    operator = by_model_name.get(base_model_name)
    if operator is None:
        return None
    return operator


def _get_moe_gating_family_model_names() -> List[str]:
    return [
        operator.profiling_name()
        for operator in MOE_FAMILY.profiling_ops()
        if operator.precision_name() == "moe_gating"
    ]


def _get_prefill_hot_moe_gating_model_names() -> List[str]:
    return [
        f"{model_name}__prefill_hot"
        for model_name in _get_moe_gating_family_model_names()
    ]


def _is_moe_gating_family_model_name(model_name: str) -> bool:
    base_model_name = get_moe_gating_base_model_name(model_name)
    return base_model_name in set(_get_moe_gating_family_model_names())


def _get_moe_required_target_columns() -> List[str]:
    return [
        f"time_stats.{model_name}.median"
        for model_name in _get_moe_family_model_names()
    ]


def _get_moe_replicated_target_columns() -> List[str]:
    return [
        f"time_stats.{operator.profiling_name()}.median"
        for operator in MOE_FAMILY.profiling_ops()
        if operator.tp_mode is TensorParallelMode.REPLICATED
    ]


def _get_moe_tp_target_columns() -> List[str]:
    return [
        f"time_stats.{operator.profiling_name()}.median"
        for operator in MOE_FAMILY.profiling_ops()
        if operator.tp_mode is TensorParallelMode.MOE_TP
    ]


class MoETrainer(BaseTrainer):
    """
    Trainer for MoE execution time prediction models.
    
    This trainer handles the training of three MoE-specific models:
    - moe_gating: Gating network execution time predictor
    - moe_shuffling: Token shuffling execution time predictor
    - moe_grouped_gemm: Expert computation execution time predictor
    
    The trainer loads profiling data from vidur/profiling/moe/ and filters it
    based on the provided configuration parameters.
    """

    # Full load-imbalance feature set produced by MoELoadImbalanceInput.to_features_dict().
    # Training strategy:
    # - If ALL features exist: train moe_grouped_gemm using this full feature set (load-imbalance mode)
    # - If NONE exist: train moe_grouped_gemm using num_tokens only (standard mode)
    # - If PARTIAL exist: fail-fast with a clear error (dataset is inconsistent)
    LOAD_IMBALANCE_FEATURES: List[str] = [
        "total_routed_tokens",
        "num_experts_per_device",
        "hidden_dim",
        "expert_hidden_dim",
        "router_topk",
        "model_expansion_ratio",
        "tokens_per_expert_avg",
        "tokens_to_experts_ratio",
        "expert_utilization",
        "min_load_ratio",
        "load_imbalance_cv",
        "max_load_ratio",
        "load_entropy",
        "load_gini_coefficient",
    ]
    REQUIRED_TARGET_COLUMNS: List[str] = _get_moe_required_target_columns()
    REPLICATED_TARGET_COLUMNS: List[str] = _get_moe_replicated_target_columns()
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        num_experts: int,
        router_topk: int,
        hidden_dim: int,
        expert_hidden_dim: int,
        moe_tensor_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        predictor_type: str = "random_forest",
        model_name: str = None,
        device: str = None,
        **kwargs
    ):
        """
        Initialize the MoE trainer.

        Args:
            dataset_path: Path to the MoE profiling dataset CSV file
            output_dir: Directory to save trained models
            num_experts: Total number of experts in the model
            router_topk: Number of experts selected per token
            hidden_dim: Model hidden dimension
            expert_hidden_dim: Expert FFN hidden dimension
            moe_tensor_parallel_size: MoE tensor parallel size (default: 1)
            expert_parallel_size: Expert parallel size (default: 1)
            predictor_type: Type of predictor ("random_forest" or "linear_regression")
            model_name: Model name (optional, for consistency with other trainers)
            device: Device SKU (optional, for consistency with other trainers)
            **kwargs: Additional configuration parameters
        """
        super().__init__(dataset_path, output_dir, predictor_type, **kwargs)

        # MoE-specific configuration
        self.num_experts = num_experts
        self.router_topk = router_topk
        self.hidden_dim = hidden_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.moe_tensor_parallel_size = moe_tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.routing_runtime_path = validate_moe_gating_routing_runtime_path(
            kwargs.get(
                "routing_runtime_path",
                STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
            )
        )
        self.gating_runtime_context = validate_moe_gating_runtime_context(
            kwargs.get(
                "gating_runtime_context",
                DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
            )
        )

        # Optional attributes for consistency with other trainers
        self.model_name = model_name
        self.device = device
        
        # Dataset will be loaded and stored in _load_dataset()
        self.df = None

        logger.info("MoE Configuration:")
        if model_name:
            logger.info(f"  - model_name: {model_name}")
        if device:
            logger.info(f"  - device: {device}")
        logger.info(f"  - num_experts: {num_experts}")
        logger.info(f"  - router_topk: {router_topk}")
        logger.info(f"  - hidden_dim: {hidden_dim}")
        logger.info(f"  - expert_hidden_dim: {expert_hidden_dim}")
        logger.info(f"  - moe_tensor_parallel_size: {moe_tensor_parallel_size}")
        logger.info(f"  - expert_parallel_size: {expert_parallel_size}")
        logger.info(f"  - routing_runtime_path: {self.routing_runtime_path}")
        logger.info(f"  - gating_runtime_context: {self.gating_runtime_context}")
    
    def _load_dataset(self) -> pd.DataFrame:
        """
        Load and filter the MoE profiling dataset.
        
        Returns:
            Filtered DataFrame ready for training
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"MoE dataset not found: {self.dataset_path}")
        
        # Load CSV
        df = pd.read_csv(self.dataset_path)
        self._set_dataset_metadata(df, source="moe_dataset")
        logger.info(f"Original MoE data: {len(df)} rows, {len(df.columns)} columns")
        
        # Display filtering conditions
        logger.info("Filtering conditions:")
        logger.info(f"  - num_experts == {self.num_experts}")
        logger.info(f"  - router_topk == {self.router_topk}")
        logger.info(f"  - hidden_dim == {self.hidden_dim}")
        logger.info(f"  - expert_hidden_dim == {self.expert_hidden_dim}")
        logger.info("  - model-specific TP/EP slices are selected during training")
        logger.info(f"  - requested moe_tensor_parallel_size == {self.moe_tensor_parallel_size}")
        logger.info(f"  - requested expert_parallel_size == {self.expert_parallel_size}")
        
        # Display available values in the dataset
        if len(df) > 0:
            if 'num_experts' in df.columns:
                logger.info(f"  - Available num_experts: {sorted(df['num_experts'].unique())}")
            if 'router_topk' in df.columns:
                logger.info(f"  - Available router_topk: {sorted(df['router_topk'].unique())}")
            if 'num_tensor_parallel_workers' in df.columns:
                logger.info(f"  - Available num_tensor_parallel_workers: {sorted(df['num_tensor_parallel_workers'].unique())}")
            if 'expert_parallel_size' in df.columns:
                logger.info(f"  - Available expert_parallel_size: {sorted(df['expert_parallel_size'].unique())}")
        
        # Apply filtering based on MoE configuration
        filtered_df = df[
            (df["num_experts"] == self.num_experts)
            & (df["router_topk"] == self.router_topk)
            & (df["hidden_dim"] == self.hidden_dim)
            & (df["expert_hidden_dim"] == self.expert_hidden_dim)
        ]
        
        logger.info(f"After filtering: {len(filtered_df)} rows")
        
        if len(filtered_df) == 0:
            logger.error("No data matches the filtering criteria!")
            logger.error("Please check if profiling data was generated with matching configuration")
            raise ValueError("No matching data found after filtering")
        
        # Verify required columns exist
        self._verify_dataset_columns(filtered_df)
        self._reject_legacy_split_row_dataset(filtered_df)
        
        # Store the filtered dataset for feature detection
        self.df = filtered_df
        
        return filtered_df
    
    def _verify_dataset_columns(self, df: pd.DataFrame) -> None:
        """
        Verify that the dataset contains all required columns.
        
        Args:
            df: DataFrame to verify
        """
        required_columns = ["num_tokens", *_get_moe_required_target_columns()]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")
        
        logger.info("Dataset column verification passed")

        # Auto-detect load-imbalance mode based on the *full* feature set presence.
        available = [feat for feat in self.LOAD_IMBALANCE_FEATURES if feat in df.columns]
        if 0 < len(available) < len(self.LOAD_IMBALANCE_FEATURES):
            missing = [feat for feat in self.LOAD_IMBALANCE_FEATURES if feat not in df.columns]
            logger.error(
                "Partial load imbalance features detected in dataset. "
                "This is an inconsistent dataset and is not supported."
            )
            logger.error(f"  - Available load-imbalance features ({len(available)}/{len(self.LOAD_IMBALANCE_FEATURES)}): {available}")
            logger.error(f"  - Missing load-imbalance features: {missing}")
            logger.error(f"  - Dataset: {self.dataset_path}")
            raise ValueError(
                "Partial load imbalance feature columns found. "
                "Please regenerate MoE profiling data with full load-imbalance features (e.g., enable load-imbalance profiling), "
                "or use a dataset without any load-imbalance columns."
            )

        if len(available) == len(self.LOAD_IMBALANCE_FEATURES):
            logger.info("Load imbalance features detected (full feature set) - will use enhanced feature set for moe_grouped_gemm")
        else:
            logger.info("Load imbalance features not detected - will use standard mode (num_tokens only) for moe_grouped_gemm")

    def _reject_legacy_split_row_dataset(self, df: pd.DataFrame) -> None:
        moe_tp_target_columns = _get_moe_tp_target_columns()
        replicated_target_columns = _get_moe_replicated_target_columns()
        if not moe_tp_target_columns or not replicated_target_columns:
            return

        tp_gt1_mask = df["num_tensor_parallel_workers"] > 1
        grouped_present_mask = df[moe_tp_target_columns].notna().any(axis=1)
        replicated_all_nan_mask = df[replicated_target_columns].isna().all(axis=1)
        legacy_split_row_mask = tp_gt1_mask & grouped_present_mask & replicated_all_nan_mask
        if not legacy_split_row_mask.any():
            return

        preview_df = df.loc[
            legacy_split_row_mask,
            ["num_tensor_parallel_workers", "expert_parallel_size", "num_tokens"],
        ].drop_duplicates()
        preview_df = preview_df.sort_values(
            ["num_tensor_parallel_workers", "expert_parallel_size", "num_tokens"]
        )

        preview_rows = [
            f"tp={int(row.num_tensor_parallel_workers)}, "
            f"ep={int(row.expert_parallel_size)}, "
            f"tokens={int(row.num_tokens)}"
            for row in preview_df.head(8).itertuples(index=False)
        ]
        raise ValueError(
            "Legacy split-row MoE profiling dataset detected: TP>1 rows have "
            "moe_grouped_gemm populated while replicated-op targets are NaN. "
            "This dataset was produced by historical split_replicated_result() "
            f"semantics and is incompatible with current trainer. file={self.dataset_path}. "
            "Please re-profile with the current frontier.profiling.moe.main. "
            "First broken rows: "
            + "; ".join(preview_rows)
        )
    
    def _has_full_load_imbalance_features(self) -> bool:
        if self.df is not None:
            columns = self.df.columns
        else:
            columns = pd.read_csv(self.dataset_path, nrows=0).columns
        return all(feat in columns for feat in self.LOAD_IMBALANCE_FEATURES)

    def _get_model_names(self) -> List[str]:
        """
        Get the list of MoE model names to train.

        Returns:
            List of model names
        """
        model_names = _get_moe_family_model_names()
        if str(getattr(self, "model_name", "")).strip() == "qwen3-a3b-30b-moe":
            if self.df is not None and has_prefill_hot_moe_gating_rows(self.df):
                model_names.extend(_get_prefill_hot_moe_gating_model_names())
            else:
                logger.warning(
                    "Prefill-hot MoE gating rows are unavailable in %s; "
                    "skipping __prefill_hot trainer pseudo-models for smoke stability.",
                    self.dataset_path,
                )
        return model_names
    
    def _get_feature_cols(self, model_name: str) -> List[str]:
        """
        Get feature column names for a specific MoE model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of feature column names
        """
        # Check if load imbalance features are available in the dataset (full feature set only).
        # Partial presence is rejected earlier in _verify_dataset_columns().
        has_load_imbalance = self._has_full_load_imbalance_features()

        if has_load_imbalance:
            if model_name in {"moe_grouped_gemm", "moe_shuffling"}:
                logger.info(f"  Using load imbalance features for {model_name} (14 features)")
                return self.LOAD_IMBALANCE_FEATURES

            logger.info(f"  Using num_tokens only for {model_name} (1 feature)")
            return ["num_tokens"]

        logger.info(f"  Using legacy features for {model_name} (1 feature)")
        return ["num_tokens"]
    
    def _get_target_col(self, model_name: str) -> str:
        """
        Get target column name for a specific MoE model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Target column name
        """
        # Target column follows the pattern: time_stats.<model_name>.median
        base_model_name = get_moe_gating_base_model_name(model_name)
        return f"time_stats.{base_model_name}.median"

    def _get_training_tp_key(self, model_name: str) -> int:
        operator = _get_moe_family_operator_by_model_name(model_name)
        if operator is None:
            raise ValueError(f"Unsupported MoE op for TP mapping: {model_name}")
        if operator.tp_mode is TensorParallelMode.REPLICATED:
            return 1
        if operator.tp_mode is TensorParallelMode.MOE_TP:
            return self.moe_tensor_parallel_size
        raise ValueError(
            f"Unsupported MoE TP mode for {model_name}: {operator.tp_mode}"
        )

    def _get_training_df_for_model(
        self,
        df: pd.DataFrame,
        model_name: str,
        feature_cols: List[str],
        target_col: str,
    ) -> pd.DataFrame:
        tp_key = self._get_training_tp_key(model_name)
        training_df = df[df["num_tensor_parallel_workers"] == tp_key].copy()
        operator = _get_moe_family_operator_by_model_name(model_name)
        if operator is None:
            raise ValueError(f"Unsupported MoE op for EP mapping: {model_name}")
        if not operator.ep_agnostic:
            training_df = training_df[
                training_df["expert_parallel_size"] == self.expert_parallel_size
            ].copy()
            ep_desc = f", ep={self.expert_parallel_size}"
        else:
            ep_desc = ", ep=ANY"

        base_model_name = get_moe_gating_base_model_name(model_name)
        if base_model_name == "moe_gating_routing_topk":
            training_df = filter_moe_gating_routing_topk_rows(
                training_df,
                requested_runtime_path=self.routing_runtime_path,
                source_name=self.dataset_path,
            )
        if _is_moe_gating_family_model_name(base_model_name):
            requested_context = self.gating_runtime_context
            if model_name.endswith("__prefill_hot"):
                requested_context = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT
            training_df = filter_moe_gating_rows_by_runtime_context(
                training_df,
                requested_context=requested_context,
                source_name=self.dataset_path,
            )
        if len(training_df) == 0:
            raise ValueError(
                f"No profiling rows remain for model {model_name} at TP={tp_key}{ep_desc}. "
                f"file={self.dataset_path}"
            )

        training_df = training_df.dropna(subset=feature_cols + [target_col]).copy()
        if len(training_df) == 0:
            raise ValueError(
                f"No valid training rows remain for model {model_name} after selecting TP={tp_key}"
                f"{ep_desc} and dropping NaN features/targets from {self.dataset_path}."
            )
        logger.info(
            "  Filtered training rows for %s: %d / %d (tp=%d%s)",
            model_name,
            len(training_df),
            len(df),
            tp_key,
            ep_desc,
        )
        return training_df

    def train(self) -> Dict[str, object]:
        """Train all MoE models with target-specific NaN filtering."""
        logger.info(f"Starting training for {self.__class__.__name__}")

        logger.info(f"Loading dataset from {self.dataset_path}")
        df = self._load_dataset()
        logger.info(f"Loaded {len(df)} rows after filtering")

        if len(df) == 0:
            raise ValueError(
                "No data available after filtering. Check dataset path and filtering criteria."
            )

        models = {}
        model_names = self._get_model_names()
        logger.info(f"Training {len(model_names)} models: {model_names}")

        for model_name in model_names:
            feature_cols = self._get_feature_cols(model_name)
            target_col = self._get_target_col(model_name)
            try:
                training_df = self._get_training_df_for_model(
                    df=df,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    target_col=target_col,
                )
            except ValueError as e:
                if model_name.endswith("__prefill_hot"):
                    logger.warning(
                        "Skipping %s: prefill-hot gating rows are unavailable for "
                        "the requested TP/EP slice (%s).",
                        model_name,
                        e,
                    )
                    continue
                raise

            logger.info(f"\n--- Training {model_name} ---")
            logger.info(f"Features: {feature_cols}")
            logger.info(f"Target: {target_col}")

            models[model_name] = self._train_single_model(
                model_name=model_name,
                df=training_df,
                feature_cols=feature_cols,
                target_col=target_col,
            )

        logger.info(f"\nTraining complete! Trained {len(models)} models")
        logger.info(f"Models saved to {self.output_dir}")

        return models


def create_moe_trainer_from_model_config(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    device: str = "a100",
    moe_tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    predictor_type: str = "random_forest",
    **kwargs
) -> MoETrainer:
    """
    Create a MoE trainer from a model configuration name.

    This is a convenience function that automatically loads model configuration
    from vidur's model registry and creates a trainer with the correct parameters.

    Args:
        dataset_path: Path to the MoE profiling dataset CSV file
        output_dir: Directory to save trained models
        model_name: Name of the model (e.g., "Qwen/Qwen1.5-MoE-A2.7B")
        device: Device SKU (e.g., "a100", "h100")
        moe_tensor_parallel_size: MoE tensor parallel size
        expert_parallel_size: Expert parallel size
        predictor_type: Type of predictor
        **kwargs: Additional configuration parameters

    Returns:
        Configured MoETrainer instance
    """
    from frontier.config.model_config import BaseModelConfig

    # Load model configuration
    model_config = BaseModelConfig.create_from_name(model_name)

    if not model_config.is_moe:
        raise ValueError(f"Model {model_name} is not a MoE model")

    logger.info(f"Creating MoE trainer for model: {model_name}")
    logger.info(f"Model configuration:")
    logger.info(f"  - num_experts: {model_config.num_experts}")
    logger.info(f"  - num_experts_per_tok: {model_config.num_experts_per_tok}")
    logger.info(f"  - embedding_dim: {model_config.embedding_dim}")
    logger.info(f"  - mlp_hidden_dim: {model_config.mlp_hidden_dim}")

    return MoETrainer(
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_experts=model_config.num_experts,
        router_topk=model_config.num_experts_per_tok,
        hidden_dim=model_config.embedding_dim,
        expert_hidden_dim=model_config.mlp_hidden_dim,
        moe_tensor_parallel_size=moe_tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        predictor_type=predictor_type,
        model_name=model_name,
        device=device,
        **kwargs
    )
