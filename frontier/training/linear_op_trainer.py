"""
Linear Operations trainer for execution time prediction models.

This module provides standalone training for linear operation-specific execution time predictors,
allowing pre-training and saving of model weights for later use in simulations.

Linear operations include:
- MLP layers: mlp_up_proj, mlp_down_proj, mlp_act
- Normalization: input_layernorm, post_attention_layernorm
- Residual: add

Note: When is_moe=True, training is skipped because MoE models use expert layers
instead of dense MLP layers.
"""

import os
from typing import Dict, List

import pandas as pd

from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.spec_decode.mtp_registry import is_target_embedded_mtp_same_tp_linear_op
from frontier.training.base_trainer import BaseTrainer
from frontier.logger import init_logger

logger = init_logger(__name__)


class LinearOpTrainer(BaseTrainer):
    """
    Trainer for linear operation execution time prediction models.
    
    This trainer handles the training of linear operation and common layer models:
    - Linear operation models: mlp_up_proj, mlp_down_proj, mlp_act
    - Common models: input_layernorm, post_attention_layernorm, add (non-RMSNorm only)
    
    The trainer loads profiling data from frontier/profiling/linear_op/ and filters it
    based on the provided configuration parameters.
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        model_name: str,
        device: str,
        tensor_parallel_size: int = 1,
        predictor_type: str = "random_forest",
        is_moe: bool = False,
        **kwargs
    ):
        """
        Initialize the Linear Operation trainer.
        
        Args:
            dataset_path: Path to the linear operation profiling dataset CSV file
            output_dir: Directory to save trained models
            model_name: Model name (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device SKU (e.g., "a100", "h100")
            tensor_parallel_size: Tensor parallel size (default: 1)
            predictor_type: Type of predictor ("random_forest" or "linear_regression")
            is_moe: If True, skip training (MoE models use expert layers instead of dense MLP)
            **kwargs: Additional configuration parameters
        """
        super().__init__(dataset_path, output_dir, predictor_type, **kwargs)
        
        # Linear operation-specific configuration
        self.model_name = model_name
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.is_moe = is_moe

        # Load model config for norm-type-aware decisions
        from frontier.config.model_config import BaseModelConfig
        self.model_config = BaseModelConfig.create_from_name(model_name)
        
        logger.info("Linear Operation Configuration:")
        logger.info(f"  - model_name: {model_name}")
        logger.info(f"  - device: {device}")
        logger.info(f"  - tensor_parallel_size: {tensor_parallel_size}")
        logger.info(f"  - is_moe: {is_moe}")
    
    def train(self) -> Dict[str, any]:
        """
        Train all linear operation models.

        If is_moe=True, skips MLP-specific models (mlp_up_proj, mlp_down_proj, mlp_act)
        but still trains common linear operations (input_layernorm, post_attention_layernorm,
        add, attn_pre_proj, attn_post_proj, attn_rope).

        Returns:
            Dictionary of trained models
        """
        if self.is_moe:
            logger.info("=" * 60)
            logger.info("NOTICE: is_moe=True, skipping MLP-specific model training.")
            logger.info("MoE models use expert layers instead of dense MLP layers.")
            logger.info("Training common linear operations only...")
            logger.info("=" * 60)
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
            training_df = self._get_training_df_for_model(
                df=df,
                model_name=model_name,
                feature_cols=feature_cols,
                target_col=target_col,
            )

            logger.info("\n" + "=" * 80)
            logger.info(f"--- Training {model_name} ---")
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

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load and filter the linear operation profiling dataset.

        Returns:
            Filtered DataFrame ready for training
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Linear operation dataset not found: {self.dataset_path}")
        
        # Load CSV
        df = pd.read_csv(self.dataset_path)
        self._set_dataset_metadata(df)
        logger.info(f"Original linear operation data: {len(df)} rows, {len(df.columns)} columns")

        # Display filtering conditions
        logger.info("Filtering conditions:")
        logger.info("  - model-specific TP slices are selected during training")
        logger.info(f"  - requested tensor_parallel_size == {self.tensor_parallel_size}")

        # Display available values in the dataset
        if len(df) > 0:
            if 'num_tensor_parallel_workers' in df.columns:
                logger.info(f"  - Available num_tensor_parallel_workers: {sorted(df['num_tensor_parallel_workers'].unique())}")

        filtered_df = df.copy()

        logger.info(f"After filtering: {len(filtered_df)} rows")
        
        if len(filtered_df) == 0:
            logger.error("No data matches the filtering criteria!")
            logger.error("Please check if profiling data was generated with matching configuration")
            raise ValueError("No matching data found after filtering")
        
        # Verify required columns exist
        self._verify_dataset_columns(filtered_df)
        
        return filtered_df
    
    def _verify_dataset_columns(self, df: pd.DataFrame) -> None:
        """
        Verify that the dataset contains all required columns.
        
        Args:
            df: DataFrame to verify
        """
        required_columns = [
            "num_tokens",
            "time_stats.emb.median",
            # Common models (always required)
            "time_stats.input_layernorm.median",
            "time_stats.post_attention_layernorm.median",
        ]
        # MLP columns only required for non-MoE models
        if not self.is_moe:
            required_columns.extend([
                "time_stats.mlp_up_proj.median",
                "time_stats.mlp_down_proj.median",
                "time_stats.mlp_act.median",
            ])
        # Only require add column for non-fused LayerNorm models
        if not self.model_config.uses_fused_add_norm:
            required_columns.append("time_stats.add.median")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        mtp_columns = [
            "time_stats.mtp_fusion_proj.median",
            "time_stats.lm_head_linear.median",
        ]
        present_mtp_columns = [col for col in mtp_columns if col in df.columns]
        if present_mtp_columns and len(present_mtp_columns) != len(mtp_columns):
            raise ValueError(
                "target-embedded MTP profiling columns must appear together: "
                f"expected {mtp_columns}, found {present_mtp_columns}"
            )
        self._has_target_embedded_mtp_ops = len(present_mtp_columns) == len(
            mtp_columns
        )
        
        logger.info("Dataset column verification passed")

    def _get_model_names(self) -> List[str]:
        """
        Get the list of linear operation model names to train.

        If is_moe=True, excludes MLP-specific models (mlp_up_proj, mlp_down_proj, mlp_act)
        but includes common linear operations.

        Returns:
            List of model names
        """
        # Common linear operation models (always trained)
        common_models = [
            "emb",
            "input_layernorm",
            "post_attention_layernorm",
            "attn_pre_proj",
            "attn_post_proj",
            "attn_rope",
        ]
        if getattr(self, "_has_target_embedded_mtp_ops", False):
            common_models.extend(
                [
                    "mtp_fusion_proj",
                    "lm_head_linear",
                ]
            )
        # add is only a separate operation for non-fused LayerNorm models
        if not self.model_config.uses_fused_add_norm:
            common_models.append("add")

        if self.is_moe:
            # For MoE models, only train common operations (skip MLP-specific)
            return common_models
        else:
            # For dense models, train all linear operations including MLP
            mlp_models = [
                "mlp_up_proj",
                "mlp_down_proj",
                "mlp_act",
            ]
            return mlp_models + common_models

    def _get_feature_cols(self, model_name: str) -> List[str]:
        """
        Get feature column names for a specific linear operation model.

        Args:
            model_name: Name of the model

        Returns:
            List of feature column names
        """
        # All linear operation models use num_tokens as the primary feature
        return ["num_tokens"]

    def _get_target_col(self, model_name: str) -> str:
        """
        Get target column name for a specific linear operation model.

        Args:
            model_name: Name of the model

        Returns:
            Target column name
        """
        # Target column follows the pattern: time_stats.<model_name>.median
        return f"time_stats.{model_name}.median"

    def _get_training_tp_key(self, model_name: str) -> int:
        replicated_ops = {
            "emb",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
            "attn_pre_proj_qkv",
            "attn_pre_proj_q_norm",
        }
        if model_name in replicated_ops:
            if (
                getattr(self, "_has_target_embedded_mtp_ops", False)
                and is_target_embedded_mtp_same_tp_linear_op(model_name)
            ):
                return resolve_effective_attention_tp_size(
                    op_name="attn_pre_proj",
                    requested_tp_size=self.tensor_parallel_size,
                    num_kv_heads=self.model_config.num_kv_heads,
                    cluster_type=None,
                    warning_cache=None,
                    include_linear_ops=True,
                )
            return 1

        if model_name.startswith("mlp_"):
            return self.tensor_parallel_size

        if model_name in {"mtp_fusion_proj", "lm_head_linear"}:
            return resolve_effective_attention_tp_size(
                op_name="attn_pre_proj",
                requested_tp_size=self.tensor_parallel_size,
                num_kv_heads=self.model_config.num_kv_heads,
                cluster_type=None,
                warning_cache=None,
                include_linear_ops=True,
            )

        if model_name.startswith("attn_"):
            return resolve_effective_attention_tp_size(
                op_name=model_name,
                requested_tp_size=self.tensor_parallel_size,
                num_kv_heads=self.model_config.num_kv_heads,
                cluster_type=None,
                warning_cache=None,
                include_linear_ops=True,
            )

        raise ValueError(f"Unsupported linear op for TP mapping: {model_name}")

    def _get_training_df_for_model(
        self,
        df: pd.DataFrame,
        model_name: str,
        feature_cols: List[str],
        target_col: str,
    ) -> pd.DataFrame:
        tp_key = self._get_training_tp_key(model_name)
        training_df = df[df["num_tensor_parallel_workers"] == tp_key].copy()
        if len(training_df) == 0:
            raise ValueError(
                f"No profiling rows remain for model {model_name} at TP={tp_key}. "
                f"file={self.dataset_path}"
            )

        expected_use_qk_norm = bool(getattr(self.model_config, "use_qk_norm", False))
        if expected_use_qk_norm and "use_qk_norm" not in training_df.columns:
            raise ValueError(
                "linear_op trainer requires 'use_qk_norm' metadata for a model "
                f"that enables QK-norm. file={self.dataset_path}, model={self.model_name}"
            )
        if "use_qk_norm" in training_df.columns:
            training_df = training_df[
                training_df["use_qk_norm"].astype(bool) == expected_use_qk_norm
            ].copy()

        expected_attn_output_gate = bool(
            getattr(self.model_config, "attn_output_gate", False)
        )
        if expected_attn_output_gate and "attn_output_gate" not in training_df.columns:
            raise ValueError(
                "linear_op trainer requires 'attn_output_gate' metadata for a model "
                "that uses gated attention output. "
                f"file={self.dataset_path}, model={self.model_name}"
            )
        if "attn_output_gate" in training_df.columns:
            training_df = training_df[
                training_df["attn_output_gate"].astype(bool)
                == expected_attn_output_gate
            ].copy()

        if len(training_df) == 0:
            raise ValueError(
                f"No profiling rows remain for model {model_name} after metadata filtering "
                f"(tp={tp_key}, use_qk_norm={expected_use_qk_norm}, "
                f"attn_output_gate={expected_attn_output_gate}) in {self.dataset_path}."
            )

        training_df = training_df.dropna(subset=feature_cols + [target_col]).copy()
        if len(training_df) == 0:
            raise ValueError(
                f"No valid training rows remain for model {model_name} after selecting TP={tp_key} "
                f"and dropping NaN features/targets from {self.dataset_path}."
            )

        logger.info(
            "  Filtered training rows for %s: %d / %d (tp=%d)",
            model_name,
            len(training_df),
            len(df),
            tp_key,
        )
        return training_df


def create_linear_op_trainer_from_model_config(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    device: str = "a100",
    tensor_parallel_size: int = 1,
    predictor_type: str = "random_forest",
    is_moe: bool = False,
    **kwargs
) -> LinearOpTrainer:
    """
    Create a Linear Operation trainer from a model configuration name.

    This is a convenience function that automatically loads model configuration
    from frontier's model registry and creates a trainer with the correct parameters.

    Args:
        dataset_path: Path to the linear operation profiling dataset CSV file
        output_dir: Directory to save trained models
        model_name: Name of the model (e.g., "meta-llama/Llama-2-7b-hf")
        device: Device SKU (e.g., "a100", "h100")
        tensor_parallel_size: Tensor parallel size
        predictor_type: Type of predictor
        is_moe: If True, skip training (MoE models use expert layers instead of dense MLP)
        **kwargs: Additional configuration parameters

    Returns:
        Configured LinearOpTrainer instance
    """
    from frontier.config.model_config import BaseModelConfig

    # Load model configuration
    model_config = BaseModelConfig.create_from_name(model_name)

    logger.info(f"Creating Linear Operation trainer for model: {model_name}")
    logger.info(f"Model configuration:")
    logger.info(f"  - embedding_dim: {model_config.embedding_dim}")
    logger.info(f"  - mlp_hidden_dim: {model_config.mlp_hidden_dim}")
    logger.info(f"  - num_layers: {model_config.num_layers}")
    logger.info(f"  - is_moe: {is_moe}")

    return LinearOpTrainer(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        device=device,
        tensor_parallel_size=tensor_parallel_size,
        predictor_type=predictor_type,
        is_moe=is_moe,
        **kwargs
    )


# Backward compatibility aliases
MLPTrainer = LinearOpTrainer
create_mlp_trainer_from_model_config = create_linear_op_trainer_from_model_config
