"""
Attention trainer for execution time prediction models.

This module provides a trainer for Attention-specific models, including:
- Compute models (attn_pre_proj, attn_post_proj, attn_rope)
- Layer models (attn_kv_cache_save, attn_prefill, attn_decode)
- Mixed-batch prefill model (attn_prefill_mixed)
- True mixed decode model (attn_decode_in_mixed)
- Common models (input_layernorm, add)
"""

import os
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.string_coercion import coerce_truthy_bool
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_feature_columns,
    get_enabled_predictor_median_column_by_role,
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_names,
    get_enabled_predictor_required_feature_columns,
)
from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.logger import init_logger
from frontier.training.base_trainer import BaseTrainer

logger = init_logger(__name__)


class AttentionTrainer(BaseTrainer):
    """
    Trainer for Attention execution time prediction models.
    
    This trainer handles both compute models (from linear_op.csv) and layer models (from attention.csv).
    It trains up to 11 models in total:
    - 3 compute models: attn_pre_proj, attn_post_proj, attn_rope
    - 3 layer models: attn_kv_cache_save, attn_prefill, attn_decode
    - 1 mixed-batch model: attn_prefill_mixed (optional, if data available)
    - 1 true mixed-batch decode model: attn_decode_in_mixed (optional, if data available)
    - 2 common models: input_layernorm, add
    """
    
    # Compute-dependent models (require compute_dataset_path / linear_op.csv)
    COMPUTE_DEPENDENT_MODELS = [
        "attn_pre_proj",
        "attn_post_proj",
        "attn_rope",
        "input_layernorm",
        # "add" is conditionally included based on norm type — see _get_model_names()
    ]
    DENSE_LAYER_MODELS = list(get_enabled_predictor_metric_names(DENSE_ATTENTION_FAMILY))
    DENSE_LAYER_TARGET_COLUMNS = list(
        get_enabled_predictor_median_columns(DENSE_ATTENTION_FAMILY)
    )
    DENSE_LAYER_TARGET_COLUMN_BY_MODEL = dict(
        zip(DENSE_LAYER_MODELS, DENSE_LAYER_TARGET_COLUMNS)
    )
    DENSE_LAYER_FEATURE_COLUMNS = get_enabled_predictor_feature_columns(
        DENSE_ATTENTION_FAMILY
    )
    DENSE_LAYER_REQUIRED_FEATURE_COLUMNS = get_enabled_predictor_required_feature_columns(
        DENSE_ATTENTION_FAMILY
    )
    DENSE_LAYER_CACHE_WRITE_TARGET_COLUMN = (
        get_enabled_predictor_median_column_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
    )

    def __init__(
        self,
        layer_dataset_path: str,
        output_dir: str,
        model_name: str,
        device: str,
        compute_dataset_path: str = None,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        predictor_type: str = "random_forest",
        **kwargs
    ):
        """
        Initialize Attention trainer.

        Args:
            layer_dataset_path: Path to layer profiling CSV (attention.csv) - REQUIRED
            output_dir: Directory to save trained models
            model_name: Model name (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device SKU (e.g., "a100", "h100")
            compute_dataset_path: Path to compute profiling CSV (linear_op.csv) - OPTIONAL
                                  When not provided, compute-dependent models are skipped
            tensor_parallel_size: Tensor parallel size
            block_size: Block size for KV cache
            predictor_type: Type of predictor ("random_forest" or "linear_regression")
            **kwargs: Additional training parameters
        """
        # Use layer dataset path as the primary dataset path for base class
        # (compute_dataset_path is now optional)
        super().__init__(layer_dataset_path, output_dir, predictor_type, **kwargs)

        self.compute_dataset_path = compute_dataset_path
        self.layer_dataset_path = layer_dataset_path
        self.model_name = model_name
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.block_size = block_size

        # Track whether compute models should be trained
        self.train_compute_models = compute_dataset_path is not None

        # Load model configuration
        from frontier.config.model_config import BaseModelConfig
        self.model_config = BaseModelConfig.create_from_name(model_name)

        logger.info(f"Attention Configuration:")
        logger.info(f"  - model_name: {self.model_name}")
        logger.info(f"  - device: {self.device}")
        logger.info(f"  - tensor_parallel_size: {self.tensor_parallel_size}")
        logger.info(f"  - block_size: {self.block_size}")
        logger.info(f"  - embedding_dim: {self.model_config.embedding_dim}")
        logger.info(f"  - num_q_heads: {self.model_config.num_q_heads}")
        logger.info(f"  - num_kv_heads: {self.model_config.num_kv_heads}")
        logger.info(f"  - train_compute_models: {self.train_compute_models}")

        if not self.train_compute_models:
            logger.info(f"  NOTE: compute_dataset_path not provided, will skip:")
            for model_name in self.COMPUTE_DEPENDENT_MODELS:
                logger.info(f"        - {model_name}")
    
    def _load_dataset(self) -> pd.DataFrame:
        """
        Load and filter compute dataset.

        This method loads the compute dataset (linear_op.csv) for compute models.
        Layer models are loaded separately in train() method.

        Returns:
            Filtered compute dataframe, or empty DataFrame if compute_dataset_path is not provided
        """
        if not self.train_compute_models:
            logger.info("compute_dataset_path not provided, returning empty DataFrame")
            return pd.DataFrame()

        logger.info(f"Loading compute data from: {self.compute_dataset_path}")
        df = pd.read_csv(self.compute_dataset_path)
        self._set_dataset_metadata(df, source="compute_dataset")

        logger.info(f"Original compute data: {len(df)} rows, {len(df.columns)} columns")

        logger.info("Filtering conditions:")
        logger.info(f"  - n_head == {self.model_config.num_q_heads}")
        logger.info(f"  - n_kv_head == {self.model_config.num_kv_heads}")
        logger.info(f"  - n_embd == {self.model_config.embedding_dim}")
        logger.info(f"  - n_expanded_embd == {self.model_config.mlp_hidden_dim}")
        logger.info(f"  - use_gated_mlp == {self.model_config.use_gated_mlp}")
        logger.info(f"  - vocab_size == {self.model_config.vocab_size}")

        filtered_df = df[
            (df["n_head"] == self.model_config.num_q_heads)
            & (df["n_kv_head"] == self.model_config.num_kv_heads)
            & (df["n_embd"] == self.model_config.embedding_dim)
            & (df["n_expanded_embd"] == self.model_config.mlp_hidden_dim)
            & (df["use_gated_mlp"] == self.model_config.use_gated_mlp)
            & (df["vocab_size"] == self.model_config.vocab_size)
        ].copy()

        expected_use_qk_norm = bool(getattr(self.model_config, "use_qk_norm", False))
        if expected_use_qk_norm and "use_qk_norm" not in filtered_df.columns:
            raise ValueError(
                "linear_op profiling data is missing 'use_qk_norm' column for a model "
                "that requires QK-norm-aware filtering. "
                f"file={self.compute_dataset_path}, model={self.model_name}"
            )
        if "use_qk_norm" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["use_qk_norm"].astype(bool) == expected_use_qk_norm
            ].copy()

        if len(filtered_df) == 0:
            raise ValueError(
                "No compute profiling rows remain after model filtering. "
                f"file={self.compute_dataset_path}, model={self.model_name}, "
                f"requested_tp={self.tensor_parallel_size}"
            )

        logger.info(f"After model filtering: {len(filtered_df)} rows")

        self._verify_compute_dataset_columns(filtered_df)

        return filtered_df
    
    def _load_layer_dataset(self) -> pd.DataFrame:
        """
        Load and filter layer dataset (attention.csv).
        
        Returns:
            Filtered layer dataframe with derived features
        """
        logger.info(f"Loading layer data from: {self.layer_dataset_path}")
        df = pd.read_csv(self.layer_dataset_path)
        self._set_dataset_metadata(df, source="layer_dataset")
        df = df.drop_duplicates()
        
        logger.info(f"Original layer data: {len(df)} rows, {len(df.columns)} columns")
        
        # Fill missing cache-write column for older attention profiling CSVs.
        for column in [self.DENSE_LAYER_CACHE_WRITE_TARGET_COLUMN]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)
        
        # Filter by model configuration
        logger.info(f"Filtering conditions:")
        logger.info(f"  - n_embd == {self.model_config.embedding_dim}")
        logger.info(f"  - n_q_head == {self.model_config.num_q_heads}")
        logger.info(f"  - n_kv_head == {self.model_config.num_kv_heads}")
        logger.info(f"  - block_size == {self.block_size}")
        logger.info(f"  - num_tensor_parallel_workers == {self.tensor_parallel_size}")
        
        filtered_df = df[
            (df["n_embd"] == self.model_config.embedding_dim)
            & (df["n_q_head"] == self.model_config.num_q_heads)
            & (df["n_kv_head"] == self.model_config.num_kv_heads)
            & (df["block_size"] == self.block_size)
            & (df["num_tensor_parallel_workers"] == self.tensor_parallel_size)
        ]
        
        logger.info(f"After filtering: {len(filtered_df)} rows")
        
        # Add derived features
        filtered_df = self._add_derived_features(filtered_df)
        
        # Verify required columns
        self._verify_layer_dataset_columns(filtered_df)
        
        return filtered_df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to layer dataframe.

        Args:
            df: Original dataframe

        Returns:
            Dataframe with derived features
        """
        df_with_features = df.copy()

        # Standard attention features
        # num_tokens = max(prefill_chunk_size, batch_size)
        df_with_features["num_tokens"] = df_with_features[["prefill_chunk_size", "batch_size"]].max(axis=1)

        # is_decode is derived from is_prefill when available.
        if "is_prefill" in df_with_features.columns:
            normalized_prefill_values = coerce_truthy_bool(
                df_with_features["is_prefill"]
            )
            df_with_features["is_decode"] = ~normalized_prefill_values
        else:
            df_with_features["is_decode"] = (df_with_features["prefill_chunk_size"] == 0)

        # prefill_chunk_size_squared = prefill_chunk_size ** 2
        df_with_features["prefill_chunk_size_squared"] = (df_with_features["prefill_chunk_size"] ** 2)

        logger.info("Added standard derived features: num_tokens, is_decode, prefill_chunk_size_squared")

        def _normalize_bool_series(series: pd.Series) -> pd.Series:
            return coerce_truthy_bool(series)

        if "is_mixed_batch" in df_with_features.columns:
            df_with_features["is_mixed_batch"] = _normalize_bool_series(
                df_with_features["is_mixed_batch"]
            )
        else:
            df_with_features["is_mixed_batch"] = False

        if "is_true_mixed_batch" in df_with_features.columns:
            df_with_features["is_true_mixed_batch"] = _normalize_bool_series(
                df_with_features["is_true_mixed_batch"]
            )
        else:
            df_with_features["is_true_mixed_batch"] = False

        # Mixed-batch specific features (if applicable)
        if "total_tokens" in df_with_features.columns:
            # total_tokens_squared for computational complexity
            df_with_features["total_tokens_squared"] = (
                df_with_features["total_tokens"] ** 2
            )

            # Interaction features between batch size and heterogeneity
            if "seq_len_variance" in df_with_features.columns:
                df_with_features["batch_variance_interaction"] = (
                    df_with_features["batch_size"] *
                    df_with_features["seq_len_variance"]
                )

            if "seq_len_cv" in df_with_features.columns:
                df_with_features["batch_cv_interaction"] = (
                    df_with_features["batch_size"] *
                    df_with_features["seq_len_cv"]
                )

                # Normalized variance
                df_with_features["variance_normalized"] = (
                    df_with_features["seq_len_variance"] /
                    (df_with_features["avg_seq_len"] ** 2 + 1e-6)
                )

            # Length range
            if "max_seq_len" in df_with_features.columns and "min_seq_len" in df_with_features.columns:
                df_with_features["seq_len_range"] = (
                    df_with_features["max_seq_len"] -
                    df_with_features["min_seq_len"]
                )

            # Tokens per sequence (average)
            df_with_features["tokens_per_seq"] = (
                df_with_features["total_tokens"] /
                df_with_features["batch_size"]
            )

            if {
                "num_prefill_seqs",
                "num_decode_seqs",
            }.issubset(df_with_features.columns) and (
                "total_batch_size" not in df_with_features.columns
            ):
                df_with_features["total_batch_size"] = (
                    df_with_features["num_prefill_seqs"]
                    + df_with_features["num_decode_seqs"]
                )

            if {
                "num_prefill_seqs",
                "total_batch_size",
            }.issubset(df_with_features.columns) and (
                "batch_composition_ratio" not in df_with_features.columns
            ):
                total_batch_size = df_with_features["total_batch_size"].replace(0, pd.NA)
                df_with_features["batch_composition_ratio"] = (
                    df_with_features["num_prefill_seqs"] / total_batch_size
                ).fillna(0.0)

            if (
                "num_decode_seqs" in df_with_features.columns
                and "decode_batch_size" not in df_with_features.columns
            ):
                df_with_features["decode_batch_size"] = df_with_features[
                    "num_decode_seqs"
                ]

            logger.info("Added mixed-batch derived features")

        return df_with_features

    def _get_required_compute_dataset_columns(self) -> List[str]:
        """Get required compute dataset columns based on model architecture."""
        required_columns = [
            "num_tokens",
            "time_stats.attn_pre_proj.median",
            "time_stats.attn_post_proj.median",
            "time_stats.attn_rope.median",
            "time_stats.input_layernorm.median",
        ]
        # Only require add column for non-fused LayerNorm models
        if not self.model_config.uses_fused_add_norm:
            required_columns.append("time_stats.add.median")

        if self.model_config.is_step2_mini():
            required_columns.extend(
                [
                    "time_stats.attn_inter_norm.median",
                    "time_stats.attn_wq_proj.median",
                ]
            )

        return required_columns

    def _verify_compute_dataset_columns(self, df: pd.DataFrame):
        """Verify that compute dataset has all required columns."""
        required_columns = self._get_required_compute_dataset_columns()

        missing_columns = [col for col in required_columns if col not in df.columns]
        all_nan_columns = [
            col for col in required_columns if col in df.columns and df[col].isna().all()
        ]

        if missing_columns or all_nan_columns:
            raise ValueError(
                "Compute dataset is missing required columns or has all-NaN values.\n"
                f"Missing columns: {missing_columns}\n"
                f"All-NaN columns: {all_nan_columns}\n"
                f"Available columns: {list(df.columns)}"
            )

        logger.info("Compute dataset column verification passed")

    def _get_compute_tp_key(self, model_name: str) -> int:
        replicated_ops = {
            "input_layernorm",
            "post_attention_layernorm",
            "add",
            "emb",
            "attn_pre_proj_qkv",
            "attn_pre_proj_q_norm",
        }
        if model_name in replicated_ops:
            return 1

        if model_name.startswith("attn_"):
            return resolve_effective_attention_tp_size(
                op_name=model_name,
                requested_tp_size=self.tensor_parallel_size,
                num_kv_heads=self.model_config.num_kv_heads,
                cluster_type=None,
                warning_cache=None,
                include_linear_ops=True,
            )

        raise ValueError(f"Unsupported compute model for TP mapping: {model_name}")

    def _get_compute_training_df(
        self,
        compute_df: pd.DataFrame,
        model_name: str,
    ) -> pd.DataFrame:
        tp_key = self._get_compute_tp_key(model_name)
        target_col = self._get_target_col(model_name)
        filtered_df = compute_df[
            compute_df["num_tensor_parallel_workers"] == tp_key
        ].copy()

        if len(filtered_df) == 0:
            raise ValueError(
                f"No compute profiling rows remain for model {model_name} at TP={tp_key}. "
                f"file={self.compute_dataset_path}"
            )
        if target_col not in filtered_df.columns:
            raise ValueError(
                f"Column '{target_col}' not found in compute dataframe for model {model_name}."
            )
        if filtered_df[target_col].isna().all():
            raise ValueError(
                f"Column '{target_col}' is all-NaN in compute dataframe for model {model_name} "
                f"at TP={tp_key}. Please re-run linear-op profiling for this TP configuration."
            )

        return filtered_df
    
    def _verify_layer_dataset_columns(self, df: pd.DataFrame):
        """Verify that layer dataset has all required columns."""
        required_columns = [
            "is_decode",
            *self.DENSE_LAYER_REQUIRED_FEATURE_COLUMNS,
            *self.DENSE_LAYER_TARGET_COLUMNS,
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Layer dataset is missing required columns: {missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        logger.info(f"Layer dataset column verification passed")
    
    def _get_model_names(self) -> List[str]:
        """
        Get list of model names to train.

        Returns:
            List of model names. When compute_dataset_path is not provided,
            compute-dependent models (attn_pre_proj, attn_post_proj, attn_rope,
            input_layernorm, add) are excluded.
        """
        # Layer models (always trained) - 4 models
        model_names = [
            # Layer models (3)
            *self.DENSE_LAYER_MODELS,
            # Mixed-batch model (1, optional - depends on data availability)
            "attn_prefill_mixed",
            # True mixed decode model (1, optional - depends on data availability)
            "attn_decode_in_mixed",
        ]

        # Compute models (only if compute_dataset_path is provided) - 6 models
        if self.train_compute_models:
            # Prepend compute models to maintain original order
            compute_models = [
                "attn_pre_proj",
                "attn_post_proj",
                "attn_rope",
            ]
            common_models = ["input_layernorm"]
            if not self.model_config.uses_fused_add_norm:
                common_models.append("add")
            model_names = compute_models + model_names + common_models

        return model_names
    
    def _get_feature_cols(self, model_name: str) -> List[str]:
        """
        Get feature columns for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of feature column names
        """
        # Compute models and common models use num_tokens
        if model_name in ["attn_pre_proj", "attn_post_proj", "attn_rope",
                         "input_layernorm", "post_attention_layernorm", "add"]:
            return ["num_tokens"]
        
        # Layer models have different features
        elif model_name in self.DENSE_LAYER_FEATURE_COLUMNS:
            return list(self.DENSE_LAYER_FEATURE_COLUMNS[model_name])
        
        elif model_name == "attn_prefill_mixed":
            # Mixed-batch prefill uses rich feature set
            return [
                # Core features
                "batch_size",
                "kv_cache_size",
                "total_tokens",
                "avg_seq_len",
                "min_seq_len",
                "max_seq_len",
                "total_tokens_squared",
                # Heterogeneity features
                "seq_len_variance",
                "seq_len_cv",
                "seq_len_range",
                # Interaction features
                "batch_variance_interaction",
                "batch_cv_interaction",
            ]

        elif model_name == "attn_decode_in_mixed":
            return [
                "decode_batch_size",
                "decode_avg_kv_cache_size",
                "num_prefill_seqs",
                "total_prefill_tokens",
                "total_batch_size",
                "batch_composition_ratio",
                "total_tokens",
            ]
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def _get_target_col(self, model_name: str) -> str:
        """
        Get target column for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Target column name
        """
        # Special case: attn_prefill_mixed uses attn_prefill stats in CSV
        # because profiling records mixed batch results under the same key
        if model_name == "attn_prefill_mixed":
            return "time_stats.attn_prefill.median"
        if model_name == "attn_decode_in_mixed":
            return "time_stats.attn_decode.median"

        if model_name in self.DENSE_LAYER_TARGET_COLUMN_BY_MODEL:
            return self.DENSE_LAYER_TARGET_COLUMN_BY_MODEL[model_name]

        return f"time_stats.{model_name}.median"

    def train(self) -> Dict[str, BaseEstimator]:
        """
        Train all Attention models.

        This method overrides the base class train() to handle dual data sources:
        - Compute models are trained on compute_dataset (linear_op.csv)
        - Layer models are trained on layer_dataset (attention.csv)
        - Mixed-batch model is trained on mixed-batch subset (if available)

        Returns:
            Dictionary mapping model names to trained models
        """
        logger.info("=" * 80)
        logger.info(f"Starting training for {self.__class__.__name__}")
        logger.info("=" * 80)

        # Load compute dataset
        logger.info("\n--- Loading Compute Dataset ---")
        compute_df = self._load_dataset()
        logger.info(f"Loaded {len(compute_df)} rows for compute models")

        # Load layer dataset
        logger.info("\n--- Loading Layer Dataset ---")
        layer_df = self._load_layer_dataset()
        logger.info(f"Loaded {len(layer_df)} rows for layer models")

        if "is_true_mixed_batch" in layer_df.columns:
            true_mixed_mask = layer_df["is_true_mixed_batch"].fillna(False).astype(bool)
        else:
            true_mixed_mask = pd.Series(False, index=layer_df.index)

        if "is_mixed_batch" in layer_df.columns:
            mixed_batch_mask = layer_df["is_mixed_batch"].fillna(False).astype(bool)
        else:
            mixed_batch_mask = (~layer_df["is_decode"]) & (layer_df["batch_size"] > 1)

        mixed_prefill_mask = mixed_batch_mask & (~true_mixed_mask)

        standard_df = layer_df[~mixed_prefill_mask & ~true_mixed_mask].copy()
        mixed_batch_df = layer_df[mixed_prefill_mask].copy()
        true_mixed_df = layer_df[true_mixed_mask].copy()
        prefill_df = standard_df[~standard_df["is_decode"]].copy()
        decode_df = standard_df[standard_df["is_decode"]].copy()

        logger.info("Split layer data by contract:")
        logger.info(f"  - Standard data: {len(standard_df)} rows")
        logger.info(f"  - Mixed prefill data: {len(mixed_batch_df)} rows")
        logger.info(f"  - True mixed prefill+decode data: {len(true_mixed_df)} rows")
        logger.info(f"  - Standard prefill rows: {len(prefill_df)}")
        logger.info(f"  - Decode rows: {len(decode_df)}")

        # Get model names
        model_names = self._get_model_names()
        logger.info(f"\nTraining up to {len(model_names)} models: {model_names}")

        models = {}

        for model_name in model_names:
            logger.info("\n" + "=" * 80)
            logger.info(f"--- Training {model_name} ---")

            # Determine which dataset to use
            if model_name in ["attn_pre_proj", "attn_post_proj", "attn_rope",
                             "input_layernorm", "post_attention_layernorm", "add"]:
                # Compute models use op-specific TP slices from compute dataset
                df = self._get_compute_training_df(compute_df, model_name)
                logger.info(
                    f"Using compute dataset ({len(df)} rows, tp={self._get_compute_tp_key(model_name)})"
                )

            elif model_name == "attn_kv_cache_save":
                # KV cache save uses full standard layer dataset
                df = standard_df
                logger.info(f"Using standard layer dataset ({len(df)} rows)")

            elif model_name == "attn_prefill":
                # Prefill uses standard prefill subset
                df = prefill_df
                logger.info(f"Using standard prefill dataset ({len(df)} rows)")

            elif model_name == "attn_decode":
                # Decode uses standard decode subset
                df = decode_df
                logger.info(f"Using standard decode dataset ({len(df)} rows)")

            elif model_name == "attn_prefill_mixed":
                # Mixed-batch prefill uses mixed-batch data (if available)
                df = mixed_batch_df
                if len(df) == 0:
                    logger.info(f"No mixed-batch data available, skipping attn_prefill_mixed")
                    continue
                logger.info(f"Using mixed-batch dataset ({len(df)} rows)")

            elif model_name == "attn_decode_in_mixed":
                df = true_mixed_df
                if len(df) == 0:
                    logger.info("No true mixed-batch data available, skipping attn_decode_in_mixed")
                    continue
                logger.info(f"Using true mixed-batch dataset ({len(df)} rows)")

            else:
                raise ValueError(f"Unknown model name: {model_name}")

            # Get features and target
            feature_cols = self._get_feature_cols(model_name)
            target_col = self._get_target_col(model_name)

            logger.info(f"Features: {feature_cols}")
            logger.info(f"Target: {target_col}")

            # Verify features exist in dataframe
            missing_features = [col for col in feature_cols if col not in df.columns]
            if missing_features:
                logger.warning(
                    f"Model {model_name} requires features {missing_features} that are not in the dataset. "
                    f"Skipping this model."
                )
                continue

            # Train model
            models[model_name] = self._train_single_model(
                model_name=model_name,
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
            )

        logger.info("\n" + "=" * 80)
        logger.info(f"Training complete! Trained {len(models)} models")
        logger.info(f"Models saved to {self.output_dir}")
        logger.info("=" * 80)

        return models


def create_attention_trainer_from_model_config(
    layer_dataset_path: str,
    output_dir: str,
    model_name: str,
    device: str = "a100",
    compute_dataset_path: str = None,
    tensor_parallel_size: int = 1,
    block_size: int = 16,
    predictor_type: str = "random_forest",
    **kwargs
) -> AttentionTrainer:
    """
    Create an Attention trainer from model configuration.

    This is a convenience function that automatically loads the model configuration
    and creates a configured AttentionTrainer instance.

    Args:
        layer_dataset_path: Path to layer profiling CSV (attention.csv) - REQUIRED
        output_dir: Directory to save trained models
        model_name: Model name (e.g., "meta-llama/Llama-2-7b-hf")
        device: Device SKU (default: "a100")
        compute_dataset_path: Path to compute profiling CSV (linear_op.csv) - OPTIONAL
                              When not provided, compute-dependent models are skipped:
                              attn_pre_proj, attn_post_proj, attn_rope,
                              input_layernorm, post_attention_layernorm, add
        tensor_parallel_size: Tensor parallel size (default: 1)
        block_size: Block size for KV cache (default: 16)
        predictor_type: Type of predictor (default: "random_forest")
        **kwargs: Additional training parameters

    Returns:
        Configured AttentionTrainer instance

    Example:
        >>> # Full training (with compute models)
        >>> trainer = create_attention_trainer_from_model_config(
        ...     layer_dataset_path="data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv",
        ...     output_dir="cache",
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     device="a100",
        ...     compute_dataset_path="data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv",
        ... )
        >>> models = trainer.train()  # Trains attention and compute models

        >>> # Attention-only training (without compute models)
        >>> trainer = create_attention_trainer_from_model_config(
        ...     layer_dataset_path="data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv",
        ...     output_dir="cache",
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     device="a100",
        ... )
        >>> models = trainer.train()  # Trains attention-layer models only
    """
    from frontier.config.model_config import BaseModelConfig

    # Load model configuration
    model_config = BaseModelConfig.create_from_name(model_name)

    logger.info(f"Creating Attention trainer for model: {model_name}")
    logger.info(f"Model configuration:")
    logger.info(f"  - embedding_dim: {model_config.embedding_dim}")
    logger.info(f"  - num_q_heads: {model_config.num_q_heads}")
    logger.info(f"  - num_kv_heads: {model_config.num_kv_heads}")
    logger.info(f"  - num_layers: {model_config.num_layers}")

    if compute_dataset_path is None:
        logger.info(f"  - compute_dataset_path: NOT PROVIDED (compute models will be skipped)")
    else:
        logger.info(f"  - compute_dataset_path: {compute_dataset_path}")

    return AttentionTrainer(
        layer_dataset_path=layer_dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        device=device,
        compute_dataset_path=compute_dataset_path,
        tensor_parallel_size=tensor_parallel_size,
        block_size=block_size,
        predictor_type=predictor_type,
        **kwargs
    )
