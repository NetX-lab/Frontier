"""
Linear Operations trainer for execution time prediction models.

This module provides standalone training for linear operation-specific execution time predictors,
allowing pre-training and saving of model weights for later use in simulations.

Linear operations include:
- MLP layers: mlp_up_proj, mlp_down_proj, mlp_act
- Normalization: input_layernorm, post_attention_layernorm
- Residual: add

Mixed-layer MoE models keep their dense boundary and shared-expert linear
operations in this trainer. Routed expert operations remain owned by
``MoETrainer``.
"""

import os
from typing import Any, Dict, List

import pandas as pd

from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.model_architectures import (
    LayerDimensionSource,
    LayerKind,
    ResolvedLayerContract,
)
from frontier.operators.binding import bind_operator_query, resolve_operator_query_tp_mode
from frontier.operators.families import (
    get_family_profiling_names,
    get_operator_family,
)
from frontier.operators.spec import TensorParallelMode
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    matches_resolved_layer_contract,
    parse_typed_operator_contracts,
    validate_typed_operator_metadata,
)
from frontier.spec_decode.mtp_registry import (
    get_target_embedded_mtp_linear_ops,
    is_target_embedded_mtp_same_tp_linear_op,
)
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
            is_moe: Legacy hint for pure MoE datasets. Active mixed-layer
                operations are resolved from the model architecture profile.
            **kwargs: Additional configuration parameters
        """
        super().__init__(dataset_path, output_dir, predictor_type, **kwargs)
        
        # Linear operation-specific configuration
        self.model_name = model_name
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.is_moe = is_moe
        self.expert_parallel_size = kwargs.get("expert_parallel_size")

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

        Active dense and shared-expert operations are selected from the model
        architecture profile. Routed MoE operations remain in ``MoETrainer``.

        Returns:
            Dictionary of trained models
        """
        if self.is_moe and not self._is_mixed_layer_model():
            logger.info("=" * 60)
            logger.info("NOTICE: pure MoE training keeps routed operations in MoETrainer.")
            logger.info("Training common linear operations only...")
            logger.info("=" * 60)
        elif self._is_mixed_layer_model():
            logger.info(
                "Mixed-layer profile detected; training profile-owned dense "
                "boundary and shared-expert linear operations."
            )
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

    def _get_architecture_profile(self):
        """Return the model's construction-time architecture profile."""

        getter = getattr(self.model_config, "get_model_architecture_profile", None)
        if not callable(getter):
            raise TypeError(
                "linear-op training requires "
                "model_config.get_model_architecture_profile()"
            )
        return getter()

    def _get_active_layer_contracts(self):
        """Return profile-owned layer contracts active for this model."""

        profile = self._get_architecture_profile()
        iterator = getattr(profile, "iter_active_layer_contracts", None)
        if not callable(iterator):
            raise TypeError(
                "model architecture profile must expose "
                "iter_active_layer_contracts()"
            )
        return tuple(iterator(self.model_config))

    def _is_mixed_layer_model(self) -> bool:
        """Return whether the profile exposes both dense and routed domains."""

        if not bool(getattr(self.model_config, "is_moe", False)):
            return False
        active_kinds = {
            contract.layer_kind for contract in self._get_active_layer_contracts()
        }
        return LayerKind.DENSE in active_kinds and LayerKind.ROUTED in active_kinds

    def _get_linear_layer_model_names(self) -> List[str]:
        """Return active dense/shared linear names from the operator registry."""

        names: List[str] = []

        # ``is_moe`` remains a CLI compatibility hint for a dense config. A
        # real MoE config is always resolved from profile-owned contracts.
        suppress_dense = self.is_moe and not bool(
            getattr(self.model_config, "is_moe", False)
        )
        for contract in self._get_active_layer_contracts():
            # Routed expert operators are owned by ``MoETrainer``. Every
            # other active layer domain is a linear-op training surface, so a
            # future profile can add a family without editing this trainer.
            if contract.layer_kind is LayerKind.ROUTED:
                continue
            if suppress_dense and contract.layer_kind is LayerKind.DENSE:
                continue
            for family_id in contract.operator_family_ids:
                family = get_operator_family(family_id)
                names.extend(get_family_profiling_names(family))
        return list(dict.fromkeys(names))

    def _get_training_layer_contract(
        self, model_name: str
    ) -> ResolvedLayerContract | None:
        """Resolve a profile-owned contract for a dense/shared operator."""

        profile = self._get_architecture_profile()
        family_matches = []
        for layer_contract in profile.layer_contracts:
            for family_id in layer_contract.operator_family_ids:
                family = get_operator_family(family_id)
                if any(
                    model_name == operator.name
                    or model_name == operator.profiling_name()
                    for operator in family.operators
                ):
                    family_matches.append((layer_contract, family_id))

        if not family_matches:
            # Non-layer aliases (for example the many-to-one ``add`` alias)
            # continue through the established generic TP resolver.
            return None

        if len(family_matches) != 1:
            raise ValueError(
                "Model architecture profile owns operator query "
                f"{model_name!r} through {len(family_matches)} layer contracts"
            )

        layer_contract, family_id = family_matches[0]
        # Bind with the profile-declared family so duplicate or disabled
        # registry entries remain visible to the caller.
        binding = bind_operator_query(model_name, family_id=family_id)
        if binding.family_id != family_id:
            raise ValueError(
                f"Operator query {model_name!r} resolved to family "
                f"{binding.family_id!r}, expected {family_id!r}"
            )
        if layer_contract.layer_kind is LayerKind.ROUTED:
            # The routed contract belongs to MoETrainer; returning it here
            # would silently move expert training across trainer boundaries.
            return None
        return profile.resolve_layer_contract(
            self.model_config,
            operator_name=model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            expert_parallel_size=getattr(self, "expert_parallel_size", None),
        )

    def _get_model_hash_identity(self, model_name: str) -> str | None:
        """Include typed identity when a mixed model trains a typed FFN op.

        Common linear operations and pure-model trainers retain the legacy
        cache key. Mixed-layer FFN operations share names across dense and
        routed domains, so their profile-owned contract must participate in
        standalone training cache identity just as it does in the shared
        prediction manager.
        """

        if not self._is_mixed_layer_model():
            return None
        layer_contract = self._get_training_layer_contract(model_name)
        if layer_contract is None:
            return None
        profile = self._get_architecture_profile()
        return profile.serialize_layer_contract_identity(
            self.model_config,
            layer_contract=layer_contract,
        )

    @staticmethod
    def _typed_row_matches_contract(
        raw_contracts: Any,
        model_name: str,
        layer_contract: ResolvedLayerContract,
    ) -> bool:
        """Match one row against the complete profile-owned layer contract."""

        contracts = parse_typed_operator_contracts(raw_contracts)
        metadata = contracts.get(model_name)
        if metadata is None:
            return False
        # Validate the complete row schema before applying the semantic match.
        # ``expected_metadata={}`` deliberately leaves sibling typed domains
        # eligible for filtering while still rejecting missing or malformed
        # required fields at the training admission boundary.
        validate_typed_operator_metadata(
            metadata,
            operator_name=model_name,
            expected_metadata={},
        )
        return matches_resolved_layer_contract(
            contracts,
            layer_contract,
            operator_name=model_name,
        )

    @staticmethod
    def _legacy_width_columns(
        layer_contract: ResolvedLayerContract,
    ) -> tuple[str, ...]:
        """Return legacy width columns ordered by the declared dimension source."""

        if layer_contract.dimension_source is LayerDimensionSource.SHARED:
            return ("share_expert_dim", "n_expanded_embd")
        return ("n_expanded_embd",)

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
        layer_model_names = self._get_linear_layer_model_names()
        required_columns.extend(
            f"time_stats.{model_name}.median"
            for model_name in layer_model_names
        )
        # Only require add column for non-fused LayerNorm models
        if not self.model_config.uses_fused_add_norm:
            required_columns.append("time_stats.add.median")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        if self._is_mixed_layer_model() and layer_model_names:
            if TYPED_OPERATOR_CONTRACTS_COLUMN not in df.columns:
                raise ValueError(
                    "mixed-layer linear-op training requires the canonical "
                    f"'{TYPED_OPERATOR_CONTRACTS_COLUMN}' column"
                )
            declared_operator_names: set[str] = set()
            for row_index, raw_contracts in df[
                TYPED_OPERATOR_CONTRACTS_COLUMN
            ].items():
                try:
                    contracts = parse_typed_operator_contracts(raw_contracts)
                except ValueError as exc:
                    raise ValueError(
                        "invalid typed operator metadata at row "
                        f"{row_index} in {self.dataset_path}"
                    ) from exc
                declared_operator_names.update(contracts)
            missing_contracts = sorted(
                set(layer_model_names).difference(declared_operator_names)
            )
            if missing_contracts:
                raise ValueError(
                    "mixed-layer linear-op metadata is missing active operator "
                    f"contracts: {missing_contracts}"
                )

        if TYPED_OPERATOR_CONTRACTS_COLUMN in df.columns:
            # Validate the canonical metadata even for a pure model when a
            # producer has opted into the typed schema.
            for row_index, raw_contracts in df[
                TYPED_OPERATOR_CONTRACTS_COLUMN
            ].items():
                try:
                    parse_typed_operator_contracts(raw_contracts)
                except ValueError as exc:
                    raise ValueError(
                        "invalid typed operator metadata at row "
                        f"{row_index} in {self.dataset_path}"
                    ) from exc

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

        Dense and shared-expert names come from the active architecture
        profile. Routed expert names remain owned by ``MoETrainer``.

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

        return self._get_linear_layer_model_names() + common_models

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
        if model_name in get_target_embedded_mtp_linear_ops():
            return resolve_effective_attention_tp_size(
                op_name="attn_pre_proj",
                requested_tp_size=self.tensor_parallel_size,
                num_kv_heads=self.model_config.num_kv_heads,
                cluster_type=None,
                warning_cache=None,
                include_linear_ops=True,
            )

        layer_contract = self._get_training_layer_contract(model_name)
        if layer_contract is not None:
            if layer_contract.tensor_parallel_size is None:
                raise ValueError(
                    "Profile-owned linear layer contract did not resolve a "
                    f"tensor parallel size for {model_name!r}"
                )
            return layer_contract.tensor_parallel_size

        try:
            tp_mode = resolve_operator_query_tp_mode(
                model_name,
                architecture_profile=self._get_architecture_profile(),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported linear op for TP mapping: {model_name}"
            ) from exc

        if tp_mode is TensorParallelMode.REPLICATED:
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

        if tp_mode is TensorParallelMode.FFN_TP:
            return self.tensor_parallel_size

        if tp_mode is TensorParallelMode.ATTENTION_TP:
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

        layer_contract = self._get_training_layer_contract(model_name)
        if layer_contract is not None:
            expected_width = layer_contract.effective_ffn_width
            typed_column = TYPED_OPERATOR_CONTRACTS_COLUMN
            available_widths: list[int] = []
            if typed_column in training_df.columns:
                typed_mask = []
                typed_widths: set[int] = set()
                for row_index, raw_contracts in training_df[typed_column].items():
                    try:
                        contracts = parse_typed_operator_contracts(raw_contracts)
                        for metadata in contracts.values():
                            width = metadata.get("effective_ffn_width")
                            if type(width) is int:
                                typed_widths.add(width)
                        typed_mask.append(
                            self._typed_row_matches_contract(
                                raw_contracts,
                                model_name,
                                layer_contract,
                            )
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "invalid typed operator metadata at row "
                            f"{row_index} in {self.dataset_path}"
                        ) from exc
                training_df = training_df.loc[typed_mask].copy()
            elif self._is_mixed_layer_model():
                raise ValueError(
                    "mixed-layer linear-op training requires the canonical "
                    f"'{typed_column}' column for {model_name}"
                )
            else:
                width_column = next(
                    (
                        column
                        for column in self._legacy_width_columns(layer_contract)
                        if column in training_df.columns
                        and training_df[column].notna().any()
                    ),
                    None,
                )
                if width_column is None:
                    raise ValueError(
                        "linear-op training requires a legacy width column for "
                        f"{model_name}; expected one of "
                        f"{self._legacy_width_columns(layer_contract)}"
                    )
                available_widths = sorted(
                    training_df[width_column].dropna().unique().tolist()
                )
                training_df = training_df[
                    training_df[width_column] == expected_width
                ].copy()
            logger.info(
                "  Typed contract filter for %s: %d rows "
                "(layer_kind=%s, width=%d, tp_mode=%s)",
                model_name,
                len(training_df),
                layer_contract.layer_kind.value,
                expected_width,
                layer_contract.tensor_parallel_mode.value,
            )
            if len(training_df) == 0:
                if typed_column in training_df.columns:
                    available_widths = sorted(typed_widths)
                raise ValueError(
                    "No profiling rows match the typed layer contract for "
                    f"{model_name}: layer_kind={layer_contract.layer_kind.value}, "
                    f"effective_ffn_width={expected_width}, "
                    f"available_widths={available_widths} in {self.dataset_path}"
                )

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
