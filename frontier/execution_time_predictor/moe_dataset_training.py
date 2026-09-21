"""Admission and training of the MoE profiling dataset.

The dataset contract decides which profiling rows a given replica may train
on, at its tensor- and expert-parallel sizes and its routing runtime.  The
trainer then fits one model per MoE operator from the admitted rows.
"""

import numpy as np
import os
import pandas as pd

from frontier.execution_time_predictor.moe_predictor_helpers import (
    _get_moe_family_model_names,
    _get_prefill_hot_moe_gating_model_names,
    _is_moe_gating_family_model_name,
    _validate_moe_columns,
)
from frontier.logger import init_logger
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
    PrefillHotRowsUnavailableError,
    filter_moe_gating_rows_by_runtime_context,
    get_moe_gating_base_model_name,
    has_prefill_hot_moe_gating_rows,
    should_enable_prefill_hot_moe_gating_contract,
)
from frontier.moe_routing_runtime import filter_moe_gating_routing_topk_rows
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    validate_typed_operator_contracts,
)
from sklearn.base import BaseEstimator
from typing import Any, Dict, List, Optional


logger = init_logger(__name__)


class MoeDatasetTraining:
    """MoE profiling dataset admission and per-operator training."""


    # Load imbalance feature columns used for MoE training (aligned with SharedPredictionModelManager)
    # Reference: frontier/training/moe_trainer.py lines 224-239 (authoritative source)
    MOE_LOAD_IMBALANCE_FEATURES = [
        # Config features (6) - describe model configuration
        "total_routed_tokens",  # Total tokens after routing (num_tokens * router_topk)
        "num_experts_per_device",  # Number of experts per device after EP sharding
        "hidden_dim",  # Model hidden dimension
        "expert_hidden_dim",  # Expert FFN hidden dimension
        "router_topk",  # Number of experts each token is routed to
        "model_expansion_ratio",  # expert_hidden_dim / hidden_dim
        # Derived features (2) - derived from config and routing
        "tokens_per_expert_avg",  # Average tokens per expert
        "tokens_to_experts_ratio",  # tokens / num_experts ratio
        # Load features (6) - describe load distribution characteristics
        "expert_utilization",  # Proportion of experts with non-zero load
        "min_load_ratio",  # Min load / average load
        "load_imbalance_cv",  # Coefficient of Variation: std/mean, key imbalance metric
        "max_load_ratio",  # Max load / average load
        "load_entropy",  # Entropy of load distribution (higher = more uniform)
        "load_gini_coefficient",  # Gini coefficient: 0=equality, 1=inequality
    ]

    def _validate_moe_dataset_contract(
        self,
        moe_df: pd.DataFrame,
        moe_input_file: str,
        model_names: List[str],
        moe_tp_size: int,
        moe_ep_size: int,
    ) -> pd.DataFrame:
        """Validate op-level MoE key coverage and return model-filtered dataframe."""
        _validate_moe_columns(moe_df)
        required_columns = [
            "num_experts",
            "router_topk",
            "hidden_dim",
            "expert_hidden_dim",
            "num_tensor_parallel_workers",
            "expert_parallel_size",
        ]
        missing_columns = [col for col in required_columns if col not in moe_df.columns]
        if missing_columns:
            raise ValueError(
                f"MoE dataset contract validation failed for {moe_input_file}: "
                f"missing required columns {missing_columns}."
            )

        model_config = self._model_config
        base_df = moe_df[
            (moe_df["num_experts"] == model_config.num_experts)
            & (moe_df["router_topk"] == model_config.num_experts_per_tok)
            & (moe_df["hidden_dim"] == model_config.embedding_dim)
            & (moe_df["expert_hidden_dim"] == model_config.mlp_hidden_dim)
        ].copy()

        if len(base_df) == 0:
            raise ValueError(
                "MoE dataset contract validation failed: no rows match model configuration in "
                f"{moe_input_file}. Required: num_experts={model_config.num_experts}, "
                f"router_topk={model_config.num_experts_per_tok}, hidden_dim={model_config.embedding_dim}, "
                f"expert_hidden_dim={model_config.mlp_hidden_dim}."
            )

        available_pairs = sorted(
            {
                (int(tp), int(ep))
                for tp, ep in base_df[
                    ["num_tensor_parallel_workers", "expert_parallel_size"]
                ].drop_duplicates().itertuples(index=False, name=None)
            }
        )
        requested_routing_runtime_path = (
            self._get_requested_moe_gating_routing_runtime_path()
        )

        missing_requirements: List[str] = []
        for model_name in model_names:
            base_model_name = get_moe_gating_base_model_name(model_name)
            tp_key = self._get_moe_op_tp_key(
                base_model_name,
                moe_tp_size,
                cluster_type=getattr(self, "_cluster_type", None),
            )
            requirement_parts = [f"TP={tp_key}"]
            if self._is_moe_op_ep_agnostic(base_model_name):
                op_df = base_df[base_df["num_tensor_parallel_workers"] == tp_key]
                requirement_parts.append("EP=ANY")
            else:
                op_df = base_df[
                    (base_df["num_tensor_parallel_workers"] == tp_key)
                    & (base_df["expert_parallel_size"] == moe_ep_size)
                ]
                requirement_parts.append(f"EP={moe_ep_size}")
            if base_model_name == "moe_gating_routing_topk":
                op_df = filter_moe_gating_routing_topk_rows(
                    op_df,
                    requested_runtime_path=requested_routing_runtime_path,
                    source_name=moe_input_file,
                )
                requirement_parts.append(
                    f"routing_runtime_path={requested_routing_runtime_path}"
                )
            if _is_moe_gating_family_model_name(base_model_name):
                op_df = filter_moe_gating_rows_by_runtime_context(
                    op_df,
                    requested_context=DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
                    source_name=moe_input_file,
                )
                requirement_parts.append(
                    "gating_runtime_context="
                    f"{DEFAULT_MOE_GATING_RUNTIME_CONTEXT}"
                )
            requirement = ", ".join(requirement_parts)
            if len(op_df) == 0:
                missing_requirements.append(f"{model_name} requires {requirement}")
                continue
            target_col = f"time_stats.{base_model_name}.median"
            if op_df[target_col].dropna().empty:
                missing_requirements.append(
                    f"{model_name} requires {requirement}, target={target_col} "
                    "to contain at least one non-NaN row"
                )

        if missing_requirements:
            requirement_text = "\n  - ".join(missing_requirements)
            raise ValueError(
                "MoE dataset contract validation failed before training.\n"
                f"File: {moe_input_file}\n"
                "Missing op-level key coverage:\n"
                f"  - {requirement_text}\n"
                f"Available (TP, EP) pairs for matched model rows: {available_pairs}"
            )

        return base_df

    def _train_moe_models(self) -> Dict[str, BaseEstimator]:
        """Train MoE-specific models (gating, shuffling, grouped_gemm) for independent training mode.

        For moe_grouped_gemm, uses 14 load-imbalance features if available in the profiling data.
        This enables simulation mode with per-expert token allocation.
        Other MoE models (gating_linear, gating_routing_topk, shuffling) use only num_tokens.
        """
        models = {}
        moe_input_file = getattr(self, "_moe_input_file", "/synthetic/moe.csv")

        if not os.path.exists(moe_input_file):
            logger.warning(f"MoE input file does not exist: {moe_input_file}")
            return models

        moe_df = pd.read_csv(moe_input_file)
        if TYPED_OPERATOR_CONTRACTS_COLUMN in moe_df.columns:
            # Validate every row before scalar or TP/EP filtering can hide a
            # malformed typed contract.
            moe_df[TYPED_OPERATOR_CONTRACTS_COLUMN].map(
                lambda raw_contracts: validate_typed_operator_contracts(
                    raw_contracts,
                    model_config=self._model_config,
                )
            )

        metadata = self._get_profiling_metadata(moe_df, moe_input_file)
        self._validate_active_measurement_type(metadata, moe_input_file)

        tp_col = "num_tensor_parallel_workers"
        ep_col = "expert_parallel_size"
        moe_tp_size = self._replica_config.moe_tensor_parallel_size
        moe_ep_size = self._replica_config.moe_expert_parallel_size

        if tp_col not in moe_df.columns:
            raise ValueError(
                f"Required column '{tp_col}' is missing in {moe_input_file}. "
                "Re-run MoE profiling with TP metadata enabled."
            )
        if ep_col not in moe_df.columns:
            raise ValueError(
                f"Required column '{ep_col}' is missing in {moe_input_file}. "
                "Re-run MoE profiling with EP metadata enabled."
            )

        base_model_names = _get_moe_family_model_names()
        model_names = list(base_model_names)
        model_filtered_df = self._validate_moe_dataset_contract(
            moe_df,
            moe_input_file,
            base_model_names,
            moe_tp_size,
            moe_ep_size,
        )
        if should_enable_prefill_hot_moe_gating_contract(
            model_config=self._model_config,
        ):
            if has_prefill_hot_moe_gating_rows(model_filtered_df):
                model_names.extend(_get_prefill_hot_moe_gating_model_names())
            else:
                logger.warning(
                    "Prefill-hot gating contract is enabled for model=%s, but "
                    "dataset %s has no usable prefill_hot rows; skipping "
                    "__prefill_hot pseudo-model training.",
                    self._replica_config.model_name,
                    moe_input_file,
                )

        self._register_profiling_metadata_for_ops(
            model_names, metadata, moe_input_file
        )

        requested_routing_runtime_path = (
            self._get_requested_moe_gating_routing_runtime_path()
        )
        moe_df_cache: Dict[
            tuple[int, Optional[int], Optional[str], Optional[str]], pd.DataFrame
        ] = {}

        def _get_moe_df_for_op(
            model_name: str,
        ) -> tuple[pd.DataFrame, int, Optional[int]]:
            base_model_name = get_moe_gating_base_model_name(model_name)
            tp_key = self._get_moe_op_tp_key(
                base_model_name,
                moe_tp_size,
                cluster_type=getattr(self, "_cluster_type", None),
            )
            ep_key: Optional[int]
            if self._is_moe_op_ep_agnostic(base_model_name):
                ep_key = None
            else:
                ep_key = moe_ep_size
            runtime_path_key: Optional[str] = None
            if base_model_name == "moe_gating_routing_topk":
                runtime_path_key = requested_routing_runtime_path
            gating_context_key: Optional[str] = None
            if _is_moe_gating_family_model_name(base_model_name):
                gating_context_key = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
                if model_name.endswith("__prefill_hot"):
                    gating_context_key = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT
            cache_key = (tp_key, ep_key, runtime_path_key, gating_context_key)
            if cache_key not in moe_df_cache:
                filtered_df = model_filtered_df[
                    model_filtered_df[tp_col] == tp_key
                ].copy()
                if ep_key is not None:
                    filtered_df = filtered_df[
                        filtered_df[ep_col] == ep_key
                    ].copy()
                if runtime_path_key is not None:
                    filtered_df = filter_moe_gating_routing_topk_rows(
                        filtered_df,
                        requested_runtime_path=runtime_path_key,
                        source_name=moe_input_file,
                    )
                if gating_context_key is not None:
                    filtered_df = filter_moe_gating_rows_by_runtime_context(
                        filtered_df,
                        requested_context=gating_context_key,
                        source_name=moe_input_file,
                    )
                if len(filtered_df) == 0:
                    ep_desc = "ANY" if ep_key is None else str(ep_key)
                    raise ValueError(
                        f"No MoE data after filtering for TP={tp_key}, EP={ep_desc}. "
                        f"Requested by op-level TP mapping in {moe_input_file}."
                    )
                filtered_df["num_tokens_rounded"] = filtered_df["num_tokens"].apply(
                    lambda x: max(1, round(x / 8) * 8)
                )
                moe_df_cache[cache_key] = filtered_df
            return moe_df_cache[cache_key], tp_key, ep_key

        for model_name in model_names:
            try:
                op_df, moe_tp_key, moe_ep_key = _get_moe_df_for_op(model_name)
            except PrefillHotRowsUnavailableError as exc:
                logger.warning(
                    "Skipping %s because prefill-hot gating rows are unavailable "
                    "for the requested TP/EP slice (%s).",
                    model_name,
                    exc,
                )
                continue
            target_op_name = get_moe_gating_base_model_name(model_name)
            target_col = f"time_stats.{target_op_name}.median"
            if target_col not in op_df.columns:
                ep_desc = "ANY" if moe_ep_key is None else str(moe_ep_key)
                raise ValueError(
                    f"Column '{target_col}' not found in MoE dataframe for TP={moe_tp_key}, EP={ep_desc}. "
                    "Re-run MoE profiling with split gating columns."
                )

            # Per-operation feature selection (aligned with SharedPredictionModelManager).
            if model_name == "moe_grouped_gemm":
                available_load_features = [
                    f for f in self.MOE_LOAD_IMBALANCE_FEATURES if f in op_df.columns
                ]
                has_load_imbalance_features = len(available_load_features) == len(
                    self.MOE_LOAD_IMBALANCE_FEATURES
                )
                if 0 < len(available_load_features) < len(self.MOE_LOAD_IMBALANCE_FEATURES):
                    missing_features = [
                        f
                        for f in self.MOE_LOAD_IMBALANCE_FEATURES
                        if f not in op_df.columns
                    ]
                    raise ValueError(
                        f"Partial load imbalance features found ({len(available_load_features)}/"
                        f"{len(self.MOE_LOAD_IMBALANCE_FEATURES)}) for TP={moe_tp_key}. "
                        f"Missing: {missing_features}."
                    )
                if has_load_imbalance_features:
                    feature_cols = available_load_features
                    logger.info(
                        f"  {model_name}: Using load imbalance features ({len(feature_cols)} features, TP={moe_tp_key})"
                    )
                else:
                    feature_cols = ["num_tokens"]
                    logger.info(
                        f"  {model_name}: Load imbalance features not found; using num_tokens only (TP={moe_tp_key})"
                    )
            elif model_name == "moe_shuffling":
                available_load_features = [
                    f for f in self.MOE_LOAD_IMBALANCE_FEATURES if f in op_df.columns
                ]
                if len(available_load_features) == len(self.MOE_LOAD_IMBALANCE_FEATURES):
                    feature_cols = available_load_features
                    logger.info(
                        f"  {model_name}: Using load imbalance features ({len(feature_cols)} features, TP={moe_tp_key})"
                    )
                else:
                    feature_cols = ["num_tokens"]
                    logger.info(
                        f"  {model_name}: Full load imbalance features unavailable; using num_tokens only (TP={moe_tp_key})"
                    )
            else:
                feature_cols = ["num_tokens"]
                logger.info(f"  {model_name}: Using num_tokens only (1 feature, TP={moe_tp_key})")

            models[model_name] = self._train_model(
                model_name=model_name,
                df=op_df,
                feature_cols=feature_cols,
                target_col=target_col,
            )
            logger.info(f"Trained MoE model: {model_name}")

        return models

    def _register_additional_profiling_metadata_from_files(self) -> None:
        moe_input_file = self._moe_input_file
        model_names = _get_moe_family_model_names()
        if should_enable_prefill_hot_moe_gating_contract(
            model_config=self._model_config,
        ):
            include_prefill_hot_models = False
            try:
                moe_df = pd.read_csv(moe_input_file)
            except FileNotFoundError:
                moe_df = None
            if moe_df is not None:
                if TYPED_OPERATOR_CONTRACTS_COLUMN in moe_df.columns:
                    moe_df[TYPED_OPERATOR_CONTRACTS_COLUMN].map(
                        lambda raw_contracts: validate_typed_operator_contracts(
                            raw_contracts,
                            model_config=self._model_config,
                        )
                    )
                include_prefill_hot_models = has_prefill_hot_moe_gating_rows(moe_df)
            if include_prefill_hot_models:
                model_names.extend(_get_prefill_hot_moe_gating_model_names())
        self._register_profiling_metadata_from_file(moe_input_file, model_names)

    def _train_models(self) -> Dict[str, BaseEstimator]:
        """Override to include MoE model training for independent training mode."""
        models = super()._train_models()

        if self._model_manager is None:
            moe_models = self._train_moe_models()
            models.update(moe_models)
            logger.info(f"Trained MoE models independently: {list(moe_models.keys())}")
        else:
            logger.info("MoE models loaded from ExecutionTimePredictionModelManager.")

        return models

    def _predict_for_compute_models(self) -> Dict[str, Any]:
        predictions = super()._predict_for_compute_models()
        extra_model_names = _get_prefill_hot_moe_gating_model_names()
        num_token_range = np.arange(1, self._max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})
        for model_name in extra_model_names:
            if model_name not in self._models:
                continue
            model = self._models[model_name]
            predictions[model_name] = self._get_model_prediction(
                model_name, model, X
            )
        return predictions
