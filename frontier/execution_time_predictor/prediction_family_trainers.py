"""Per-family training of the execution-time prediction models.

One method per operator family: dense FFN and MoE, dense MLP, attention,
latent MLA attention, residual, pipeline- and tensor-parallel communication,
and CPU overhead.  Each selects the rows its family owns, builds the feature
frame, and hands one model at a time to the shared fitting routine.
"""

import os
import pandas as pd

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.attention.model_binding import resolve_runtime_attention_family
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_name_by_role,
    get_enabled_predictor_metric_names,
    get_enabled_shared_predictor_feature_columns,
    validate_attention_profiling_dataframe,
)
from frontier.attention.string_coercion import coerce_truthy_int
from frontier.execution_time_predictor.prediction_model_identity import (
    _add_layer_contract_to_training_context,
    _build_exact_feature_lookup,
    _get_contract_hash,
    _get_moe_family_model_names,
    _get_prefill_hot_moe_gating_model_names,
    _is_moe_gating_family_model_name,
    _layer_contract_kwargs,
    _normalize_layer_contract_context,
    _resolve_model_architecture_profile,
    _resolve_model_architecture_profile_id,
    _serialize_selected_layer_cache_identity,
)
from frontier.logger import init_logger
from frontier.model_architectures import ResolvedLayerContract
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
    PrefillHotRowsUnavailableError,
    filter_moe_gating_rows_by_runtime_context,
    get_moe_gating_base_model_name,
    has_prefill_hot_moe_gating_rows,
    should_enable_prefill_hot_moe_gating_contract,
)
from frontier.moe_routing_runtime import (
    filter_moe_gating_routing_topk_rows,
    resolve_moe_gating_routing_runtime_path,
)
from frontier.operators.families import (
    FFN_FAMILY,
    SHARE_EXPERT_FAMILY,
    get_family_profiling_names,
)
from frontier.spec_decode.runtime import is_target_embedded_mtp_enabled
from frontier.types import ClusterType, MeasurementType
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, List, Optional, Tuple


logger = init_logger(__name__)


class PredictionFamilyTrainers:
    """Per-family model training for the execution-time predictor."""

    def _train_ffn_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config,
                                        linear_ops_file: str, moe_file: str,
                                        is_moe_model: bool, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """
        Train FFN/MoE models for a specific cluster.

        This function handles FFN-related operations in the Transformer layer:
        - FFN core operations (from linear_op.csv): mlp_up_proj, mlp_down_proj, mlp_act
        - MoE core operations (from moe.csv): moe_gating_linear, moe_gating_routing_topk, moe_shuffling, moe_grouped_gemm
        - Pre-FFN normalization (from linear_op.csv): post_attention_layernorm

        Transformer layer context:
            ... → Attention → add → [post_attention_layernorm] → [FFN/MoE] → add → ...
        """
        models = {}

        ffn_tp_key = self._get_ffn_tp_key(cluster_type, replica_config, is_moe_model)
        tp_size = ffn_tp_key

        # Create a signature for this FFN model configuration.
        model_config = replica_config.model_config
        model_arch = model_config.get_model_arch() if model_config is not None else "generic"
        architecture_profile_id = _resolve_model_architecture_profile_id(model_config)
        primary_contract = self._resolve_typed_layer_contract(
            "moe_grouped_gemm" if is_moe_model else "mlp_up_proj",
            cluster_type,
            replica_config,
            is_moe_model=is_moe_model,
        )
        typed_contract_hash = self._get_ffn_contract_signature(
            cluster_type,
            replica_config,
            is_moe_model,
        )
        active_measurement_type = getattr(
            self, "_active_measurement_type", MeasurementType.CUDA_EVENT
        )
        ffn_signature = (
            f"ffn_{replica_config.device}_{replica_config.model_name}_{tp_size}"
            f"_moe{is_moe_model}_arch_profile{architecture_profile_id}"
            f"_layer_contracts{typed_contract_hash}"
            f"_family{self._measurement_family_name(active_measurement_type)}"
        )

        if ffn_signature in trained_model_signatures:
            logger.info(f"Skipping FFN models training for {cluster_type} - already trained with signature {ffn_signature}")
            return models

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'is_moe_model': is_moe_model,
            'model_arch': model_arch,
            'model_architecture_profile': architecture_profile_id,
            'use_qk_norm': bool(getattr(model_config, 'use_qk_norm', False)),
        }

        # Choose input file based on model type
        if is_moe_model:
            moe_input_file = moe_file
            if not os.path.exists(moe_input_file):
                raise FileNotFoundError(f"MoE input file {moe_input_file} not found")
            logger.info(f"Loading MoE data for {cluster_type} from: {moe_input_file}")
            training_context['input_file'] = moe_input_file

            # MoE core operations with per-operation feature selection
            # Split gating into moe_gating_linear and moe_gating_routing_topk (Step 1.6)
            # Aligned with frontier/training/moe_trainer.py _get_feature_cols() method
            base_moe_model_names = _get_moe_family_model_names()
            moe_model_names = list(base_moe_model_names)
            if should_enable_prefill_hot_moe_gating_contract(
                model_config=model_config,
                model_arch=model_arch,
                model_name=replica_config.model_name,
            ):
                prefill_hot_probe_df = pd.read_csv(moe_input_file)
                include_prefill_hot_models = has_prefill_hot_moe_gating_rows(
                    prefill_hot_probe_df
                )

                if include_prefill_hot_models:
                    moe_model_names.extend(_get_prefill_hot_moe_gating_model_names())
                else:
                    logger.warning(
                        "Prefill-hot gating contract enabled for model=%s, but "
                        "dataset %s has no usable prefill_hot rows; skipping "
                        "__prefill_hot pseudo-models in shared-manager training.",
                        replica_config.model_name,
                        moe_input_file,
                    )
            self._validate_moe_dataset_contract(
                moe_input_file,
                replica_config,
                base_moe_model_names,
                cluster_type,
                **_layer_contract_kwargs(primary_contract),
            )
            requested_routing_runtime_path = resolve_moe_gating_routing_runtime_path(
                getattr(replica_config, "moe_routing_distribution_type", "balanced")
            )

            moe_df_cache: Dict[
                Tuple[
                    int,
                    Optional[int],
                    Optional[str],
                    Optional[str],
                    Optional[str],
                ],
                pd.DataFrame,
            ] = {}

            def _get_moe_df_for_op(
                model_name: str,
            ) -> Tuple[
                pd.DataFrame,
                int,
                Optional[int],
                Optional[ResolvedLayerContract],
            ]:
                base_model_name = get_moe_gating_base_model_name(model_name)
                op_layer_contract = self._resolve_typed_layer_contract(
                    base_model_name,
                    cluster_type,
                    replica_config,
                    is_moe_model=True,
                )
                tp_key = self._get_moe_op_tp_key(
                    base_model_name,
                    replica_config,
                    cluster_type,
                )
                if tp_key <= 0:
                    raise ValueError(
                        f"Invalid TP key for MoE training: {tp_key} (op={model_name})"
                    )

                ep_key: Optional[int]
                if self._is_moe_op_ep_agnostic(base_model_name):
                    ep_key = None
                else:
                    ep_key = replica_config.moe_expert_parallel_size

                runtime_path_key: Optional[str] = None
                if base_model_name == "moe_gating_routing_topk":
                    runtime_path_key = requested_routing_runtime_path

                gating_context_key: Optional[str] = None
                if _is_moe_gating_family_model_name(base_model_name):
                    gating_context_key = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
                    if model_name.endswith("__prefill_hot"):
                        gating_context_key = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT

                contract_identity = _serialize_selected_layer_cache_identity(
                    op_layer_contract
                )
                cache_key = (
                    tp_key,
                    ep_key,
                    runtime_path_key,
                    gating_context_key,
                    contract_identity,
                )
                if cache_key not in moe_df_cache:
                    op_df = self._load_moe_df(
                        moe_input_file,
                        replica_config,
                        load_imbalance=False,
                        tensor_parallel_size=tp_key,
                        expert_parallel_size=ep_key,
                        **_layer_contract_kwargs(
                            op_layer_contract,
                            operator_name=base_model_name,
                        ),
                    )
                    if runtime_path_key is not None:
                        op_df = filter_moe_gating_routing_topk_rows(
                            op_df,
                            requested_runtime_path=runtime_path_key,
                            source_name=moe_input_file,
                        )
                    if gating_context_key is not None:
                        op_df = filter_moe_gating_rows_by_runtime_context(
                            op_df,
                            requested_context=gating_context_key,
                            source_name=moe_input_file,
                        )
                    moe_df_cache[cache_key] = op_df
                    ep_desc = "ANY" if ep_key is None else str(ep_key)
                    logger.info(
                        f"Loaded {len(moe_df_cache[cache_key])} rows for MoE training "
                        f"(op={model_name}, tp_key={tp_key}, ep_key={ep_desc}, "
                        f"routing_runtime_path={runtime_path_key or 'ANY'}, "
                        f"gating_runtime_context={gating_context_key or 'ANY'}, "
                        "auto feature mode)"
                    )
                return moe_df_cache[cache_key], tp_key, ep_key, op_layer_contract

            for model_name in moe_model_names:
                model_signature = f"{model_name}_{ffn_signature}"
                if model_signature not in trained_model_signatures:
                    try:
                        (
                            op_moe_df,
                            moe_tp_key,
                            moe_ep_key,
                            op_layer_contract,
                        ) = _get_moe_df_for_op(model_name)
                    except PrefillHotRowsUnavailableError as exc:
                        logger.warning(
                            "Skipping %s because prefill-hot gating rows are unavailable "
                            "for the requested TP/EP slice (%s).",
                            model_name,
                            exc,
                        )
                        continue
                    op_training_context = _add_layer_contract_to_training_context(
                        training_context,
                        op_layer_contract,
                    )
                    op_training_context['tensor_parallel_size'] = moe_tp_key
                    op_training_context['expert_parallel_size'] = (
                        "ANY" if moe_ep_key is None else moe_ep_key
                    )

                    # Per-operation feature selection.
                    if model_name == "moe_grouped_gemm":
                        available_load_features = [
                            f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                            if f in op_moe_df.columns
                        ]
                        has_load_imbalance_features = (
                            len(available_load_features)
                            == len(self.MOE_LOAD_IMBALANCE_FEATURES)
                        )
                        if 0 < len(available_load_features) < len(self.MOE_LOAD_IMBALANCE_FEATURES):
                            missing_features = [
                                f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                                if f not in op_moe_df.columns
                            ]
                            raise ValueError(
                                f"Partial load imbalance features found ({len(available_load_features)}/"
                                f"{len(self.MOE_LOAD_IMBALANCE_FEATURES)}) for {model_name} at TP={moe_tp_key}. "
                                f"Missing: {missing_features}."
                            )

                        if has_load_imbalance_features:
                            op_feature_cols = available_load_features
                            logger.info(
                                f"  {model_name}: Using load imbalance features "
                                f"({len(op_feature_cols)} features, TP={moe_tp_key})"
                            )
                        else:
                            op_feature_cols = ["num_tokens"]
                            logger.info(
                                f"  {model_name}: Load imbalance features not found; "
                                f"using num_tokens only (TP={moe_tp_key})."
                            )
                    elif model_name == "moe_shuffling":
                        available_load_features = [
                            f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                            if f in op_moe_df.columns
                        ]
                        if len(available_load_features) == len(self.MOE_LOAD_IMBALANCE_FEATURES):
                            op_feature_cols = available_load_features
                            logger.info(
                                f"  {model_name}: Using load imbalance features "
                                f"({len(op_feature_cols)} features, TP={moe_tp_key})"
                            )
                        else:
                            # For shuffling we allow partial/legacy datasets and fall back to
                            # num_tokens-only training when the full load feature set is absent.
                            op_feature_cols = ["num_tokens"]
                            logger.info(
                                f"  {model_name}: Full load imbalance features unavailable; "
                                f"using num_tokens only (TP={moe_tp_key})."
                            )
                    else:
                        op_feature_cols = ["num_tokens"]
                        logger.info(
                            f"  {model_name}: Using num_tokens only (1 feature, TP={moe_tp_key})"
                        )

                    # Store feature_cols in training_context for this specific operation
                    op_training_context['feature_cols'] = op_feature_cols

                    target_op_name = get_moe_gating_base_model_name(model_name)
                    train_kwargs: Dict[str, Any] = dict(
                        model_name=model_name,
                        df=op_moe_df,
                        feature_cols=op_feature_cols,
                        target_col=f"time_stats.{target_op_name}.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=op_training_context,
                    )
                    train_kwargs.update(_layer_contract_kwargs(op_layer_contract))
                    models[model_name] = self._train_single_model(
                        **train_kwargs,
                    )
                    trained_model_signatures.add(model_signature)
                    logger.info(f"Trained {model_name} for {cluster_type} with features: {op_feature_cols}")

            # Step2Mini/Step3 share_expert operations (forward_3: shared expert alongside routed experts)
            model_config = replica_config.model_config
            if model_config is not None and model_config.supports_share_expert():
                # share_expert operations are trained from linear_op.csv (not moe.csv)
                if not os.path.exists(linear_ops_file):
                    raise FileNotFoundError(
                        f"Linear ops input file {linear_ops_file} not found for share_expert"
                    )

                step2mini_share_expert_model_names = list(
                    get_family_profiling_names(SHARE_EXPERT_FAMILY)
                )
                if not step2mini_share_expert_model_names:
                    raise ValueError("Shared-expert operator family has no profiling names")
                share_expert_tp_key = self._get_linear_op_tp_key(
                    step2mini_share_expert_model_names[0],
                    cluster_type,
                    replica_config,
                    is_moe_model,
                )
                shared_layer_contract = self._resolve_typed_layer_contract(
                    step2mini_share_expert_model_names[0],
                    cluster_type,
                    replica_config,
                    is_moe_model=True,
                )
                if (
                    shared_layer_contract is None
                    and _resolve_model_architecture_profile(model_config) is not None
                ):
                    raise ValueError(
                        "Missing shared layer contract for share-expert training"
                    )
                share_expert_linear_ops_df = self._load_linear_op_df(
                    linear_ops_file,
                    share_expert_tp_key,
                    **_layer_contract_kwargs(
                        shared_layer_contract,
                        operator_name=step2mini_share_expert_model_names[0],
                    ),
                )
                logger.info(f"Loaded {len(share_expert_linear_ops_df)} rows for share_expert training")

                for model_name in step2mini_share_expert_model_names:
                    model_signature = f"{model_name}_{ffn_signature}"
                    if model_signature not in trained_model_signatures:
                        # Update training context to reflect linear_op.csv source.
                        shared_training_context = _add_layer_contract_to_training_context(
                            training_context,
                            shared_layer_contract,
                        )
                        shared_training_context['input_file'] = linear_ops_file
                        shared_training_context['tensor_parallel_size'] = share_expert_tp_key
                        target_col = f"time_stats.{model_name}.median"
                        if target_col not in share_expert_linear_ops_df.columns:
                            raise ValueError(
                                f"share_expert operation '{model_name}' column '{target_col}' not found in profiling data. "
                                f"Ensure profiling was run with a model architecture that includes share_expert. "
                                f"Available columns: {list(share_expert_linear_ops_df.columns)}"
                            )
                        train_kwargs: Dict[str, Any] = dict(
                            model_name=model_name,
                            df=share_expert_linear_ops_df,
                            feature_cols=["num_tokens"],
                            target_col=target_col,
                            execution_time_predictor_config=execution_time_predictor_config,
                            training_context=shared_training_context,
                        )
                        train_kwargs.update(
                            _layer_contract_kwargs(shared_layer_contract)
                        )
                        models[model_name] = self._train_single_model(
                            **train_kwargs,
                        )
                        trained_model_signatures.add(model_signature)
                        logger.info(f"Trained {model_name} for {cluster_type}")

            # Mixed-layer MoE models (for example step-moe-noquant) also have
            # dense boundary layers.  Train these additions after the legacy
            # MoE/share-expert families so RandomForest training order remains
            # compatible with the historical predictor artifact contract.
            if self._is_mixed_layer_moe_model(model_config, is_moe_model):
                dense_ffn_tp_key = self._get_ffn_tp_key(
                    cluster_type, replica_config, is_moe_model=False
                )
                dense_layer_contract = self._resolve_typed_layer_contract(
                    "mlp_up_proj",
                    cluster_type,
                    replica_config,
                    is_moe_model=False,
                )
                dense_ffn_signature = (
                    f"ffn_{replica_config.device}_{replica_config.model_name}_{dense_ffn_tp_key}"
                    f"_moeFalse_arch_profile{architecture_profile_id}"
                    f"_layer_contracts{_get_contract_hash(dense_layer_contract)}"
                    f"_family{self._measurement_family_name(active_measurement_type)}"
                )
                dense_training_context = dict(training_context)
                dense_training_context["is_moe_model"] = False
                dense_training_context["tensor_parallel_size"] = dense_ffn_tp_key
                dense_training_context = _add_layer_contract_to_training_context(
                    dense_training_context,
                    dense_layer_contract,
                )
                self._train_dense_mlp_models_for_cluster(
                    cluster_type=cluster_type,
                    replica_config=replica_config,
                    execution_time_predictor_config=execution_time_predictor_config,
                    linear_ops_file=linear_ops_file,
                    ffn_signature=dense_ffn_signature,
                    ffn_tp_key=dense_ffn_tp_key,
                    training_context=dense_training_context,
                    trained_model_signatures=trained_model_signatures,
                    models=models,
                    layer_contract=dense_layer_contract,
                )
        else:
            self._train_dense_mlp_models_for_cluster(
                cluster_type=cluster_type,
                replica_config=replica_config,
                execution_time_predictor_config=execution_time_predictor_config,
                linear_ops_file=linear_ops_file,
                ffn_signature=ffn_signature,
                ffn_tp_key=ffn_tp_key,
                training_context=training_context,
                trained_model_signatures=trained_model_signatures,
                models=models,
                layer_contract=primary_contract,
            )

        # Pre-FFN normalization (post_attention_layernorm) - always from linear_op.csv
        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found for post_attention_layernorm")
        layernorm_tp_key = self._get_linear_op_tp_key(
            "post_attention_layernorm",
            cluster_type,
            replica_config,
            is_moe_model,
        )
        linear_ops_df = self._load_linear_op_df(linear_ops_file, layernorm_tp_key)
        layernorm_context = dict(training_context)
        layernorm_context["input_file"] = linear_ops_file
        layernorm_context["tensor_parallel_size"] = layernorm_tp_key

        layernorm_model_name = "post_attention_layernorm"
        layernorm_signature = f"{layernorm_model_name}_{ffn_signature}"
        if layernorm_signature not in trained_model_signatures:
            models[layernorm_model_name] = self._train_single_model(
                model_name=layernorm_model_name,
                df=linear_ops_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{layernorm_model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=layernorm_context,
            )
            trained_model_signatures.add(layernorm_signature)
            logger.info(f"Trained {layernorm_model_name} for {cluster_type}")

        # Mark this FFN configuration as trained
        trained_model_signatures.add(ffn_signature)
        return models

    def _train_dense_mlp_models_for_cluster(
        self,
        *,
        cluster_type: ClusterType,
        replica_config,
        execution_time_predictor_config,
        linear_ops_file: str,
        ffn_signature: str,
        ffn_tp_key: int,
        training_context: Dict[str, Any],
        trained_model_signatures: set,
        models: Dict[str, BaseEstimator],
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> None:
        """Materialize dense MLP predictors from the linear-op profile."""
        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found")
        if (
            layer_contract is None
            and _resolve_model_architecture_profile(
                getattr(replica_config, "model_config", None)
            )
            is not None
        ):
            layer_contract = self._resolve_typed_layer_contract(
                "mlp_up_proj",
                cluster_type,
                replica_config,
                is_moe_model=False,
            )
        if (
            layer_contract is not None
            and layer_contract.tensor_parallel_size is not None
            and layer_contract.tensor_parallel_size != ffn_tp_key
        ):
            raise ValueError(
                "Dense FFN training TP conflicts with its typed layer contract: "
                f"ffn_tp_key={ffn_tp_key}, "
                f"contract_tp={layer_contract.tensor_parallel_size}"
            )
        logger.info(f"Loading MLP data for {cluster_type} from: {linear_ops_file}")
        dense_model_names = tuple(get_family_profiling_names(FFN_FAMILY))
        if not dense_model_names:
            raise ValueError("FFN operator family has no profiling names")
        linear_ops_df = self._load_linear_op_df(
            linear_ops_file,
            ffn_tp_key,
            **_layer_contract_kwargs(
                layer_contract,
                operator_name=dense_model_names[0],
            ),
        )
        logger.info(f"Loaded {len(linear_ops_df)} rows for MLP training")
        dense_training_context = _add_layer_contract_to_training_context(
            training_context,
            layer_contract,
        )
        dense_training_context["input_file"] = linear_ops_file
        dense_training_context["tensor_parallel_size"] = ffn_tp_key

        missing_standard_columns = [
            f"time_stats.{model_name}.median"
            for model_name in dense_model_names
            if f"time_stats.{model_name}.median" not in linear_ops_df.columns
        ]
        if missing_standard_columns:
            model_config = getattr(replica_config, "model_config", None)
            supports_share_expert = bool(
                model_config is not None
                and model_config.supports_share_expert()
            )
            if supports_share_expert:
                logger.info(
                    "Skipping standard dense MLP training for %s: profile provides "
                    "shared-expert operations instead; missing columns=%s",
                    cluster_type,
                    missing_standard_columns,
                )
                return
            raise ValueError(
                "Dense MLP profiling data is incomplete; missing columns: "
                + ", ".join(missing_standard_columns)
            )

        for model_name in dense_model_names:
            model_signature = f"{model_name}_{ffn_signature}"
            if model_signature in trained_model_signatures:
                continue
            train_kwargs: Dict[str, Any] = dict(
                model_name=model_name,
                df=linear_ops_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=dense_training_context,
            )
            train_kwargs.update(_layer_contract_kwargs(layer_contract))
            models[model_name] = self._train_single_model(**train_kwargs)
            trained_model_signatures.add(model_signature)
            logger.info(f"Trained {model_name} for {cluster_type}")

    def _train_attn_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, replica_scheduler_config, linear_ops_file: str, attn_file: str, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """
        Train attention-related models for a cluster.

        This function handles Attention-related operations in the Transformer layer:
        - Pre-attention normalization (from linear_op.csv): input_layernorm
        - Attention projections (from linear_op.csv): attn_pre_proj, attn_post_proj, attn_rope
        - Attention core operations (from attention.csv): attn_kv_cache_save, attn_prefill, attn_decode

        Transformer layer context:
            Input → [input_layernorm] → [attn_pre_proj → attn_rope → attn_prefill/decode → attn_kv_cache_save → attn_post_proj] → add → ...
        """
        models = {}
        tp_size = replica_config.attn_tensor_parallel_size

        model_config = replica_config.model_config
        model_arch = model_config.get_model_arch() if model_config is not None else "generic"
        architecture_profile_id = _resolve_model_architecture_profile_id(model_config)
        attention_signature = (
            f"attention_{replica_config.device}_{replica_config.model_name}_{tp_size}"
            f"_{replica_scheduler_config.block_size}_arch_profile{architecture_profile_id}"
            f"_family{self._measurement_family_name(self._active_measurement_type)}"
        )

        if attention_signature in trained_model_signatures:
            logger.info(f"Skipping attention models training for {cluster_type} - already trained")
            return models

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'block_size': replica_scheduler_config.block_size,
            'model_arch': model_arch,
            'model_architecture_profile': architecture_profile_id,
            'use_qk_norm': bool(getattr(model_config, 'use_qk_norm', False)),
        }

        # ========== Part 1: Linear operations from linear_op.csv ==========
        # These include: input_layernorm, attn_pre_proj, attn_post_proj, attn_rope
        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found")

        logger.info(f"Loading sharded attention linear-op data from: {linear_ops_file}")
        attn_tp_key = self._get_linear_op_tp_key(
            "attn_pre_proj",
            cluster_type,
            replica_config,
            is_moe_model=False,
        )
        required_columns = self._get_required_attn_linear_op_columns(model_config)
        attn_linear_ops_df = self._load_linear_op_df(
            linear_ops_file,
            attn_tp_key,
            required_columns=required_columns,
            training_context=training_context,
        )
        logger.info(
            f"Loaded {len(attn_linear_ops_df)} rows for sharded attention ops training"
        )

        # Pre-attention normalization: input_layernorm
        input_layernorm_tp_key = self._get_linear_op_tp_key(
            "input_layernorm",
            cluster_type,
            replica_config,
            is_moe_model=False,
        )
        input_layernorm_df = self._load_linear_op_df(
            linear_ops_file,
            input_layernorm_tp_key,
            required_columns=["time_stats.input_layernorm.median"],
            training_context=training_context,
        )
        input_layernorm_context = dict(training_context)
        input_layernorm_context["input_file"] = linear_ops_file
        input_layernorm_context["tensor_parallel_size"] = input_layernorm_tp_key

        layernorm_model_name = "input_layernorm"
        layernorm_signature = f"{layernorm_model_name}_{attention_signature}"
        if layernorm_signature not in trained_model_signatures:
            models[layernorm_model_name] = self._train_single_model(
                model_name=layernorm_model_name,
                df=input_layernorm_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{layernorm_model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=input_layernorm_context,
            )
            trained_model_signatures.add(layernorm_signature)
            logger.info(f"Trained {layernorm_model_name} for {cluster_type}")

        # Attention projections: attn_pre_proj, attn_post_proj, attn_rope
        attn_proj_context = dict(training_context)
        attn_proj_context["input_file"] = linear_ops_file
        attn_proj_context["tensor_parallel_size"] = attn_tp_key
        attn_proj_model_names = ["attn_pre_proj", "attn_post_proj", "attn_rope"]
        for model_name in attn_proj_model_names:
            model_signature = f"{model_name}_{attention_signature}"
            if model_signature not in trained_model_signatures:
                models[model_name] = self._train_single_model(
                    model_name=model_name,
                    df=attn_linear_ops_df,
                    feature_cols=["num_tokens"],
                    target_col=f"time_stats.{model_name}.median",
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=attn_proj_context,
                )
                trained_model_signatures.add(model_signature)
                logger.info(f"Trained {model_name} for {cluster_type}")

        if is_target_embedded_mtp_enabled(
            getattr(replica_config, "speculative_decoding_config", None)
        ):
            required_mtp_columns = (
                self._get_required_target_embedded_mtp_linear_op_columns()
            )
            missing_mtp_columns = [
                col for col in required_mtp_columns if col not in attn_linear_ops_df.columns
            ]
            all_nan_mtp_columns = [
                col
                for col in required_mtp_columns
                if col in attn_linear_ops_df.columns
                and attn_linear_ops_df[col].isna().all()
            ]
            if missing_mtp_columns or all_nan_mtp_columns:
                raise ValueError(
                    "target-embedded MTP compute profiling columns are missing or all-NaN in "
                    f"{linear_ops_file}. "
                    f"Missing columns: {missing_mtp_columns}. "
                    f"All-NaN columns: {all_nan_mtp_columns}. "
                    "Re-run linear-op profiling with --include_target_embedded_mtp."
                )
            for model_name in ["mtp_fusion_proj", "lm_head_linear"]:
                model_signature = f"{model_name}_{attention_signature}"
                if model_signature not in trained_model_signatures:
                    models[model_name] = self._train_single_model(
                        model_name=model_name,
                        df=attn_linear_ops_df,
                        feature_cols=["num_tokens"],
                        target_col=f"time_stats.{model_name}.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=attn_proj_context,
                    )
                    trained_model_signatures.add(model_signature)
                    logger.info(
                        "Trained %s for %s (target-embedded MTP)",
                        model_name,
                        cluster_type,
                    )

        model_config = replica_config.model_config
        architecture_profile = _resolve_model_architecture_profile(model_config)
        predictor_attention_extra_ops = (
            architecture_profile.predictor_attention_extra_ops
            if architecture_profile is not None
            else ()
        )
        for model_name in predictor_attention_extra_ops:
                model_signature = f"{model_name}_{attention_signature}"
                if model_signature not in trained_model_signatures:
                    target_col = f"time_stats.{model_name}.median"
                    if target_col not in attn_linear_ops_df.columns:
                        raise ValueError(
                            f"Architecture-profile operation '{model_name}' column '{target_col}' not found in profiling data. "
                            f"Ensure profiling was run with the selected model architecture profile. "
                            f"Available columns: {list(attn_linear_ops_df.columns)}"
                        )
                    models[model_name] = self._train_single_model(
                        model_name=model_name,
                        df=attn_linear_ops_df,
                        feature_cols=["num_tokens"],
                        target_col=target_col,
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=attn_proj_context,
                    )
                    trained_model_signatures.add(model_signature)
                    logger.info("Trained architecture-profile %s for %s", model_name, cluster_type)

        # ========== Part 2: Attention core operations from attention.csv ==========
        if not os.path.exists(attn_file):
            raise FileNotFoundError(f"Attention input file {attn_file} not found")

        logger.info(f"Loading attention data from: {attn_file}")
        attention_df = self._load_attention_df(
            attn_file,
            replica_config,
            replica_scheduler_config,
            cluster_type=cluster_type,
        )
        training_context['input_file'] = attn_file

        # Family-aware attention-core training. Latent-MLA profiles carry six
        # ``attn_mla_*`` operators with a structural layout the dense block cannot
        # consume; route them through the MLA branch before the dense derive (which
        # assumes dense feature columns such as ``prefill_chunk_size``).
        if self._is_mla_family(replica_config.model_config):
            attention_df = self._get_mla_attention_df_with_derived_features(
                attention_df
            )
            logger.info(
                f"Loaded {len(attention_df)} rows for latent-MLA attention core training"
            )
            models.update(
                self._train_mla_attention_core_models(
                    attention_df=attention_df,
                    attention_signature=attention_signature,
                    cluster_type=cluster_type,
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                    trained_model_signatures=trained_model_signatures,
                )
            )
            trained_model_signatures.add(attention_signature)
            return models

        attention_df = self._get_attention_df_with_derived_features(attention_df)
        logger.info(f"Loaded {len(attention_df)} rows for attention core training")
        measurement_type = self._active_measurement_type
        dense_attention_model_names = get_enabled_predictor_metric_names(
            DENSE_ATTENTION_FAMILY
        )
        dense_attention_target_columns = dict(
            zip(
                dense_attention_model_names,
                get_enabled_predictor_median_columns(DENSE_ATTENTION_FAMILY),
            )
        )
        dense_attention_feature_columns = get_enabled_shared_predictor_feature_columns(
            DENSE_ATTENTION_FAMILY
        )

        # Train kv_cache_save model
        kv_cache_model_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
        kv_cache_model_signature = f"{kv_cache_model_name}_{attention_signature}"
        if kv_cache_model_signature not in trained_model_signatures:
            kv_cache_feature_cols = list(
                dense_attention_feature_columns[kv_cache_model_name]
            )
            missing_cols = [
                col for col in kv_cache_feature_cols if col not in attention_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing columns for {kv_cache_model_name} training: {missing_cols}. "
                    "Re-run attention profiling with mixed-batch metadata."
                )
            models[kv_cache_model_name] = self._train_single_model(
                model_name=kv_cache_model_name,
                df=attention_df,
                feature_cols=kv_cache_feature_cols,
                target_col=dense_attention_target_columns[kv_cache_model_name],
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=training_context,
                persist_exact_lookup=True,
            )
            trained_model_signatures.add(kv_cache_model_signature)
            logger.info(f"Trained {kv_cache_model_name} for {cluster_type}")

        # Split data for prefill and decode.
        # Mixed-batch prefill rows in attention_combined.csv use prefill_chunk_size=0,
        # so standard prefill training must keep only rows with positive chunk size.
        true_mixed_df = attention_df[attention_df["is_true_mixed_batch"]].copy()
        standard_df = attention_df[~attention_df["is_true_mixed_batch"]].copy()
        prefill_df = standard_df[~standard_df["is_decode"]].copy()
        decode_df = standard_df[standard_df["is_decode"]].copy()
        standard_prefill_df = pd.DataFrame()
        if measurement_type in (MeasurementType.CUDA_EVENT, MeasurementType.DEVICE_EVENT):
            if "prefill_chunk_size" not in prefill_df.columns:
                raise ValueError(
                    "Missing required column 'prefill_chunk_size' in attention profiling data."
                )
            standard_prefill_df = prefill_df[prefill_df["prefill_chunk_size"] > 0].copy()

            prefill_model_name = get_enabled_predictor_metric_name_by_role(
                DENSE_ATTENTION_FAMILY,
                AttentionOperatorRole.PREFILL_KERNEL,
            )
            prefill_model_signature = f"{prefill_model_name}_{attention_signature}"
            if prefill_model_signature not in trained_model_signatures:
                if len(standard_prefill_df) == 0:
                    raise ValueError(
                        "No standard prefill rows (prefill_chunk_size > 0) found in eager attention profiling data."
                    )
                models[prefill_model_name] = self._train_single_model(
                    model_name=prefill_model_name,
                    df=standard_prefill_df,
                    feature_cols=list(dense_attention_feature_columns[prefill_model_name]),
                    target_col=dense_attention_target_columns[prefill_model_name],
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                )
                trained_model_signatures.add(prefill_model_signature)
                logger.info(f"Trained {prefill_model_name} for {cluster_type}")

            decode_model_name = get_enabled_predictor_metric_name_by_role(
                DENSE_ATTENTION_FAMILY,
                AttentionOperatorRole.DECODE_KERNEL,
            )
            decode_model_signature = f"{decode_model_name}_{attention_signature}"
            if decode_model_signature not in trained_model_signatures:
                if len(decode_df) == 0:
                    logger.info(
                        "Skipping eager %s training for %s - no standard decode rows",
                        decode_model_name,
                        cluster_type,
                    )
                else:
                    decode_feature_cols = list(
                        dense_attention_feature_columns[decode_model_name]
                    )
                    missing_decode_cols = [
                        col
                        for col in [
                            *decode_feature_cols,
                            dense_attention_target_columns[decode_model_name],
                        ]
                        if col not in decode_df.columns
                    ]
                    if missing_decode_cols:
                        logger.info(
                            "Skipping eager %s training for %s - missing decode feature columns %s",
                            decode_model_name,
                            cluster_type,
                            missing_decode_cols,
                        )
                    else:
                        models[decode_model_name] = self._train_single_model(
                            model_name=decode_model_name,
                            df=decode_df,
                            feature_cols=decode_feature_cols,
                            target_col=dense_attention_target_columns[decode_model_name],
                            execution_time_predictor_config=execution_time_predictor_config,
                            training_context=training_context,
                        )
                        trained_model_signatures.add(decode_model_signature)
                        logger.info(f"Trained eager {decode_model_name} for {cluster_type}")
        elif measurement_type == MeasurementType.KERNEL_ONLY:
            decode_model_name = get_enabled_predictor_metric_name_by_role(
                DENSE_ATTENTION_FAMILY,
                AttentionOperatorRole.DECODE_KERNEL,
            )
            decode_model_signature = f"{decode_model_name}_{attention_signature}"
            if decode_model_signature not in trained_model_signatures:
                if len(decode_df) == 0:
                    raise ValueError(
                        "No standard decode rows found in kernel-only attention profiling data."
                    )
                models[decode_model_name] = self._train_single_model(
                    model_name=decode_model_name,
                    df=decode_df,
                    feature_cols=list(dense_attention_feature_columns[decode_model_name]),
                    target_col=dense_attention_target_columns[decode_model_name],
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                )
                trained_model_signatures.add(decode_model_signature)
                logger.info(f"Trained {decode_model_name} for {cluster_type}")
        else:
            raise ValueError(f"Unsupported measurement_type={measurement_type!r}")

        # ========== Part 3: Mixed-batch prefill model (optional, high-dimensional) ==========
        # attn_prefill_mixed uses 12 features and requires on-demand prediction at runtime
        # Check if profiling data contains mixed-batch features
        mixed_batch_model_signature = f"attn_prefill_mixed_{attention_signature}"
        if measurement_type in (MeasurementType.CUDA_EVENT, MeasurementType.DEVICE_EVENT) and mixed_batch_model_signature not in trained_model_signatures:
            # Check for mixed-batch specific columns in the dataframe
            required_mixed_features = self.ATTN_PREFILL_MIXED_FEATURES
            has_mixed_batch_data = all(feat in prefill_df.columns for feat in required_mixed_features)
            
            if has_mixed_batch_data:
                logger.info(f"Training attn_prefill_mixed with {len(required_mixed_features)} features for {cluster_type}")
                
                # Filter for mixed-prefill rows (exclude true mixed prefill+decode rows)
                mixed_batch_df = prefill_df[
                    prefill_df["is_mixed_batch"] | (prefill_df["batch_size"] > 1)
                ].copy()
                
                if len(mixed_batch_df) > 0:
                    models["attn_prefill_mixed"] = self._train_single_model(
                        model_name="attn_prefill_mixed",
                        df=mixed_batch_df,
                        feature_cols=required_mixed_features,
                        target_col="time_stats.attn_prefill.median",  # Same target column as attn_prefill
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=training_context,
                        persist_exact_lookup=True,
                    )
                    trained_model_signatures.add(mixed_batch_model_signature)
                    logger.info(f"Trained attn_prefill_mixed with {len(mixed_batch_df)} samples for {cluster_type}")
                else:
                    logger.warning(f"No mixed-batch data (batch_size > 1) available for attn_prefill_mixed in {cluster_type}")
            else:
                missing_features = [f for f in required_mixed_features if f not in prefill_df.columns]
                logger.info(f"Skipping attn_prefill_mixed for {cluster_type} - missing features: {missing_features}")

        decode_in_mixed_signature = f"attn_decode_in_mixed_{attention_signature}"
        if measurement_type in (MeasurementType.CUDA_EVENT, MeasurementType.DEVICE_EVENT) and decode_in_mixed_signature not in trained_model_signatures:
            required_decode_mixed_features = self.ATTN_DECODE_IN_MIXED_FEATURES
            has_decode_mixed_data = all(
                feat in true_mixed_df.columns for feat in required_decode_mixed_features
            )
            if has_decode_mixed_data:
                if len(true_mixed_df) > 0:
                    models["attn_decode_in_mixed"] = self._train_single_model(
                        model_name="attn_decode_in_mixed",
                        df=true_mixed_df,
                        feature_cols=required_decode_mixed_features,
                        target_col="time_stats.attn_decode.median",
                        execution_time_predictor_config=execution_time_predictor_config,
                        training_context=training_context,
                        persist_exact_lookup=True,
                    )
                    trained_model_signatures.add(decode_in_mixed_signature)
                    logger.info(
                        f"Trained attn_decode_in_mixed with {len(true_mixed_df)} samples for {cluster_type}"
                    )
                else:
                    logger.info(
                        f"Skipping attn_decode_in_mixed for {cluster_type} - no true mixed rows"
                    )
            else:
                missing_features = [
                    f for f in required_decode_mixed_features if f not in true_mixed_df.columns
                ]
                logger.info(
                    f"Skipping attn_decode_in_mixed for {cluster_type} - missing features: {missing_features}"
                )

        trained_model_signatures.add(attention_signature)
        return models

    @staticmethod

    def _is_mla_family(model_config) -> bool:
        """Return True when the model binds to the latent-MLA attention family."""
        if model_config is None:
            return False
        return (
            resolve_runtime_attention_family(model_config).family_id
            == LATENT_MLA_ATTENTION_FAMILY.family_id
        )

    def _get_mla_attention_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Derive latent-MLA attention features (normalize ``is_prefill`` to int).

        Mirrors the monolithic ``SklearnExecutionTimePredictor`` MLA early-return:
        latent-MLA training keys on the imported structural columns directly and must
        NOT add the dense ``num_tokens`` / ``prefill_chunk_size`` derived features.
        """
        df_with_derived_features = df.copy()
        if "is_prefill" in df_with_derived_features.columns:
            df_with_derived_features["is_prefill"] = coerce_truthy_int(
                df_with_derived_features["is_prefill"]
            )
        return df_with_derived_features

    def _filter_mla_attention_df(
        self,
        df: pd.DataFrame,
        file_path: str,
        replica_config,
        replica_scheduler_config,
    ) -> pd.DataFrame:
        """Filter an imported latent-MLA profile to the requested structural layout.

        Verbatim port of the monolithic ``_filter_mla_attention_df`` (adapted to the
        shared-manager's per-cluster ``replica_config`` / ``replica_scheduler_config``
        instead of instance state). Fail-fast on missing structural columns or an empty
        post-filter frame per §7 (no silent fallback).
        """
        validate_attention_profiling_dataframe(
            df,
            LATENT_MLA_ATTENTION_FAMILY,
            measurement_type=self._active_measurement_type,
        )

        model_config = replica_config.model_config
        expected_values = {
            "n_q_head": int(getattr(model_config, "num_q_heads")),
            "n_kv_head": int(
                model_config.get_runtime_num_kv_heads()
                if hasattr(model_config, "get_runtime_num_kv_heads")
                else 1
            ),
            "head_size": int(
                model_config.get_runtime_head_size()
                if hasattr(model_config, "get_runtime_head_size")
                else int(getattr(model_config, "kv_lora_rank"))
                + int(getattr(model_config, "qk_rope_head_dim"))
            ),
            "qk_nope_head_dim": int(getattr(model_config, "qk_nope_head_dim")),
            "qk_rope_head_dim": int(getattr(model_config, "qk_rope_head_dim")),
            "qk_head_dim": int(model_config.get_qk_head_dim()),
            "kv_lora_rank": int(getattr(model_config, "kv_lora_rank")),
            "v_head_dim": int(getattr(model_config, "v_head_dim")),
            "block_size": int(replica_scheduler_config.block_size),
            "num_tensor_parallel_workers": int(
                replica_config.attn_tensor_parallel_size
            ),
        }
        missing_columns = [
            column for column in expected_values if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "MLA attention profiling data is missing structural columns: "
                f"{missing_columns}. file={file_path}"
            )

        filtered = df.copy()
        for column, expected_value in expected_values.items():
            filtered = filtered[filtered[column].astype(int) == expected_value]

        if filtered.empty:
            raise ValueError(
                "No MLA attention profiling rows remain after structural filtering. "
                f"file={file_path}, expected={expected_values}"
            )
        return filtered

    def _train_mla_attention_core_models(
        self,
        attention_df: pd.DataFrame,
        attention_signature: str,
        cluster_type: ClusterType,
        execution_time_predictor_config,
        training_context: Dict[str, Any],
        trained_model_signatures: set,
    ) -> Dict[str, BaseEstimator]:
        """Train the six latent-MLA attention-core operators (training-only A2 fix).

        Mirrors the monolithic ``_train_mla_attention_layer_models`` (sparse-by-target
        row filtering + exact-row memoization). The on-demand consumer pairs each
        estimator's ``_frontier_exact_lookup`` with the same module-level builder, so
        the disaggregation prediction path works unchanged once these models exist.
        """
        model_names = list(
            get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY)
        )
        target_columns = dict(
            zip(
                model_names,
                get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY),
            )
        )
        feature_columns = get_enabled_shared_predictor_feature_columns(
            LATENT_MLA_ATTENTION_FAMILY
        )

        models: Dict[str, BaseEstimator] = {}
        for model_name in model_names:
            model_signature = f"{model_name}_{attention_signature}"
            if model_signature in trained_model_signatures:
                continue

            feature_cols = list(feature_columns[model_name])
            target_col = target_columns[model_name]
            required_columns = [*feature_cols, target_col]
            missing_columns = [
                column
                for column in required_columns
                if column not in attention_df.columns
            ]
            all_nan_columns = [
                column
                for column in required_columns
                if column in attention_df.columns
                and attention_df[column].isna().all()
            ]
            if missing_columns or all_nan_columns:
                raise ValueError(
                    "MLA attention profiling data cannot train "
                    f"{model_name}."
                    f"\nMissing columns: {missing_columns}"
                    f"\nAll-NaN columns: {all_nan_columns}"
                )

            op_attention_df = attention_df.dropna(subset=[target_col]).copy()
            if op_attention_df.empty:
                raise ValueError(
                    "MLA attention profiling data cannot train "
                    f"{model_name}: target column {target_col!r} has no "
                    "observed timing rows."
                )
            nan_feature_columns = [
                column
                for column in feature_cols
                if op_attention_df[column].isna().any()
            ]
            if nan_feature_columns:
                raise ValueError(
                    "MLA attention profiling data cannot train "
                    f"{model_name}: feature columns contain NaN after "
                    f"target filtering: {nan_feature_columns}"
                )

            model = self._train_single_model(
                model_name=model_name,
                df=op_attention_df,
                feature_cols=feature_cols,
                target_col=target_col,
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=training_context,
                persist_exact_lookup=True,
            )
            if not hasattr(model, "_frontier_exact_lookup"):
                model._frontier_exact_lookup = _build_exact_feature_lookup(
                    op_attention_df,
                    feature_cols,
                    target_col,
                )
            models[model_name] = model
            trained_model_signatures.add(model_signature)
            logger.info(f"Trained {model_name} for {cluster_type}")

        return models

    def _train_residual_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config,
                                           linear_ops_file: str, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """
        Train residual connection models for a cluster.

        This function handles residual connection operations in the Transformer layer:
        - Residual add operation (from linear_op.csv): add

        Transformer layer context:
            ... → Attention → [add] → LayerNorm → FFN/MoE → [add] → ...

        The residual add operation is used after both Attention and FFN blocks,
        making it a common operation that serves both sub-layers.
        """
        models = {}

        model_config = replica_config.model_config

        # RMSNorm: add is fused into layernorm, no separate add model needed
        if model_config is not None and model_config.uses_fused_add_norm:
            logger.info(f"Skipping residual add model training for {cluster_type} "
                        f"— model uses fused add+norm (RMSNorm)")
            return models

        is_moe_model = model_config is not None and model_config.is_moe
        tp_size = self._get_linear_op_tp_key(
            "add",
            cluster_type,
            replica_config,
            is_moe_model,
        )

        # Create a signature for this residual model configuration
        residual_signature = f"residual_{replica_config.device}_{replica_config.model_name}_{tp_size}_family{self._measurement_family_name(self._active_measurement_type)}"

        if residual_signature in trained_model_signatures:
            logger.info(f"Skipping residual models training for {cluster_type} - already trained with signature {residual_signature}")
            return models

        if not os.path.exists(linear_ops_file):
            raise FileNotFoundError(f"Linear ops input file {linear_ops_file} not found for residual models")

        logger.info(f"Loading linear ops data for residual models from: {linear_ops_file}")
        linear_ops_df = self._load_linear_op_df(linear_ops_file, tp_size)
        logger.info(f"Loaded {len(linear_ops_df)} rows for residual training")

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'input_file': linear_ops_file,
        }

        # Train the residual add model
        add_model_name = "add"
        add_signature = f"{add_model_name}_{residual_signature}"
        if add_signature not in trained_model_signatures:
            models[add_model_name] = self._train_single_model(
                model_name=add_model_name,
                df=linear_ops_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{add_model_name}.median",
                execution_time_predictor_config=execution_time_predictor_config,
                training_context=training_context,
            )
            trained_model_signatures.add(add_signature)
            logger.info(f"Trained {add_model_name} for {cluster_type}")

        # Mark this residual configuration as trained
        trained_model_signatures.add(residual_signature)
        return models

    def _train_pipeline_parallel_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """Train pipeline parallel communication models for a cluster."""
        models = {}

        _, _, _, send_recv_input_file, _, _ = self._get_input_files_for_config(replica_config, execution_time_predictor_config)
        
        pp_signature = f"send_recv_{replica_config.network_device}_{replica_config.num_pipeline_stages}_{replica_config.attn_tensor_parallel_size}_family{self._measurement_family_name(self._active_measurement_type)}"
        
        if pp_signature in trained_model_signatures:
            logger.info(f"Skipping send_recv model training for {cluster_type} - already trained")
            return models
        
        send_recv_df = self._load_send_recv_df(send_recv_input_file, replica_config)
        send_recv_df = self._get_send_recv_df_with_derived_features(send_recv_df, replica_config)

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'pipeline_stages': replica_config.num_pipeline_stages,
            'tensor_parallel_size': replica_config.attn_tensor_parallel_size,
            'network_device': replica_config.network_device,
            'input_file': send_recv_input_file,
        }

        models["send_recv"] = self._train_single_model(
            model_name="send_recv",
            df=send_recv_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.send_recv.median",
            execution_time_predictor_config=execution_time_predictor_config,
            training_context=training_context,
        )

        trained_model_signatures.add(pp_signature)
        return models

    def _train_tensor_parallel_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, use_attn_tp: bool, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """Train tensor parallel communication models for a cluster."""
        models = {}

        _, _, all_reduce_input_file, _, _, _ = self._get_input_files_for_config(replica_config, execution_time_predictor_config)
        
        # Use different tensor parallel size based on cluster type
        tp_size = replica_config.attn_tensor_parallel_size if use_attn_tp else replica_config.moe_tensor_parallel_size
        
        tp_signature = f"all_reduce_{replica_config.network_device}_{tp_size}_family{self._measurement_family_name(self._active_measurement_type)}"
        
        if tp_signature in trained_model_signatures:
            logger.info(f"Skipping all_reduce model training for {cluster_type} - already trained")
            return models
        
        # 添加详细的上下文信息
        training_context = {
            'cluster_type': cluster_type,
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': tp_size,
            'network_device': replica_config.network_device,
            'input_file': all_reduce_input_file,
            'use_attn_tp': use_attn_tp
        }
        
        logger.info(f"Loading all_reduce data for {cluster_type}: file={all_reduce_input_file}, tp_size={tp_size}")
        
        all_reduce_df = self._load_all_reduce_df(all_reduce_input_file, replica_config, tp_size)
        logger.info(f"Loaded {len(all_reduce_df)} rows for all_reduce training")
        
        all_reduce_df = self._get_all_reduce_df_with_derived_features(all_reduce_df, replica_config)
        logger.info(f"After feature engineering: {len(all_reduce_df)} rows")
        
        models["all_reduce"] = self._train_single_model(
            model_name="all_reduce",
            df=all_reduce_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.all_reduce.median",
            execution_time_predictor_config=execution_time_predictor_config,
            training_context=training_context
        )
        
        trained_model_signatures.add(tp_signature)
        return models

    def _train_cpu_overhead_models_for_cluster(self, cluster_type: ClusterType, replica_config, execution_time_predictor_config, trained_model_signatures: set) -> Dict[str, BaseEstimator]:
        """Train CPU overhead models for a cluster."""
        models = {}

        if execution_time_predictor_config.skip_cpu_overhead_modeling:
            return models

        _, _, _, _, cpu_overhead_input_file, _ = self._get_input_files_for_config(replica_config, execution_time_predictor_config)
        
        cpu_signature = f"cpu_overhead_{replica_config.network_device}_{replica_config.model_name}_{replica_config.attn_tensor_parallel_size}_family{self._measurement_family_name(self._active_measurement_type)}"
        
        if cpu_signature in trained_model_signatures:
            logger.info(f"Skipping CPU overhead models training for {cluster_type} - already trained")
            return models
        
        cpu_overhead_df = self._load_cpu_overhead_df(cpu_overhead_input_file, replica_config)
        if cpu_overhead_df.empty:
            logger.warning(
                "Skipping CPU overhead model training for cluster %s due to missing/empty CPU overhead profiling data. file=%s",
                cluster_type,
                cpu_overhead_input_file,
            )
            trained_model_signatures.add(cpu_signature)
            return models

        # Build training context for error messages
        training_context = {
            'cluster_type': str(cluster_type),
            'device': replica_config.device,
            'model_name': replica_config.model_name,
            'tensor_parallel_size': replica_config.attn_tensor_parallel_size,
            'network_device': replica_config.network_device,
            'input_file': cpu_overhead_input_file,
        }

        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        for model_name in model_names:
            target_col = "ray_comm_time_mean" if model_name == "ray_comm_time" else f"{model_name}_median"

            model_signature = f"{model_name}_{cpu_signature}"
            if model_signature not in trained_model_signatures:
                feature_cols = [
                    "batch_size",
                    "num_prefill_tokens",
                    "num_decode_tokens",
                ]
                model = self._train_single_model(
                    model_name=model_name,
                    df=cpu_overhead_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    execution_time_predictor_config=execution_time_predictor_config,
                    training_context=training_context,
                    persist_exact_lookup=True,
                )
                if not hasattr(model, "_frontier_exact_lookup"):
                    model._frontier_exact_lookup = _build_exact_feature_lookup(
                        cpu_overhead_df,
                        feature_cols,
                        target_col,
                    )
                models[model_name] = model
                trained_model_signatures.add(model_signature)

        trained_model_signatures.add(cpu_signature)
        return models

    def _train_single_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        execution_time_predictor_config,
        training_context: Optional[Dict[str, Any]] = None,
        persist_exact_lookup: bool = True,
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> BaseEstimator:
        """Train a single model with given data and configuration."""
        layer_contract, training_context = _normalize_layer_contract_context(
            training_context,
            explicit_layer_contract=layer_contract,
        )
        if len(df) == 0:
            # 提供详细的错误信息，以便调试
            context_info = ""
            if training_context:
                context_info = f"""
                Training Context:
                - Cluster Type: {training_context.get('cluster_type', 'Unknown')}
                - Device: {training_context.get('device', 'Unknown')}
                - Model Name: {training_context.get('model_name', 'Unknown')}
                - Pipeline Stages: {training_context.get('pipeline_stages', 'Unknown')}
                - Network Device: {training_context.get('network_device', 'Unknown')}
                - Tensor Parallel Size: {training_context.get('tensor_parallel_size', 'Unknown')}
                - Input File: {training_context.get('input_file', 'Unknown')}
                - Block Size: {training_context.get('block_size', 'Unknown')}
                - Feature Columns: {feature_cols}
                - Target Column: {target_col}
                """

            raise Exception(f"Training data for model {model_name} is empty.{context_info}")

        required_cols = feature_cols + [target_col]
        nan_row_mask = df[required_cols].isna().any(axis=1)
        nan_row_count = int(nan_row_mask.sum())
        if nan_row_count > 0:
            logger.warning(
                "Dropping %d/%d rows with NaN feature/target values before training %s "
                "(target=%s).",
                nan_row_count,
                len(df),
                model_name,
                target_col,
            )
            df = df.loc[~nan_row_mask].copy()
        if len(df) == 0:
            raise ValueError(
                f"Training data for model {model_name} is empty after dropping NaN rows "
                f"(target={target_col})."
            )

        profiling_precision = self._get_profiling_precision_from_df(df)
        measurement_type = self._validate_active_measurement_type(df)
        hash_args = (
            model_name,
            df,
            execution_time_predictor_config,
            profiling_precision,
            measurement_type,
        )
        model_hash = self._get_model_hash(
            *hash_args,
            **_layer_contract_kwargs(layer_contract),
        )
        cached_model = self._load_model_from_cache(model_name, model_hash)
        if cached_model is not None:
            if layer_contract is not None:
                requested_identity = _serialize_selected_layer_cache_identity(
                    layer_contract
                )
                if requested_identity is None:
                    raise ValueError(
                        "resolved layer contract did not produce a cache identity"
                    )
                self._validate_cached_layer_cache_identity(
                    model_name=model_name,
                    model=cached_model,
                    requested_identity=requested_identity,
                )
            if persist_exact_lookup:
                self._ensure_exact_lookup_metadata(
                    model_name=model_name,
                    model_hash=model_hash,
                    model=cached_model,
                    df=df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                )
            self._store_model_precision(
                model_name,
                profiling_precision,
                cached_model,
                **_layer_contract_kwargs(layer_contract),
            )
            return cached_model

        # ============================================================
        # CACHE MISS: Model not found in cache
        # ============================================================
        # When running in production mode (non-dummy mode), we expect all models
        # to be pre-trained using the standalone training module and cached.
        # If a model is not found in cache, it indicates a configuration mismatch
        # or missing profiling/training step.
        #
        # To train models, use the standalone training workflow:
        # 1. Run profiling: tests/test_pd_af_profiling.sh
        # 2. Run training: tests/test_pd_af_training.sh
        # 3. Run simulation: tests/test_small_scale_pd_af_disaggregation_cluster_parallel.sh
        # ============================================================

        error_msg = f"""
        ❌ MODEL CACHE MISS ERROR ❌

        Model '{model_name}' with hash '{model_hash}' not found in cache directory: {self._cache_dir}

        Configuration Details:
        - Model Name: {model_name}
        - Cache Hash: {model_hash}
        - Cache Directory: {self._cache_dir}
        - Expected Cache File: {self._cache_dir}/{model_name}_{model_hash}.pkl
        """

        if training_context:
            error_msg += f"""
        Training Context:
        - Cluster Type: {training_context.get('cluster_type', 'Unknown')}
        - Device: {training_context.get('device', 'Unknown')}
        - Model Name: {training_context.get('model_name', 'Unknown')}
        - Tensor Parallel Size: {training_context.get('tensor_parallel_size', 'Unknown')}
        - Expert Parallel Size: {training_context.get('moe_expert_parallel_size', 'N/A')}
        - Input File: {training_context.get('input_file', 'Unknown')}
        - Feature Columns: {feature_cols}
        - Target Column: {target_col}
        """

        error_msg += f"""

        ⚠️  REQUIRED ACTION ⚠️

        This error indicates that the required model has not been pre-trained.
        Please follow the complete workflow:

        ============================================================

        NOTE: Real-time training is TEMPORARILY ENABLED for cache generation.
        """

        logger.warning(error_msg)
        logger.info(f"CACHE MISS: Training model '{model_name}' with hash '{model_hash}' in real-time...")

        # ============================================================
        # TEMPORARILY ENABLED: Real-time training code
        # ============================================================
        # This code performs real-time model training during simulation
        # initialization to generate missing cache files.
        # ============================================================

        estimator, grid_search_params = self._create_estimator_and_params(execution_time_predictor_config)

        cv = min(execution_time_predictor_config.k_fold_cv_splits, len(df)) if len(df) >= 2 else 2

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=execution_time_predictor_config.num_training_job_threads,
        )

        X, y = df[feature_cols], df[target_col]
        grid_search.fit(X, y)
        score = grid_search.score(X, y)

        logger.info(f"✓ Trained model {model_name} with MAPE {-score}%")

        best_estimator = grid_search.best_estimator_
        # Persist feature metadata for runtime on-demand prediction (e.g., moe_grouped_gemm load imbalance mode).
        setattr(best_estimator, "_frontier_feature_names", list(feature_cols))
        setattr(best_estimator, "_frontier_target_col", target_col)
        # Tie the trained estimator to its cache hash so prediction caches can include model identity.
        setattr(best_estimator, "_frontier_model_hash", model_hash)
        if layer_contract is not None:
            self._model_contract_identity(best_estimator, layer_contract)

        if persist_exact_lookup:
            setattr(
                best_estimator,
                "_frontier_exact_lookup",
                _build_exact_feature_lookup(df, feature_cols, target_col),
            )

        self._store_model_in_cache(model_name, model_hash, best_estimator)
        self._store_model_precision(
            model_name,
            profiling_precision,
            best_estimator,
            **_layer_contract_kwargs(layer_contract),
        )
        return best_estimator
