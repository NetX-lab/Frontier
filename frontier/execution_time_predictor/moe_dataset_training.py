"""Admission and training of the MoE profiling dataset.

The dataset contract decides which profiling rows a given replica may train
on, at its tensor- and expert-parallel sizes and its routing runtime.  The
trainer then fits one model per MoE operator from the admitted rows.
"""

import json
import hashlib
from dataclasses import replace
import numpy as np
import os
import pandas as pd

from frontier.execution_time_predictor.prediction_model_identity import (
    _get_moe_family_model_names,
    _get_prefill_hot_moe_gating_model_names,
    _is_moe_gating_family_model_name,
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
from frontier.moe_load_imbalance import (
    MOE_GROUPED_GEMM_PADDED_FEATURES,
    MOE_LOAD_IMBALANCE_FEATURES,
    block_size_m_for_tokens,
)
from frontier.moe_routing_runtime import filter_moe_gating_routing_topk_rows
from frontier.operators.families import MOE_FAMILY, get_family_profiling_names
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    validate_typed_operator_contracts,
)
from sklearn.base import BaseEstimator
from typing import Any, Dict, List, Optional


logger = init_logger(__name__)

NUM_TOKENS_POST_PADDED_COLUMN = "time_stats.moe_grouped_gemm.num_tokens_post_padded"
BLOCK_SIZE_M_COLUMN = "time_stats.moe_grouped_gemm.block_size_m"
BLOCK_SIZE_M_RANGES_COLUMN = "time_stats.moe_grouped_gemm.block_size_m_ranges"


def position_moe_operator_facts(frame, replica_config):
    """Bind producer contexts and source groups before any legacy family helpers."""
    from frontier.operators.families import get_moe_profiling_operators
    from frontier.operators.binding import bind_operator_query
    from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
    from frontier.moe_load_imbalance import moe_native_binding_context
    from frontier.config import PrecisionType
    from frontier.moe_gating_runtime import get_moe_gating_prediction_model_name
    from frontier.execution_time_predictor.step5_cost import STEP5_NATIVE_EXPERT_SCHEMAS
    model = replica_config.model_config
    groups = {group.layer_range: group for group in resolve_routed_layer_groups(model)}
    if frame[["moe_layer_ids", "native_binding", "native_context"]].isna().any().any() or set(frame.moe_layer_ids) != set(groups):
        raise ValueError("Position MoE rows require every actual routed group/native binding")
    bound = []
    for (group_name, native_binding), rows in frame.groupby(["moe_layer_ids", "native_binding"], sort=False):
        group = groups[group_name]
        expected = dict(num_experts=model.num_experts,router_topk=model.num_experts_per_tok,
            hidden_dim=model.embedding_dim,expert_hidden_dim=model.get_model_architecture_profile().resolve_layer_contract(
                model,layer_id=group.layer_ids[0],operator_name="moe_grouped_gemm",moe_tp_size=replica_config.moe_tensor_parallel_size,
                expert_parallel_size=replica_config.moe_expert_parallel_size).effective_ffn_width,
            **group.row_metadata())
        if any(name not in rows or rows[name].ne(value).any() for name,value in expected.items()):
            raise ValueError("Position MoE rows differ from actual model/routed-group dimensions")
        context = json.loads(native_binding)
        if any(moe_native_binding_context(json.loads(value)) != context for value in rows["native_context"]):
            raise ValueError("MoE row context differs from its native binding")
        facts = {}
        for spec in get_moe_profiling_operators(context):
            contract = model.get_model_architecture_profile().resolve_layer_contract(model,
                layer_id=group.layer_ids[0], operator_name=spec.name,
                moe_tp_size=replica_config.moe_tensor_parallel_size,
                expert_parallel_size=replica_config.moe_expert_parallel_size)
            contexts = rows.loc[rows.moe_frame_scope.eq("source_local"), "gating_runtime_context"].unique() if spec.precision_name() == "moe_gating" else [None]
            for gating_context in contexts:
                name = get_moe_gating_prediction_model_name(spec.name, requested_context=gating_context) if gating_context else spec.name
                precision = PrecisionType.FP32 if spec.precision_name() == "moe_gating" else PrecisionType.FP8 if spec.name == "moe_input_quantize" else group.precision
                features = ("num_tokens", "live_routing_token_count") if spec.ep_agnostic else STEP5_NATIVE_EXPERT_SCHEMAS[context["experts_class"]]
                facts[name] = dict(model_name=name,spec=spec,binding=bind_operator_query(spec.name),
                    layer_contract=contract,precision=precision,native_context=context,
                    frame_scope="source_local" if spec.ep_agnostic else "dp_gathered",
                    row_metadata=dict(moe_layer_ids=group_name,native_binding=native_binding),
                    gating_runtime_context=gating_context,feature_columns=features)
        bound.append(facts)
    if not bound:
        raise ValueError("Position MoE data contains no selected native bindings")
    return bound


def select_moe_operator_features(
    model_name: str, op_df: pd.DataFrame, *, operator_facts=None,
) -> tuple[pd.DataFrame, List[str]]:
    """Return the training rows and feature columns of one MoE operator model.

    The grouped GEMM trains on the routed and padded token counts when the table
    records moe_align_block_size's padded count (decision T33-C2-F), else on the
    full load-imbalance feature set, else on ``num_tokens``; a partial
    load-imbalance set is an inconsistent table. Shuffling uses the full
    load-imbalance set when present. Every other operator uses ``num_tokens``.
    """
    if operator_facts is not None:
        if model_name != operator_facts["model_name"]:
            raise ValueError("MoE feature selection conflicts with bound model context")
        features = list(operator_facts["feature_columns"])
        if not operator_facts["spec"].ep_agnostic:
            from frontier.moe_load_imbalance import moe_grouped_gemm_schema
            moe_grouped_gemm_schema(features, native_context=operator_facts["native_context"])
            if "native_binding" in op_df:
                from frontier.execution_time_predictor.step5_cost import step5_native_expert_features
                for _, row in op_df.iterrows():
                    expected = step5_native_expert_features(json.loads(row.native_context),
                        physical_rows=int(row.num_tokens), live_assignments=int(row.total_routed_tokens))
                    if any(row[name] != value for name, value in expected.items()):
                        raise ValueError("MoE features differ from actual native padding/capacity receipt")
        if not features or set(features) - set(op_df.columns) or op_df[features].isna().any().any():
            raise ValueError("MoE selected native feature schema lacks complete physical coverage")
        return op_df, features
    load_features = [name for name in MOE_LOAD_IMBALANCE_FEATURES if name in op_df.columns]
    has_load_features = len(load_features) == len(MOE_LOAD_IMBALANCE_FEATURES)
    if model_name == "moe_grouped_gemm":
        if NUM_TOKENS_POST_PADDED_COLUMN in op_df.columns:
            unpadded_rows = int(op_df[NUM_TOKENS_POST_PADDED_COLUMN].isna().sum())
            if unpadded_rows:
                raise ValueError(
                    f"{unpadded_rows} of {len(op_df)} MoE rows lack {NUM_TOKENS_POST_PADDED_COLUMN}; "
                    "the grouped GEMM needs it on every row or on none"
                )
            padded_df = op_df.assign(num_tokens_post_padded=op_df[NUM_TOKENS_POST_PADDED_COLUMN])
            from frontier.moe_load_imbalance import moe_grouped_gemm_schema
            return padded_df, list(moe_grouped_gemm_schema(MOE_GROUPED_GEMM_PADDED_FEATURES).features)
        if load_features and not has_load_features:
            missing = [name for name in MOE_LOAD_IMBALANCE_FEATURES if name not in op_df.columns]
            raise ValueError(
                f"Partial load imbalance features found ({len(load_features)}/"
                f"{len(MOE_LOAD_IMBALANCE_FEATURES)}) for {model_name}. Missing: {missing}."
            )
    if model_name in ("moe_grouped_gemm", "moe_shuffling") and has_load_features:
        return op_df, load_features
    return op_df, ["num_tokens"]


def block_size_m_ranges(op_df: pd.DataFrame) -> tuple[tuple[int, int, int], ...]:
    """vLLM's BLOCK_SIZE_M ranges up to the largest ``num_tokens`` of a padded grouped-GEMM table.

    Every row records the ``(first_num_tokens, last_num_tokens, block_size_m)``
    ranges of the kernel config up to its own ``num_tokens``. The largest row's
    ranges cover the table, and every row's ``block_size_m`` must agree with them.
    """
    if BLOCK_SIZE_M_RANGES_COLUMN not in op_df.columns:
        raise ValueError(
            f"the padded grouped-GEMM table lacks {BLOCK_SIZE_M_RANGES_COLUMN}; re-profile the MoE table "
            "so the simulator can select vLLM's BLOCK_SIZE_M between profiled num_tokens"
        )
    rows_without_ranges = int(op_df[BLOCK_SIZE_M_RANGES_COLUMN].isna().sum())
    if rows_without_ranges:
        raise ValueError(
            f"{rows_without_ranges} of {len(op_df)} MoE rows lack {BLOCK_SIZE_M_RANGES_COLUMN}; "
            "the padded grouped GEMM needs it on every row"
        )
    largest_row = op_df.loc[op_df["num_tokens"].idxmax()]
    ranges = tuple(
        (int(first), int(last), int(block_size))
        for first, last, block_size in json.loads(largest_row[BLOCK_SIZE_M_RANGES_COLUMN])
    )
    disagreeing = sorted(
        {
            (int(num_tokens), int(block_size))
            for num_tokens, block_size in op_df[["num_tokens", BLOCK_SIZE_M_COLUMN]].itertuples(index=False)
            if block_size_m_for_tokens(ranges, int(num_tokens)) != block_size
        }
    )
    if disagreeing:
        raise ValueError(
            f"MoE rows (num_tokens, {BLOCK_SIZE_M_COLUMN}) {disagreeing} disagree with the "
            f"{BLOCK_SIZE_M_RANGES_COLUMN} of the {int(largest_row['num_tokens'])}-token row {list(ranges)}"
        )
    return ranges


def _validate_moe_columns(moe_df: pd.DataFrame) -> None:
    """
    Validate that MoE DataFrame contains required split gating columns.

    This function enforces fail-fast behavior by rejecting legacy moe_gating
    column format and requiring the split columns (moe_gating_linear and
    moe_gating_routing_topk).

    Args:
        moe_df: DataFrame containing MoE profiling data

    Raises:
        ValueError: If required split columns are missing or if legacy
                   moe_gating column is present without split columns
    """
    required_columns = [
        f"time_stats.{operator_name}.median"
        for operator_name in get_family_profiling_names(MOE_FAMILY)
    ]

    missing_columns = [col for col in required_columns if col not in moe_df.columns]

    if missing_columns:
        # Check if legacy moe_gating column exists (for better error message)
        legacy_col = "time_stats.moe_gating.median"
        if legacy_col in moe_df.columns:
            raise ValueError(
                f"Missing required MoE columns: {missing_columns}. "
                f"Found legacy '{legacy_col}' column which is no longer supported. "
                f"Re-run MoE profiling with split gating scopes enabled to generate "
                f"'moe_gating_linear' and 'moe_gating_routing_topk' columns."
            )
        else:
            raise ValueError(
                f"Missing required MoE columns: {missing_columns}. "
                f"Re-run MoE profiling with split gating scopes enabled."
            )


class MoeDatasetTraining:
    """MoE profiling dataset admission and per-operator training."""

    def _get_model_hash(self, model_name, df=None):
        identity = getattr(self, "_moe_training_identity", None)
        model_hash = super()._get_model_hash(model_name, df)
        if identity is None:
            return model_hash
        return hashlib.md5(f"{model_hash}_{json.dumps(identity, sort_keys=True)}".encode()).hexdigest()[:8]

    def _validate_moe_dataset_contract(
        self,
        moe_df: pd.DataFrame,
        moe_input_file: str,
        model_names: List[str],
        moe_tp_size: int,
        moe_ep_size: int,
        *, operator_facts=None,
    ) -> pd.DataFrame:
        """Validate op-level MoE key coverage and return model-filtered dataframe."""
        if operator_facts is not None:
            from frontier.moe_routing_runtime import filter_moe_operator_rows
            if not operator_facts:
                raise ValueError("MoE preflight requires nonempty selected operator_facts")
            for facts in operator_facts.values():
                filter_moe_operator_rows(moe_df, operator_facts=facts,
                    requested_runtime_path=facts["native_context"]["routing_runtime_path"], source_name=moe_input_file)
            return moe_df
        if self._get_requested_moe_gating_routing_runtime_path() == "position_cyclic":
            raise ValueError("Position MoE preflight requires selected operator_facts before legacy helpers")
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

    def _train_moe_models(self, *, operator_facts=None) -> Dict[str, BaseEstimator]:
        """Train MoE-specific models (gating, shuffling, grouped_gemm) for independent training mode.

        Features per operator follow select_moe_operator_features.
        """
        models = {}
        moe_input_file = getattr(self, "_moe_input_file", "/synthetic/moe.csv")

        if not os.path.exists(moe_input_file):
            logger.warning(f"MoE input file does not exist: {moe_input_file}")
            return models

        moe_df = pd.read_csv(moe_input_file)
        if operator_facts is None and self._get_requested_moe_gating_routing_runtime_path() == "position_cyclic":
            for facts in position_moe_operator_facts(moe_df, self._replica_config):
                models.update(self._train_moe_models(operator_facts=facts))
            return models
        if TYPED_OPERATOR_CONTRACTS_COLUMN in moe_df.columns:
            # Validate every row before scalar or TP/EP filtering can hide a
            # malformed typed contract.
            moe_df[TYPED_OPERATOR_CONTRACTS_COLUMN].map(
                lambda raw_contracts: validate_typed_operator_contracts(
                    raw_contracts,
                    model_config=self._model_config,
                )
            )

        if operator_facts is not None:
            from frontier.moe_routing_runtime import filter_moe_operator_rows
            self._validate_moe_dataset_contract(moe_df, moe_input_file, list(operator_facts),
                self._replica_config.moe_tensor_parallel_size, self._replica_config.moe_expert_parallel_size,
                operator_facts=operator_facts)
            for name, facts in operator_facts.items():
                rows = filter_moe_operator_rows(moe_df, operator_facts=facts,
                    requested_runtime_path=facts["native_context"]["routing_runtime_path"], source_name=moe_input_file)
                rows, features = select_moe_operator_features(name, rows, operator_facts=facts)
                target = facts["spec"].profiling_name()
                if rows.groupby(features)[f"time_stats.{target}.kernel_count"].nunique().gt(1).any():
                    raise ValueError("MoE selected physical key has conflicting measured kernel counts")
                from frontier.execution_time_predictor.step5 import step5_operator_training_rows
                _, bound_context = step5_operator_training_rows(name, rows,
                    dict(operator_facts=facts, input_file=moe_input_file), self._active_measurement_type)[0]
                identity = {**bound_context["measurement_identity"],
                            "typed_contract": json.dumps(facts["layer_contract"].typed_metadata_identity(), sort_keys=True)}
                metadata = self._get_profiling_metadata(rows, moe_input_file)
                self._validate_active_measurement_type(metadata, moe_input_file)
                self._register_profiling_metadata_for_ops([name], replace(metadata, profiling_precision=facts["precision"]), moe_input_file)
                self._moe_training_identity = identity
                try:
                    duration = self._train_model(name, rows, features, f"time_stats.{target}.median")
                    count = getattr(duration, "_frontier_kernel_count_model", None)
                    if count is None:
                        count = self._train_model(f"{name}_kernel_count", rows, features, f"time_stats.{target}.kernel_count")
                finally:
                    self._moe_training_identity = None
                if set(duration._frontier_exact_lookup) != set(count._frontier_exact_lookup):
                    raise ValueError("MoE physical duration/count exact feature keys differ")
                for fitted in (duration, count):
                    cached_identity = getattr(fitted, "_frontier_measurement_identity", None)
                    if cached_identity is not None and json.loads(cached_identity) != identity:
                        raise ValueError("MoE cached selected measurement identity conflicts with the actual binding")
                    fitted._frontier_measurement_identity = json.dumps(identity, sort_keys=True)
                    fitted._frontier_feature_bounds = {col: (float(rows[col].min()), float(rows[col].max())) for col in features}
                    if "native_binding" in rows:
                        from frontier.moe_load_imbalance import moe_physical_row_receipts
                        fitted._frontier_moe_physical_rows = moe_physical_row_receipts(rows)
                count._frontier_operator_name = target
                duration._frontier_kernel_count_model = count
                for fitted_name, fitted in ((name, duration), (f"{name}_kernel_count", count)):
                    fitted._frontier_target_col = f"time_stats.{target}.{'median' if fitted_name == name else 'kernel_count'}"
                    self._store_model_in_cache(fitted_name, fitted._frontier_model_hash, fitted)
                models[name] = duration
            return models
        if self._get_requested_moe_gating_routing_runtime_path() == "position_cyclic":
            raise ValueError("Position MoE training requires committed native row bindings before legacy enumeration")

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

            op_df, feature_cols = select_moe_operator_features(model_name, op_df)
            logger.info(f"  {model_name}: features {feature_cols} (TP={moe_tp_key})")

            models[model_name] = self._train_model(
                model_name=model_name,
                df=op_df,
                feature_cols=feature_cols,
                target_col=target_col,
            )
            from frontier.moe_load_imbalance import moe_grouped_gemm_schema
            schema = moe_grouped_gemm_schema(feature_cols)
            if schema is not None and schema.uses_block_size_m_ranges:
                models[model_name]._frontier_block_size_m_ranges = block_size_m_ranges(op_df)
            logger.info(f"Trained MoE model: {model_name}")

        return models

    def _register_additional_profiling_metadata_from_files(self, *, operator_facts=None) -> None:
        moe_input_file = self._moe_input_file
        if operator_facts is not None:
            from frontier.moe_routing_runtime import filter_moe_operator_rows
            for name, facts in operator_facts.items():
                rows = filter_moe_operator_rows(pd.read_csv(moe_input_file), operator_facts=facts,
                    requested_runtime_path=facts["native_context"]["routing_runtime_path"], source_name=moe_input_file)
                metadata = self._get_profiling_metadata(rows, moe_input_file)
                self._register_profiling_metadata_for_ops([name], replace(metadata, profiling_precision=facts["precision"]), moe_input_file)
            return
        if self._get_requested_moe_gating_routing_runtime_path() == "position_cyclic":
            raise ValueError("Position MoE metadata requires selected operator_facts")
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
