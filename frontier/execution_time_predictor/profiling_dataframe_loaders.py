"""Loading and validation of the profiling CSVs the predictors train on.

Each loader reads one file of the canonical profiling taxonomy, validates the
columns its consumers require, applies the typed operator contract filter
when the file carries one, and derives the feature columns training needs.
"""

import os
import pandas as pd

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_median_column_by_role,
    get_enabled_predictor_metric_name_by_role,
)
from frontier.attention.string_coercion import coerce_truthy_bool
from frontier.execution_time_predictor.attention_dataset_contract import (
    enforce_mixed_attention_input_contract,
)
from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.execution_time_predictor.prediction_model_identity import (
    _resolve_model_architecture_profile,
    _typed_row_matches_contract,
    _validate_typed_parallel_selection,
)
from frontier.execution_time_predictor.profiling_metadata import (
    infer_single_runtime_model_config,
    infer_single_runtime_profile,
    validate_model_architecture_profile,
)
from frontier.logger import init_logger
from frontier.model_architectures import ResolvedLayerContract
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    validate_typed_operator_contracts,
)
from frontier.profiling.cpu_overhead.validation import (
    apply_cpu_overhead_schema_v2_defaults,
    validate_cpu_overhead_dataframe,
)
from frontier.types import ClusterType
from typing import Any, Dict, List, Optional, cast


logger = init_logger(__name__)


class ProfilingDataFrameLoaders:
    """Profiling CSV loading, validation and derived features."""

    def _load_linear_op_df(
        self,
        file_path: str,
        tensor_parallel_size: int,
        required_columns: Optional[List[str]] = None,
        training_context: Optional[Dict[str, Any]] = None,
        layer_contract: Optional[ResolvedLayerContract] = None,
        operator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load linear operation dataframe (linear_op.csv or mlp.csv) with tensor parallel filtering.

        This function loads profiling data for linear operations including:
        - Attention projections: attn_pre_proj, attn_post_proj, attn_rope
        - MLP operations: mlp_up_proj, mlp_down_proj, mlp_act
        - LayerNorm operations: input_layernorm, post_attention_layernorm
        - Residual operations: add

        Note: This function is for linear_op.csv data only. For MoE data, use _load_moe_df().

        Args:
            file_path: Path to the profiling CSV file (linear_op.csv or mlp.csv)
            tensor_parallel_size: Required tensor parallel size for filtering

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If required columns are missing or no data matches filtering criteria
        """
        if layer_contract is not None:
            _validate_typed_parallel_selection(
                layer_contract,
                tensor_parallel_size=tensor_parallel_size,
            )

        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Linear ops input file does not exist: {file_path}\n"
                f"Please run profiling first to generate this file.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_linear_op.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original linear ops data: {len(df)} rows, {len(df.columns)} columns")
        expected_profile = (
            layer_contract.profile_id
            if layer_contract is not None
            else infer_single_runtime_profile(self)
        )
        if expected_profile is not None and "model_architecture_profile" in df.columns:
            validate_model_architecture_profile(
                df,
                file_path=file_path,
                expected_profile=expected_profile,
            )

        # Check required column
        if 'num_tensor_parallel_workers' not in df.columns:
            raise ValueError(
                f"Column 'num_tensor_parallel_workers' not found in {file_path}\n"
                f"Available columns: {list(df.columns)}\n"
                f"This may indicate a corrupted or incompatible profiling file."
            )

        has_typed_contracts = TYPED_OPERATOR_CONTRACTS_COLUMN in df.columns
        parsed_typed_contracts: Optional[pd.Series] = None
        if has_typed_contracts:
            if not operator_name and layer_contract is not None:
                raise ValueError(
                    "typed profiling loading requires operator_name when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            if operator_name is not None and layer_contract is None:
                raise ValueError(
                    "typed profiling loading requires layer_contract when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            # Parse every row before applying scalar filters so malformed metadata
            # cannot be hidden by an unrelated TP or width selector.
            parsed_typed_contracts = cast(
                pd.Series,
                df[TYPED_OPERATOR_CONTRACTS_COLUMN].map(
                    lambda raw_contracts: validate_typed_operator_contracts(
                        raw_contracts,
                        model_config=infer_single_runtime_model_config(self),
                    )
                ),
            )

        # Show filtering conditions
        available_tp = sorted(df['num_tensor_parallel_workers'].unique())
        logger.info(f"Filtering conditions:")
        logger.info(f"  - num_tensor_parallel_workers == {tensor_parallel_size}")
        logger.info(f"  - Available num_tensor_parallel_workers: {available_tp}")

        # Apply filtering
        filtered_df: pd.DataFrame = cast(
            pd.DataFrame,
            df[df["num_tensor_parallel_workers"] == tensor_parallel_size],
        )
        if parsed_typed_contracts is not None and layer_contract is not None:
            selected_layer_contract = layer_contract
            if not isinstance(operator_name, str) or not operator_name:
                raise ValueError(
                    "typed profiling loading requires a non-empty operator_name "
                    "for contract matching"
                )
            typed_mask = parsed_typed_contracts.loc[filtered_df.index].map(
                lambda raw_contracts: _typed_row_matches_contract(
                    raw_contracts,
                    selected_layer_contract,
                    operator_name=operator_name,
                )
            )
            filtered_df = cast(pd.DataFrame, filtered_df[typed_mask])
            if filtered_df.empty:
                raise ValueError(
                    "No linear-op rows match the selected typed layer contract "
                    f"for operator={operator_name!r}, TP={tensor_parallel_size} "
                    f"in {file_path}"
                )
        elif layer_contract is not None:
            if "n_expanded_embd" not in filtered_df.columns:
                raise ValueError(
                    "Legacy linear-op profiling data is missing 'n_expanded_embd' "
                    f"for typed contract loading in {file_path}"
                )
            filtered_df = cast(
                pd.DataFrame,
                filtered_df[
                    filtered_df["n_expanded_embd"]
                    == layer_contract.effective_ffn_width
                ],
            )
        logger.info(f"After filtering: {len(filtered_df)} rows")

        expected_use_qk_norm = None
        if training_context is not None and "use_qk_norm" in training_context:
            expected_use_qk_norm = bool(training_context["use_qk_norm"])

        if expected_use_qk_norm is True and "use_qk_norm" not in filtered_df.columns:
            raise ValueError(
                "linear_op profiling data is missing 'use_qk_norm' column for a model "
                "that requires QK-norm-aware filtering. "
                f"file={file_path}, model={training_context.get('model_name') if training_context else 'unknown'}"
            )

        if expected_use_qk_norm is not None and "use_qk_norm" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["use_qk_norm"].astype(bool) == expected_use_qk_norm
            ]
            logger.info(
                "After use_qk_norm filtering: %s rows (expected_use_qk_norm=%s)",
                len(filtered_df),
                expected_use_qk_norm,
            )

        if len(filtered_df) == 0:
            width_requirement = (
                layer_contract.effective_ffn_width
                if layer_contract is not None
                else "legacy model width"
            )
            raise ValueError(
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required tensor_parallel_size: {tensor_parallel_size}\n"
                f"Available tensor_parallel_sizes: {available_tp}\n"
                f"Required effective_ffn_width: {width_requirement}\n"
                f"Please run profiling with the correct configuration."
            )

        if required_columns:
            self._validate_required_linear_op_columns(
                filtered_df,
                required_columns,
                file_path,
                training_context=training_context,
            )

        return filtered_df

    def _get_required_attn_linear_op_columns(self, model_config) -> List[str]:
        required_columns = [
            "time_stats.attn_pre_proj.median",
            "time_stats.attn_post_proj.median",
            "time_stats.attn_rope.median",
        ]
        if model_config is not None and bool(getattr(model_config, "use_qk_norm", False)):
            required_columns.append("use_qk_norm")
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is not None:
            required_columns.extend(
                f"time_stats.{op_name}.median"
                for op_name in architecture_profile.predictor_attention_extra_ops
            )
        return required_columns

    @staticmethod

    def _get_required_target_embedded_mtp_linear_op_columns() -> List[str]:
        return [
            "time_stats.mtp_fusion_proj.median",
            "time_stats.lm_head_linear.median",
        ]

    @staticmethod

    def _validate_required_linear_op_columns(
        df: pd.DataFrame,
        required_columns: List[str],
        file_path: str,
        training_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        all_nan_columns = [
            col
            for col in required_columns
            if col in df.columns and df[col].isna().all()
        ]

        if missing_columns or all_nan_columns:
            context_text = ""
            if training_context:
                context_text = f"\nTraining context: {training_context}"

            raise ValueError(
                "Required attention linear op columns are missing or all-NaN in "
                f"{file_path}."
                f"\nMissing columns: {missing_columns}"
                f"\nAll-NaN columns: {all_nan_columns}"
                f"{context_text}"
            )

    def _load_attention_df(
        self,
        file_path: str,
        replica_config,
        replica_scheduler_config,
        cluster_type: Optional[ClusterType] = None,
    ) -> pd.DataFrame:
        """
        Load attention dataframe (attention.csv) with model configuration filtering.

        Args:
            file_path: Path to the attention profiling CSV file
            replica_config: Replica configuration for filtering
            replica_scheduler_config: Replica scheduler configuration for block size
            cluster_type: Cluster type for policy warning context

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If no data matches filtering criteria
        """
        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Attention input file does not exist: {file_path}\n"
                f"Please run attention profiling first to generate this file.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_attention.sh"
            )

        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        logger.info(f"Original attention data: {len(df)} rows, {len(df.columns)} columns")

        enforce_mixed_attention_input_contract(
            attention_file_path=file_path,
            available_columns=df.columns,
        )

        # Latent-MLA profiles use a distinct structural schema (runtime kv heads = 1,
        # head size = kv_lora_rank + qk_rope_head_dim); route them to the MLA
        # structural filter before the dense cache-write fill / dense filter.
        model_config = replica_config.model_config
        if self._is_mla_family(model_config):
            return self._filter_mla_attention_df(
                df, file_path, replica_config, replica_scheduler_config
            )

        # Fill missing cache-write column for older attention profiling CSVs.
        cache_write_median_column = get_enabled_predictor_median_column_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
        for column in [cache_write_median_column]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)

        model_config = replica_config.model_config
        requested_tp = replica_config.attn_tensor_parallel_size
        prefill_op_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.PREFILL_KERNEL,
        )
        effective_tp = resolve_effective_attention_tp_size(
            op_name=prefill_op_name,
            requested_tp_size=requested_tp,
            num_kv_heads=model_config.num_kv_heads,
            cluster_type=cluster_type,
            warning_cache=getattr(self, "_attention_tp_warning_cache", None),
            include_linear_ops=False,
        )

        # Show filtering conditions
        logger.info(f"Filtering conditions:")
        logger.info(f"  - n_embd == {model_config.embedding_dim}")
        logger.info(f"  - n_q_head == {model_config.num_q_heads}")
        logger.info(f"  - n_kv_head == {model_config.num_kv_heads}")
        logger.info(f"  - block_size == {replica_scheduler_config.block_size}")
        logger.info(
            "  - num_tensor_parallel_workers == %s (requested_tp=%s)",
            effective_tp,
            requested_tp,
        )

        filtered_df = df[
            (df["n_embd"] == model_config.embedding_dim)
            & (df["n_q_head"] == model_config.num_q_heads)
            & (df["n_kv_head"] == model_config.num_kv_heads)
            & (df["block_size"] == replica_scheduler_config.block_size)
            & (df["num_tensor_parallel_workers"] == effective_tp)
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")

        if len(filtered_df) == 0:
            # Surface what is available to make debugging explicit.
            available = {
                "n_embd": sorted(df["n_embd"].unique().tolist()) if "n_embd" in df else [],
                "n_q_head": sorted(df["n_q_head"].unique().tolist()) if "n_q_head" in df else [],
                "n_kv_head": sorted(df["n_kv_head"].unique().tolist()) if "n_kv_head" in df else [],
                "block_size": sorted(df["block_size"].unique().tolist()) if "block_size" in df else [],
                "num_tensor_parallel_workers": sorted(df["num_tensor_parallel_workers"].unique().tolist()) if "num_tensor_parallel_workers" in df else [],
            }

            logger.error(
                "Attention profiling rows are missing for the requested configuration. "
                "Available values: %s", available
            )

            raise ValueError(
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required configuration:\n"
                f"  - n_embd: {model_config.embedding_dim}\n"
                f"  - n_q_head: {model_config.num_q_heads}\n"
                f"  - n_kv_head: {model_config.num_kv_heads}\n"
                f"  - block_size: {replica_scheduler_config.block_size}\n"
                f"  - tensor_parallel_size(requested): {requested_tp}\n"
                f"  - tensor_parallel_size(effective): {effective_tp}\n"
                f"Available values: {available}\n"
                f"Please run attention profiling with the correct configuration."
            )

        return filtered_df

    def _load_all_reduce_df(self, file_path: str, replica_config, tensor_parallel_size: int) -> pd.DataFrame:
        """
        Load all_reduce dataframe with cluster-specific tensor parallel size.

        Args:
            file_path: Path to the communication profiling CSV file
            replica_config: Replica configuration
            tensor_parallel_size: Required tensor parallel size for filtering

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If no data matches filtering criteria
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"All-reduce input file does not exist: {file_path}\n"
                f"Please run communication profiling first.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_communication.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original all_reduce data: {len(df)} rows")

        # Show filtering conditions
        logger.info(f"Filtering conditions:")
        logger.info(f"  - num_workers == {tensor_parallel_size}")
        logger.info(f"  - devices_per_node == {tensor_parallel_size}")
        logger.info(f"  - collective == 'all_reduce'")

        filtered_df = df[
            (df["num_workers"] == tensor_parallel_size)
            & (df["devices_per_node"] == tensor_parallel_size)
            & (df["collective"] == "all_reduce")
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")

        if len(filtered_df) == 0:
            available_info = ""
            if len(df) > 0:
                available_info = (
                    f"Available values in file:\n"
                    f"  - num_workers: {sorted(df['num_workers'].unique())}\n"
                    f"  - devices_per_node: {sorted(df['devices_per_node'].unique())}\n"
                    f"  - collective: {sorted(df['collective'].unique())}"
                )
            raise ValueError(
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required: num_workers={tensor_parallel_size}, devices_per_node={tensor_parallel_size}, collective='all_reduce'\n"
                f"{available_info}"
            )

        return filtered_df

    def _load_send_recv_df(self, file_path: str, replica_config) -> pd.DataFrame:
        """
        Load send_recv dataframe for pipeline parallel communication.

        Args:
            file_path: Path to the communication profiling CSV file
            replica_config: Replica configuration

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Send/recv input file does not exist: {file_path}\n"
                f"Please run communication profiling first.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_communication.sh"
            )

        num_workers = replica_config.num_pipeline_stages * replica_config.attn_tensor_parallel_size
        devices_per_node = replica_config.node_config.num_devices_per_node
        is_multi_node = num_workers > devices_per_node

        if is_multi_node:
            devices_per_node = 1
        else:
            devices_per_node = 2

        df = pd.read_csv(file_path)
        logger.info(f"Original send_recv data: {len(df)} rows")
        logger.info(f"Filtering conditions: collective='send_recv', devices_per_node={devices_per_node}")

        filtered_df = df[
            (df["collective"] == "send_recv")
            & (df["devices_per_node"] == devices_per_node)
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")
        return filtered_df

    def _load_cpu_overhead_df(self, file_path: str, replica_config) -> pd.DataFrame:
        """
        Load CPU overhead dataframe with model configuration filtering.

        Args:
            file_path: Path to the CPU overhead profiling CSV file
            replica_config: Replica configuration

        Returns:
            Filtered DataFrame

        Raises:
            FileNotFoundError: If the input file does not exist
        """
        if not os.path.exists(file_path):
            logger.warning(
                "CPU overhead input file does not exist: %s. "
                "Skipping CPU overhead model training for this cluster.",
                file_path,
            )
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(
                "CPU overhead input file is empty: %s. "
                "Skipping CPU overhead model training for this cluster.",
                file_path,
            )
            return pd.DataFrame()

        df = apply_cpu_overhead_schema_v2_defaults(
            df,
            warn_fn=logger.warning,
            context=file_path,
        )
        df = validate_cpu_overhead_dataframe(df)

        model_config = replica_config.model_config

        logger.info(f"Original CPU overhead data: {len(df)} rows")
        logger.info(f"Filtering conditions: model_name='{model_config.get_name()}', tensor_parallel_degree={replica_config.attn_tensor_parallel_size}")

        filtered_df = df[
            (df["model_name"] == model_config.get_name())
            & (df["tensor_parallel_degree"] == replica_config.attn_tensor_parallel_size)
        ]

        logger.info(f"After filtering: {len(filtered_df)} rows")
        if filtered_df.empty:
            logger.warning(
                "No CPU overhead profiling rows found for model_name='%s', "
                "tensor_parallel_degree=%s in file '%s'.",
                model_config.get_name(),
                replica_config.attn_tensor_parallel_size,
                file_path,
            )
        return filtered_df

    # Load imbalance feature columns used for MoE training
    # These features describe the load distribution across experts
    # Reference: frontier/training/moe_trainer.py lines 224-239 (authoritative source)
    # Reference: frontier/profiling/moe/LOAD_IMBALANCE_GUIDE.md
    MOE_LOAD_IMBALANCE_FEATURES = [
        # Config features (6) - describe model configuration
        "total_routed_tokens",      # Total tokens after routing (num_tokens * router_topk)
        "num_experts_per_device",   # Number of experts per device after EP sharding
        "hidden_dim",               # Model hidden dimension
        "expert_hidden_dim",        # Expert FFN hidden dimension
        "router_topk",              # Number of experts each token is routed to
        "model_expansion_ratio",    # expert_hidden_dim / hidden_dim
        # Derived features (2) - derived from config and routing
        "tokens_per_expert_avg",    # Average tokens per expert
        "tokens_to_experts_ratio",  # tokens / num_experts ratio
        # Load features (6) - describe load distribution characteristics
        "expert_utilization",       # Proportion of experts with non-zero load
        "min_load_ratio",           # Min load / average load
        "load_imbalance_cv",        # Coefficient of Variation: std/mean, key imbalance metric
        "max_load_ratio",           # Max load / average load
        "load_entropy",             # Entropy of load distribution (higher = more uniform)
        "load_gini_coefficient",    # Gini coefficient: 0=equality, 1=inequality
    ]

    # Feature columns for mixed-batch attention prefill model
    # These features capture batch heterogeneity characteristics together with
    # the uniform KV-cache context used by MixedAttentionInput profiling.
    # Reference: frontier/training/attention_trainer.py lines 362-375 (authoritative source)
    ATTN_PREFILL_MIXED_FEATURES = [
        # Core features (7)
        "batch_size",               # Number of sequences in batch
        "kv_cache_size",            # Uniform KV cache context for the mixed batch
        "total_tokens",             # Total tokens across all sequences
        "avg_seq_len",              # Average sequence length
        "min_seq_len",              # Minimum sequence length
        "max_seq_len",              # Maximum sequence length
        "total_tokens_squared",     # Computational complexity proxy
        # Heterogeneity features (3)
        "seq_len_variance",         # Variance of sequence lengths
        "seq_len_cv",               # Coefficient of variation (std/mean)
        "seq_len_range",            # max_seq_len - min_seq_len
        # Interaction features (2)
        "batch_variance_interaction",   # batch_size * seq_len_variance
        "batch_cv_interaction",         # batch_size * seq_len_cv
    ]

    ATTN_DECODE_IN_MIXED_FEATURES = [
        "decode_batch_size",
        "decode_avg_kv_cache_size",
        "num_prefill_seqs",
        "total_prefill_tokens",
        "total_batch_size",
        "batch_composition_ratio",
        "total_tokens",
    ]

    def _load_moe_df(
        self,
        file_path: str,
        replica_config,
        load_imbalance: bool = True,
        tensor_parallel_size: Optional[int] = None,
        expert_parallel_size: Optional[int] = None,
        layer_contract: Optional[ResolvedLayerContract] = None,
        operator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load MoE dataframe with cluster-specific configuration filtering.

        This function loads and filters MoE profiling data based on the model configuration
        and parallelism settings. It supports two training modes controlled by `load_imbalance`:

        1. **Load Imbalance Mode (default, load_imbalance=True)**:
           - Uses profiling data that includes load imbalance features
           - Training will use features like `load_imbalance_cv`, `load_gini_coefficient`, etc.
           - Recommended for accurate MoE execution time prediction under real-world scenarios
           - Requires profiling with `--enable_load_imbalance` flag

        2. **Standard Mode (load_imbalance=False)**:
           - Uses basic profiling data without load imbalance features
           - Training only uses `num_tokens` as feature
           - Simpler but less accurate for imbalanced workloads
           - Compatible with legacy profiling data

        The difference is in the **training features used**, not data row filtering.
        Load imbalance mode uses additional features to capture expert load distribution.

        Reference: frontier/profiling/moe/LOAD_IMBALANCE_GUIDE.md

        Args:
            file_path: Path to the MoE profiling CSV file
            replica_config: Replica configuration containing model and parallelism settings
            load_imbalance: Training mode flag:
                - True (default): Load imbalance mode - use load imbalance features
                - False: Standard mode - only use basic num_tokens feature
            tensor_parallel_size: Optional TP override for op-specific MoE training.
                If None, uses replica_config.moe_tensor_parallel_size.
            expert_parallel_size: Optional EP filter for op-specific MoE training.
                If None, EP filtering is skipped (used for EP-agnostic replicated ops).

        Returns:
            Filtered DataFrame ready for MoE model training

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If no data matches filtering criteria or required features are missing
        """
        if layer_contract is not None:
            _validate_typed_parallel_selection(
                layer_contract,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"MoE input file does not exist: {file_path}\n"
                f"Please run MoE profiling first.\n"
                f"Suggested command: bash frontier/profiling/example/test_profiling_moe.sh"
            )

        df = pd.read_csv(file_path)
        logger.info(f"Original MoE data: {len(df)} rows, {len(df.columns)} columns")
        expected_profile = (
            layer_contract.profile_id
            if layer_contract is not None
            else infer_single_runtime_profile(self)
        )
        if expected_profile is not None and "model_architecture_profile" in df.columns:
            validate_model_architecture_profile(
                df,
                file_path=file_path,
                expected_profile=expected_profile,
            )

        has_typed_contracts = TYPED_OPERATOR_CONTRACTS_COLUMN in df.columns
        parsed_typed_contracts: Optional[pd.Series] = None
        if has_typed_contracts:
            if not operator_name:
                raise ValueError(
                    "typed profiling loading requires operator_name when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            if layer_contract is None:
                raise ValueError(
                    "typed profiling loading requires layer_contract when the "
                    f"canonical {TYPED_OPERATOR_CONTRACTS_COLUMN!r} column is present"
                )
            # Parse every row before applying scalar filters so malformed metadata
            # cannot be hidden by an unrelated TP, EP, or width selector.
            parsed_typed_contracts = cast(
                pd.Series,
                df[TYPED_OPERATOR_CONTRACTS_COLUMN].map(
                    lambda raw_contracts: validate_typed_operator_contracts(
                        raw_contracts,
                        model_config=replica_config.model_config,
                    )
                ),
            )

        model_config = replica_config.model_config
        training_mode = "load_imbalance (load_imbalance=True)" if load_imbalance else "standard (load_imbalance=False)"
        if tensor_parallel_size is None:
            tensor_parallel_size = replica_config.moe_tensor_parallel_size
        if tensor_parallel_size <= 0:
            raise ValueError(
                f"Invalid tensor_parallel_size for MoE data loading: {tensor_parallel_size}"
            )

        # Display filtering conditions
        logger.info(f"Filtering conditions:")
        logger.info(f"  - num_experts == {model_config.num_experts}")
        logger.info(f"  - router_topk == {model_config.num_experts_per_tok}")
        logger.info(f"  - hidden_dim == {model_config.embedding_dim}")
        expected_expert_width = (
            layer_contract.effective_ffn_width
            if layer_contract is not None
            else model_config.mlp_hidden_dim
        )
        logger.info(f"  - expert_hidden_dim == {expected_expert_width}")
        logger.info(f"  - num_tensor_parallel_workers == {tensor_parallel_size}")
        if expert_parallel_size is None:
            logger.info("  - expert_parallel_size == ANY (EP-agnostic op)")
        else:
            logger.info(f"  - expert_parallel_size == {expert_parallel_size}")
        logger.info(f"  - training_mode: {training_mode}")

        # Display available values in the dataset
        available_info = []
        if len(df) > 0:
            if 'num_experts' in df.columns:
                available_info.append(f"  - Available num_experts: {sorted(df['num_experts'].unique())}")
            if 'router_topk' in df.columns:
                available_info.append(f"  - Available router_topk: {sorted(df['router_topk'].unique())}")
            if 'num_tensor_parallel_workers' in df.columns:
                available_info.append(f"  - Available num_tensor_parallel_workers: {sorted(df['num_tensor_parallel_workers'].unique())}")
            if 'expert_parallel_size' in df.columns:
                available_info.append(f"  - Available expert_parallel_size: {sorted(df['expert_parallel_size'].unique())}")
            if 'load_distribution' in df.columns:
                available_info.append(f"  - Available load_distribution: {sorted(df['load_distribution'].unique())}")

        for info in available_info:
            logger.info(info)

        # Apply filtering based on MoE configuration
        filtered_df = cast(pd.DataFrame, df[
            (df["num_experts"] == model_config.num_experts)
            & (df["router_topk"] == model_config.num_experts_per_tok)
            & (df["hidden_dim"] == model_config.embedding_dim)
            & (df["num_tensor_parallel_workers"] == tensor_parallel_size)
        ])
        if not has_typed_contracts:
            filtered_df = filtered_df[
                filtered_df["expert_hidden_dim"] == expected_expert_width
            ]
        else:
            if parsed_typed_contracts is None:
                raise RuntimeError(
                    "typed MoE metadata column was detected but could not be parsed"
                )
            if layer_contract is None:
                raise ValueError(
                    "typed MoE filtering requires a resolved layer contract"
                )
            selected_layer_contract = layer_contract
            typed_mask = parsed_typed_contracts.loc[filtered_df.index].map(
                lambda raw_contracts: _typed_row_matches_contract(
                    raw_contracts,
                    selected_layer_contract,
                    operator_name=operator_name,
                )
            )
            filtered_df = cast(pd.DataFrame, filtered_df[typed_mask])
        filtered_df = cast(pd.DataFrame, filtered_df)
        if expert_parallel_size is not None:
            if "expert_parallel_size" not in filtered_df.columns:
                raise ValueError(
                    "MoE profiling data is missing 'expert_parallel_size' while "
                    f"EP={expert_parallel_size} is required in {file_path}"
                )
            filtered_df = filtered_df[
                filtered_df["expert_parallel_size"] == expert_parallel_size
            ]

        logger.info(f"After config filtering: {len(filtered_df)} rows")

        # Check for load imbalance features if load_imbalance mode is enabled
        if load_imbalance:
            missing_features = [
                f for f in self.MOE_LOAD_IMBALANCE_FEATURES
                if f not in filtered_df.columns
            ]
            if missing_features:
                logger.warning(
                    f"Load imbalance mode requested but missing features: {missing_features}\n"
                    f"Available columns: {list(filtered_df.columns)}\n"
                    f"Please run MoE profiling with --enable_load_imbalance flag.\n"
                    f"Use load_imbalance=False (standard mode) explicitly if you want to train without load imbalance features."
                )
                raise ValueError("Missing load imbalance features")
                # Note: We don't change load_imbalance here, caller should handle feature selection
            else:
                logger.info(f"Load imbalance features available: {self.MOE_LOAD_IMBALANCE_FEATURES}")

        if len(filtered_df) == 0:
            ep_requirement = "ANY" if expert_parallel_size is None else expert_parallel_size
            available_info_text = "\n".join(available_info)
            message = (
                f"No data matches the filtering criteria in {file_path}\n"
                f"Required MoE configuration:\n"
                f"  - num_experts: {model_config.num_experts}\n"
                f"  - router_topk: {model_config.num_experts_per_tok}\n"
                f"  - hidden_dim: {model_config.embedding_dim}\n"
                f"  - expert_hidden_dim: {expected_expert_width}\n"
                f"  - tensor_parallel_size: {tensor_parallel_size}\n"
                f"  - expert_parallel_size: {ep_requirement}\n"
                f"  - training_mode: {training_mode}\n"
            )
            if has_typed_contracts:
                message += (
                    f"  - typed operator: {operator_name!r}\n"
                    "  - typed layer contract admission: required\n"
                )
            if available_info_text:
                message += available_info_text
            raise ValueError(
                message
            )

        return filtered_df

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to attention dataframe.

        Standard features for attn_prefill and attn_decode:
            - num_tokens: max(prefill_chunk_size, batch_size)
            - is_decode: derived from is_prefill when available, else prefill_chunk_size == 0
            - prefill_chunk_size_squared: prefill_chunk_size ** 2

        Mixed-batch features for attn_prefill_mixed (12 features):
            Reference: frontier/training/attention_trainer.py lines 362-375
            These features capture batch heterogeneity for accurate prefill time prediction.
        """
        df_with_derived_features = df.copy()

        # Standard attention features
        df_with_derived_features["num_tokens"] = df_with_derived_features[["prefill_chunk_size", "batch_size"]].max(axis=1)
        if "is_prefill" in df_with_derived_features.columns:
            normalized_prefill_values = coerce_truthy_bool(
                df_with_derived_features["is_prefill"]
            )
            df_with_derived_features["is_decode"] = ~normalized_prefill_values
        else:
            df_with_derived_features["is_decode"] = (df_with_derived_features["prefill_chunk_size"] == 0)
        df_with_derived_features["prefill_chunk_size_squared"] = (df_with_derived_features["prefill_chunk_size"] ** 2)

        def _normalize_bool_series(series: pd.Series) -> pd.Series:
            return coerce_truthy_bool(series)

        if "is_mixed_batch" in df_with_derived_features.columns:
            df_with_derived_features["is_mixed_batch"] = _normalize_bool_series(
                df_with_derived_features["is_mixed_batch"]
            )
        else:
            df_with_derived_features["is_mixed_batch"] = False

        if "is_true_mixed_batch" in df_with_derived_features.columns:
            df_with_derived_features["is_true_mixed_batch"] = _normalize_bool_series(
                df_with_derived_features["is_true_mixed_batch"]
            )
        else:
            df_with_derived_features["is_true_mixed_batch"] = False

        # Mixed-batch features for attn_prefill_mixed (if applicable)
        # Check if the profiling data contains mixed-batch specific columns
        has_mixed_batch_data = "total_tokens" in df_with_derived_features.columns

        if has_mixed_batch_data:
            logger.info("Adding mixed-batch derived features for attn_prefill_mixed")

            # total_tokens_squared for computational complexity
            if "total_tokens" in df_with_derived_features.columns:
                df_with_derived_features["total_tokens_squared"] = (
                    df_with_derived_features["total_tokens"] ** 2
                )

            # seq_len_range = max_seq_len - min_seq_len
            if "max_seq_len" in df_with_derived_features.columns and "min_seq_len" in df_with_derived_features.columns:
                df_with_derived_features["seq_len_range"] = (
                    df_with_derived_features["max_seq_len"] -
                    df_with_derived_features["min_seq_len"]
                )

            # Interaction features: batch_size * heterogeneity metrics
            if "seq_len_variance" in df_with_derived_features.columns:
                df_with_derived_features["batch_variance_interaction"] = (
                    df_with_derived_features["batch_size"] *
                    df_with_derived_features["seq_len_variance"]
                )

            if "seq_len_cv" in df_with_derived_features.columns:
                df_with_derived_features["batch_cv_interaction"] = (
                    df_with_derived_features["batch_size"] *
                    df_with_derived_features["seq_len_cv"]
                )

            if {
                "num_prefill_seqs",
                "num_decode_seqs",
            }.issubset(df_with_derived_features.columns) and (
                "total_batch_size" not in df_with_derived_features.columns
            ):
                df_with_derived_features["total_batch_size"] = (
                    df_with_derived_features["num_prefill_seqs"]
                    + df_with_derived_features["num_decode_seqs"]
                )

            if {
                "num_prefill_seqs",
                "total_batch_size",
            }.issubset(df_with_derived_features.columns) and (
                "batch_composition_ratio" not in df_with_derived_features.columns
            ):
                total_batch_size = df_with_derived_features["total_batch_size"].replace(0, pd.NA)
                df_with_derived_features["batch_composition_ratio"] = (
                    df_with_derived_features["num_prefill_seqs"] / total_batch_size
                ).fillna(0.0)

            if (
                "num_decode_seqs" in df_with_derived_features.columns
                and "decode_batch_size" not in df_with_derived_features.columns
            ):
                df_with_derived_features["decode_batch_size"] = df_with_derived_features[
                    "num_decode_seqs"
                ]

        return df_with_derived_features

    def _get_all_reduce_df_with_derived_features(self, df: pd.DataFrame, replica_config) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / replica_config.model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame, replica_config) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / replica_config.model_config.embedding_dim / 2
        )
        return df_with_derived_features
