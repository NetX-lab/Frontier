"""Resolution of the typed layer contract a training or lookup call applies.

A profiling row is admitted for a model only when its typed operator
contract matches the layer the caller asked about, at the tensor- and
expert-parallel sizes that layer actually uses.  These methods derive those
keys and the contract signature that identifies a trained model.
"""

import hashlib
import json
import pandas as pd

from frontier.execution_time_predictor.attention_tp_policy import (
    resolve_effective_attention_tp_size,
)
from frontier.execution_time_predictor.prediction_model_identity import (
    _resolve_model_architecture_profile,
    _resolve_profile_typed_family_for_query,
    _serialize_selected_layer_cache_identity,
)
from frontier.model_architectures import LayerKind, ResolvedLayerContract
from frontier.moe_routing_runtime import (
    filter_moe_gating_routing_topk_rows,
    resolve_moe_gating_routing_runtime_path,
)
from frontier.operators.binding import resolve_operator_query_tp_mode
from frontier.operators.families import (
    MOE_FAMILY,
    get_operator_family,
    is_moe_operator_ep_agnostic,
    resolve_moe_operator_tp_key,
)
from frontier.operators.spec import TensorParallelMode
from frontier.operators.typed_contracts import TYPED_OPERATOR_CONTRACTS_COLUMN
from frontier.spec_decode.mtp_registry import (
    get_target_embedded_mtp_linear_ops,
    is_target_embedded_mtp_same_tp_linear_op,
)
from frontier.spec_decode.runtime import is_target_embedded_mtp_enabled
from frontier.types import ClusterType
from typing import List, Optional, Tuple


class LayerContractResolution:
    """Typed layer contract and parallel-size resolution."""

    def _get_ffn_tp_key(self, cluster_type: ClusterType, replica_config, is_moe_model: bool) -> int:
        if cluster_type == ClusterType.DECODE_FFN:
            # In the FFN-only PD-AF cluster, dense FFN tensor parallelism is
            # carried by moe_tensor_parallel_size as the FFN-domain TP field.
            # attn_tensor_parallel_size can remain at its default because this
            # cluster owns no attention weights.  Use the FFN-domain TP for
            # both dense and MoE DECODE_FFN profiling selection.
            return replica_config.moe_tensor_parallel_size
        if (
            is_moe_model
            and cluster_type in {
                ClusterType.PREFILL,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            }
        ):
            return replica_config.moe_tensor_parallel_size
        return replica_config.attn_tensor_parallel_size

    def _resolve_typed_layer_contract(
        self,
        op_name: str,
        cluster_type: ClusterType,
        replica_config,
        *,
        is_moe_model: bool,
        layer_id: Optional[int] = None,
    ) -> Optional[ResolvedLayerContract]:
        """Resolve a typed FFN contract through the architecture profile."""

        model_config = getattr(replica_config, "model_config", None)
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is None:
            return None

        typed_family = _resolve_profile_typed_family_for_query(
            architecture_profile, op_name
        )
        if typed_family is None:
            return None
        typed_family_id, _ = typed_family

        # DECODE_ATTN is attention-only. Its exact zero domain is a deliberate
        # sentinel; any non-zero value indicates a malformed configuration.
        if cluster_type == ClusterType.DECODE_ATTN:
            zero_fields = (
                "attn_tensor_parallel_size",
                "moe_tensor_parallel_size",
                "moe_expert_parallel_size",
            )
            invalid = {
                field_name: getattr(replica_config, field_name, None)
                for field_name in zero_fields
                if getattr(replica_config, field_name, None) != 0
            }
            if invalid:
                raise ValueError(
                    "DECODE_ATTN typed FFN resolution requires exact zero "
                    f"parallel sizes, got {invalid!r}"
                )
            return None

        from frontier.operators.binding import bind_operator_query

        binding = bind_operator_query(op_name, family_id=typed_family_id)
        if binding.family_id != typed_family_id:
            raise ValueError(
                f"Operator query {op_name!r} resolved to family "
                f"{binding.family_id!r}, expected {typed_family_id!r}"
            )

        moe_tp_size = getattr(replica_config, "moe_tensor_parallel_size", None)
        attention_tp_size = getattr(
            replica_config, "attn_tensor_parallel_size", None
        )
        if cluster_type == ClusterType.DECODE_FFN:
            # The FFN-only role stores its domain size in the existing MoE TP
            # field, while the profile still owns the semantic TP mode.
            attention_tp_size = moe_tp_size
        ffn_tp_size = self._get_ffn_tp_key(
            cluster_type, replica_config, is_moe_model
        )
        return architecture_profile.resolve_layer_contract(
            model_config,
            layer_id=layer_id,
            operator_name=op_name,
            attention_tp_size=attention_tp_size,
            moe_tp_size=moe_tp_size,
            ffn_tp_size=ffn_tp_size,
            expert_parallel_size=getattr(
                replica_config, "moe_expert_parallel_size", None
            ),
        )

    def _resolve_ffn_layer_contracts(
        self,
        cluster_type: ClusterType,
        replica_config,
        is_moe_model: bool,
    ) -> Tuple[Tuple[str, ResolvedLayerContract], ...]:
        """Resolve each profile-owned FFN domain used by one training pass."""

        if cluster_type == ClusterType.DECODE_ATTN:
            zero_fields = (
                "attn_tensor_parallel_size",
                "moe_tensor_parallel_size",
                "moe_expert_parallel_size",
            )
            invalid = {
                field_name: getattr(replica_config, field_name, None)
                for field_name in zero_fields
                if getattr(replica_config, field_name, None) != 0
            }
            if invalid:
                raise ValueError(
                    "DECODE_ATTN FFN contract resolution requires exact zero "
                    f"parallel sizes, got {invalid!r}"
                )
            return ()

        model_config = getattr(replica_config, "model_config", None)
        if model_config is None:
            return ()
        architecture_profile = _resolve_model_architecture_profile(model_config)
        if architecture_profile is None:
            return ()
        if bool(is_moe_model) != bool(getattr(model_config, "is_moe", False)):
            raise ValueError(
                "is_moe_model does not match model configuration while resolving "
                "typed FFN contracts"
            )

        contracts: list[Tuple[str, ResolvedLayerContract]] = []
        for spec in architecture_profile.iter_active_layer_contracts(model_config):
            family_is_moe = spec.layer_kind is not LayerKind.DENSE
            for family_id in spec.operator_family_ids:
                family = get_operator_family(family_id)
                profiling_ops = tuple(family.profiling_ops())
                if not profiling_ops:
                    raise ValueError(
                        f"Typed operator family {family_id!r} has no profiling operators"
                    )
                contract = self._resolve_typed_layer_contract(
                    profiling_ops[0].name,
                    cluster_type,
                    replica_config,
                    is_moe_model=family_is_moe,
                )
                if contract is None:
                    raise ValueError(
                        f"Missing typed layer contract for operator family {family_id!r}"
                    )
                if contract.operator_family_id != family_id:
                    raise ValueError(
                        f"Operator family {family_id!r} resolved to "
                        f"{contract.operator_family_id!r}"
                    )
                contracts.append((family_id, contract))
        return tuple(contracts)

    def _get_ffn_contract_signature(
        self,
        cluster_type: ClusterType,
        replica_config,
        is_moe_model: bool,
    ) -> str:
        """Return a deterministic signature for the active FFN domains."""

        entries = self._resolve_ffn_layer_contracts(
            cluster_type, replica_config, is_moe_model
        )
        if not entries:
            return "none"
        payload = []
        for family_id, contract in entries:
            family = get_operator_family(family_id)
            profiling_ops = tuple(family.profiling_ops())
            if not profiling_ops:
                raise ValueError(
                    f"Typed operator family {family_id!r} has no profiling operators"
                )

            # The first operator is the compatibility representative returned
            # by _resolve_ffn_layer_contracts().  Include every sibling as
            # well: a family may mix EP-agnostic routing operators with an
            # EP-sensitive grouped GEMM, and the cache signature must retain
            # both semantics.
            for operator in profiling_ops:
                operator_contract = contract
                if operator is not profiling_ops[0]:
                    operator_contract = self._resolve_typed_layer_contract(
                        operator.profiling_name(),
                        cluster_type,
                        replica_config,
                        is_moe_model=contract.layer_kind is not LayerKind.DENSE,
                    )
                    if operator_contract is None:
                        raise ValueError(
                            "Missing typed layer contract for operator "
                            f"{operator.profiling_name()!r} in family {family_id!r}"
                        )
                payload.append(
                    {
                        "family_id": family_id,
                        "operator_name": operator.profiling_name(),
                        "identity": _serialize_selected_layer_cache_identity(
                            operator_contract
                        ),
                    }
                )
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @staticmethod

    def _is_mixed_layer_moe_model(model_config, is_moe_model: bool) -> bool:
        """Return whether a model needs both MoE and dense FFN predictors.

        Some MoE architectures keep dense FFN layers at the model boundaries.
        Their runtime dispatch is layer-specific, so model-level ``is_moe`` is
        insufficient to decide which predictor families must be materialized.
        Keep the legacy pure-MoE path unchanged when the layer-count contract
        is unavailable.
        """
        if not is_moe_model or model_config is None:
            return False
        get_num_moe_layers = getattr(model_config, "get_num_moe_layers", None)
        num_layers = getattr(model_config, "num_layers", None)
        if callable(get_num_moe_layers) and isinstance(num_layers, int):
            return int(get_num_moe_layers()) < int(num_layers)
        return False

    def _get_linear_op_tp_key(self, op_name: str, cluster_type: ClusterType, replica_config, is_moe_model: bool) -> int:
        model_config = getattr(replica_config, "model_config", None)
        # Lightweight configs retain the scalar FFN compatibility path, but
        # generic linear attention names still require a profile declaration
        # for TP-mode lookup.
        architecture_profile = _resolve_model_architecture_profile(
            model_config,
            allow_generic=True,
        )
        if op_name in get_target_embedded_mtp_linear_ops():
            return resolve_effective_attention_tp_size(
                op_name="attn_pre_proj",
                requested_tp_size=replica_config.attn_tensor_parallel_size,
                num_kv_heads=replica_config.model_config.num_kv_heads,
                cluster_type=cluster_type,
                warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                include_linear_ops=True,
            )

        try:
            tp_mode = resolve_operator_query_tp_mode(
                op_name,
                architecture_profile=architecture_profile,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported linear op for TP mapping: {op_name}") from exc

        typed_contract = self._resolve_typed_layer_contract(
            op_name,
            cluster_type,
            replica_config,
            is_moe_model=is_moe_model,
        )
        if (
            typed_contract is not None
            and typed_contract.tensor_parallel_size is not None
        ):
            return typed_contract.tensor_parallel_size

        if tp_mode is TensorParallelMode.REPLICATED:
            if (
                is_target_embedded_mtp_enabled(
                    getattr(replica_config, "speculative_decoding_config", None)
                )
                and is_target_embedded_mtp_same_tp_linear_op(op_name)
            ):
                return resolve_effective_attention_tp_size(
                    op_name="attn_pre_proj",
                    requested_tp_size=replica_config.attn_tensor_parallel_size,
                    num_kv_heads=replica_config.model_config.num_kv_heads,
                    cluster_type=cluster_type,
                    warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                    include_linear_ops=True,
                )
            return 1

        if tp_mode is TensorParallelMode.FFN_TP:
            return self._get_ffn_tp_key(cluster_type, replica_config, is_moe_model)

        if tp_mode is TensorParallelMode.ATTENTION_TP:
            return resolve_effective_attention_tp_size(
                op_name=op_name,
                requested_tp_size=replica_config.attn_tensor_parallel_size,
                num_kv_heads=replica_config.model_config.num_kv_heads,
                cluster_type=cluster_type,
                warning_cache=getattr(self, "_attention_tp_warning_cache", None),
                include_linear_ops=True,
            )

        raise ValueError(f"Unsupported linear op for TP mapping: {op_name}")

    @staticmethod

    def _get_moe_op_tp_key(
        op_name: str,
        replica_config,
        cluster_type: ClusterType | None = None,
    ) -> int:
        try:
            return resolve_moe_operator_tp_key(
                op_name,
                moe_tp_size=replica_config.moe_tensor_parallel_size,
                cluster_type=cluster_type,
                family=MOE_FAMILY,
            )
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for TP mapping: {op_name}"
                ) from exc
            raise

    @staticmethod

    def _is_moe_op_ep_agnostic(op_name: str) -> bool:
        try:
            return is_moe_operator_ep_agnostic(op_name, family=MOE_FAMILY)
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for EP mapping: {op_name}"
                ) from exc
            raise

    def _validate_moe_dataset_contract(
        self,
        file_path: str,
        replica_config,
        model_names: List[str],
        cluster_type: ClusterType,
        layer_contract: Optional[ResolvedLayerContract] = None,
    ) -> None:
        """Validate op-level MoE profiling key coverage before model training."""
        df = pd.read_csv(file_path)
        required_columns = [
            "num_experts",
            "router_topk",
            "hidden_dim",
            "expert_hidden_dim",
            "num_tensor_parallel_workers",
            "expert_parallel_size",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"MoE dataset contract validation failed for {file_path}: "
                f"missing required columns {missing_columns}."
            )

        model_config = replica_config.model_config
        if layer_contract is None:
            # A legacy caller has no profile-owned contract to validate.  Keep
            # the historical scalar admission rule, while refusing to guess
            # when the dataset advertises typed metadata.
            if TYPED_OPERATOR_CONTRACTS_COLUMN in df.columns:
                raise ValueError(
                    "typed MoE profiling data requires an explicit routed layer contract"
                )
            expected_expert_width = getattr(model_config, "mlp_hidden_dim", None)
            if type(expected_expert_width) is not int or expected_expert_width <= 0:
                raise ValueError(
                    "legacy MoE dataset validation requires a positive model_config.mlp_hidden_dim"
                )
        else:
            if layer_contract.layer_kind is not LayerKind.ROUTED:
                raise ValueError(
                    "MoE dataset contract validation requires a routed layer contract"
                )
            expected_expert_width = layer_contract.effective_ffn_width
        base_df = df[
            (df["num_experts"] == model_config.num_experts)
            & (df["router_topk"] == model_config.num_experts_per_tok)
            & (df["hidden_dim"] == model_config.embedding_dim)
            & (df["expert_hidden_dim"] == expected_expert_width)
        ]

        if len(base_df) == 0:
            raise ValueError(
                "MoE dataset contract validation failed: no rows match model configuration in "
                f"{file_path}. Required: num_experts={model_config.num_experts}, "
                f"router_topk={model_config.num_experts_per_tok}, hidden_dim={model_config.embedding_dim}, "
                f"expert_hidden_dim={expected_expert_width}."
            )

        available_pairs = sorted(
            {
                (int(tp), int(ep))
                for tp, ep in base_df[
                    ["num_tensor_parallel_workers", "expert_parallel_size"]
                ].drop_duplicates().itertuples(index=False, name=None)
            }
        )
        requested_routing_runtime_path = resolve_moe_gating_routing_runtime_path(
            getattr(replica_config, "moe_routing_distribution_type", "balanced")
        )

        missing_requirements: List[str] = []
        for model_name in model_names:
            tp_key = self._get_moe_op_tp_key(
                model_name,
                replica_config,
                cluster_type,
            )
            if self._is_moe_op_ep_agnostic(model_name):
                op_df = base_df[
                    base_df["num_tensor_parallel_workers"] == tp_key
                ]
                requirement = f"TP={tp_key}, EP=ANY"
            else:
                ep_key = replica_config.moe_expert_parallel_size
                op_df = base_df[
                    (base_df["num_tensor_parallel_workers"] == tp_key)
                    & (base_df["expert_parallel_size"] == ep_key)
                ]
                requirement = f"TP={tp_key}, EP={ep_key}"
            if model_name == "moe_gating_routing_topk":
                op_df = filter_moe_gating_routing_topk_rows(
                    op_df,
                    requested_runtime_path=requested_routing_runtime_path,
                    source_name=file_path,
                )
                requirement = (
                    f"{requirement}, routing_runtime_path="
                    f"{requested_routing_runtime_path}"
                )
            if len(op_df) == 0:
                missing_requirements.append(f"{model_name} requires {requirement}")

        if missing_requirements:
            requirement_text = "\n  - ".join(missing_requirements)
            raise ValueError(
                "MoE dataset contract validation failed before training.\n"
                f"File: {file_path}\n"
                "Missing op-level key coverage:\n"
                f"  - {requirement_text}\n"
                f"Available (TP, EP) pairs for matched model rows: {available_pairs}"
            )
