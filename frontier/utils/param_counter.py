from math import ceil
from types import SimpleNamespace
from typing import Callable, Iterable, cast

from frontier.config import ReplicaConfig
from frontier.model_architectures import (
    LayerKind,
    ModelArchitectureProfile,
    ResolvedLayerContract,
)
from frontier.operators.spec import TensorParallelMode
from frontier.spec_decode.mtp_registry import is_target_embedded_mtp_method
from frontier.spec_decode.mtp_runtime import (
    build_mtp_runtime_contract,
    load_mtp_structural_model_config,
)
from frontier.spec_decode.runtime import MTP_METHOD_FAMILIES, SUPPORTED_SPEC_METHODS
from frontier.types import ClusterType


class ParamCounter:
    def __init__(
        self,
        replica_config: ReplicaConfig,
        cluster_type: ClusterType | None = None
    ) -> None:
        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self._cluster_type = cluster_type or ClusterType.MONOLITHIC

        # Validate configuration based on cluster type
        if self._cluster_type != ClusterType.DECODE_FFN:
            if (hasattr(self._model_config, "num_q_heads") and
                self._model_config.num_q_heads > 0 and
                hasattr(self._replica_config, "attn_tensor_parallel_size") and
                self._replica_config.attn_tensor_parallel_size > 0):
                assert (
                    self._model_config.num_q_heads % self._replica_config.attn_tensor_parallel_size
                    == 0
                ), (
                    f"Tensor parallelism configuration error: num_q_heads ({self._model_config.num_q_heads}) "
                    f"must be evenly divisible by attn_tensor_parallel_size ({self._replica_config.attn_tensor_parallel_size})"
                )
                assert (
                    self._model_config.embedding_dim % self._replica_config.attn_tensor_parallel_size
                    == 0
                ), (
                    f"Tensor parallelism configuration error: embedding_dim ({self._model_config.embedding_dim}) "
                    f"must be evenly divisible by attn_tensor_parallel_size ({self._replica_config.attn_tensor_parallel_size})"
                )
                assert self._model_config.embedding_dim % self._model_config.num_q_heads == 0, (
                    f"Model configuration error: embedding_dim ({self._model_config.embedding_dim}) "
                    f"must be evenly divisible by num_q_heads ({self._model_config.num_q_heads})"
                )

        # Validate pipeline parallelism configuration (applies to all cluster types)
        # This should have been caught earlier in ReplicaConfig.__post_init__, but we check again for safety
        if self._model_config.num_layers % self._replica_config.num_pipeline_stages != 0:
            raise ValueError(
                f"Pipeline parallelism configuration error in {self._cluster_type.name} cluster: "
                f"num_layers ({self._model_config.num_layers}) must be evenly divisible by "
                f"num_pipeline_stages ({self._replica_config.num_pipeline_stages}). "
                f"This should have been caught during configuration validation."
            )

        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )

        if self._cluster_type != ClusterType.DECODE_FFN:
            if (hasattr(self._model_config, "num_q_heads") and
                self._model_config.num_q_heads > 0):
                # Use model_config.get_head_dim() to prioritize explicit head_dim from JSON config
                # This ensures consistency with the profiling module's ModelConfig.get_head_size()
                self._attention_head_dim = self._model_config.get_head_dim()
                self._q_heads_per_tensor_parallel_worker = (
                    self._model_config.num_q_heads // self._replica_config.attn_tensor_parallel_size
                )
                self._kv_heads_per_tensor_parallel_worker = ceil(
                    self._model_config.num_kv_heads / self._replica_config.attn_tensor_parallel_size
                )
            else:
                self._attention_head_dim = 0
                self._q_heads_per_tensor_parallel_worker = 0
                self._kv_heads_per_tensor_parallel_worker = 0
        else:
            # DECODE_FFN cluster does not process attention params.
            self._attention_head_dim = 0
            self._q_heads_per_tensor_parallel_worker = 0
            self._kv_heads_per_tensor_parallel_worker = 0

    @staticmethod
    def _normalize_parallel_size(parallel_size: int) -> int:
        normalized = int(parallel_size)
        if normalized <= 0:
            return 1
        return normalized

    def _get_attn_tp_size(self) -> int:
        return self._normalize_parallel_size(
            getattr(self._replica_config, "attn_tensor_parallel_size", 1)
        )

    def _get_moe_tp_size(self) -> int:
        return self._normalize_parallel_size(
            getattr(self._replica_config, "moe_tensor_parallel_size", 1)
        )

    def _get_ep_size(self) -> int:
        return self._normalize_parallel_size(
            getattr(self._replica_config, "moe_expert_parallel_size", 1)
        )

    def _select_profile_layer_id(self, layer_kind: LayerKind) -> int | None:
        """Select a representative physical layer for a typed contract."""

        if not bool(getattr(self._model_config, "is_moe", False)):
            return 0

        get_layer_ids = getattr(self._model_config, "get_moe_layer_ids", None)
        if not callable(get_layer_ids):
            # A model without an explicit layer map is an all-routed model.
            return None

        typed_get_layer_ids = cast(Callable[[], Iterable[int]], get_layer_ids)
        raw_layer_ids = tuple(typed_get_layer_ids())
        if layer_kind is LayerKind.DENSE:
            moe_layer_ids = set(raw_layer_ids)
            num_layers = getattr(self._model_config, "num_layers", None)
            if type(num_layers) is not int or num_layers <= 0:
                raise ValueError(
                    "ParamCounter requires a positive num_layers to resolve dense layers"
                )
            return next(
                (
                    layer_id
                    for layer_id in range(num_layers)
                    if layer_id not in moe_layer_ids
                ),
                None,
            )

        if layer_kind in (LayerKind.ROUTED, LayerKind.SHARED):
            if not raw_layer_ids:
                raise ValueError(
                    f"ParamCounter cannot resolve {layer_kind.value} layer without an MoE layer"
                )
            return raw_layer_ids[0]

        raise ValueError(f"Unsupported typed FFN layer kind: {layer_kind!r}")

    def _resolve_profile_layer_contract(
        self,
        layer_kind: LayerKind,
        tensor_parallel_size: int,
        *,
        expert_parallel_size: int | None = None,
    ) -> ResolvedLayerContract | None:
        """Resolve one FFN domain through the model architecture profile."""

        get_profile = getattr(
            self._model_config, "get_model_architecture_profile", None
        )
        if not callable(get_profile):
            # Synthetic configs used by isolated MTP tests predate typed profiles.
            return None

        profile = cast(ModelArchitectureProfile, get_profile())
        resolver = getattr(profile, "resolve_layer_contract", None)
        if not callable(resolver):
            raise TypeError(
                "model architecture profile must expose resolve_layer_contract()"
            )
        typed_resolver = cast(Callable[..., ResolvedLayerContract], resolver)

        spec = profile.get_layer_contract(layer_kind)
        attention_tp_size = self._get_attn_tp_size()
        moe_tp_size = self._get_moe_tp_size()
        ffn_tp_size = tensor_parallel_size
        if spec.tensor_parallel_mode is TensorParallelMode.ATTENTION_TP:
            attention_tp_size = tensor_parallel_size
        elif spec.tensor_parallel_mode is TensorParallelMode.MOE_TP:
            moe_tp_size = tensor_parallel_size

        layer_id = self._select_profile_layer_id(layer_kind)
        if layer_kind is LayerKind.DENSE and layer_id is None:
            # An all-MoE model has no dense layer contract to account for.
            return None

        resolver_kwargs = {
            "layer_kind": layer_kind,
            "attention_tp_size": attention_tp_size,
            "moe_tp_size": moe_tp_size,
            "ffn_tp_size": ffn_tp_size,
            "tensor_parallel_size": tensor_parallel_size,
            "expert_parallel_size": expert_parallel_size,
        }
        if layer_id is not None:
            resolver_kwargs["layer_id"] = layer_id
        contract = typed_resolver(self._model_config, **resolver_kwargs)
        if not isinstance(contract, ResolvedLayerContract):
            raise TypeError(
                "model architecture profile resolver must return a "
                "ResolvedLayerContract"
            )
        return contract

    @staticmethod
    def _get_ffn_weight_params(
        width: int,
        embedding_dim: int,
        use_gated_mlp: bool,
        tensor_parallel_size: int,
    ) -> int:
        """Return one FFN's local weight count for a typed width."""

        if type(width) is not int or width <= 0:
            return 0
        if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
            raise ValueError(
                "tensor_parallel_size must be a positive int, "
                f"got {tensor_parallel_size!r}"
            )
        multiplier = 3 if use_gated_mlp else 2
        return multiplier * embedding_dim * width // tensor_parallel_size

    def _get_dense_mlp_params_per_layer(self, tensor_parallel_size: int) -> int:
        contract = self._resolve_profile_layer_contract(
            LayerKind.DENSE,
            tensor_parallel_size,
        )
        width = (
            contract.effective_ffn_width
            if contract is not None
            else getattr(
                self._model_config,
                "dense_mlp_hidden_dim",
                getattr(self._model_config, "mlp_hidden_dim", 0),
            )
        )
        return self._get_ffn_weight_params(
            width,
            self._model_config.embedding_dim,
            bool(getattr(self._model_config, "use_gated_mlp", False)),
            tensor_parallel_size,
        )

    def _get_share_expert_params_per_layer(self, tensor_parallel_size: int) -> int:
        if not getattr(self._model_config, "is_moe", False):
            return 0

        get_model_architecture_profile = getattr(
            self._model_config, "get_model_architecture_profile", None
        )
        if not callable(get_model_architecture_profile):
            raise TypeError(
                "ParamCounter requires model_config.get_model_architecture_profile()"
            )
        profile = get_model_architecture_profile()
        if not getattr(profile, "counts_share_expert_param_memory", False):
            return 0

        contract = self._resolve_profile_layer_contract(
            LayerKind.SHARED,
            tensor_parallel_size,
        )
        share_expert_dim = int(
            contract.effective_ffn_width
            if contract is not None
            else getattr(self._model_config, "share_expert_dim", 0) or 0
        )
        if share_expert_dim <= 0:
            return 0

        return self._get_ffn_weight_params(
            share_expert_dim,
            self._model_config.embedding_dim,
            bool(getattr(self._model_config, "use_gated_mlp", False)),
            tensor_parallel_size,
        )

    def _get_num_moe_layers_per_pipeline_stage(self) -> int:
        if not getattr(self._model_config, "is_moe", False):
            return 0

        num_layers_per_stage = int(self._num_layers_per_pipeline_stage)
        total_layers = int(getattr(self._model_config, "num_layers", num_layers_per_stage))

        num_moe_layers_total = total_layers
        if hasattr(self._model_config, "get_num_moe_layers"):
            num_moe_layers_total = int(self._model_config.get_num_moe_layers())
        elif hasattr(self._model_config, "get_moe_layer_ids"):
            num_moe_layers_total = len(self._model_config.get_moe_layer_ids())

        if num_moe_layers_total <= 0:
            return 0
        if num_moe_layers_total >= total_layers:
            return num_layers_per_stage

        num_pipeline_stages = int(getattr(self._replica_config, "num_pipeline_stages", 1))
        if num_pipeline_stages <= 1:
            return min(num_moe_layers_total, num_layers_per_stage)

        # Approximate MoE layer distribution by ratio when explicit stage slicing is unavailable.
        ratio = float(num_moe_layers_total) / float(total_layers)
        estimate = int(round(ratio * num_layers_per_stage))
        return max(0, min(num_layers_per_stage, estimate))

    def _get_routed_moe_params_per_layer(self, tensor_parallel_size: int) -> int:
        is_moe = getattr(self._model_config, "is_moe", False)
        num_experts = int(getattr(self._model_config, "num_experts", 0))
        if not is_moe or num_experts <= 0:
            return self._get_dense_mlp_params_per_layer(tensor_parallel_size)

        contract = self._resolve_profile_layer_contract(
            LayerKind.ROUTED,
            tensor_parallel_size,
            expert_parallel_size=self._get_ep_size(),
        )
        routed_width = (
            contract.effective_ffn_width
            if contract is not None
            else getattr(
                self._model_config,
                "routed_mlp_hidden_dim",
                getattr(self._model_config, "mlp_hidden_dim", 0),
            )
        )
        num_parameters = self._get_ffn_weight_params(
            routed_width,
            self._model_config.embedding_dim,
            bool(getattr(self._model_config, "use_gated_mlp", False)),
            tensor_parallel_size,
        )

        ep_size = self._get_ep_size()
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
            )
        num_experts_per_device = num_experts // ep_size

        # num_parameters currently holds one expert's FFN params.
        num_parameters *= num_experts_per_device
        # Add router gate: Linear(embedding_dim, num_experts, bias=False).
        num_parameters += self._model_config.embedding_dim * num_experts
        return num_parameters

    def get_num_attention_params_per_layer(self) -> int:
        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0

        if (self._q_heads_per_tensor_parallel_worker == 0 or
            self._kv_heads_per_tensor_parallel_worker == 0 or
            self._attention_head_dim == 0):
            return 0

        num_parameters = 0
        # Weights for attention matrices Wq, Wk, Wv.
        num_parameters += (
            self._model_config.embedding_dim
            * self._attention_head_dim
            * (
                self._q_heads_per_tensor_parallel_worker
                + 2 * self._kv_heads_per_tensor_parallel_worker
            )
        )
        # Weights for attention output projection Wo.
        num_parameters += (
            self._model_config.embedding_dim
            * self._attention_head_dim
            * self._q_heads_per_tensor_parallel_worker
        )
        return num_parameters

    def get_num_mlp_params_per_layer(self) -> int:
        # For DECODE_ATTN cluster, there are no MLP parameters.
        if self._cluster_type == ClusterType.DECODE_ATTN:
            return 0

        if self._cluster_type == ClusterType.DECODE_FFN or getattr(
            self._model_config, "is_moe", False
        ):
            tensor_parallel_size = self._get_moe_tp_size()
        else:
            tensor_parallel_size = self._get_attn_tp_size()

        return self._get_routed_moe_params_per_layer(tensor_parallel_size)

    def get_num_parameters_per_layer(self) -> int:
        return (
            self.get_num_attention_params_per_layer()
            + self.get_num_mlp_params_per_layer()
        )

    def get_num_parameters_per_device(self) -> int:
        if not getattr(self._model_config, "is_moe", False):
            num_parameters_per_layer = self.get_num_parameters_per_layer()
            return (
                num_parameters_per_layer * self._num_layers_per_pipeline_stage
                + self.get_num_mtp_parameters_per_device()
            )

        num_attention_parameters = (
            self.get_num_attention_params_per_layer() * self._num_layers_per_pipeline_stage
        )
        num_mlp_parameters = self.get_num_mlp_parameters_per_device()
        return (
            num_attention_parameters
            + num_mlp_parameters
            + self.get_num_mtp_parameters_per_device()
        )

    def get_num_mlp_parameters_per_device(self) -> int:
        if self._cluster_type == ClusterType.DECODE_ATTN:
            return 0

        is_moe = bool(getattr(self._model_config, "is_moe", False))
        if not is_moe:
            num_parameters_per_layer = self.get_num_mlp_params_per_layer()
            return num_parameters_per_layer * self._num_layers_per_pipeline_stage

        num_moe_layers = self._get_num_moe_layers_per_pipeline_stage()
        num_non_moe_layers = self._num_layers_per_pipeline_stage - num_moe_layers

        # Each layer owns only the FFN domains that execute on that layer.
        # Dense boundaries do not carry routed or shared-expert weights, and
        # pure-MoE models do not have an inactive dense contract to resolve.
        dense_mlp_params = (
            self._get_dense_mlp_params_per_layer(self._get_attn_tp_size())
            if num_non_moe_layers
            else 0
        )
        routed_moe_params = (
            self._get_routed_moe_params_per_layer(self._get_moe_tp_size())
            if num_moe_layers
            else 0
        )
        share_expert_params = (
            self._get_share_expert_params_per_layer(self._get_attn_tp_size())
            if num_moe_layers
            else 0
        )

        return (
            num_non_moe_layers * dense_mlp_params
            + num_moe_layers * (routed_moe_params + share_expert_params)
        )

    def _get_mtp_embed_params(self, proposer_model_config) -> int:
        return (
            int(proposer_model_config.vocab_size)
            * int(proposer_model_config.embedding_dim)
            // self._get_attn_tp_size()
        )

    def _get_mtp_fusion_proj_params(self, contract, proposer_model_config) -> int:
        hidden_dim = int(proposer_model_config.embedding_dim)
        params = 2 * hidden_dim * hidden_dim
        if contract.fusion_is_tp_sharded:
            params //= int(contract.attn_tp_size)
        return params

    def _get_mtp_lm_head_params(self, proposer_model_config) -> int:
        if getattr(proposer_model_config, "tie_word_embeddings", True):
            return 0
        return (
            int(proposer_model_config.vocab_size)
            * int(proposer_model_config.embedding_dim)
            // self._get_attn_tp_size()
        )

    def _build_mtp_replica_config(self, proposer_model_config) -> ReplicaConfig:
        return cast(
            ReplicaConfig,
            SimpleNamespace(
                model_config=proposer_model_config,
                attn_tensor_parallel_size=self._get_attn_tp_size(),
                moe_tensor_parallel_size=self._get_moe_tp_size(),
                moe_expert_parallel_size=self._get_ep_size(),
                num_pipeline_stages=1,
            ),
        )

    def get_num_mtp_parameters_per_device(self) -> int:
        spec_config = getattr(self._replica_config, "speculative_decoding_config", None)
        if spec_config is None or not getattr(spec_config, "enabled", False):
            return 0
        if self._cluster_type in {ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN}:
            return 0

        method = str(spec_config.method)
        if method not in MTP_METHOD_FAMILIES:
            if method not in SUPPORTED_SPEC_METHODS:
                raise ValueError(
                    "Speculative method name must match vLLM names, "
                    f"got={method!r}, supported={sorted(SUPPORTED_SPEC_METHODS)}"
                )
            return 0

        contract = build_mtp_runtime_contract(
            method=method,
            target_model_name=str(self._replica_config.model_name),
            spec_model_name=str(getattr(spec_config, "spec_model_name", "")),
            attn_tp_size=self._get_attn_tp_size(),
            mtp_n_predict=int(spec_config.mtp_n_predict),
            mtp_num_layers=int(spec_config.mtp_num_layers),
        )

        proposer_model_config = (
            self._model_config
            if str(contract.proposer_model_name) == str(self._replica_config.model_name)
            else load_mtp_structural_model_config(str(contract.proposer_model_name))
        )
        mtp_replica_config = self._build_mtp_replica_config(proposer_model_config)
        mtp_layer_counter = ParamCounter(
            mtp_replica_config,
            cluster_type=self._cluster_type,
        )
        decoder_layer_params = int(mtp_layer_counter.get_num_parameters_per_layer())

        hidden_dim = int(proposer_model_config.embedding_dim)
        shared_embed_params = self._get_mtp_embed_params(proposer_model_config)
        shared_fusion_params = self._get_mtp_fusion_proj_params(
            contract,
            proposer_model_config,
        )
        shared_lm_head_params = 0
        per_layer_params = decoder_layer_params
        total_norm_params = (
            int(contract.num_pre_fusion_norms) + int(contract.num_post_decoder_norms)
        ) * hidden_dim

        if is_target_embedded_mtp_method(str(contract.method)):
            shared_lm_head_params = self._get_mtp_lm_head_params(proposer_model_config)
            shared_extra_params = (
                shared_embed_params
                + shared_fusion_params
                + total_norm_params
                + shared_lm_head_params
            )
            return shared_extra_params + per_layer_params * int(contract.mtp_num_layers)

        per_layer_shared_head_params = shared_fusion_params + total_norm_params

        if str(contract.method) == "deepseek_mtp":
            per_layer_shared_head_params += self._get_mtp_lm_head_params(
                proposer_model_config
            )
            return (
                shared_embed_params
                + int(contract.mtp_num_layers)
                * (per_layer_params + per_layer_shared_head_params)
            )

        if str(contract.method) == "ernie_mtp":
            shared_lm_head_params = self._get_mtp_lm_head_params(proposer_model_config)
            return (
                shared_embed_params
                + shared_lm_head_params
                + int(contract.mtp_num_layers)
                * (per_layer_params + per_layer_shared_head_params)
            )

        raise ValueError(
            f"Unsupported MTP method for parameter counting: {contract.method!r}"
        )
