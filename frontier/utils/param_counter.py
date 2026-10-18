from math import ceil
from types import SimpleNamespace
from typing import Callable, Iterable, cast

from frontier.config import ReplicaConfig
from frontier.config.precision_type import PrecisionType
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
from frontier.types import ActivationType, ClusterType, NormType


class ParamCounter:
    def __init__(
        self,
        replica_config: ReplicaConfig,
        cluster_type: ClusterType | None = None
    ) -> None:
        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self._cluster_type = cluster_type or ClusterType.MONOLITHIC
        get_specs = getattr(self._model_config, "get_layer_attention_specs", None)
        specs = tuple(get_specs()) if callable(get_specs) else ()
        self._uses_v4_parameter_contract = any(
            spec.family_id == "deepseek_v4_attention" for spec in specs
        )
        get_family = getattr(self._model_config, "get_attention_family", None)
        self._uses_dsa_parameter_contract = (
            not self._uses_v4_parameter_contract
            and callable(get_family) and get_family().family_id == "dsa_attention"
        )
        if self._uses_dsa_parameter_contract:
            self._validate_dsa_parameter_contract()
        if self._uses_v4_parameter_contract:
            self._validate_v4_parameter_contract()

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
                if not (
                    self._uses_dsa_parameter_contract
                    or self._uses_v4_parameter_contract
                ):
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

    def _validate_dsa_parameter_contract(self) -> None:
        """Limit source estimates to the represented BF16/block-FP8 topology."""

        model = self._model_config
        if (
            self._replica_config.num_pipeline_stages != 1
            or self._cluster_type is not ClusterType.MONOLITHIC
        ):
            raise NotImplementedError(
                "DSA parameter estimates require PP1 MONOLITHIC ownership"
            )
        if self._get_moe_tp_size() != 1:
            raise NotImplementedError("DSA parameter estimates require MoE TP1")
        for name in ("attn_tensor_parallel_size", "moe_expert_parallel_size"):
            value = getattr(self._replica_config, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DSA {name} must be a positive int")
        spec = getattr(self._replica_config, "speculative_decoding_config", None)
        if spec is not None and spec.enabled:
            raise NotImplementedError("DSA parameter estimates exclude speculative decoding")
        if model.tie_word_embeddings is not False:
            raise NotImplementedError("DSA parameter estimates require untied embeddings")
        if PrecisionType.from_torch_dtype(model.torch_dtype) is not PrecisionType.BF16:
            raise NotImplementedError("DSA parameter estimates require BF16 model dtype")
        if model.topk_method != "noaux_tc":
            raise NotImplementedError("DSA parameter estimates require noaux_tc routing")
        if (
            model.use_bias or model.use_qkv_bias or not model.use_gated_mlp
            or model.activation is not ActivationType.SILU
            or model.norm is not NormType.RMS_NORM
        ):
            raise NotImplementedError("DSA parameter estimates require bias-free gated FFNs")
        for name in (
            "embedding_dim", "num_q_heads", "q_lora_rank", "kv_lora_rank",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
            "index_n_heads", "index_head_dim", "vocab_size", "num_experts",
        ):
            value = getattr(model, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DSA parameter dimension {name} must be a positive int")
        if model.qk_head_dim != model.qk_nope_head_dim + model.qk_rope_head_dim:
            raise ValueError("DSA qk_head_dim must equal non-RoPE plus RoPE dimensions")
        if model.num_q_heads % self._get_attn_tp_size():
            raise ValueError("DSA attention heads must be divisible by attention TP")
        # The pinned native embedding constructor pads vocabulary to multiples of 64.
        if model.vocab_size % 64 or model.vocab_size % self._get_attn_tp_size():
            raise NotImplementedError("DSA parameter estimates require unpadded vocabulary shards")
        quant = model.quantization_config
        if (
            quant is None or quant.quant_method != "fp8"
            or not quant.is_checkpoint_fp8_serialized
            or quant.activation_scheme != "dynamic"
            or quant.scale_fmt is not None or quant.quantized_operations is not None
        ):
            raise NotImplementedError(
                "DSA parameter estimates require serialized dynamic block FP8 "
                "with initial FP32 scales and native operation selection"
            )
        if quant.weight_block_size != (128, 128):
            raise NotImplementedError("DSA parameter estimates require native 128x128 FP8 blocks")
        if any(not isinstance(prefix, str) or not prefix for prefix in quant.ignored_layers):
            raise ValueError("DSA quantization exclusions must be non-empty exact prefixes")
        specs = model.get_layer_attention_specs()
        if tuple(spec.global_layer_id for spec in specs) != tuple(range(model.num_layers)):
            raise ValueError("DSA parameter estimates require every backbone layer in order")
        latest_producer = None
        for layer_id, attention in enumerate(specs):
            producer = attention.index_producer_layer_id
            if attention.family_id != "dsa_attention" or attention.variant_id not in (
                "full_index", "shared_index",
            ):
                raise NotImplementedError("DSA parameter estimates require full/shared index bindings")
            if attention.variant_id == "full_index":
                latest_producer = layer_id
            if (
                attention.global_layer_id != layer_id or type(producer) is not int
                or producer != latest_producer
            ):
                raise ValueError("DSA parameter estimates require canonical index ownership")

    def _dsa_is_quantized(self, prefix: str, packed: tuple[str, ...] = ()) -> bool:
        """Resolve native exact exclusions, including uniform packed shards."""

        ignored = self._model_config.quantization_config.ignored_layers
        if prefix in ignored:
            return False
        if packed:
            parent = prefix.rsplit(".", 1)[0]
            skipped = tuple(f"{parent}.{name}" in ignored for name in packed)
            if any(skipped) and not all(skipped):
                raise ValueError(f"DSA fused projection {prefix} has mixed shard precision")
            return not all(skipped)
        if "experts" in prefix:
            return not any("experts" in name and prefix in name for name in ignored)
        return True

    def _dsa_linear_totals(
        self, prefix: str, input_dim: int, outputs: tuple[int, ...],
        *, packed: tuple[str, ...] = (), allow_partial_last: bool = False,
    ) -> tuple[int, int]:
        """Return learned elements and initial logical storage, excluding padding."""

        elements = input_dim * sum(outputs)
        if not self._dsa_is_quantized(prefix, packed):
            return elements, 2 * elements
        block_n, block_k = self._model_config.quantization_config.weight_block_size
        checked_outputs = outputs[:-1] if allow_partial_last else outputs
        if any(width % block_n for width in checked_outputs):
            raise ValueError(f"DSA FP8 output partitions must align to block_n: {prefix}")
        scale_elements = ceil(sum(outputs) / block_n) * ceil(input_dim / block_k)
        return elements, elements + 4 * scale_elements

    def _get_dsa_attention_layer_totals(self, layer_id: int) -> tuple[int, int]:
        model = self._model_config
        attention = model.get_layer_attention_spec(layer_id)
        hidden = model.embedding_dim
        heads = model.num_q_heads // self._get_attn_tp_size()
        prefix = f"model.layers.{layer_id}.self_attn"
        if (heads * model.v_head_dim) % model.quantization_config.weight_block_size[1]:
            raise ValueError("DSA row-parallel O input must align to FP8 block_k")
        kv_b_prefix = f"{prefix}.kv_b_proj"
        if not self._dsa_is_quantized(kv_b_prefix):
            raise NotImplementedError("DSA BF16 KV-B alias storage is not represented")
        matrices = (
            self._dsa_linear_totals(
                f"{prefix}.fused_qkv_a_proj", hidden,
                (model.q_lora_rank, model.kv_lora_rank + model.qk_rope_head_dim),
                packed=("q_a_proj", "kv_a_proj_with_mqa"), allow_partial_last=True,
            ),
            self._dsa_linear_totals(
                f"{prefix}.q_b_proj", model.q_lora_rank, (heads * model.qk_head_dim,),
            ),
            self._dsa_linear_totals(
                kv_b_prefix, model.kv_lora_rank,
                (heads * (model.qk_nope_head_dim + model.v_head_dim),),
            ),
            self._dsa_linear_totals(
                f"{prefix}.o_proj", heads * model.v_head_dim, (hidden,),
            ),
        )
        norms = model.q_lora_rank + model.kv_lora_rank
        elements = sum(count for count, _ in matrices) + norms
        storage = sum(memory for _, memory in matrices) + 2 * norms
        # W_UK_T/W_UV share one BF16 dequantized KV-B backing; not learned again.
        storage += 2 * heads * (model.qk_nope_head_dim + model.v_head_dim) * model.kv_lora_rank
        if attention.variant_id == "full_index":
            query_elements, query_bytes = self._dsa_linear_totals(
                f"{prefix}.indexer.wq_b", model.q_lora_rank,
                (model.index_n_heads * model.index_head_dim,),
            )
            bf16_elements = hidden * (model.index_head_dim + model.index_n_heads)
            bf16_elements += 2 * model.index_head_dim
            elements += query_elements + bf16_elements
            storage += query_bytes + 2 * bf16_elements
        return elements, storage

    def _get_dsa_mlp_layer_totals(self, layer_id: int) -> tuple[int, int]:
        model = self._model_config
        model.get_layer_attention_spec(layer_id)
        profile = model.get_model_architecture_profile()
        routed = model.is_moe_layer(layer_id)
        kinds = (LayerKind.ROUTED, LayerKind.SHARED) if routed else (LayerKind.DENSE,)
        elements = storage = 0
        for kind in kinds:
            contract = profile.resolve_layer_contract(
                model, layer_id=layer_id, layer_kind=kind,
                attention_tp_size=self._get_attn_tp_size(),
                moe_tp_size=self._get_moe_tp_size(), expert_parallel_size=self._get_ep_size(),
            )
            expected_mode = (
                TensorParallelMode.MOE_TP if kind is LayerKind.ROUTED
                else TensorParallelMode.ATTENTION_TP
            )
            if contract.tensor_parallel_mode is not expected_mode:
                raise NotImplementedError("DSA parameter estimates support non-SP typed FFN ownership only")
            tp = contract.tensor_parallel_size
            width = contract.effective_ffn_width // tp
            block_n, block_k = model.quantization_config.weight_block_size
            if width % block_n or width % block_k:
                raise ValueError("DSA local FFN width must align to FP8 block dimensions")
            prefix = f"model.layers.{layer_id}.mlp"
            experts = 1
            if kind is LayerKind.ROUTED:
                prefix += ".experts"
                experts = model.num_experts // contract.expert_parallel_size
                if not self._dsa_is_quantized(prefix):
                    raise NotImplementedError("DSA unquantized routed expert storage is not represented")
                packed = ()
            else:
                if kind is LayerKind.SHARED:
                    prefix += ".shared_experts"
                packed = ("gate_proj", "up_proj")
            up = self._dsa_linear_totals(
                f"{prefix}.gate_up_proj", model.embedding_dim, (width, width), packed=packed,
            )
            down = self._dsa_linear_totals(
                f"{prefix}.down_proj", width, (model.embedding_dim,),
            )
            elements += experts * (up[0] + down[0])
            storage += experts * (up[1] + down[1])
        if routed:
            gate = model.embedding_dim * model.num_experts
            correction = model.num_experts  # noaux_tc owns one FP32 parameter per expert.
            elements += gate + correction
            storage += 2 * gate + 4 * correction
        return elements, storage

    def _get_dsa_other_parameters(self) -> int:
        model = self._model_config
        endpoints = 2 * model.vocab_size * model.embedding_dim // self._get_attn_tp_size()
        norms = (2 * model.num_layers + 1) * model.embedding_dim
        return endpoints + norms

    def _validate_v4_parameter_contract(self) -> None:
        """Limit V4 constructor estimates to the represented H800 SP contract."""

        model = self._model_config
        replica = self._replica_config
        if (
            self._cluster_type is not ClusterType.MONOLITHIC
            or replica.num_pipeline_stages != 1
            or getattr(replica, "device", None) != "h800"
        ):
            raise NotImplementedError("V4 parameter estimates require H800 PP1 MONOLITHIC")
        for name in (
            "attn_tensor_parallel_size", "attn_dp", "moe_tensor_parallel_size",
            "moe_expert_parallel_size",
        ):
            value = getattr(replica, name, None)
            if type(value) is not int or value <= 0:
                raise ValueError(f"V4 {name} must be a positive int")
        if (
            replica.attn_tensor_parallel_size != 8 or replica.attn_dp not in (2, 4)
            or replica.moe_tensor_parallel_size != 1
            or replica.moe_expert_parallel_size
            != replica.attn_tensor_parallel_size * replica.attn_dp
        ):
            raise NotImplementedError("V4 parameter estimates require TP8 DP2/4 SP with expert TP1")
        speculative = getattr(replica, "speculative_decoding_config", None)
        if speculative is not None and speculative.enabled:
            raise NotImplementedError("V4 parameter estimates exclude speculative decoding")
        if (
            model.tie_word_embeddings is not False
            or PrecisionType.from_torch_dtype(model.torch_dtype) is not PrecisionType.BF16
            or model.expert_dtype != "fp4" or model.topk_method != "noaux_tc"
            or model.use_bias or model.use_qkv_bias or not model.use_gated_mlp
            or model.activation is not ActivationType.SILU
            or model.norm is not NormType.RMS_NORM
        ):
            raise NotImplementedError("V4 parameter estimates require the BF16/FP8/MXFP4 source contract")
        for name in (
            "num_layers", "embedding_dim", "num_q_heads", "head_dim", "q_lora_rank",
            "o_groups", "o_lora_rank", "index_n_heads", "index_head_dim",
            "routed_mlp_hidden_dim", "share_expert_dim", "num_experts",
            "num_experts_per_tok", "vocab_size", "hc_mult",
        ):
            value = getattr(model, name, None)
            if type(value) is not int or value <= 0:
                raise ValueError(f"V4 parameter dimension {name} must be a positive int")
        if (
            type(model.num_hash_layers) is not int
            or not 0 <= model.num_hash_layers <= model.num_layers
        ):
            raise ValueError("V4 num_hash_layers must be within the backbone layer count")
        tp = self._get_attn_tp_size()
        if (
            model.num_q_heads % tp or model.o_groups % tp
            or (model.num_q_heads * model.head_dim) % model.o_groups
            or model.num_experts % self._get_ep_size()
            or model.num_experts_per_tok > model.num_experts
        ):
            raise ValueError("V4 parameter dimensions do not divide their TP/EP ownership")
        if any(
            dimension % 256
            for dimension in (model.embedding_dim, model.routed_mlp_hidden_dim)
        ):
            raise NotImplementedError("V4 expert dimensions require unrounded CUDA alignment")
        if model.vocab_size % tp:
            raise NotImplementedError("V4 logical endpoint counts require even vocabulary shards")
        quant = model.quantization_config
        if (
            quant is None or quant.quant_method != "fp8"
            or not quant.is_checkpoint_fp8_serialized
            or quant.activation_scheme != "dynamic" or quant.scale_fmt != "ue8m0"
            or quant.weight_block_size != (128, 128) or quant.ignored_layers
            or quant.quantized_operations is not None
        ):
            raise NotImplementedError("V4 parameter estimates require native FP8 UE8M0 block scales")
        block_n, block_k = quant.weight_block_size
        local_q_width = model.num_q_heads // tp * model.head_dim
        local_o_rank = model.o_groups // tp * model.o_lora_rank
        fp8_axes = (
            ("q_lora_rank output", model.q_lora_rank, block_n),
            ("local Q output", local_q_width, block_n),
            ("local output rank N", local_o_rank, block_n),
            ("local output rank K", local_o_rank, block_k),
            ("shared expert merged width", model.share_expert_dim, block_n),
            ("shared expert down input", model.share_expert_dim, block_k),
        )
        for axis_name, dimension, block in fp8_axes:
            if dimension % block:
                raise ValueError(
                    f"V4 FP8 {axis_name} ({dimension}) must align to block {block}"
                )
        specs = model.get_layer_attention_specs()
        if tuple(spec.global_layer_id for spec in specs) != tuple(range(model.num_layers)):
            raise ValueError("V4 parameter estimates require every backbone layer in order")
        variants = {0: "uncompressed", 4: "csa", 128: "hca"}
        for spec, ratio in zip(specs, model.compress_ratios):
            if (
                spec.family_id != "deepseek_v4_attention"
                or spec.variant_id != variants.get(ratio)
            ):
                raise ValueError("V4 parameter estimates require canonical compression bindings")
        if model.get_moe_layer_ids() != list(range(model.num_layers)):
            raise NotImplementedError("V4 parameter estimates require all backbone FFNs routed/shared")

    def _v4_fp8_linear_totals(self, output_dim: int, input_dim: int) -> tuple[int, int]:
        """Count FP8 coefficients and initial one-byte UE8M0 block scales."""

        block_n, block_k = self._model_config.quantization_config.weight_block_size
        elements = output_dim * input_dim
        scales = ceil(output_dim / block_n) * ceil(input_dim / block_k)
        return elements, elements + scales

    def _get_v4_compressor_totals(self, width: int, ratio: int) -> tuple[int, int]:
        overlap = 1 + (ratio == 4)
        projection = 2 * overlap * width * self._model_config.embedding_dim
        ape = ratio * overlap * width
        norm = width
        return projection + ape + norm, 2 * (projection + norm) + 4 * ape

    def _get_v4_attention_layer_totals(self, layer_id: int) -> tuple[int, int]:
        model = self._model_config
        attention = model.get_layer_attention_spec(layer_id)
        tp = self._get_attn_tp_size()
        hidden, width = model.embedding_dim, model.head_dim
        heads = model.num_q_heads // tp
        padded_heads = 64 if heads <= 64 else 128
        if heads > 128:
            raise NotImplementedError("V4 FlashMLA source sink supports at most 128 local heads")
        matrices = (
            self._v4_fp8_linear_totals(model.q_lora_rank + width, hidden),
            self._v4_fp8_linear_totals(heads * width, model.q_lora_rank),
            self._v4_fp8_linear_totals(
                model.o_groups * model.o_lora_rank // tp,
                model.num_q_heads * width // model.o_groups,
            ),
            self._v4_fp8_linear_totals(hidden, model.o_groups * model.o_lora_rank // tp),
        )
        norms = model.q_lora_rank + width
        elements = sum(count for count, _ in matrices) + norms + heads
        storage = sum(memory for _, memory in matrices) + 2 * norms + 4 * padded_heads
        if attention.variant_id in ("csa", "hca"):
            ratio = model.compress_ratios[layer_id]
            count, memory = self._get_v4_compressor_totals(width, ratio)
            elements += count
            storage += memory
        if attention.variant_id == "csa":
            query = self._v4_fp8_linear_totals(
                model.index_n_heads * model.index_head_dim, model.q_lora_rank
            )
            index_weights = model.index_n_heads * hidden
            compressor = self._get_v4_compressor_totals(model.index_head_dim, ratio)
            elements += query[0] + index_weights + compressor[0]
            storage += query[1] + 2 * index_weights + compressor[1]
        return elements, storage

    def _get_v4_mlp_layer_totals(self, layer_id: int) -> tuple[int, int]:
        model = self._model_config
        model.get_layer_attention_spec(layer_id)
        profile = model.get_model_architecture_profile()
        contract_args = dict(
            layer_id=layer_id, attention_tp_size=self._get_attn_tp_size(),
            moe_tp_size=self._get_moe_tp_size(), expert_parallel_size=self._get_ep_size(),
            attention_dp_size=self._replica_config.attn_dp,
            num_pipeline_stages=self._replica_config.num_pipeline_stages,
        )
        routed = profile.resolve_layer_contract(model, layer_kind=LayerKind.ROUTED, **contract_args)
        shared = profile.resolve_layer_contract(model, layer_kind=LayerKind.SHARED, **contract_args)
        if (
            routed.tensor_parallel_mode is not TensorParallelMode.MOE_TP
            or routed.tensor_parallel_size != 1
            or shared.tensor_parallel_mode is not TensorParallelMode.REPLICATED
            or shared.tensor_parallel_size != 1
        ):
            raise NotImplementedError("V4 source storage requires routed TP1 and replicated SP shared weights")
        local_experts = model.num_experts // routed.expert_parallel_size
        routed_elements = local_experts * 3 * model.embedding_dim * routed.effective_ffn_width
        # MXFP4 stores two coefficients per byte and one UE8M0 scale per 32.
        routed_bytes = routed_elements // 2 + routed_elements // 32
        shared_up = self._v4_fp8_linear_totals(
            2 * shared.effective_ffn_width, model.embedding_dim
        )
        shared_down = self._v4_fp8_linear_totals(
            model.embedding_dim, shared.effective_ffn_width
        )
        gate = model.num_experts * model.embedding_dim
        elements = routed_elements + shared_up[0] + shared_down[0] + gate
        storage = routed_bytes + shared_up[1] + shared_down[1] + 2 * gate
        if layer_id < model.num_hash_layers:
            storage += 4 * model.vocab_size * model.num_experts_per_tok
        else:
            elements += model.num_experts
            storage += 4 * model.num_experts
        return elements, storage

    def _get_v4_mhc_layer_elements(self) -> int:
        model = self._model_config
        mixture = (2 + model.hc_mult) * model.hc_mult
        return 2 * (mixture * model.hc_mult * model.embedding_dim + mixture + 3)

    def _get_v4_other_totals(self) -> tuple[int, int]:
        model = self._model_config
        tp = self._get_attn_tp_size()
        endpoints = 2 * model.vocab_size * model.embedding_dim // tp
        padded_vocab = ceil(model.vocab_size / 64) * 64
        endpoint_storage = 4 * padded_vocab * model.embedding_dim // tp
        norms = (2 * model.num_layers + 1) * model.embedding_dim
        mhc = model.num_layers * self._get_v4_mhc_layer_elements()
        mhc_head = model.hc_mult**2 * model.embedding_dim + model.hc_mult + 1
        return endpoints + norms + mhc + mhc_head, endpoint_storage + 2 * norms + 4 * (mhc + mhc_head)

    def get_conditional_mhc_broadcast_memory_bytes(self) -> int:
        """Return source-required derived storage; presence is native-unverified.

        This storage is excluded from constructor bytes and coefficient counts.
        The first-layer finalizer must actually run before it exists natively.
        """

        if not self._uses_v4_parameter_contract:
            return 0
        model = self._model_config
        mixture = (2 + model.hc_mult) * model.hc_mult
        return 4 * mixture * model.embedding_dim

    def get_num_attention_params_per_layer(self, layer_id: int = 0) -> int:
        """Return learned elements; DSA defaults to canonical layer0's full indexer."""

        if self._uses_v4_parameter_contract:
            return self._get_v4_attention_layer_totals(layer_id)[0]
        if self._uses_dsa_parameter_contract:
            return self._get_dsa_attention_layer_totals(layer_id)[0]
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

    def _get_pipeline_stage_layer_ids(self, stage_id: int) -> range:
        first_layer_id = int(stage_id) * self._num_layers_per_pipeline_stage
        return range(
            first_layer_id,
            first_layer_id + self._num_layers_per_pipeline_stage,
        )

    def get_attention_stage_layer_counts(self) -> tuple[tuple[int, int], ...]:
        """Return ``(GDN layers, full-attention layers)`` for every PP stage."""

        if self._cluster_type == ClusterType.DECODE_FFN:
            return ((0, 0),)
        stage_counts = []
        for stage_id in range(int(self._replica_config.num_pipeline_stages)):
            gdn_count = 0
            full_count = 0
            for layer_id in self._get_pipeline_stage_layer_ids(stage_id):
                if self._model_config.is_gdn_layer(layer_id):
                    gdn_count += 1
                else:
                    full_count += 1
            stage_counts.append((gdn_count, full_count))
        return tuple(stage_counts)

    def _get_attention_stage_totals(self) -> tuple[tuple[int, int], ...]:
        """Return ``(parameter elements, parameter bytes)`` per PP stage."""

        if self._uses_v4_parameter_contract:
            layer_totals = tuple(
                self._get_v4_attention_layer_totals(layer_id)
                for layer_id in self._get_pipeline_stage_layer_ids(0)
            )
            return ((sum(count for count, _ in layer_totals),
                     sum(memory for _, memory in layer_totals)),)
        if self._uses_dsa_parameter_contract:
            layer_totals = tuple(
                self._get_dsa_attention_layer_totals(layer_id)
                for layer_id in self._get_pipeline_stage_layer_ids(0)
            )
            return ((sum(count for count, _ in layer_totals),
                     sum(memory for _, memory in layer_totals)),)
        if self._cluster_type == ClusterType.DECODE_FFN:
            return ((0, 0),)
        full_attention_params = self.get_num_attention_params_per_layer()
        gdn_params = self.get_num_gdn_params_per_layer()
        full_attention_bytes = 2 * full_attention_params
        gdn_bytes = self.get_gdn_parameter_memory_bytes_per_layer()
        totals = []
        for gdn_count, full_count in self.get_attention_stage_layer_counts():
            totals.append(
                (
                    gdn_count * gdn_params + full_count * full_attention_params,
                    gdn_count * gdn_bytes + full_count * full_attention_bytes,
                )
            )
        return tuple(totals)

    def get_resident_attention_stage_id(self) -> int:
        """Return the stage selected by resident attention parameter memory."""
        totals = self._get_attention_stage_totals()
        return max(range(len(totals)), key=lambda stage_id: totals[stage_id][1])

    def get_num_attention_parameters_per_device(self) -> int:
        """Return the largest resident attention/GDN shard across PP stages."""

        return max(elements for elements, _ in self._get_attention_stage_totals())

    def get_attention_parameter_memory_per_device_bytes(self) -> int:
        """Return resident attention/GDN bytes for the largest PP stage."""

        return max(memory_bytes for _, memory_bytes in self._get_attention_stage_totals())

    def get_num_gdn_params_per_layer(self) -> int:
        """Return local Qwen GDN parameter count for one layer."""

        if self._cluster_type == ClusterType.DECODE_FFN:
            return 0
        gdn_config = self._model_config.get_gdn_config()
        if gdn_config is None:
            return 0
        tp_size = self._get_attn_tp_size()
        sharded_dimensions = (
            gdn_config.conv_dim,
            gdn_config.key_dim,
            gdn_config.value_dim,
            gdn_config.num_key_heads,
            gdn_config.num_value_heads,
        )
        if any(int(dimension) % tp_size for dimension in sharded_dimensions):
            raise ValueError(
                "GDN dimensions must be divisible by attention TP size, "
                f"tp={tp_size}, dimensions={sharded_dimensions}"
            )

        hidden_size = int(self._model_config.embedding_dim)
        local_conv_dim = int(gdn_config.conv_dim) // tp_size
        local_key_dim = int(gdn_config.key_dim) // tp_size
        local_value_dim = int(gdn_config.value_dim) // tp_size
        local_value_heads = int(gdn_config.num_value_heads) // tp_size
        return (
            local_conv_dim * int(gdn_config.conv_kernel_size)
            + hidden_size * (2 * local_key_dim + 2 * local_value_dim)
            + hidden_size * (2 * local_value_heads)
            + hidden_size * local_value_dim
            + int(gdn_config.value_head_dim)
            + 2 * local_value_heads
        )

    def get_gdn_parameter_memory_bytes_per_layer(self) -> int:
        """Return local GDN bytes with FP32 ``A_log`` accounted separately."""

        num_parameters = self.get_num_gdn_params_per_layer()
        if num_parameters == 0:
            return 0
        gdn_config = self._model_config.get_gdn_config()
        local_value_heads = int(gdn_config.num_value_heads) // self._get_attn_tp_size()
        # A_log is FP32; other pinned runtime GDN weights use BF16.
        return 2 * (num_parameters - local_value_heads) + 4 * local_value_heads

    def get_num_mlp_params_per_layer(self, layer_id: int = 0) -> int:
        if self._uses_v4_parameter_contract:
            return self._get_v4_mlp_layer_totals(layer_id)[0]
        if self._uses_dsa_parameter_contract:
            return self._get_dsa_mlp_layer_totals(layer_id)[0]
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

    def get_num_parameters_per_layer(self, layer_id: int = 0) -> int:
        if self._uses_v4_parameter_contract:
            return (
                self.get_num_attention_params_per_layer(layer_id)
                + self.get_num_mlp_params_per_layer(layer_id)
                + 2 * self._model_config.embedding_dim
                + self._get_v4_mhc_layer_elements()
            )
        if self._uses_dsa_parameter_contract:
            return (
                self.get_num_attention_params_per_layer(layer_id)
                + self.get_num_mlp_params_per_layer(layer_id)
                + 2 * self._model_config.embedding_dim
            )
        return (
            self.get_num_attention_params_per_layer()
            + self.get_num_mlp_params_per_layer()
        )

    def get_num_parameters_per_device(self) -> int:
        if self._uses_v4_parameter_contract:
            return (
                self.get_num_attention_parameters_per_device()
                + self.get_num_mlp_parameters_per_device()
                + self._get_v4_other_totals()[0]
            )
        if self._uses_dsa_parameter_contract:
            return (
                self.get_num_attention_parameters_per_device()
                + self.get_num_mlp_parameters_per_device()
                + self._get_dsa_other_parameters()
            )
        num_attention_parameters = self.get_num_attention_parameters_per_device()
        num_mlp_parameters = self.get_num_mlp_parameters_per_device()
        return (
            num_attention_parameters
            + num_mlp_parameters
            + self.get_num_mtp_parameters_per_device()
        )

    def get_num_mlp_parameters_per_device(self) -> int:
        if self._uses_v4_parameter_contract:
            return sum(
                self._get_v4_mlp_layer_totals(layer_id)[0]
                for layer_id in self._get_pipeline_stage_layer_ids(0)
            )
        if self._uses_dsa_parameter_contract:
            return sum(
                self._get_dsa_mlp_layer_totals(layer_id)[0]
                for layer_id in self._get_pipeline_stage_layer_ids(0)
            )
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

    def get_mlp_parameter_memory_per_device_bytes(self) -> int:
        """Return source-defined V4/DSA FFN bytes or the legacy estimate."""

        if self._uses_v4_parameter_contract:
            return sum(
                self._get_v4_mlp_layer_totals(layer_id)[1]
                for layer_id in self._get_pipeline_stage_layer_ids(0)
            )
        if self._uses_dsa_parameter_contract:
            return sum(
                self._get_dsa_mlp_layer_totals(layer_id)[1]
                for layer_id in self._get_pipeline_stage_layer_ids(0)
            )
        return 2 * self.get_num_mlp_parameters_per_device()

    def get_parameter_memory_per_device_bytes(self) -> int:
        """Return source estimates or the accepted legacy D57 policy.

        DSA bytes include initial FP32 scales and one derived BF16 KV-B backing.
        Final backend padding/scale layouts and native unique storage remain
        unqualified; these source estimates do not establish runtime fit.
        V4 returns source-declared constructor bytes, including UE8M0 scales
        and integer auxiliaries, excluding conditional mHC broadcast storage.
        V4 native unique resident storage remains unverified.
        """

        if self._uses_v4_parameter_contract:
            return (
                self.get_attention_parameter_memory_per_device_bytes()
                + self.get_mlp_parameter_memory_per_device_bytes()
                + self._get_v4_other_totals()[1]
            )
        if self._uses_dsa_parameter_contract:
            return (
                self.get_attention_parameter_memory_per_device_bytes()
                + self.get_mlp_parameter_memory_per_device_bytes()
                + 2 * self._get_dsa_other_parameters()
            )

        # Qwen3.8 routed experts physically use MXFP4, but the integration
        # keeps the established 2-byte MLP/MoE/MTP approximation, which can
        # overestimate resident model-weight memory. Capacity checks remain
        # active; no MXFP4 capacity helper is introduced here.
        return (
            self.get_attention_parameter_memory_per_device_bytes()
            + 2 * self.get_num_mlp_parameters_per_device()
            + 2 * self.get_num_mtp_parameters_per_device()
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
        if getattr(proposer_model_config, "tie_word_embeddings", None) is not False:
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
