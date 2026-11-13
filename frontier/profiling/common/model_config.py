from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from frontier.attention.model_binding import (
    resolve_runtime_attention_family,
    resolve_attention_topology,
    resolve_attention_head_partition,
)
from frontier.attention.gdn import (
    GatedDeltaNetConfig,
    LayerAttentionSpec,
)
from frontier.attention.ops import AttentionMemoryLayout
from frontier.config.model_config import (
    BaseModelConfig,
    FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST,
    NON_FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST,
    QuantizationConfig,
    _infer_attn_output_gate_from_hf_config,
    _infer_share_expert_dim_from_hf_config,
    _infer_use_qk_norm_from_hf_config,
    _normalize_hf_text_config,
)
from frontier.config.utils import dataclass_to_dict
from frontier.model_architectures import get_model_architecture_profile
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.types import ActivationType, NormType


class ModelConfig:
    def __init__(
        self,
        name: str,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        embedding_dim: int,
        mlp_hidden_dim: int,
        max_position_embeddings: int,
        use_gated_mlp: bool,
        use_bias: bool,
        use_qkv_bias: bool,
        activation: ActivationType,
        norm: NormType,
        post_attn_norm: bool,
        vocab_size: int,
        is_neox_style: Optional[bool] = True,
        rope_theta: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        partial_rotary_factor: float = 1.0,
        no_tensor_parallel: bool = False,
        is_moe: bool = False,
        num_experts: int = 0,
        num_experts_per_tok: int = 0,
        moe_layers_enum: Optional[str] = None,
        use_qk_norm: bool = False,
        attn_output_gate: bool = False,
        rms_norm_eps: float = 1e-6,
        dtype: Optional[str] = None,
        model_type: Optional[str] = None,
        model_architecture_profile: Optional[str] = None,
        architectures: Optional[tuple[str, ...] | list[str]] = None,
        fused_add_norm_capability: Optional[bool] = None,
        # Architecture-specific structural fields
        model_arch: Optional[str] = None,
        share_expert_dim: Optional[int] = None,
        share_q_dim: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_mla: bool = False,
        use_mfa: bool = False,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        qk_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        index_topk: Optional[int] = None,
        index_n_heads: Optional[int] = None,
        index_head_dim: Optional[int] = None,
        index_topk_freq: Optional[int] = None,
        index_topk_pattern: Optional[str] = None,
        index_skip_topk_offset: Optional[int] = None,
        indexer_rope_interleave: Optional[bool] = None,
        compress_ratios: Optional[tuple[int, ...] | list[int]] = None,
        sliding_window: Optional[int] = None,
        num_hash_layers: Optional[int] = None,
        hc_mult: Optional[int] = None,
        hc_sinkhorn_iters: Optional[int] = None,
        hc_eps: Optional[float] = None,
        o_groups: Optional[int] = None,
        o_lora_rank: Optional[int] = None,
        compress_rope_theta: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
        expert_dtype: Optional[str] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        projector_bias: Optional[bool] = None,
        sparse_config: Optional[Dict[str, Any]] = None,
        attention_other_setting: Optional[Dict[str, Any]] = None,
        att_impl_type: Optional[str] = None,
        num_nextn_predict_layers: Optional[int] = None,
        mtp_layer_types: Optional[tuple[str, ...]] = None,
        layer_rope_thetas: Optional[tuple[float, ...]] = None,
        partial_rotary_factors: Optional[tuple[float, ...]] = None,
        swiglu_limits: Optional[tuple[float, ...]] = None,
        swiglu_limits_shared: Optional[tuple[float, ...]] = None,
        norm_dtype: Optional[str] = None,
        fp32_residual_connection: Optional[bool] = None,
        need_fp32_gate: Optional[bool] = None,
        use_moe_router_bias: Optional[bool] = None,
        moe_router_activation: Optional[str] = None,
        moe_router_scaling_factor: Optional[float] = None,
        zero_centered: Optional[bool] = None,
        yarn_only_types: Optional[tuple[str, ...]] = None,
        use_rope_layers: Optional[tuple[bool, ...]] = None,
        swa_num_attention_heads: Optional[int] = None,
        norm_expert_weight: bool = False,
        layer_types: Optional[tuple[str, ...] | list[str]] = None,
        full_attention_interval: Optional[int] = None,
        linear_conv_kernel_dim: Optional[int] = None,
        linear_key_head_dim: Optional[int] = None,
        linear_value_head_dim: Optional[int] = None,
        linear_num_key_heads: Optional[int] = None,
        linear_num_value_heads: Optional[int] = None,
        gdn_output_gate_type: str = GatedDeltaNetConfig.output_gate_type,
        # Quantization config for metadata tracking
        quantization_config: Optional[QuantizationConfig] = None,
        # Whether lm_head shares weights with embed_tokens (HF standard field)
        tie_word_embeddings: Optional[bool] = True,
        # Typed FFN widths. Keep the legacy scalar for compatibility while
        # preserving dense and routed domains independently.
        dense_mlp_hidden_dim: Optional[int] = None,
        routed_mlp_hidden_dim: Optional[int] = None,
        topk_method: Optional[str] = None,
        scoring_func: Optional[str] = None,
        norm_topk_prob: Optional[bool] = None,
        routed_scaling_factor: Optional[float] = None,
    ):
        self.name = name
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dense_mlp_hidden_dim = dense_mlp_hidden_dim
        self.routed_mlp_hidden_dim = routed_mlp_hidden_dim
        for field_name in ("dense_mlp_hidden_dim", "routed_mlp_hidden_dim"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(
                    f"{field_name} must be a positive int when provided, got {value!r}"
                )
        self.max_position_embeddings = max_position_embeddings
        self.use_gated_mlp = use_gated_mlp
        self.vocab_size = vocab_size
        self.use_bias = use_bias
        self.use_qkv_bias = use_qkv_bias
        self.activation = str(activation)
        self.norm = str(norm)
        self.post_attn_norm = post_attn_norm
        self.no_tensor_parallel = no_tensor_parallel
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.is_neox_style = is_neox_style

        # MoE-specific parameters
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layers_enum = moe_layers_enum

        # QK-norm support (for Qwen3, Gemma3, OLMo2, etc.)
        self.use_qk_norm = use_qk_norm
        self.attn_output_gate = bool(attn_output_gate)
        self.rms_norm_eps = rms_norm_eps
        self._dtype = self._parse_dtype(dtype)
        self.model_type = (
            str(model_type).lower() if model_type is not None else None
        )
        self.model_architecture_profile = (
            str(model_architecture_profile).lower()
            if model_architecture_profile is not None
            else None
        )
        self.architectures = (
            tuple(str(value) for value in architectures)
            if architectures is not None
            else ()
        )
        if fused_add_norm_capability is not None and not isinstance(
            fused_add_norm_capability, bool
        ):
            raise ValueError(
                "fused_add_norm_capability must be bool or None, "
                f"got {type(fused_add_norm_capability)}"
            )
        self.fused_add_norm_capability = fused_add_norm_capability

        # Architecture-specific structural fields
        self.model_arch = model_arch
        self.share_expert_dim = share_expert_dim
        self.share_q_dim = share_q_dim
        # head_dim: If provided, use it; otherwise compute from embedding_dim/num_q_heads
        self._head_dim = head_dim
        self.use_mla = bool(use_mla)
        self.use_mfa = bool(use_mfa)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk_freq = index_topk_freq
        self.index_topk_pattern = index_topk_pattern
        self.index_skip_topk_offset = index_skip_topk_offset
        self.indexer_rope_interleave = indexer_rope_interleave
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.compress_ratios = (
            tuple(compress_ratios) if compress_ratios is not None else None
        )
        self.sliding_window = sliding_window
        self.num_hash_layers = num_hash_layers
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.compress_rope_theta = compress_rope_theta
        self.swiglu_limit = swiglu_limit
        self.expert_dtype = expert_dtype
        self.vision_config = vision_config
        self.projector_bias = projector_bias
        self.sparse_config = sparse_config
        self.attention_other_setting = attention_other_setting
        self.att_impl_type = att_impl_type
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_layer_types = tuple(mtp_layer_types) if mtp_layer_types is not None else None
        self.layer_rope_thetas = tuple(layer_rope_thetas) if layer_rope_thetas is not None else None
        self.partial_rotary_factors = tuple(partial_rotary_factors) if partial_rotary_factors is not None else None
        self.swiglu_limits = tuple(swiglu_limits) if swiglu_limits is not None else None
        self.swiglu_limits_shared = tuple(swiglu_limits_shared) if swiglu_limits_shared is not None else None
        self.norm_dtype = norm_dtype
        self.fp32_residual_connection = fp32_residual_connection
        self.need_fp32_gate = need_fp32_gate
        self.use_moe_router_bias = use_moe_router_bias
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.zero_centered = zero_centered
        self.yarn_only_types = tuple(yarn_only_types) if yarn_only_types is not None else None
        self.use_rope_layers = tuple(use_rope_layers) if use_rope_layers is not None else None
        self.swa_num_attention_heads = swa_num_attention_heads
        self.norm_expert_weight = norm_expert_weight
        self.layer_types = (
            tuple(str(value) for value in layer_types)
            if layer_types is not None
            else None
        )
        self.full_attention_interval = (
            int(full_attention_interval)
            if full_attention_interval is not None
            else None
        )
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.gdn_output_gate_type = str(gdn_output_gate_type)
        # Keep resolution lazy to preserve the existing profiling config
        # construction and binder validation order.
        self._layer_attention_specs: tuple[LayerAttentionSpec, ...] | None = None
        self._gdn_config_cache: GatedDeltaNetConfig | None = None

        # Quantization config for metadata tracking
        if quantization_config is not None and not isinstance(
            quantization_config, QuantizationConfig
        ):
            if isinstance(quantization_config, dict):
                quantization_config = QuantizationConfig.from_dict(quantization_config)
            else:
                raise TypeError(
                    "quantization_config must be QuantizationConfig or dict, "
                    f"got {type(quantization_config)}"
                )
        self.quantization_config = quantization_config

        # Whether lm_head shares weights with embed_tokens
        self.tie_word_embeddings = (
            True if tie_word_embeddings is None else bool(tie_word_embeddings)
        )

        if self.use_mla and self.use_mfa:
            raise ValueError("use_mla and use_mfa are mutually exclusive")

        assert self.norm in ["layer_norm", "rms_norm"]
        assert self.activation in ["gelu", "silu"]

        if self.use_gated_mlp:
            assert self.activation == "silu"
        else:
            assert self.activation == "gelu"

        architecture_profile = self.get_model_architecture_profile()
        architecture_profile.validate_structural_requirements(self)

        if self.use_mla:
            missing_mla_fields = [
                field_name
                for field_name in (
                    "kv_lora_rank",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                )
                if getattr(self, field_name) is None
            ]
            if missing_mla_fields:
                raise ValueError(
                    "MLA profiling ModelConfig requires fields: "
                    f"{missing_mla_fields}"
                )
            qk_nope_head_dim = self.qk_nope_head_dim
            qk_rope_head_dim = self.qk_rope_head_dim
            assert qk_nope_head_dim is not None
            assert qk_rope_head_dim is not None
            if self.qk_head_dim is None:
                self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            expected_qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            if self.qk_head_dim != expected_qk_head_dim:
                raise ValueError(
                    "qk_head_dim must equal qk_nope_head_dim + "
                    "qk_rope_head_dim for MLA. "
                    f"qk_head_dim={self.qk_head_dim}, "
                    f"expected={expected_qk_head_dim}"
                )

    def supports_share_expert(self) -> bool:
        """Check if the model uses share_expert in the FFN path."""
        return self.get_model_architecture_profile().supports_share_expert(self)

    def get_model_architecture_profile(self):
        """Return plugin-style model architecture semantics for this config."""
        return get_model_architecture_profile(self)

    @property
    def uses_fused_add_norm(self) -> bool:
        """Whether the model uses fused add+norm kernel (RMSNorm).

        Decision source priority:
        1) Explicit capability flag from model metadata (`uses_fused_add_norm`)
        2) Model-type allowlist/denylist
        3) Legacy norm heuristic (backward compatibility)
        """
        if self.fused_add_norm_capability is not None:
            return self.fused_add_norm_capability
        if self.model_type in NON_FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST:
            return False
        if self.model_type in FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST:
            return True
        return self.norm == "rms_norm"

    def get_head_dim(self) -> int:
        """Get the head dimension, either explicit or computed."""
        if self._head_dim is not None:
            return self._head_dim
        return self.embedding_dim // self.num_q_heads

    def get_attention_family(self):
        """Return the family used by model-wide profiling cache semantics."""
        return resolve_runtime_attention_family(self)

    def get_gdn_config(self) -> Optional[GatedDeltaNetConfig]:
        """Return the shape normalized together with the model-owned schedule."""
        self.get_layer_attention_specs()
        return self._gdn_config_cache

    def get_layer_attention_specs(self) -> tuple[LayerAttentionSpec, ...]:
        if self._layer_attention_specs is None:
            self._layer_attention_specs, self._gdn_config_cache = resolve_attention_topology(self)
        return self._layer_attention_specs

    def get_layer_attention_spec(self, global_layer_id: int) -> LayerAttentionSpec:
        if type(global_layer_id) is not int:
            raise ValueError(
                f"global_layer_id must be an int, got {global_layer_id!r}"
            )
        specs = self.get_layer_attention_specs()
        if global_layer_id < 0 or global_layer_id >= len(specs):
            raise ValueError(
                f"global_layer_id {global_layer_id} out of range "
                f"[0, {len(specs)})"
            )
        return specs[global_layer_id]

    def get_num_gdn_layers(self) -> int:
        return sum(spec.is_gdn for spec in self.get_layer_attention_specs())

    def get_num_full_attention_layers(self) -> int:
        return sum(
            spec.is_full_attention for spec in self.get_layer_attention_specs()
        )

    def is_gdn_layer(self, global_layer_id: int) -> bool:
        return self.get_layer_attention_spec(global_layer_id).is_gdn

    def get_runtime_num_kv_heads(self) -> int:
        """Return runtime KV heads for cache allocation."""
        family = self.get_attention_family()
        return family.resolve_runtime_num_kv_heads(self)

    def get_runtime_head_size(self) -> int:
        """Return vLLM runtime KV-cache head size."""
        family = self.get_attention_family()
        return family.resolve_runtime_head_size(self)

    def get_qk_head_dim(self) -> int:
        """Return the full QK head dimension."""
        if self.use_mla:
            if self.qk_head_dim is None:
                raise ValueError("MLA qk_head_dim is not configured")
            return self.qk_head_dim
        return self.get_head_dim()

    def get_moe_layer_ids(self) -> List[int]:
        """Return sorted MoE layer IDs for this model config.

        Semantics mirror BaseModelConfig.get_moe_layer_ids():
        - Non-MoE models: empty list
        - MoE without moe_layers_enum: all layers are MoE
        - MoE with moe_layers_enum: parse explicit IDs, keep within [0, num_layers)
        """
        if not self.is_moe:
            return []
        from frontier.model_architectures import parse_moe_layer_ids

        return list(parse_moe_layer_ids(self.moe_layers_enum, self.num_layers))

    def get_quant_signature(self) -> str:
        """Get the quantization signature for this model config.

        Returns:
            A stable string identifier for the quantization configuration.
            Returns "none" if no quantization is configured.
        """
        if self.quantization_config is None:
            return "none"
        return self.quantization_config.get_quant_signature()

    @staticmethod
    def from_model_name(model_name: str):
        model_config: BaseModelConfig = BaseModelConfig.create_from_name(model_name)
        model_config_dict = asdict(model_config)

        # Capture torch_dtype from BaseModelConfig before it gets removed.
        # BaseModelConfig._create_from_hf_json already parses torch_dtype correctly.
        base_torch_dtype: str | None = model_config_dict.get('torch_dtype', None)

        # Extract quantization_config before removing it from dict
        # We need to preserve it for metadata tracking
        quant_config_dict = model_config_dict.pop('quantization_config', None)
        quantization_config = None
        if quant_config_dict is not None:
            quantization_config = QuantizationConfig.from_dict(quant_config_dict)

        # Remove fields not supported by profiling ModelConfig
        # These are BaseModelConfig-specific fields not needed for profiling
        unsupported_fields = [
            '_model_name',
            '_moe_layer_ids_cache',
            '_layer_attention_specs_cache',
            '_gdn_config_cache',
            'torch_dtype',
        ]
        for field in unsupported_fields:
            model_config_dict.pop(field, None)
        if model_config_dict.get("rms_norm_eps") is None:
            model_config_dict.pop("rms_norm_eps", None)
        if model_config_dict.get("tie_word_embeddings") is None:
            model_config_dict.pop("tie_word_embeddings", None)

        # Read additional fields from JSON config that are not in BaseModelConfig
        # These fields are needed for profiling but not part of the core model config
        import os
        import json
        safe_name = model_name.replace("/", "__")
        json_path = os.path.join("data", "config", "models", f"{safe_name}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_cfg = _normalize_hf_text_config(json.load(f))
            # QK-norm support (for Qwen3, Gemma3, OLMo2, etc.)
            model_config_dict['use_qk_norm'] = _infer_use_qk_norm_from_hf_config(
                json_cfg
            )
            model_config_dict['attn_output_gate'] = _infer_attn_output_gate_from_hf_config(
                json_cfg
            )
            # RMS norm epsilon
            model_config_dict['rms_norm_eps'] = json_cfg.get('rms_norm_eps', 1e-6)
            # Precision dtype: prefer torch_dtype (HF standard), fallback to dtype
            dtype_value = json_cfg.get('torch_dtype') or json_cfg.get('dtype')
            if dtype_value is None:
                # Use the value already parsed by BaseModelConfig
                dtype_value = base_torch_dtype
            if dtype_value is None:
                raise ValueError(
                    f"Model config for '{model_name}' has neither 'torch_dtype' nor "
                    f"'dtype' in JSON ({json_path}) or BaseModelConfig. "
                    f"Please specify the model precision."
                )
            model_config_dict['dtype'] = dtype_value
            # Model type used by architecture profile matching
            model_config_dict['model_type'] = json_cfg.get(
                'model_type', model_config_dict.get('model_type')
            )
            # Explicit fused-add capability override
            explicit_fused_add_norm = json_cfg.get('uses_fused_add_norm')
            if explicit_fused_add_norm is not None:
                if not isinstance(explicit_fused_add_norm, bool):
                    raise ValueError(
                        "uses_fused_add_norm must be bool when present in model config, "
                        f"got {type(explicit_fused_add_norm)} for '{model_name}'"
                    )
                model_config_dict['fused_add_norm_capability'] = explicit_fused_add_norm
            # Architecture-specific structural fields
            model_config_dict['model_arch'] = json_cfg.get(
                'model_arch', model_config_dict.get('model_arch')
            )
            model_config_dict['share_expert_dim'] = (
                _infer_share_expert_dim_from_hf_config(json_cfg)
            )
            model_config_dict['share_q_dim'] = json_cfg.get('share_q_dim')
            model_config_dict['head_dim'] = json_cfg.get('head_dim')
            for field_name in [
                'use_mla',
                'use_mfa',
                'q_lora_rank',
                'kv_lora_rank',
                'qk_nope_head_dim',
                'qk_rope_head_dim',
                'qk_head_dim',
                'v_head_dim',
                'linear_conv_kernel_dim',
                'linear_key_head_dim',
                'linear_value_head_dim',
                'linear_num_key_heads',
                'linear_num_value_heads',
            ]:
                if field_name in json_cfg:
                    model_config_dict[field_name] = json_cfg[field_name]
            if (
                'use_mla' not in model_config_dict
                and str(json_cfg.get('model_type', '')).lower()
                in {'deepseek_v2', 'deepseek_v3', 'deepseek_mtp', 'kimi_k2'}
                and json_cfg.get('kv_lora_rank') is not None
            ):
                model_config_dict['use_mla'] = True
            # Whether lm_head shares weights with embed_tokens (HF standard)
            model_config_dict['tie_word_embeddings'] = json_cfg.get(
                'tie_word_embeddings', True
            )
            # Only Step opts into this source routing field; legacy payloads stay unchanged.
            if model_config_dict.get("model_type") == "step5":
                model_config_dict["norm_expert_weight"] = json_cfg["norm_expert_weight"]
            # Load quantization_config from JSON if not already loaded from BaseModelConfig
            if quantization_config is None and 'quantization_config' in json_cfg:
                quantization_config = QuantizationConfig.from_dict(json_cfg['quantization_config'])
        else:
            # No JSON config file found; use BaseModelConfig's torch_dtype
            if base_torch_dtype is None:
                raise ValueError(
                    f"Model config for '{model_name}' has no JSON config at "
                    f"data/config/models/{model_name.replace('/', '__')}.json and "
                    f"BaseModelConfig has no torch_dtype. "
                    f"Please specify the model precision."
                )
            model_config_dict['dtype'] = base_torch_dtype

        return ModelConfig(model_name, quantization_config=quantization_config, **model_config_dict)

    def get_num_q_heads(self, parallel_config: ParallelConfig):
        tp_size = parallel_config.tensor_parallel_size
        if tp_size <= 0:
            raise ValueError(
                f"tensor_parallel_size must be positive, got {tp_size}"
            )
        if self.num_q_heads <= 0:
            raise ValueError(f"num_q_heads must be positive, got {self.num_q_heads}")
        if self.num_q_heads % tp_size != 0:
            raise ValueError(
                f"Q heads must be divisible by tensor_parallel_size. "
                f"num_q_heads={self.num_q_heads}, tensor_parallel_size={tp_size}"
            )
        return self.num_q_heads // tp_size

    def get_num_kv_heads(self, parallel_config: ParallelConfig):
        family = self.get_attention_family()
        if family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
            return family.resolve_runtime_num_kv_heads(self)

        local_heads, _ = resolve_attention_head_partition(
            self.num_kv_heads, parallel_config.tensor_parallel_size,
        )
        return local_heads

    def get_head_size(self):
        family = self.get_attention_family()
        if family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
            return family.resolve_runtime_head_size(self)
        # Use explicit head_dim if provided by the model architecture
        # Otherwise compute from embedding_dim / num_q_heads
        if self._head_dim is not None:
            return self._head_dim
        return self.embedding_dim // self.num_q_heads

    @property
    def dtype(self):
        return self._dtype

    @staticmethod
    def _import_torch():
        try:
            import torch  # type: ignore
            return torch
        except Exception:
            return None

    @staticmethod
    def _parse_dtype(dtype: Optional[str]) -> Any:
        torch_mod = ModelConfig._import_torch()
        if dtype is None:
            return torch_mod.float16 if torch_mod is not None else "float16"
        if (
            torch_mod is not None
            and hasattr(torch_mod, "dtype")
            and isinstance(dtype, torch_mod.dtype)
        ):
            return dtype
        dtype_str = str(dtype).lower()
        if dtype_str in {"float16", "fp16", "half"}:
            return torch_mod.float16 if torch_mod is not None else "float16"
        if dtype_str in {"bfloat16", "bf16"}:
            return torch_mod.bfloat16 if torch_mod is not None else "bfloat16"
        if dtype_str in {"float32", "fp32"}:
            return torch_mod.float32 if torch_mod is not None else "float32"
        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _dtype_to_str(dtype: Any) -> str:
        torch_mod = ModelConfig._import_torch()
        if torch_mod is not None and dtype == torch_mod.float16:
            return "FP16"
        if torch_mod is not None and dtype == torch_mod.bfloat16:
            return "BF16"
        if torch_mod is not None and dtype == torch_mod.float32:
            return "FP32"
        dtype_str = str(dtype).lower()
        if dtype_str in {"float16", "fp16", "half"}:
            return "FP16"
        if dtype_str in {"bfloat16", "bf16"}:
            return "BF16"
        if dtype_str in {"float32", "fp32"}:
            return "FP32"
        return str(dtype)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ModelConfig to a dictionary for serialization.

        This is useful for passing ModelConfig across process boundaries
        in multiprocessing scenarios.
        """
        return {
            "name": self.name,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "dense_mlp_hidden_dim": self.dense_mlp_hidden_dim,
            "routed_mlp_hidden_dim": self.routed_mlp_hidden_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "use_gated_mlp": self.use_gated_mlp,
            "use_bias": self.use_bias,
            "use_qkv_bias": self.use_qkv_bias,
            "activation": self.activation,
            "norm": self.norm,
            "post_attn_norm": self.post_attn_norm,
            "vocab_size": self.vocab_size,
            "is_neox_style": self.is_neox_style,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "partial_rotary_factor": self.partial_rotary_factor,
            "no_tensor_parallel": self.no_tensor_parallel,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_layers_enum": self.moe_layers_enum,
            "use_qk_norm": self.use_qk_norm,
            "attn_output_gate": self.attn_output_gate,
            "rms_norm_eps": self.rms_norm_eps,
            "dtype": self._dtype_to_str(self._dtype),
            "model_type": self.model_type,
            "model_architecture_profile": self.model_architecture_profile,
            "architectures": list(self.architectures),
            "fused_add_norm_capability": self.fused_add_norm_capability,
            # Architecture-specific structural fields
            "model_arch": self.model_arch,
            "share_expert_dim": self.share_expert_dim,
            "share_q_dim": self.share_q_dim,
            "head_dim": self._head_dim,
            "use_mla": self.use_mla,
            "use_mfa": self.use_mfa,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            **{
                name: getattr(self, name)
                for name in (
                    "index_topk", "index_n_heads", "index_head_dim",
                    "index_topk_freq", "index_topk_pattern", "index_skip_topk_offset",
                    "indexer_rope_interleave",
                    "topk_method", "scoring_func", "norm_topk_prob",
                    "routed_scaling_factor",
                    "compress_ratios", "sliding_window", "num_hash_layers",
                    "hc_mult", "hc_sinkhorn_iters", "hc_eps", "o_groups",
                    "o_lora_rank", "compress_rope_theta", "swiglu_limit",
                    "expert_dtype", "vision_config", "projector_bias",
                    "sparse_config", "attention_other_setting", "att_impl_type",
                    "num_nextn_predict_layers", "mtp_layer_types", "layer_rope_thetas",
                    "partial_rotary_factors", "swiglu_limits", "swiglu_limits_shared",
                    "norm_dtype", "fp32_residual_connection", "need_fp32_gate",
                    "use_moe_router_bias", "moe_router_activation", "moe_router_scaling_factor",
                    "zero_centered", "yarn_only_types", "use_rope_layers", "swa_num_attention_heads",
                )
                if getattr(self, name) is not None
            },
            "layer_types": list(self.layer_types) if self.layer_types is not None else None,
            "full_attention_interval": self.full_attention_interval,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "gdn_output_gate_type": self.gdn_output_gate_type,
            **({"norm_expert_weight": self.norm_expert_weight} if self.model_type == "step5" else {}),
            # Quantization config
            "quantization_config": (
                dataclass_to_dict(self.quantization_config)
                if self.quantization_config else None
            ),
            # LM head weight sharing
            "tie_word_embeddings": self.tie_word_embeddings,
        }
