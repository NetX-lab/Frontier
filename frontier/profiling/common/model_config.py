from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from frontier.config.model_config import (
    BaseModelConfig,
    FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST,
    NON_FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST,
    QuantizationConfig,
    _infer_attn_output_gate_from_hf_config,
    _infer_share_expert_dim_from_hf_config,
    _infer_use_qk_norm_from_hf_config,
)
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
        fused_add_norm_capability: Optional[bool] = None,
        # Step2Mini-specific fields
        model_arch: Optional[str] = None,
        share_expert_dim: Optional[int] = None,
        share_q_dim: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_mla: bool = False,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        qk_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        # Quantization config for metadata tracking
        quantization_config: Optional[QuantizationConfig] = None,
        # Whether lm_head shares weights with embed_tokens (HF standard field)
        tie_word_embeddings: bool = True,
    ):
        self.name = name
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
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
        if fused_add_norm_capability is not None and not isinstance(
            fused_add_norm_capability, bool
        ):
            raise ValueError(
                "fused_add_norm_capability must be bool or None, "
                f"got {type(fused_add_norm_capability)}"
            )
        self.fused_add_norm_capability = fused_add_norm_capability

        # Step2Mini-specific fields
        self.model_arch = model_arch
        self.share_expert_dim = share_expert_dim
        self.share_q_dim = share_q_dim
        # head_dim: If provided, use it; otherwise compute from embedding_dim/num_q_heads
        self._head_dim = head_dim
        self.use_mla = bool(use_mla)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

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
        self.tie_word_embeddings = bool(tie_word_embeddings)

        assert self.norm in ["layer_norm", "rms_norm"]
        assert self.activation in ["gelu", "silu"]

        if self.use_gated_mlp:
            assert self.activation == "silu"
        else:
            assert self.activation == "gelu"

        # Step2Mini validation: if model_arch is step2_mini, require share_expert_dim
        if self.model_arch == "step2_mini":
            if self.share_expert_dim is None:
                raise ValueError(
                    "Step2Mini model requires share_expert_dim to be specified"
                )
            if not self.is_moe:
                raise ValueError(
                    "Step2Mini model requires is_moe=True"
                )

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
            if self.qk_head_dim is None:
                self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            expected_qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            if self.qk_head_dim != expected_qk_head_dim:
                raise ValueError(
                    "qk_head_dim must equal qk_nope_head_dim + "
                    "qk_rope_head_dim for MLA. "
                    f"qk_head_dim={self.qk_head_dim}, "
                    f"expected={expected_qk_head_dim}"
                )

    @property
    def is_step2_mini(self) -> bool:
        """Check if this is a Step2Mini model architecture."""
        return self.model_arch == "step2_mini"

    def is_step3_text(self) -> bool:
        """Check if this is a Step3Text model architecture."""
        return self.model_type == "step3_text"

    def supports_share_expert(self) -> bool:
        """Check if the model uses share_expert in the FFN path."""
        return (
            self.is_step2_mini
            or self.is_step3_text()
            or (self.is_moe and int(self.share_expert_dim or 0) > 0)
        )

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

    def get_runtime_head_size(self) -> int:
        """Return vLLM runtime KV-cache head size."""
        if self.use_mla:
            if self.kv_lora_rank is None or self.qk_rope_head_dim is None:
                raise ValueError(
                    "MLA runtime head size requires kv_lora_rank and "
                    "qk_rope_head_dim"
                )
            return self.kv_lora_rank + self.qk_rope_head_dim
        return self.get_head_dim()

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
        raw = self.moe_layers_enum
        if raw is None or str(raw).strip() == "":
            return list(range(self.num_layers))
        parsed = []
        for token in str(raw).split(","):
            token = token.strip()
            if token == "":
                continue
            layer_id = int(token)
            if 0 <= layer_id < self.num_layers:
                parsed.append(layer_id)
        return sorted(set(parsed))

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
        base_torch_dtype: str = model_config_dict.get('torch_dtype', None)

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
            'norm_expert_weight',  # Step2Mini field not used in profiling
            'torch_dtype',
        ]
        for field in unsupported_fields:
            model_config_dict.pop(field, None)

        # Read additional fields from JSON config that are not in BaseModelConfig
        # These fields are needed for profiling but not part of the core model config
        import os
        import json
        safe_name = model_name.replace("/", "__")
        json_path = os.path.join("data", "config", "models", f"{safe_name}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_cfg = json.load(f)
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
            # Model type (e.g., step3_text) for Step3-only gating
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
            # Step2Mini-specific fields
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
                'q_lora_rank',
                'kv_lora_rank',
                'qk_nope_head_dim',
                'qk_rope_head_dim',
                'qk_head_dim',
                'v_head_dim',
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
        if self.use_mla:
            return 1

        tp_size = parallel_config.tensor_parallel_size
        if tp_size <= 0:
            raise ValueError(
                f"tensor_parallel_size must be positive, got {tp_size}"
            )
        if self.num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be positive, got {self.num_kv_heads}"
            )

        # Match vLLM v1 semantics:
        # - total_num_kv_heads >= tp_size: partition KV heads, require divisibility
        # - total_num_kv_heads < tp_size: replicate KV heads, require tp_size % total_num_kv_heads == 0
        if self.num_kv_heads >= tp_size:
            if self.num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"KV heads must be divisible by tensor_parallel_size when KV heads are partitioned. "
                    f"num_kv_heads={self.num_kv_heads}, tensor_parallel_size={tp_size}"
                )
        else:
            if tp_size % self.num_kv_heads != 0:
                raise ValueError(
                    f"KV heads replication requires tensor_parallel_size to be divisible by num_kv_heads. "
                    f"num_kv_heads={self.num_kv_heads}, tensor_parallel_size={tp_size}"
                )

        return max(1, self.num_kv_heads // tp_size)

    def get_head_size(self):
        if self.use_mla:
            return self.get_runtime_head_size()
        # Use explicit head_dim if provided (e.g., for Step3 models with MLA)
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
            "fused_add_norm_capability": self.fused_add_norm_capability,
            # Step2Mini-specific fields
            "model_arch": self.model_arch,
            "share_expert_dim": self.share_expert_dim,
            "share_q_dim": self.share_q_dim,
            "head_dim": self._head_dim,
            "use_mla": self.use_mla,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            # Quantization config
            "quantization_config": asdict(self.quantization_config) if self.quantization_config else None,
            # LM head weight sharing
            "tie_word_embeddings": self.tie_word_embeddings,
        }
