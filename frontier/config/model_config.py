from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import os

from frontier.attention.model_binding import bind_attention_family
from frontier.attention.ops import AttentionMemoryLayout
from frontier.config.base_fixed_config import BaseFixedConfig
from frontier.config.precision_type import PrecisionType
from frontier.logger import init_logger
from frontier.model_architectures import get_model_architecture_profile
from frontier.types import ActivationType, NormType

logger = init_logger(__name__)


# ============================================================================
# Quantization Configuration Types
# ============================================================================

@dataclass
class QuantizationConfig:
    """Configuration for model quantization, aligned with stepfun-vllm Fp8Config semantics.

    Attributes:
        quant_method: Quantization method (None for no quantization, "fp8" for FP8)
        activation_scheme: Activation quantization scheme ("dynamic" or "static")
        is_checkpoint_fp8_serialized: Whether checkpoint has pre-quantized FP8 weights
        weight_block_size: Block dimensions for block-wise quantization, e.g., (128, 128)
        ignored_layers: Layer prefixes to skip quantization (e.g., ["lm_head", "router"])
    """
    quant_method: Optional[str] = None
    activation_scheme: Optional[str] = None
    is_checkpoint_fp8_serialized: bool = False
    weight_block_size: Optional[Tuple[int, int]] = None
    ignored_layers: List[str] = field(default_factory=list)

    @staticmethod
    def _normalize_quant_method(quant_method: Optional[str]) -> Optional[str]:
        if quant_method is None:
            return None
        normalized = str(quant_method).strip().lower()
        alias_map = {
            "fbgemm_fp8": "fp8",
        }
        return alias_map.get(normalized, normalized)

    def __post_init__(self):
        """Validate quantization configuration and enforce stepfun-vllm constraints."""
        self.quant_method = self._normalize_quant_method(self.quant_method)

        # Validate quant_method
        valid_quant_methods = {None, "fp8"}
        if self.quant_method not in valid_quant_methods:
            raise ValueError(
                f"Invalid quant_method '{self.quant_method}'. "
                f"Must be one of: {valid_quant_methods}"
            )

        # Validate activation_scheme
        valid_schemes = {None, "dynamic", "static"}
        if self.activation_scheme not in valid_schemes:
            raise ValueError(
                f"Invalid activation_scheme '{self.activation_scheme}'. "
                f"Must be one of: {valid_schemes}"
            )

        # stepfun-vllm constraint: block-wise FP8 requires specific configuration
        # See: stepfun-vllm/vllm/model_executor/layers/quantization/fp8.py lines 76-89
        if self.quant_method == "fp8" and self.weight_block_size is not None:
            if not self.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Block-wise FP8 quantization (weight_block_size is set) requires "
                    "is_checkpoint_fp8_serialized=True. This is a stepfun-vllm constraint."
                )
            if self.activation_scheme != "dynamic":
                raise ValueError(
                    "Block-wise FP8 quantization (weight_block_size is set) requires "
                    f"activation_scheme='dynamic', got '{self.activation_scheme}'. "
                    "This is a stepfun-vllm constraint."
                )

    def get_quant_signature(self) -> str:
        """Generate a stable, hashable signature for this quantization configuration.

        This signature is used for:
        1. Profiling CSV metadata
        2. Training data filtering (strict match)
        3. Model cache key
        4. Predictor model family selection

        Returns:
            A stable string identifier for this quantization configuration.
        """
        if self.quant_method is None:
            return "none"

        # Build a canonical representation
        parts = [
            f"method={self.quant_method}",
            f"act={self.activation_scheme or 'none'}",
            f"serialized={self.is_checkpoint_fp8_serialized}",
        ]

        if self.weight_block_size is not None:
            parts.append(f"block={self.weight_block_size[0]}x{self.weight_block_size[1]}")
        else:
            parts.append("block=none")

        # Sort ignored_layers for stability
        if self.ignored_layers:
            sorted_ignored = sorted(self.ignored_layers)
            parts.append(f"ignored={','.join(sorted_ignored)}")

        signature_str = "|".join(parts)
        return signature_str

    def get_quant_signature_hash(self) -> str:
        """Generate a short hash of the quant_signature for cache keys."""
        sig = self.get_quant_signature()
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> "QuantizationConfig":
        """Create QuantizationConfig from a dictionary (e.g., from JSON config)."""
        if config_dict is None:
            return cls()

        weight_block_size = config_dict.get("weight_block_size")
        if weight_block_size is not None:
            if isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2:
                weight_block_size = tuple(weight_block_size)
            else:
                raise ValueError(
                    f"weight_block_size must be a list/tuple of 2 integers, "
                    f"got {weight_block_size}"
                )

        return cls(
            quant_method=cls._normalize_quant_method(config_dict.get("quant_method")),
            activation_scheme=config_dict.get("activation_scheme"),
            is_checkpoint_fp8_serialized=bool(config_dict.get("is_checkpoint_fp8_serialized", False)),
            weight_block_size=weight_block_size,
            ignored_layers=list(config_dict.get("ignored_layers", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "quant_method": self.quant_method,
            "activation_scheme": self.activation_scheme,
            "is_checkpoint_fp8_serialized": self.is_checkpoint_fp8_serialized,
            "weight_block_size": list(self.weight_block_size) if self.weight_block_size else None,
            "ignored_layers": self.ignored_layers,
        }


# ============================================================================
# Model Architecture Constants
# ============================================================================

class ModelArch:
    """Model architecture identifiers for op_name isolation."""
    GENERIC = "generic"  # Default for Llama, Qwen, etc.
    STEP2_MINI = "step2_mini"  # Step2Mini with inter_norm + wq + share_expert

    VALID_ARCHS = {GENERIC, STEP2_MINI}


# Explicit fused add+norm capability source.
# Use model-level metadata/allowlist first; only fall back to norm heuristic
# for backward compatibility with legacy model registrations.
FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST = {
    "deepseek_v3",
    "llama",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3_moe",
    "step2_mini",
    "step3_text",
}
NON_FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST = {
    "phimoe",
}
QK_NORM_MODEL_TYPE_ALLOWLIST = {
    "qwen3_moe",
    "qwen3_next",
}


def _infer_attn_output_gate_from_hf_config(cfg: Dict[str, Any]) -> bool:
    explicit_value = cfg.get("attn_output_gate")
    if explicit_value is not None:
        return bool(explicit_value)

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type == "qwen3_next":
        return True

    architectures = [str(value).lower() for value in cfg.get("architectures", [])]
    return any("qwen3next" in architecture for architecture in architectures)


def _infer_use_qk_norm_from_hf_config(cfg: Dict[str, Any]) -> bool:
    explicit_value = cfg.get("use_qk_norm")
    if explicit_value is not None:
        return bool(explicit_value)

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type in QK_NORM_MODEL_TYPE_ALLOWLIST:
        return True

    architectures = [str(value).lower() for value in cfg.get("architectures", [])]
    return any("qwen3next" in architecture for architecture in architectures)


def _infer_share_expert_dim_from_hf_config(
    cfg: Dict[str, Any],
) -> Optional[int]:
    raw_value = cfg.get("share_expert_dim")
    if raw_value is None:
        raw_value = cfg.get("shared_expert_intermediate_size")
    if raw_value is None:
        return None

    share_expert_dim = int(raw_value)
    if share_expert_dim <= 0:
        return None
    return share_expert_dim


@dataclass
class BaseModelConfig(BaseFixedConfig):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    mlp_hidden_dim: int
    max_position_embeddings: int
    use_gated_mlp: bool
    use_bias: bool
    use_qkv_bias: bool
    activation: ActivationType
    norm: NormType
    post_attn_norm: bool
    vocab_size: int
    use_qk_norm: bool = False
    attn_output_gate: bool = False
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_layers_enum: Optional[str] = None

    # Model type from config.json (normalized to lowercase when provided)
    model_type: Optional[str] = None
    model_architecture_profile: Optional[str] = None

    # Architecture-specific structural fields used by registered profiles.
    model_arch: str = ModelArch.GENERIC
    share_expert_dim: Optional[int] = None  # Shared expert intermediate dimension
    share_q_dim: Optional[int] = None  # Shared Q dimension for inter_norm + wq path
    norm_expert_weight: bool = False  # Whether to normalize expert weights

    # Explicit head dimension for architectures where head_dim differs from
    # embedding_dim // num_q_heads.
    # If None, head_dim is computed as embedding_dim // num_q_heads
    head_dim: Optional[int] = None

    # Attention topology fields.
    # use_mla follows vLLM latent-MLA cache semantics: one latent KV head with
    # cache head size kv_lora_rank + qk_rope_head_dim.
    # use_mfa models Step3Text's dense-KV MFA attention path: shared-Q
    # projection plus a single dense KV head.
    use_mla: bool = False
    use_mfa: bool = False
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # Default model precision from model config (e.g., torch_dtype in HF config)
    torch_dtype: str = "float16"

    # Quantization configuration (aligned with stepfun-vllm Fp8Config)
    quantization_config: Optional[QuantizationConfig] = None

    # Explicit fused add+norm capability. When None, decide from model-type allowlist.
    fused_add_norm_capability: Optional[bool] = None

    # Internal fields (excluded from hash/comparison)
    _model_name: Optional[str] = field(default=None, compare=False, hash=False)
    _moe_layer_ids_cache: Optional[List[int]] = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self):
        """Validate model configuration after initialization."""
        if self.model_type is not None:
            self.model_type = str(self.model_type).lower()
        if self.model_architecture_profile is not None:
            self.model_architecture_profile = str(self.model_architecture_profile).lower()

        # Validate model_arch
        if self.model_arch not in ModelArch.VALID_ARCHS:
            raise ValueError(
                f"Invalid model_arch '{self.model_arch}'. "
                f"Must be one of: {ModelArch.VALID_ARCHS}"
            )
        architecture_profile = self.get_model_architecture_profile()

        # Validate torch_dtype
        PrecisionType.from_torch_dtype(self.torch_dtype)

        if self.fused_add_norm_capability is not None and not isinstance(
            self.fused_add_norm_capability, bool
        ):
            raise ValueError(
                "fused_add_norm_capability must be bool or None, "
                f"got {type(self.fused_add_norm_capability)}"
            )

        if self.use_mla and self.use_mfa:
            raise ValueError("use_mla and use_mfa are mutually exclusive")

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
                    "MLA model configuration requires fields: "
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
            # share_q_dim is optional but recommended for accurate inter_norm + wq modeling

    @property
    def uses_fused_add_norm(self) -> bool:
        """Whether the model uses fused add+norm kernel (RMSNorm).
        When True, input_layernorm/post_attention_layernorm time already includes
        residual addition, so 'add' should NOT be profiled or predicted separately.

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
        return self.norm == NormType.RMS_NORM

    def get_quant_signature(self) -> str:
        """Get quantization signature for this model configuration.

        Used for strict filtering in training/prediction pipelines.
        """
        if self.quantization_config is None:
            return "none"
        return self.quantization_config.get_quant_signature()

    def get_head_dim(self) -> int:
        """Get the head dimension for attention.

        Returns the explicit head_dim if provided in the model configuration,
        otherwise computes it as embedding_dim // num_q_heads.

        This method ensures consistency with the profiling module's ModelConfig.get_head_size()
        which also prioritizes explicit head_dim from JSON config.

        Returns:
            int: The head dimension for attention operations.
        """
        if self.head_dim is not None:
            return self.head_dim
        return self.embedding_dim // self.num_q_heads

    def get_attention_family(self):
        """Return the bound attention family for runtime cache semantics."""
        return bind_attention_family(self).family

    def uses_mla(self) -> bool:
        """Return whether this model uses vLLM-style MLA cache semantics."""
        family = self.get_attention_family()
        return family.memory_layout is AttentionMemoryLayout.LATENT_MLA

    def get_runtime_num_kv_heads(self) -> int:
        """Return runtime KV heads for cache allocation."""
        family = self.get_attention_family()
        return family.resolve_runtime_num_kv_heads(self)

    def get_runtime_head_size(self) -> int:
        """Return runtime KV-cache head size."""
        family = self.get_attention_family()
        return family.resolve_runtime_head_size(self)

    def get_qk_head_dim(self) -> int:
        """Return the full QK head dimension."""
        family = self.get_attention_family()
        if family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
            if self.qk_head_dim is None:
                raise ValueError("MLA qk_head_dim is not configured")
            return self.qk_head_dim
        return self.get_head_dim()

    def get_default_precision(self) -> PrecisionType:
        """Get default model precision derived from torch_dtype."""
        return PrecisionType.from_torch_dtype(self.torch_dtype)

    def get_quant_signature_hash(self) -> str:
        """Get short hash of quantization signature for cache keys."""
        if self.quantization_config is None:
            return hashlib.sha256(b"none").hexdigest()[:16]
        return self.quantization_config.get_quant_signature_hash()

    def get_model_arch(self) -> str:
        """Get model architecture identifier for op_name isolation."""
        return self.model_arch

    def get_model_architecture_profile(self):
        """Return plugin-style model architecture semantics for this config."""
        return get_model_architecture_profile(self)

    def supports_share_expert(self) -> bool:
        """Check if the model uses share_expert in the FFN path."""
        return self.get_model_architecture_profile().supports_share_expert(self)

    def get_moe_layer_ids(self) -> List[int]:
        """Return sorted MoE layer IDs covered by this model config.

        Semantics:
        - Non-MoE models: empty list
        - MoE models without explicit `moe_layers_enum`: all layers are MoE
        - MoE models with `moe_layers_enum`: parse explicit layer IDs and keep
          those within `[0, num_layers)`.
        """
        if not self.is_moe:
            return []

        if self._moe_layer_ids_cache is not None:
            return self._moe_layer_ids_cache

        raw_moe_layers = self.moe_layers_enum
        if raw_moe_layers is None or str(raw_moe_layers).strip() == "":
            self._moe_layer_ids_cache = list(range(self.num_layers))
            return self._moe_layer_ids_cache

        parsed_layer_ids: List[int] = []
        seen_layer_ids = set()
        for token in str(raw_moe_layers).split(","):
            token = token.strip()
            if token == "":
                continue
            try:
                layer_id = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid moe_layers_enum token '{token}' for model {self.get_name()}"
                ) from exc
            if layer_id < 0:
                raise ValueError(
                    f"moe_layers_enum contains negative layer id {layer_id} for model {self.get_name()}"
                )
            if layer_id >= self.num_layers:
                continue
            if layer_id in seen_layer_ids:
                continue
            seen_layer_ids.add(layer_id)
            parsed_layer_ids.append(layer_id)

        if not parsed_layer_ids:
            raise ValueError(
                "moe_layers_enum does not include any layer within the current model depth "
                f"[0, {self.num_layers}) for model {self.get_name()}"
            )

        self._moe_layer_ids_cache = sorted(parsed_layer_ids)
        return self._moe_layer_ids_cache

    def is_moe_layer(self, layer_id: int) -> bool:
        """Return whether a specific layer index uses MoE FFN."""
        if not self.is_moe:
            return False
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range for model with num_layers={self.num_layers}"
            )
        return layer_id in self.get_moe_layer_ids()

    def get_num_moe_layers(self) -> int:
        """Return the number of layers that use MoE FFN."""
        return len(self.get_moe_layer_ids())

    def get_name(self):
        # For dynamically loaded models, return the stored name
        if self._model_name is not None:
            return self._model_name
        # For predefined models, return the class-level name
        return self.__class__.get_static_name()

    @staticmethod
    def get_static_name():
        # Provide a default to satisfy BaseFixedConfig iteration over subclasses.
        # Concrete predefined models override this. Dynamic HF JSON path will not match this.
        return "__base_model_config__"

    @classmethod
    def create_from_name(cls, name: str) -> "BaseModelConfig":
        """Create from registered name or from a HuggingFace config.json in data/config/models.
        If not found among registered subclasses, will try loading JSON at
        data/config/models/{sanitized(name)}.json where sanitized(name) replaces '/' with '__'.
        """
        # Try registered subclasses first
        try:
            # Call BaseFixedConfig.create_from_name directly to avoid recursion
            return BaseFixedConfig.create_from_name.__func__(cls, name)  # type: ignore[attr-defined]
        except ValueError:
            pass
        # Fallback to HF JSON loader (explicit behavior by design)
        return cls._create_from_hf_json(name)

    @classmethod
    def _create_from_hf_json(cls, name: str) -> "BaseModelConfig":
        safe_name = cls._sanitize_name(name)
        base_dir = os.path.join("data", "config", "models")
        file_path = os.path.join(base_dir, f"{safe_name}.json")
        if not os.path.exists(file_path):
            raise ValueError(f"[BaseModelConfig] Unknown model '{name}' and no JSON found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Required fields (fail fast)
        required = [
            ("num_hidden_layers", int),
            ("num_attention_heads", int),
            ("num_key_value_heads", int),
            ("hidden_size", int),
            ("intermediate_size", int),  # for dense; see MoE below
            ("max_position_embeddings", int),
            ("vocab_size", int),
            ("hidden_act", str),
        ]
        for k, _t in required:
            if k not in cfg:
                # For MoE models, intermediate_size may be absent but moe_intermediate_size required instead.
                if k == "intermediate_size" and ("num_experts" in cfg and isinstance(cfg["num_experts"], int) and cfg["num_experts"] > 1):
                    if "moe_intermediate_size" not in cfg:
                        raise ValueError(f"Missing required key 'moe_intermediate_size' in {file_path} for MoE model")
                    continue
                raise ValueError(f"Missing required key '{k}' in {file_path}")

        # Detect MoE
        is_moe = int(cfg.get("num_experts", 0)) > 1
        model_type_lower = str(cfg.get("model_type", "")).lower()

        # Map common fields
        num_layers = int(cfg["num_hidden_layers"])
        num_q_heads = int(cfg["num_attention_heads"])
        num_kv_heads = int(cfg["num_key_value_heads"])
        embedding_dim = int(cfg["hidden_size"])
        if is_moe:
            mlp_hidden_dim = int(cfg["moe_intermediate_size"])  # per-expert FFN dim
        else:
            mlp_hidden_dim = int(cfg["intermediate_size"])
        max_pos = int(cfg["max_position_embeddings"])
        vocab_size = int(cfg["vocab_size"])

        # Infer enums and booleans
        activation = cls._map_activation(str(cfg["hidden_act"]))
        norm = cls._infer_norm(cfg)
        post_attn_norm = True if norm == NormType.RMS_NORM else False
        use_gated_mlp = True if activation == ActivationType.SILU else False

        # Heuristics for bias and qkv bias with explicit override when keys exist
        use_bias = bool(cfg.get("use_bias", False))
        use_qkv_bias = bool(cfg.get("use_qkv_bias", cfg.get("qkv_bias", False)))
        use_qk_norm = _infer_use_qk_norm_from_hf_config(cfg)
        attn_output_gate = _infer_attn_output_gate_from_hf_config(cfg)
        if "use_qkv_bias" not in cfg and "qkv_bias" not in cfg:
            # Check attention_bias first (explicit field in HF configs)
            if "attention_bias" in cfg:
                use_qkv_bias = bool(cfg["attention_bias"])
            else:
                # Some models like Qwen/Qwen2 use QKV bias by design
                model_type = str(cfg.get("model_type", "")).lower()
                arch0 = str(next(iter(cfg.get("architectures", [])), "")).lower()
                if "qwen" in model_type or "qwen" in arch0:
                    use_qkv_bias = True

        rope_theta = cfg.get("rope_theta")
        rope_scaling = cfg.get("rope_scaling")

        # Parse model architecture (for op_name isolation)
        model_arch = cfg.get("model_arch")
        if model_arch is None:
            # Infer from model_type if model_arch not explicitly set
            if model_type_lower == "step2_mini":
                model_arch = ModelArch.STEP2_MINI
            else:
                model_arch = ModelArch.GENERIC
        elif model_arch not in ModelArch.VALID_ARCHS:
            raise ValueError(
                f"Invalid model_arch '{model_arch}' in config. "
                f"Must be one of: {ModelArch.VALID_ARCHS}"
            )

        # Parse architecture-specific structural fields.
        share_expert_dim = _infer_share_expert_dim_from_hf_config(cfg)
        share_q_dim = cfg.get("share_q_dim")
        if share_q_dim is not None:
            share_q_dim = int(share_q_dim)
        norm_expert_weight = bool(cfg.get("norm_expert_weight", False))

        # Parse default model precision.
        # Resolution order is explicit and fail-fast:
        #   1) torch_dtype (preferred)
        #   2) dtype (fallback for legacy configs)
        #   3) float16 default when neither field exists
        torch_dtype = cls._resolve_model_torch_dtype(cfg, file_path)

        # Parse explicit head_dim for architectures where head_dim differs from
        # embedding_dim // num_q_heads.
        # If not provided, get_head_dim() will compute it as embedding_dim // num_q_heads
        explicit_head_dim = cfg.get("head_dim")
        if explicit_head_dim is not None:
            explicit_head_dim = int(explicit_head_dim)

        use_mla = bool(
            cfg.get("use_mla", False)
            or (
                model_type_lower
                in {"deepseek_v2", "deepseek_v3", "deepseek_mtp", "kimi_k2"}
                and cfg.get("kv_lora_rank") is not None
            )
        )
        use_mfa = bool(cfg.get("use_mfa", False))
        q_lora_rank = cfg.get("q_lora_rank")
        kv_lora_rank = cfg.get("kv_lora_rank")
        qk_nope_head_dim = cfg.get("qk_nope_head_dim")
        qk_rope_head_dim = cfg.get("qk_rope_head_dim")
        qk_head_dim = cfg.get("qk_head_dim")
        v_head_dim = cfg.get("v_head_dim")
        if q_lora_rank is not None:
            q_lora_rank = int(q_lora_rank)
        if kv_lora_rank is not None:
            kv_lora_rank = int(kv_lora_rank)
        if qk_nope_head_dim is not None:
            qk_nope_head_dim = int(qk_nope_head_dim)
        if qk_rope_head_dim is not None:
            qk_rope_head_dim = int(qk_rope_head_dim)
        if qk_head_dim is not None:
            qk_head_dim = int(qk_head_dim)
        elif qk_nope_head_dim is not None and qk_rope_head_dim is not None:
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        if v_head_dim is not None:
            v_head_dim = int(v_head_dim)

        # Parse quantization configuration
        quant_config_dict = cfg.get("quantization_config")
        quantization_config = QuantizationConfig.from_dict(quant_config_dict)
        fused_add_norm_capability = cls._resolve_fused_add_norm_capability(cfg)

        base_kwargs = dict(
            num_layers=num_layers,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            embedding_dim=embedding_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            max_position_embeddings=max_pos,
            use_gated_mlp=use_gated_mlp,
            use_bias=use_bias,
            use_qkv_bias=use_qkv_bias,
            use_qk_norm=use_qk_norm,
            attn_output_gate=attn_output_gate,
            activation=activation,
            norm=norm,
            post_attn_norm=post_attn_norm,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            model_type=model_type_lower or None,
            model_architecture_profile=cfg.get("model_architecture_profile"),
            model_arch=model_arch,
            share_expert_dim=share_expert_dim,
            share_q_dim=share_q_dim,
            norm_expert_weight=norm_expert_weight,
            head_dim=explicit_head_dim,
            use_mla=use_mla,
            use_mfa=use_mfa,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            fused_add_norm_capability=fused_add_norm_capability,
            moe_layers_enum=cfg.get("moe_layers_enum"),
        )

        if is_moe:
            num_experts = int(cfg["num_experts"])  # already validated
            num_experts_per_tok = int(cfg.get("num_experts_per_tok", 0))
            return MoEModelConfig(
                **base_kwargs,
                is_moe=True,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                _model_name=name,  # Store the original model name
            )
        else:
            return DenseModelConfig(**base_kwargs, _model_name=name)  # Store the original model name

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("/", "__")

    @staticmethod
    def _resolve_model_torch_dtype(cfg: Dict[str, Any], file_path: str) -> str:
        """Resolve torch dtype from HF config with explicit conflict checks.

        Priority:
        1. ``torch_dtype`` when present
        2. ``dtype`` as backward-compatible fallback
        3. ``float16`` default when both are absent

        If both ``torch_dtype`` and ``dtype`` are present, they must resolve to
        the same precision (aliases like ``bf16`` and ``bfloat16`` are treated
        as equivalent). Otherwise, fail fast.
        """
        torch_dtype_raw = cfg.get("torch_dtype")
        dtype_raw = cfg.get("dtype")

        if torch_dtype_raw is not None and dtype_raw is not None:
            resolved_torch = PrecisionType.from_torch_dtype(str(torch_dtype_raw))
            resolved_dtype = PrecisionType.from_torch_dtype(str(dtype_raw))
            if resolved_torch != resolved_dtype:
                raise ValueError(
                    "Conflicting precision fields in "
                    f"{file_path}: torch_dtype={torch_dtype_raw}, dtype={dtype_raw}"
                )
            return str(torch_dtype_raw)

        if torch_dtype_raw is not None:
            PrecisionType.from_torch_dtype(str(torch_dtype_raw))
            return str(torch_dtype_raw)

        if dtype_raw is not None:
            PrecisionType.from_torch_dtype(str(dtype_raw))
            return str(dtype_raw)

        return "float16"

    @staticmethod
    def _map_activation(act: str) -> ActivationType:
        a = act.lower()
        if a in {"silu", "swish"}:
            return ActivationType.SILU
        if a.startswith("gelu") or a == "gelu":
            return ActivationType.GELU
        raise ValueError(f"Unsupported hidden_act '{act}' in HuggingFace config")

    @staticmethod
    def _infer_norm(cfg: Dict[str, Any]) -> NormType:
        model_type = str(cfg.get("model_type", "")).lower()
        # Presence of RMS norm epsilon is a strong signal
        if "rms_norm_eps" in cfg or "rmsnorm_eps" in cfg or "rms_norm_epsilon" in cfg:
            return NormType.RMS_NORM
        if "layer_norm_eps" in cfg or "layernorm_eps" in cfg:
            return NormType.LAYER_NORM
        # Heuristic by model family
        if "qwen" in model_type or "llama" in model_type:
            return NormType.RMS_NORM
        if "phi" in model_type:
            return NormType.LAYER_NORM
        # Default to RMS to match LLaMA-style models
        return NormType.RMS_NORM

    @staticmethod
    def _resolve_fused_add_norm_capability(cfg: Dict[str, Any]) -> Optional[bool]:
        """Resolve fused add+norm capability from explicit metadata or model allowlist."""
        explicit_flag = cfg.get("uses_fused_add_norm")
        if explicit_flag is not None:
            if not isinstance(explicit_flag, bool):
                raise ValueError(
                    "uses_fused_add_norm must be bool when present in model config, "
                    f"got {type(explicit_flag)}"
                )
            return explicit_flag

        model_type = str(cfg.get("model_type", "")).lower()
        if model_type in NON_FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST:
            return False
        if model_type in FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST:
            return True
        return None



@dataclass
class DenseModelConfig(BaseModelConfig):
    is_moe: bool = False


@dataclass
class MoEModelConfig(BaseModelConfig):
    is_moe: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.num_experts <= 1:
            raise ValueError("MoEModelConfig requires num_experts > 1")

@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = 16384
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 32768
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_static_name():
        return "meta-llama/Llama-2-Config"


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_static_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096

    @staticmethod
    def get_static_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_TinyModelConfig(Llama2ModelConfig):
    num_layers: int = 2
    num_q_heads: int = 2
    num_kv_heads: int = 2
    embedding_dim: int = 2
    mlp_hidden_dim: int = 2
    max_position_embeddings: int = 2

    @staticmethod
    def get_static_name():
        return "meta-llama/Llama-2-tiny"



@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 4096

    @staticmethod
    def get_static_name():
        return "meta-llama/Llama-2-70b-hf"


@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_static_name():
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_static_name():
        return "meta-llama/Meta-Llama-3-70B"


@dataclass
class InternLMModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 4096
    vocab_size: int = 103168


@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):
    num_layers: int = 60
    num_q_heads: int = 40
    num_kv_heads: int = 40
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 13824

    @staticmethod
    def get_static_name():
        return "internlm/internlm-20b"


@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 32768
    vocab_size: int = 92544


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 16384
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_static_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240
    max_position_embeddings: int = 2048
    use_gated_mlp: bool = False
    use_bias: bool = True
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.GELU
    norm: NormType = NormType.LAYER_NORM
    post_attn_norm: bool = False
    vocab_size: int = 51200
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = 10000
    partial_rotary_factor: float = 0.4
    no_tensor_parallel: bool = True

    @staticmethod
    def get_static_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = True
    max_position_embeddings: int = 32768
    vocab_size: int = 152064

    @staticmethod
    def get_static_name():
        return "Qwen/Qwen-Config"


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 64
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 24576
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_static_name():
        return "Qwen/Qwen-72B"


@dataclass
class Qwen3_4BModelConfig(QwenModelConfig):
    num_layers: int = 36
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 9728
    max_position_embeddings: int = 262144
    rope_theta: Optional[float] = 5000000
    vocab_size: int = 151936
    use_qkv_bias: bool = False

    @staticmethod
    def get_static_name():
        return "Qwen/Qwen3-4B"


@dataclass
class Qwen3_32BModelConfig(QwenModelConfig):
    num_layers: int = 64
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 25600
    max_position_embeddings: int = 40960
    rope_theta: Optional[float] = 1000000
    vocab_size: int = 151936
    use_qkv_bias: bool = False

    @staticmethod
    def get_static_name():
        return "Qwen/Qwen3-32B"
