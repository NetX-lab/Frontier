from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass

from frontier.config.model_config import (
    _infer_attn_output_gate_from_hf_config,
    _infer_share_expert_dim_from_hf_config,
    _infer_use_qk_norm_from_hf_config,
)
from frontier.config.precision_type import PrecisionType
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig
from frontier.spec_decode.runtime import (
    get_mtp_method_family,
    method_requires_prefix_matching_disabled,
)


@dataclass(frozen=True)
class MTPRuntimeContract:
    method: str
    mtp_family: str
    target_model_name: str
    proposer_model_name: str
    spec_model_name: str
    attn_tp_size: int
    mtp_n_predict: int
    mtp_num_layers: int
    requires_prefix_caching_disabled: bool
    fusion_op_name: str
    fusion_is_tp_sharded: bool
    fusion_requires_allgather: bool
    norm_op_name: str
    num_pre_fusion_norms: int
    num_post_decoder_norms: int
    embedding_requires_allreduce: bool
    lm_head_op_name: str
    lm_head_requires_allgather: bool


class StructuralModelConfigAdapter:
    def __init__(self, profiling_config: ProfilingModelConfig) -> None:
        self._profiling_config = profiling_config

    def __getattr__(self, name: str):
        return getattr(self._profiling_config, name)

    def get_name(self) -> str:
        return str(self._profiling_config.name)

    @property
    def torch_dtype(self) -> str:
        return str(
            getattr(
                self._profiling_config,
                "torch_dtype",
                getattr(self._profiling_config, "dtype", "float16"),
            )
        )

    @property
    def quantization_config(self):
        return getattr(self._profiling_config, "quantization_config", None)

    def is_step2_mini(self) -> bool:
        return bool(getattr(self._profiling_config, "model_arch", None) == "step2_mini")

    def is_step3_text(self) -> bool:
        return bool(getattr(self._profiling_config, "model_type", None) == "step3_text")

    def supports_share_expert(self) -> bool:
        return (
            self.is_step2_mini()
            or self.is_step3_text()
            or (
                bool(getattr(self._profiling_config, "is_moe", False))
                and int(getattr(self._profiling_config, "share_expert_dim", 0) or 0) > 0
            )
        )

    def get_head_dim(self) -> int:
        return int(self._profiling_config.get_head_dim())

    def get_moe_layer_ids(self):
        return self._profiling_config.get_moe_layer_ids()

    def is_moe_layer(self, layer_id: int) -> bool:
        return int(layer_id) in self.get_moe_layer_ids()

    def get_num_moe_layers(self) -> int:
        return len(self.get_moe_layer_ids())

    def get_default_precision(self) -> PrecisionType:
        return PrecisionType.from_torch_dtype(self.torch_dtype)

    def get_quant_signature(self) -> str:
        quantization_config = self.quantization_config
        if quantization_config is None:
            return "none"
        if hasattr(quantization_config, "get_quant_signature"):
            return str(quantization_config.get_quant_signature())
        raise ValueError(
            "Structural MTP profiling model config has unsupported quantization_config"
        )

    def get_quant_signature_hash(self) -> str:
        return hashlib.sha256(self.get_quant_signature().encode()).hexdigest()[:16]


_MTP_RUNTIME_METHOD_REGISTRY = {
    "qwen3_next_mtp": {
        "fusion_op_name": "mtp_fusion_proj",
        "fusion_is_tp_sharded": True,
        "fusion_requires_allgather": True,
        "norm_op_name": "input_layernorm",
        "num_pre_fusion_norms": 2,
        "num_post_decoder_norms": 1,
        "embedding_requires_allreduce": True,
        "lm_head_op_name": "lm_head_linear",
        "lm_head_requires_allgather": True,
    },
    "qwen3_moe_mtp": {
        "fusion_op_name": "mtp_fusion_proj",
        "fusion_is_tp_sharded": True,
        "fusion_requires_allgather": True,
        "norm_op_name": "input_layernorm",
        "num_pre_fusion_norms": 2,
        "num_post_decoder_norms": 1,
        "embedding_requires_allreduce": True,
        "lm_head_op_name": "lm_head_linear",
        "lm_head_requires_allgather": True,
    },
    "deepseek_mtp": {
        "fusion_op_name": "mtp_fusion_proj",
        "fusion_is_tp_sharded": False,
        "fusion_requires_allgather": False,
        "norm_op_name": "input_layernorm",
        "num_pre_fusion_norms": 2,
        "num_post_decoder_norms": 1,
        "embedding_requires_allreduce": True,
        "lm_head_op_name": "lm_head_linear",
        "lm_head_requires_allgather": True,
    },
    "ernie_mtp": {
        "fusion_op_name": "mtp_fusion_proj",
        "fusion_is_tp_sharded": False,
        "fusion_requires_allgather": False,
        "norm_op_name": "input_layernorm",
        "num_pre_fusion_norms": 2,
        "num_post_decoder_norms": 0,
        "embedding_requires_allreduce": True,
        "lm_head_op_name": "lm_head_linear",
        "lm_head_requires_allgather": True,
    },
}


def _load_structural_model_config_from_json(model_name: str) -> StructuralModelConfigAdapter:
    safe_name = str(model_name).replace("/", "__")
    config_path = os.path.join("data", "config", "models", f"{safe_name}.json")
    if not os.path.exists(config_path):
        raise ValueError(
            f"Could not find structural model config JSON for model={model_name!r}: "
            f"{config_path!r}"
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    num_layers = int(raw["num_hidden_layers"])
    num_q_heads = int(raw["num_attention_heads"])
    num_kv_heads = int(raw.get("num_key_value_heads", num_q_heads))
    embedding_dim = int(raw["hidden_size"])
    num_experts = int(raw.get("n_routed_experts", raw.get("num_experts", 0)) or 0)
    is_moe = num_experts > 0
    mlp_hidden_dim = int(
        raw.get(
            "moe_intermediate_size" if is_moe else "intermediate_size",
            raw.get("intermediate_size", 0),
        )
    )
    hidden_act = str(raw.get("hidden_act", "silu")).lower()
    use_gated_mlp = hidden_act == "silu"
    if hidden_act not in {"silu", "gelu"}:
        hidden_act = "silu" if use_gated_mlp else "gelu"

    profiling_config = ProfilingModelConfig(
        name=str(model_name),
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        embedding_dim=embedding_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        max_position_embeddings=int(raw["max_position_embeddings"]),
        use_gated_mlp=use_gated_mlp,
        use_bias=bool(raw.get("attention_bias", raw.get("use_bias", False))),
        use_qkv_bias=bool(raw.get("attention_bias", raw.get("use_qkv_bias", False))),
        activation=hidden_act,
        norm="rms_norm" if "rms_norm_eps" in raw else "layer_norm",
        post_attn_norm=bool(raw.get("post_attn_norm", True)),
        vocab_size=int(raw["vocab_size"]),
        rope_theta=raw.get("rope_theta"),
        rope_scaling=raw.get("rope_scaling"),
        is_moe=is_moe,
        num_experts=num_experts,
        num_experts_per_tok=int(raw.get("num_experts_per_tok", 0) or 0),
        moe_layers_enum=raw.get("moe_layers_enum"),
        use_qk_norm=_infer_use_qk_norm_from_hf_config(raw),
        attn_output_gate=_infer_attn_output_gate_from_hf_config(raw),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        dtype=raw.get("torch_dtype", raw.get("dtype", "float16")),
        model_type=raw.get("model_type"),
        model_arch=raw.get("model_arch"),
        share_expert_dim=_infer_share_expert_dim_from_hf_config(raw),
        share_q_dim=raw.get("share_q_dim"),
        head_dim=raw.get("head_dim"),
        quantization_config=None,
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
    )
    return StructuralModelConfigAdapter(profiling_config)


def load_mtp_structural_model_config(model_name: str) -> StructuralModelConfigAdapter:
    try:
        profiling_config = ProfilingModelConfig.from_model_name(model_name)
        return StructuralModelConfigAdapter(profiling_config)
    except Exception:
        return _load_structural_model_config_from_json(model_name)


def build_mtp_runtime_contract(
    *,
    method: str,
    target_model_name: str,
    spec_model_name: str,
    attn_tp_size: int,
    mtp_n_predict: int,
    mtp_num_layers: int,
) -> MTPRuntimeContract:
    normalized_method = str(method)
    if normalized_method not in _MTP_RUNTIME_METHOD_REGISTRY:
        raise ValueError(
            f"No MTP runtime contract registered for method={normalized_method!r}"
        )

    target_model_name_normalized = str(target_model_name).strip()
    if not target_model_name_normalized:
        raise ValueError("target_model_name must be non-empty for MTP runtime")
    if int(attn_tp_size) <= 0:
        raise ValueError(
            f"attn_tp_size must be > 0 for MTP runtime, got={attn_tp_size!r}"
        )
    if int(mtp_n_predict) <= 0:
        raise ValueError(
            f"mtp_n_predict must be > 0 for MTP runtime, got={mtp_n_predict!r}"
        )
    if int(mtp_num_layers) <= 0:
        raise ValueError(
            f"mtp_num_layers must be > 0 for MTP runtime, got={mtp_num_layers!r}"
        )

    registry_entry = _MTP_RUNTIME_METHOD_REGISTRY[normalized_method]
    spec_model_name_normalized = str(spec_model_name).strip()
    requires_draft_model = normalized_method in {"deepseek_mtp", "ernie_mtp"}
    if requires_draft_model and not spec_model_name_normalized:
        raise ValueError(
            "draft-model MTP requires non-empty spec_model_name for runtime "
            f"contract, got method={normalized_method!r}"
        )
    proposer_model_name = (
        spec_model_name_normalized
        if requires_draft_model
        else target_model_name_normalized
    )

    return MTPRuntimeContract(
        method=normalized_method,
        mtp_family=get_mtp_method_family(normalized_method),
        target_model_name=target_model_name_normalized,
        proposer_model_name=proposer_model_name,
        spec_model_name=spec_model_name_normalized,
        attn_tp_size=int(attn_tp_size),
        mtp_n_predict=int(mtp_n_predict),
        mtp_num_layers=int(mtp_num_layers),
        requires_prefix_caching_disabled=method_requires_prefix_matching_disabled(
            normalized_method
        ),
        fusion_op_name=str(registry_entry["fusion_op_name"]),
        fusion_is_tp_sharded=bool(registry_entry["fusion_is_tp_sharded"]),
        fusion_requires_allgather=bool(
            registry_entry["fusion_requires_allgather"]
        ),
        norm_op_name=str(registry_entry["norm_op_name"]),
        num_pre_fusion_norms=int(registry_entry["num_pre_fusion_norms"]),
        num_post_decoder_norms=int(registry_entry["num_post_decoder_norms"]),
        embedding_requires_allreduce=bool(
            registry_entry["embedding_requires_allreduce"]
        ),
        lm_head_op_name=str(registry_entry["lm_head_op_name"]),
        lm_head_requires_allgather=bool(
            registry_entry["lm_head_requires_allgather"]
        ),
    )
