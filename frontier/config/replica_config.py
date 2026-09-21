"""Per-replica hardware, parallelism and model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from frontier.config.device_sku_config import BaseDeviceSKUConfig
from frontier.config.model_config import BaseModelConfig
from frontier.config.node_sku_config import BaseNodeSKUConfig
from frontier.config.speculative_decoding_config import (
    SpeculativeDecodingConfig,
)


@dataclass
class ReplicaConfig:
    memory_margin_fraction: float = field(
        default=0.1,
        metadata={"help": "Memory margin fraction."},
    )
    num_pipeline_stages: int = field(
        default=1,
        metadata={"help": "Number of pipeline stages (pp size)."},
    )
    attn_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Attention tensor parallel size (attn_tp size)."},
    )
    attn_dp: int = field(
        default=1,
        metadata={
            "help": "Attention data-parallel lanes owned by one Replica.",
        },
    )
    moe_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "MoE tensor parallel size (moe_tp size)."},
    )
    moe_expert_parallel_size: int = field(
        default=1,
        metadata={"help": "MoE expert parallel size (moe_ep size)."},
    )
    total_expert_num: int = field(
        default=1,
        metadata={"help": "Total expert number."},
    )
    router_load_balancing_type: str = field(
        default="None",
        metadata={"help": "MOE router load balancing type."},
    )
    router_topk: int = field(
        default=0,
        metadata={"help": "Router topk. Set to 0 to inherit from model config."},
    )
    moe_routing_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed for deterministic MoE routing distribution generation. "
            "Must be a non-negative integer."
        },
    )
    moe_routing_trace_path: str = field(
        default="",
        metadata={
            "help": "Deferred StepFun merged trace JSONL for unsupported trace replay. "
            "A non-empty path fails fast at the architecture boundary."
        },
    )
    decode_attn_initial_lane_trace_path: str = field(
        default="",
        metadata={
            "help": "Optional StepFun attention trace JSONL for trace-driven "
            "decode-attn initial lane occupancy and warmup replay."
        },
    )
    decode_attn_steady_state_snapshot_path: str = field(
        default="",
        metadata={
            "help": "Optional StepFun attention trace JSONL for explicit "
            "decode-attn steady-state snapshot hydration."
        },
    )
    decode_attn_steady_state_measurement_report_path: str = field(
        default="",
        metadata={
            "help": "Optional StepFun measurement JSON for post-boundary "
            "decode-attn request arrival replay."
        },
    )
    moe_routing_distribution_type: str = field(
        default="balanced",
        metadata={
            "help": "MoE expert-load distribution for disaggregated routing simulation. "
            "Valid values: 'balanced', 'random', 'skewed', or 'zipf'. This controls "
            "token-to-expert load skew without changing router_topk/model semantics."
        },
    )
    device: str = field(
        default="a100",
        metadata={"help": "Device."},
    )
    network_device: str = field(
        default="a100_pairwise_nvlink",
        metadata={"help": "Network device."},
    )
    speculative_decoding_config: SpeculativeDecodingConfig = field(
        default_factory=SpeculativeDecodingConfig,
        metadata={"help": "Speculative decoding simulation configuration."},
    )

    # configs should be set by the user
    cluster_prefix: str = None
    local_expert_num: int = None
    model_name: str = "meta-llama/Llama-2-7b-hf"

    def __post_init__(self):
        if type(self.attn_dp) is not int or self.attn_dp <= 0:
            raise ValueError(
                "attn_dp must be a positive integer, "
                f"got {self.attn_dp!r}"
            )
        if self.cluster_prefix == "decode_attn" and self.attn_dp != 1:
            raise ValueError(
                "DECODE_ATTN requires attn_dp=1 because it is the PD-AF attention role"
            )
        # Load model and device configs first (needed for validation)
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model_name
        )
        self.device_config: BaseDeviceSKUConfig = (
            BaseDeviceSKUConfig.create_from_type_string(self.device)
        )
        self.node_config: BaseNodeSKUConfig = BaseNodeSKUConfig.create_from_type_string(
            self.network_device
        )

        # Auto-set total_expert_num from model config if not explicitly set and model is MoE
        if (
            self.total_expert_num == 1
            and self.model_config.is_moe
            and self.model_config.num_experts > 0
        ):
            self.total_expert_num = self.model_config.num_experts

        # Align router_topk with model config when not explicitly set.
        if self.model_config.is_moe:
            if self.router_topk is None or int(self.router_topk) <= 0:
                if self.model_config.num_experts_per_tok > 0:
                    self.router_topk = int(self.model_config.num_experts_per_tok)
                else:
                    raise ValueError(
                        "router_topk is not set and model_config.num_experts_per_tok is missing"
                    )
        else:
            if self.router_topk is None or int(self.router_topk) <= 0:
                self.router_topk = 1

        valid_moe_routing_distribution_types = {
            "balanced",
            "random",
            "skewed",
            "zipf",
        }
        self.moe_routing_distribution_type = str(
            self.moe_routing_distribution_type
        ).strip().lower()
        if self.moe_routing_distribution_type not in valid_moe_routing_distribution_types:
            raise ValueError(
                "moe_routing_distribution_type must be one of "
                f"{sorted(valid_moe_routing_distribution_types)}, "
                f"got {self.moe_routing_distribution_type!r}"
            )

        # Validate pipeline parallelism configuration early
        if self.model_config.num_layers % self.num_pipeline_stages != 0:
            raise ValueError(
                f"Pipeline parallelism configuration error: "
                f"num_layers ({self.model_config.num_layers}) must be evenly divisible by "
                f"num_pipeline_stages ({self.num_pipeline_stages}). "
                f"Current configuration would result in uneven layer distribution across pipeline stages. "
                f"Please adjust num_pipeline_stages to be a divisor of {self.model_config.num_layers}."
            )

        # Note: this world_size only limits in replica dimension.
        if self.cluster_prefix == "prefill":
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_dp
            )
        elif self.cluster_prefix == "decode_attn":
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_dp
            )
        elif self.cluster_prefix == "decode_ffn":
            self.world_size = (
                self.num_pipeline_stages
                * self.moe_tensor_parallel_size
                * self.moe_expert_parallel_size
            )
        elif self.cluster_prefix == "decode":
            # Unified decode cluster (PD-disaggregation): similar to prefill, includes both Attention and FFN
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_dp
            )
        else:  # Monolithic
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_dp
            )

        # Validate expert parallelism configuration for MoE models
        # Use model_config.is_moe for MoE detection - NOT total_expert_num
        if self.cluster_prefix != "decode_attn" and self.model_config.is_moe:
            if self.total_expert_num > 1:
                assert (
                    self.total_expert_num % self.moe_expert_parallel_size == 0
                ), "total_expert_num must be divisible by moe_expert_parallel_size"
                self.local_expert_num = (
                    self.total_expert_num // self.moe_expert_parallel_size
                )

        if (
            self.speculative_decoding_config.enabled
            and self.cluster_prefix in {"decode_attn", "decode_ffn"}
        ):
            raise ValueError(
                "Speculative decoding Phase 1 supports only co-location and "
                "pd-disaggregation decode path. decode_attn/decode_ffn are not "
                f"supported, cluster_prefix={self.cluster_prefix!r}."
            )

        from frontier.attention.gdn.guards import validate_gdn_runtime_support

        validate_gdn_runtime_support(
            self.model_config,
            speculative_enabled=bool(self.speculative_decoding_config.enabled),
            num_pipeline_stages=self.num_pipeline_stages,
            moe_expert_parallel_size=self.moe_expert_parallel_size,
            attn_dp=self.attn_dp,
            cross_node=(
                int(self.world_size) > int(self.node_config.num_devices_per_node)
            ),
        )
