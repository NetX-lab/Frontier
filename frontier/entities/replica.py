from math import ceil

from frontier.config import BaseRequestGeneratorConfig, ReplicaConfig
from frontier.entities.base_entity import BaseEntity
from frontier.logger import init_logger
from frontier.types import ClusterType

logger = init_logger(__name__)


class Replica(BaseEntity):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        generator_config: BaseRequestGeneratorConfig,
        cluster_type: ClusterType,
    ) -> None:
        self._id = Replica.generate_id()

        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self._device_config = replica_config.device_config
        self._generator_config = generator_config
        self._cluster_type = cluster_type

        if self._cluster_type == ClusterType.DECODE_FFN:
            # This replica only has FFN. Attention properties are not applicable.
            self._model_config.num_q_heads = 0
            self._model_config.num_kv_heads = 0

        # Validate pipeline parallelism configuration
        # This should have been caught earlier in ReplicaConfig.__post_init__, but we check again for safety
        if self._model_config.num_layers % self._replica_config.num_pipeline_stages != 0:
            raise ValueError(
                f"Pipeline parallelism configuration error in {self._cluster_type.name} replica: "
                f"num_layers ({self._model_config.num_layers}) must be evenly divisible by "
                f"num_pipeline_stages ({self._replica_config.num_pipeline_stages}). "
                f"This should have been caught during configuration validation."
            )

        if self._cluster_type != ClusterType.DECODE_FFN:
            assert (
                self._model_config.embedding_dim % self._replica_config.attn_tensor_parallel_size
                == 0
            ), (
                f"Tensor parallelism configuration error: embedding_dim ({self._model_config.embedding_dim}) "
                f"must be evenly divisible by attn_tensor_parallel_size ({self._replica_config.attn_tensor_parallel_size})"
            )

    @property
    def id(self) -> int:
        return self._id

    @property
    def cluster_type(self) -> ClusterType:
        return self._cluster_type

    @property
    def num_layers(self) -> int:
        return self._model_config.num_layers

    @property
    def num_q_heads(self) -> int:
        return self._model_config.num_q_heads

    @property
    def num_kv_heads(self) -> int:
        return self._model_config.num_kv_heads

    @property
    def embedding_dim(self) -> int:
        return self._model_config.embedding_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return self._model_config.mlp_hidden_dim

    @property
    def use_gated_mlp(self) -> int:
        return self._model_config.use_gated_mlp

    @property
    def vocab_size(self) -> int:
        return self._model_config.vocab_size

    @property
    def num_pipeline_stages(self) -> int:
        return self._replica_config.num_pipeline_stages

    @property
    def num_layers_per_pipeline_stage(self) -> int:
        return self._model_config.num_layers // self._replica_config.num_pipeline_stages

    @property
    def attention_head_dim(self) -> int:
        if self.num_q_heads == 0:
            return 0
        # Use model_config.get_head_dim() to prioritize explicit head_dim from JSON config
        # This ensures consistency with the profiling module's ModelConfig.get_head_size()
        return self._model_config.get_head_dim()

    @property
    def q_heads_per_tensor_parallel_worker(self) -> int:
        if self.num_attn_tensor_parallel_workers == 0:
            return self._model_config.num_q_heads
        return (
            self._model_config.num_q_heads // self.num_attn_tensor_parallel_workers
        )

    @property
    def kv_heads_per_tensor_parallel_worker(self) -> int:
        if self.num_kv_heads == 0:
            return 0
        if self.num_attn_tensor_parallel_workers == 0:
            return self._model_config.num_kv_heads
        return ceil(
            self._model_config.num_kv_heads / self.num_attn_tensor_parallel_workers
        )

    @property
    def num_attn_tensor_parallel_workers(self) -> int:
        return self._replica_config.attn_tensor_parallel_size

    @property
    def total_memory_gb(self) -> int:
        return self._device_config.total_memory_gb

    @property
    def memory_margin_fraction(self) -> float:
        return self._replica_config.memory_margin_fraction

    @property
    def max_request_tokens(self) -> int:
        return self._generator_config.max_tokens

    @property
    def per_device_flops(self) -> float:
        return self._device_config.fp16_tflops * 2**40

    # New prop. for MoE
    @property
    def is_moe(self) -> bool:
        """
        Check if the model is a Mixture of Experts (MoE) model.
        
        IMPORTANT: MoE detection is based SOLELY on model architecture (model_config.is_moe),
        NOT on parallelism settings like moe_expert_parallel_size or total_expert_num.
        A MoE model remains a MoE model regardless of how it's parallelized.
        
        This property first checks the global IS_MOE flag (set during config initialization),
        then falls back to model_config.is_moe for backward compatibility.
        
        Returns:
            bool: True if the model is MoE, False otherwise.
        """
        from frontier.config import global_vars
        
        # Use global IS_MOE if initialized (preferred - single source of truth)
        global_is_moe = global_vars.get_is_moe()
        if global_is_moe is not None:
            return global_is_moe
        
        # Fallback to model_config.is_moe for backward compatibility
        # This handles cases where global_vars hasn't been initialized yet
        return self._model_config.is_moe

    @property
    def num_moe_expert_parallel_size(self) -> int:
        return self._replica_config.moe_expert_parallel_size
    
    @property
    def num_moe_tensor_parallel_workers(self) -> int:
        return self._replica_config.moe_tensor_parallel_size
    
    @property
    def total_num_experts(self) -> int:
        return self._replica_config.total_expert_num
    
    @property
    def local_num_experts(self) -> int:
        return self._replica_config.local_expert_num

    @property
    def router_topk(self) -> int:
        return self._replica_config.router_topk

    @property
    def router_load_balancing_type(self) -> str:
        return self._replica_config.router_load_balancing_type

    @property
    def extend_ep_across_dp(self) -> bool:
        if not self.is_moe:
            return False
        return self._replica_config.extend_ep_across_dp

    @property
    def dp_size(self) -> int:
        """Get the data parallel size (number of DP replicas within this replica)."""
        return self._replica_config.attn_data_parallel_size

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "use_gated_mlp": self.use_gated_mlp,
            "vocab_size": self.vocab_size,
            "num_pipeline_stages": self.num_pipeline_stages,
            "num_attn_tensor_parallel_workers": self.num_attn_tensor_parallel_workers,
        }
