from frontier.config import ReplicaConfig
from frontier.entities import BatchStage
from frontier.utils.param_counter import ParamCounter
from frontier.types import ClusterType


class MFUCalculator:

    def __init__(self, replica_config: ReplicaConfig, cluster_type: ClusterType):
        self._cluster_type = cluster_type
        self._replica_config = replica_config
        model_config = self._replica_config.model_config

        self._num_layers_per_device = (
            model_config.num_layers // self._replica_config.num_pipeline_stages
        )
        self._device_flops = self._replica_config.device_config.fp16_tflops * 2**40

    def _get_mlp_flops(self, batch_stage: BatchStage) -> float:
        param_counter = ParamCounter(self._replica_config, self._cluster_type)
        num_mlp_params_per_device = param_counter.get_num_mlp_parameters_per_device()
        num_tokens = sum(batch_stage.num_tokens)
        return 2 * num_tokens * num_mlp_params_per_device

    def _get_attention_flops(self, batch_stage: BatchStage) -> float:
        param_counter = ParamCounter(self._replica_config, self._cluster_type)
        model_config = self._replica_config.model_config

        num_heads_per_device = (
            model_config.num_q_heads // self._replica_config.attn_tensor_parallel_size
        )
        # Use model_config.get_head_dim() to prioritize explicit head_dim from JSON config
        head_dimension = model_config.get_head_dim()

        total_flops = 0
        for request, num_tokens in zip(batch_stage.requests, batch_stage.num_tokens):
            total_flops += (
                4  # for number of ops in attention
                * self._num_layers_per_device
                * num_heads_per_device
                * head_dimension
                * num_tokens  # q length
                * (num_tokens + request.num_processed_tokens)  # kv length
            )

        return total_flops

    def get_mfu(self, batch_stage: BatchStage) -> float:
        total_flops = 0
        if self._cluster_type in [
            ClusterType.PREFILL,
            ClusterType.MONOLITHIC,
        ]:
            total_flops = self._get_mlp_flops(batch_stage) + self._get_attention_flops(
                batch_stage
            )
        elif self._cluster_type == ClusterType.DECODE_ATTN:
            total_flops = self._get_attention_flops(batch_stage)
        elif self._cluster_type == ClusterType.DECODE_FFN:
            total_flops = self._get_mlp_flops(batch_stage)

        if batch_stage.execution_time == 0:
            return 0.0

        total_flops_per_second = total_flops / batch_stage.execution_time
        return total_flops_per_second * 100 / self._device_flops
