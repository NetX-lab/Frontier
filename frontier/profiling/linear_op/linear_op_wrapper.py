import os

import torch

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_utils.tensor_parallel_utils import (
    get_padded_vocab_size,
)
from frontier.profiling.common.utils import (
    configure_quantization_manager_for_model_name,
    initialize_dummy_weights,
)
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.linear_op.linear_op_impl import GPTModel
from frontier.profiling.linear_op.profiling_plan import _share_expert_profiling_names
from frontier.profiling.utils import ProfileMethod, normalize_profile_method
from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 3
ACTIVE_STEPS = 20


class LinearOpWrapper:
    """
    Wrapper for profiling linear operations in LLM models.
    
    This class profiles all linear-complexity operations including:
    - MLP layers: mlp_up_proj, mlp_down_proj, mlp_act
    - Normalization: input_layernorm, post_attention_layernorm
    - Attention projections: attn_pre_proj, attn_post_proj, attn_rope
    - Residual: add
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        profile_method: str,
        rank: int,
        output_dir: str,
        profiling_plan: dict | None = None,
    ):
        super().__init__()

        self.profile_method = normalize_profile_method(profile_method)
        self.timer_stats_store = TimerStatsStore(profile_method=self.profile_method)

        self.model_config = model_config
        configure_quantization_manager_for_model_name(self.model_config.name)
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.rank = rank
        self.output_dir = output_dir
        self.profiling_plan = profiling_plan
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.pad_vocab_size = (
            self.model_config.vocab_size % self.num_tensor_parallel_workers != 0
        )
        self.padded_vocab_size = self.model_config.vocab_size
        if self.pad_vocab_size:
            self.padded_vocab_size = get_padded_vocab_size(
                self.model_config.vocab_size, self.num_tensor_parallel_workers
            )
            print(
                f"[WARNING] vocab_size {self.model_config.vocab_size} is not divisible by "
                f"TP={self.num_tensor_parallel_workers}. Padding to {self.padded_vocab_size} for profiling."
            )

        # Initialize a complete GPT model to profile linear operations in context
        self.model = GPTModel(
            model_config,
            num_tensor_parallel_workers,
            (
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                else 1
            ),
            pad_vocab_size=self.pad_vocab_size,
            profiling_plan=self.profiling_plan,
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=self.model_config.dtype).cuda().eval()

    def _get_expected_keys(self) -> list[str]:
        if self.profiling_plan is not None and "enabled_ops" in self.profiling_plan:
            return list(self.profiling_plan["enabled_ops"])

        expected_keys = [
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_pre_proj",
            "attn_post_proj",
            "attn_rope",
        ]
        architecture_profile = self.model_config.get_model_architecture_profile()
        expected_keys.extend(
            op_name
            for op_name in architecture_profile.linear_attention.sharded_ops
            if op_name not in expected_keys
        )
        if (
            getattr(self.model_config, "is_moe", False)
            and hasattr(self.model_config, "supports_share_expert")
            and self.model_config.supports_share_expert()
        ):
            expected_keys.extend(_share_expert_profiling_names())
        return expected_keys

    @torch.inference_mode()  # disable gradient calculation
    def profile(self, num_tokens: int):
        vocab_range = self.padded_vocab_size // self.num_tensor_parallel_workers
        input_ids = torch.randint(
            low=0,
            high=vocab_range,
            size=(num_tokens,),
            device="cuda",
            dtype=torch.long,
        )
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Run the model once without capturing the graph.
            # This is to make sure that the captured graph does not include the
            # kernel launches for initial benchmarking (e.g., Triton autotune).
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)

            with record_function_tracer:
                self.model(
                    input_ids,
                    positions,
                )

            time_stats = record_function_tracer.get_operation_time_stats(debug=True)

            # Check for missing expected operations
            expected_keys = self._get_expected_keys()
            missing_keys = [k for k in expected_keys if k not in time_stats]
            if missing_keys:
                print(f"[WARNING] num_tokens={num_tokens}: Missing operations: {missing_keys}")
        else:
            for _ in range(WARMUP_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

            time_stats = self.timer_stats_store.get_stats()


        stats = {
            "time_stats": time_stats,
            "n_head": self.model_config.num_q_heads,
            "n_kv_head": self.model_config.num_kv_heads,
            "n_embd": self.model_config.embedding_dim,
            "n_expanded_embd": self.model_config.mlp_hidden_dim,
            "vocab_size": self.model_config.vocab_size,
            "use_gated_mlp": self.model_config.use_gated_mlp,
            "use_qk_norm": getattr(self.model_config, "use_qk_norm", False),
            "attn_output_gate": getattr(self.model_config, "attn_output_gate", False),
            "num_tokens": num_tokens,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
            "padded_n_embd": (
                self.profiling_plan.get("padded_n_embd", self.model_config.embedding_dim)
                if self.profiling_plan is not None
                else self.model_config.embedding_dim
            ),
            "padded_n_expanded_embd": (
                self.profiling_plan.get(
                    "padded_n_expanded_embd", self.model_config.mlp_hidden_dim
                )
                if self.profiling_plan is not None
                else self.model_config.mlp_hidden_dim
            ),
            "model_arch": self.model_config.model_arch,
            "model_architecture_profile": (
                self.model_config.get_model_architecture_profile().profile_id
            ),
            "share_expert_dim": self.model_config.share_expert_dim,
            "share_q_dim": self.model_config.share_q_dim,
        }
        self.timer_stats_store.clear_stats()

        return stats
