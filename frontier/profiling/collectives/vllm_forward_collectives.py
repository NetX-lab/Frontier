"""Collectives that vLLM v0.28 issues in one DeepseekV2-family forward pass.

The shapes follow the sequence-parallel MoE path that vLLM selects when expert
parallelism spans tensor- and data-parallel ranks (``ParallelConfig.
use_sequence_parallel_moe``). Source references are to vLLM v0.28.0:

* ``model_executor/models/deepseek_v2.py``: ``DeepseekV2DecoderLayer.forward``
  (attention reduce-scatter, next-layer all-gather), ``DeepseekV2Model.forward``
  (final all-gather of hidden and residual).
* ``distributed/device_communicators/all2all.py``: ``AgRsAll2AllManager``
  dispatch (``all_gatherv`` of hidden, top-k weights and ids over the EP group)
  and combine (``reduce_scatterv``).
* ``forward_context.py``: ``_compute_sp_num_tokens`` (per-EP-rank rows).
* ``v1/worker/dp_utils.py``: ``_run_ar`` (per-step DP synchronization).
* ``model_executor/layers/logits_processor.py``: logits all-gather.

All functions here are pure; ``vllm_collective_benchmark`` executes the calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

# vLLM v0.28 config/parallel.py:188 default when the engine omits the field.
VLLM_DEFAULT_ALL2ALL_BACKEND = "allgather_reducescatter"
# Backends for which ParallelConfig.use_sequence_parallel_moe can be true and
# whose dispatch/combine are all_gatherv/reduce_scatterv (all2all.py:44).
SEQUENCE_PARALLEL_ALL2ALL_BACKENDS = ("allgather_reducescatter",)
# Router outputs: fused_topk_bias_router.py returns float32 weights, int32 ids.
TOPK_WEIGHTS_DTYPE = "float32"
TOPK_IDS_DTYPE = "int32"
# dp_utils._run_ar all-reduces an int32 tensor of shape [4, dp_size].
DP_SYNC_ROWS = 4
DP_SYNC_DTYPE = "int32"
# vocab_parallel_embedding.DEFAULT_VOCAB_PADDING_SIZE.
VOCAB_PADDING_SIZE = 64

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4}

# ``all_reduce`` etc. go through vLLM's GroupCoordinator, which picks the
# communicator. ``device_group_all_reduce`` and ``cpu_all_reduce`` are plain
# torch.distributed all-reduces on the group's NCCL and gloo process groups,
# as dp_utils._run_ar issues them.
OPS = ("all_reduce", "reduce_scatter", "all_gather", "all_gatherv",
       "reduce_scatterv", "device_group_all_reduce", "cpu_all_reduce")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    num_hidden_layers: int
    first_k_dense_replace: int
    moe_layer_freq: int
    num_experts_per_tok: int
    vocab_size: int

    @classmethod
    def from_hf_config(cls, config: dict) -> "ModelShape":
        if config.get("n_routed_experts") is None:
            raise ValueError("Config has no routed experts; this path covers MoE models")
        return cls(
            hidden_size=int(config["hidden_size"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            first_k_dense_replace=int(config["first_k_dense_replace"]),
            # deepseek_v2.py reads getattr(config, "moe_layer_freq", 1).
            moe_layer_freq=int(config.get("moe_layer_freq", 1)),
            num_experts_per_tok=int(config["num_experts_per_tok"]),
            vocab_size=int(config["vocab_size"]),
        )

    def moe_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers)
                if i >= self.first_k_dense_replace and i % self.moe_layer_freq == 0]


@dataclass(frozen=True)
class ServingParallelism:
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    enable_expert_parallel: bool
    all2all_backend: str
    activation_dtype: str
    async_scheduling: bool

    @classmethod
    def from_engine(cls, engine: dict) -> "ServingParallelism":
        return cls(
            tensor_parallel_size=int(engine["tensor_parallel_size"]),
            data_parallel_size=int(engine["data_parallel_size"]),
            pipeline_parallel_size=int(engine["pipeline_parallel_size"]),
            enable_expert_parallel=bool(engine["enable_expert_parallel"]),
            all2all_backend=engine.get("all2all_backend", VLLM_DEFAULT_ALL2ALL_BACKEND),
            activation_dtype=str(engine["dtype"]),
            async_scheduling=bool(engine["async_scheduling"]),
        )

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def expert_parallel_size(self) -> int:
        return self.world_size if self.enable_expert_parallel else 1

    @property
    def dp_sync_on_cpu(self) -> bool:
        # vllm/config/vllm.py:1236-1247: async scheduling moves DP sync to gloo.
        return self.async_scheduling

    def validate_sequence_parallel_moe(self) -> None:
        """Reject topologies where vLLM does not take the SP MoE path."""
        if not (self.enable_expert_parallel and self.tensor_parallel_size > 1
                and self.data_parallel_size > 1 and self.pipeline_parallel_size == 1):
            raise ValueError(
                "Sequence-parallel MoE requires EP, TP>1, DP>1 and PP=1 "
                "(parallel.py use_sequence_parallel_moe, deepseek_v2.py:1240)")
        if self.all2all_backend not in SEQUENCE_PARALLEL_ALL2ALL_BACKENDS:
            raise ValueError(
                f"all2all_backend={self.all2all_backend!r} is not modeled; supported: "
                f"{SEQUENCE_PARALLEL_ALL2ALL_BACKENDS}")
        if self.activation_dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported activation dtype {self.activation_dtype!r}")


@dataclass(frozen=True)
class ForwardScenario:
    """Tokens and logits rows each DP rank runs in one engine step."""

    name: str
    tokens_per_dp_rank: tuple[int, ...]
    logit_rows_per_dp_rank: tuple[int, ...]
    cuda_graph: bool


@dataclass(frozen=True)
class CollectiveCall:
    """One distinct collective and how often a forward pass issues it.

    ``shard_rows`` holds, per global rank, the rows on the sharded side: the
    input of all_reduce/all_gather(v), the output of reduce_scatter(v). A rank
    with zero rows does not issue a TP-local call. ``captured`` marks calls
    inside the CUDA-graph-captured model forward; logits and DP synchronization
    run outside it (gpu_model_runner.py ``compute_logits`` after
    ``_model_forward``).
    """

    name: str
    op: str
    group: str
    dim: int
    columns: tuple[int, ...]
    dtypes: tuple[str, ...]
    shard_rows: tuple[int, ...]
    calls_per_forward: int
    source: str
    captured: bool = False

    def row_bytes(self) -> int:
        return sum(c * DTYPE_BYTES[d] for c, d in zip(self.columns, self.dtypes))


def _group_members(parallel: ServingParallelism, group: str, rank: int) -> list[int]:
    tp = parallel.tensor_parallel_size
    if group == "tp":
        start = rank - rank % tp
        return list(range(start, start + tp))
    if group == "ep":
        return list(range(parallel.world_size))
    if group == "dp":
        return list(range(rank % tp, parallel.world_size, tp))
    raise ValueError(f"Unknown group {group!r}")


def call_bytes(call: CollectiveCall, parallel: ServingParallelism,
               rank: int) -> tuple[int, int]:
    """Return (input_bytes, output_bytes) of ``call`` on ``rank``."""
    members = _group_members(parallel, call.group, rank)
    own = call.shard_rows[rank] * call.row_bytes()
    total = sum(call.shard_rows[m] for m in members) * call.row_bytes()
    if call.op in ("all_reduce", "device_group_all_reduce", "cpu_all_reduce"):
        return own, own
    if call.op in ("all_gather", "all_gatherv"):
        return own, total
    if call.op == "reduce_scatter":
        return own * len(members), own
    if call.op == "reduce_scatterv":
        return total, own
    raise ValueError(f"Unknown op {call.op!r}")


def forward_collectives(model: ModelShape, parallel: ServingParallelism,
                        scenario: ForwardScenario) -> list[CollectiveCall]:
    """List the distinct collectives of one forward pass for ``scenario``."""
    parallel.validate_sequence_parallel_moe()
    tp, dp = parallel.tensor_parallel_size, parallel.data_parallel_size
    if len(scenario.tokens_per_dp_rank) != dp or len(scenario.logit_rows_per_dp_rank) != dp:
        raise ValueError("Scenario must give one token and logits count per DP rank")
    if min(scenario.tokens_per_dp_rank) <= 0:
        raise ValueError("Every DP rank runs at least one token (idle ranks run a dummy batch)")
    if scenario.cuda_graph and len(set(scenario.tokens_per_dp_rank)) != 1:
        raise ValueError("CUDA graph steps are DP-padded to a uniform size (dp_utils.py:80)")

    act = parallel.activation_dtype
    hidden = model.hidden_size
    moe_layers = model.moe_layer_ids()
    dense_layers = model.num_hidden_layers - len(moe_layers)
    if moe_layers and moe_layers != list(range(moe_layers[0], model.num_hidden_layers)):
        raise ValueError("Interleaved dense/MoE layers change the SP gather pattern")

    def per_rank(values_per_dp: Iterable[int]) -> tuple[int, ...]:
        return tuple(v for v in values_per_dp for _ in range(tp))

    tokens = per_rank(scenario.tokens_per_dp_rank)
    sp_rows = per_rank(_ceil_div(n, tp) for n in scenario.tokens_per_dp_rank)
    logit_rows = per_rank(scenario.logit_rows_per_dp_rank)
    padded_vocab = _ceil_div(model.vocab_size, VOCAB_PADDING_SIZE) * VOCAB_PADDING_SIZE
    if padded_vocab % tp:
        raise ValueError("Padded vocab is not divisible by TP")

    calls = [
        CollectiveCall("embedding_all_reduce", "all_reduce", "tp", 0, (hidden,), (act,),
                       tokens, 1, "vocab_parallel_embedding.py:506"),
    ]
    if dense_layers:
        calls.append(CollectiveCall(
            "dense_layer_all_reduce", "all_reduce", "tp", 0, (hidden,), (act,), tokens,
            2 * dense_layers, "RowParallelLinear o_proj and down_proj, reduce_results=True"))
    if moe_layers:
        n_moe = len(moe_layers)
        k = model.num_experts_per_tok
        calls += [
            CollectiveCall("moe_attention_reduce_scatter", "reduce_scatter", "tp", 0,
                           (hidden,), (act,), sp_rows, n_moe, "deepseek_v2.py:1335"),
            # Layers after the first MoE layer receive SP-sharded hidden states.
            CollectiveCall("moe_layer_input_all_gather", "all_gather", "tp", 0,
                           (hidden,), (act,), sp_rows, n_moe - 1, "deepseek_v2.py:1308"),
            CollectiveCall("final_hidden_residual_all_gather", "all_gather", "tp", 0,
                           (2 * hidden,), (act,), sp_rows, 1, "deepseek_v2.py:1499"),
            CollectiveCall("moe_dispatch_all_gatherv", "all_gatherv", "ep", 0,
                           (hidden, k, k), (act, TOPK_WEIGHTS_DTYPE, TOPK_IDS_DTYPE),
                           sp_rows, n_moe, "all2all.py AgRsAll2AllManager.dispatch"),
            CollectiveCall("moe_combine_reduce_scatterv", "reduce_scatterv", "ep", 0,
                           (hidden,), (act,), sp_rows, n_moe,
                           "all2all.py AgRsAll2AllManager.combine"),
        ]
    calls = [replace(call, captured=scenario.cuda_graph) for call in calls]
    if any(logit_rows):
        calls.append(CollectiveCall(
            "logits_all_gather", "all_gather", "tp", -1, (padded_vocab // tp,), (act,),
            logit_rows, 1, "logits_processor.py:126"))
    dp_sync_op = "cpu_all_reduce" if parallel.dp_sync_on_cpu else "device_group_all_reduce"
    calls.append(CollectiveCall(
        "dp_batch_sync_all_reduce", dp_sync_op, "dp", 0, (dp,), (DP_SYNC_DTYPE,),
        (DP_SYNC_ROWS,) * parallel.world_size, 1, "dp_utils.py:_run_ar"))
    return calls


def scenarios_from_grid(parallel: ServingParallelism, grid: dict) -> list[ForwardScenario]:
    """Build step scenarios from a grid spec.

    ``grid["eager"]`` lists ``[tokens, requests]`` points; each point runs with
    every DP rank busy (``uniform``) and with only DP rank 0 busy while the
    others run vLLM's one-token dummy batch (``single_active``,
    gpu_worker.py:1210). ``grid["cuda_graph_batch_sizes"]`` lists decode batch
    sizes, which vLLM pads to the same size on every DP rank.
    """
    dp = parallel.data_parallel_size
    scenarios = []
    for tokens, requests in grid["eager"]:
        if not 0 < requests <= tokens:
            raise ValueError(f"Invalid eager point tokens={tokens} requests={requests}")
        scenarios.append(ForwardScenario(
            f"eager_uniform_t{tokens}_r{requests}", (tokens,) * dp, (requests,) * dp, False))
        scenarios.append(ForwardScenario(
            f"eager_single_active_t{tokens}_r{requests}", (tokens,) + (1,) * (dp - 1),
            (requests,) + (0,) * (dp - 1), False))
    for batch in grid["cuda_graph_batch_sizes"]:
        scenarios.append(ForwardScenario(
            f"graph_decode_b{batch}", (batch,) * dp, (batch,) * dp, True))
    return scenarios
