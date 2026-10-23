"""Native vLLM forward collectives for DeepseekV2-family SP and Step5 MoE.

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

The Step5 image instead selects non-sequence-parallel MoE unless
``kernel_config.moe_backend == "deep_gemm_mega_moe"``. Its model issues separate
attention, dense, shared and routed TP reductions; AgRsAll2AllManager dispatches
unsharded tokens over DP, and the V2 runner synchronizes six int32 rows on gloo.
FP8 dispatch scales assume the per-token block layout of the SM90
Triton/DeepGemm path; native DP>1 shape confirmation remains a collection gate.

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
# Step5 image v1/worker/gpu/dp_utils.py:67 (V2 model runner, always host-side).
STEP5_DP_SYNC_ROWS = 6
DP_SYNC_DTYPE = "int32"
# vocab_parallel_embedding.DEFAULT_VOCAB_PADDING_SIZE.
VOCAB_PADDING_SIZE = 64

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4,
               "float8_e4m3fn": 1}

# ``all_reduce`` etc. go through vLLM's GroupCoordinator, which picks the
# communicator. ``device_group_all_reduce`` and ``cpu_all_reduce`` are plain
# torch.distributed all-reduces on the group's NCCL and gloo process groups,
# as dp_utils._run_ar issues them.
OPS = ("all_reduce", "reduce_scatter", "all_gather", "all_gatherv",
       "reduce_scatterv", "reduce_scatterv_into_output", "device_group_all_reduce", "cpu_all_reduce")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _layer_indices(value: str | Iterable[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    values = value.split(",") if isinstance(value, str) else value
    indices = tuple(int(i) for i in values if str(i).strip())
    if len(indices) != len(set(indices)):
        raise ValueError("MoE layer schedule contains duplicate indices")
    return tuple(sorted(indices))


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    num_hidden_layers: int
    first_k_dense_replace: int
    moe_layer_freq: int
    num_experts_per_tok: int
    vocab_size: int
    model_type: str = ""
    moe_layers: tuple[int, ...] | None = None
    fp8_moe_layers: tuple[int, ...] = ()
    fp8_group_size: int | None = None

    @classmethod
    def from_hf_config(cls, config: dict) -> "ModelShape":
        text = config.get("text_config", config)
        model_type = str(text.get("model_type", "")).lower()
        if text.get("n_routed_experts", text.get("moe_num_experts")) is None:
            raise ValueError("Config has no routed experts; this path covers MoE models")
        listed = _layer_indices(text.get("moe_layer_list"))
        enumerated = _layer_indices(text.get("moe_layers_enum"))
        if listed is not None and enumerated is not None and listed != enumerated:
            raise ValueError("moe_layers_enum and moe_layer_list describe different layers")
        layers = enumerated if enumerated is not None else listed
        n_layers = int(text["num_hidden_layers"])
        if layers is None and model_type == "step5":
            # Step5 _get_step5_moe_layer_indices uses this native default.
            layers = tuple(range(1, n_layers))
        if layers is not None:
            total_layers = n_layers + int(text.get("num_nextn_predict_layers", 0) or 0)
            if any(not 0 <= i < total_layers for i in layers):
                raise ValueError(f"MoE layer schedule must be in [0, {total_layers})")
            # MTP layers do not run in the ordinary model forward.
            layers = tuple(i for i in layers if i < n_layers)
        model = cls(
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=n_layers,
            first_k_dense_replace=int(text.get("first_k_dense_replace", 0)),
            # deepseek_v2.py reads getattr(config, "moe_layer_freq", 1).
            moe_layer_freq=int(text.get("moe_layer_freq", 1)),
            num_experts_per_tok=int(text.get("moe_top_k", text.get("num_experts_per_tok"))),
            vocab_size=int(text["vocab_size"]),
            model_type=model_type,
            moe_layers=layers,
        )
        if model_type != "step5":
            return model
        quant = config.get("quantization_config", text.get("quantization_config")) or {}
        if not quant.get("quant_method"):
            return model
        if (quant["quant_method"] != "fp8" or quant.get("fmt", "e4m3") != "e4m3"
                or quant.get("activation_scheme") != "dynamic"):
            raise ValueError("Step5 dispatch models only dynamic e4m3 block FP8 or unquantized inputs")
        block = quant.get("weight_block_size")
        if not block or len(block) != 2 or any(int(n) <= 0 for n in block):
            raise ValueError("FP8 dispatch requires positive weight_block_size dimensions")
        excluded = quant.get("ignored_layers") or quant.get("modules_to_not_convert") or []
        # Native FusedMoE matches its exact module prefix inside excluded weight
        # names, including w13_weight/w2_weight, not attention/shared exclusions.
        fp8_layers = tuple(i for i in model.moe_layer_ids() if not any(
            f"model.layers.{i}.moe.experts" in name for name in excluded))
        return replace(model, fp8_moe_layers=fp8_layers, fp8_group_size=int(block[1]))

    def moe_layer_ids(self) -> list[int]:
        if self.moe_layers is not None:
            return list(self.moe_layers)
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
    moe_backend: str = "auto"

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
            moe_backend=str(engine.get("moe_backend", "auto")),
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

    def validate_step5_moe(self) -> None:
        """Limit the model to Step5's separate-reduction, DP AgRs path."""
        if self.moe_backend == "deep_gemm_mega_moe":
            raise ValueError("Step5 deep_gemm_mega_moe sequence-parallel collectives are not modeled")
        if not (self.enable_expert_parallel and self.tensor_parallel_size > 1
                and self.data_parallel_size > 1 and self.pipeline_parallel_size == 1):
            raise ValueError("Step5 DP MoE requires EP, TP>1, DP>1 and PP=1")
        if self.all2all_backend != VLLM_DEFAULT_ALL2ALL_BACKEND:
            raise ValueError(f"all2all_backend={self.all2all_backend!r} is not modeled for Step5")
        if self.activation_dtype != "bfloat16":
            raise ValueError("Step5 dispatch/combine currently models bfloat16 activations")


@dataclass(frozen=True)
class ForwardScenario:
    """Wire tokens and logits rows each DP rank runs in one engine step.

    Useful tokens exclude graph padding and idle-rank dummy work. None preserves
    the original interpretation (all wire tokens are useful). ``active_dp_rails``
    selects TP indices whose device DP groups run; None means all rails, while
    (0,) measures isolation. It does not suppress TP calls or host DP sync.
    """

    name: str
    tokens_per_dp_rank: tuple[int, ...]
    logit_rows_per_dp_rank: tuple[int, ...]
    cuda_graph: bool
    useful_tokens_per_dp_rank: tuple[int, ...] | None = None
    active_dp_rails: tuple[int, ...] | None = None

    def validate(self, parallel: ServingParallelism) -> None:
        dp = parallel.data_parallel_size
        if len(self.tokens_per_dp_rank) != dp or len(self.logit_rows_per_dp_rank) != dp:
            raise ValueError("Scenario must give one token and logits count per DP rank")
        if min(self.tokens_per_dp_rank) <= 0:
            raise ValueError("Every DP rank runs at least one token (idle ranks run a dummy batch)")
        if self.cuda_graph and len(set(self.tokens_per_dp_rank)) != 1:
            raise ValueError("CUDA graph steps are DP-padded to a uniform size (dp_utils.py:80)")
        useful = self.useful_tokens_per_dp_rank
        if useful is not None:
            if len(useful) != dp or any(not 0 <= u <= t for u, t in zip(useful, self.tokens_per_dp_rank)):
                raise ValueError("Useful token counts must be between zero and wire tokens per DP rank")
            if not any(useful):
                raise ValueError("A forward scenario requires at least one DP rank with useful tokens")
            if not self.cuda_graph and any(u == 0 and t != 1 for u, t in zip(useful, self.tokens_per_dp_rank)):
                raise ValueError("Idle eager ranks must run a one-token dummy batch")
            if any(not 0 <= r <= u for r, u in zip(self.logit_rows_per_dp_rank, useful)):
                raise ValueError("Logits rows must not exceed useful tokens per DP rank")
        rails = self.active_dp_rails
        if rails is not None and (not rails or len(rails) != len(set(rails))
                                  or any(not 0 <= i < parallel.tensor_parallel_size for i in rails)):
            raise ValueError("active_dp_rails must contain distinct TP indices")


@dataclass(frozen=True)
class CollectiveCall:
    """One distinct collective and how often a forward pass issues it.

    ``shard_rows`` holds, per global rank, the rows on the sharded side: the
    input of all_reduce/all_gather(v), the output of reduce_scatter(v). A rank
    with zero rows skips the call; DP rail isolation zeros an entire DP group.
    ``captured`` marks calls
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


def group_members(parallel: ServingParallelism, group: str, rank: int) -> list[int]:
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
    members = group_members(parallel, call.group, rank)
    own = call.shard_rows[rank] * call.row_bytes()
    total = sum(call.shard_rows[m] for m in members) * call.row_bytes()
    if call.op in ("all_reduce", "device_group_all_reduce", "cpu_all_reduce"):
        return own, own
    if call.op in ("all_gather", "all_gatherv"):
        return own, total
    if call.op == "reduce_scatter":
        return own * len(members), own
    if call.op in ("reduce_scatterv", "reduce_scatterv_into_output"):
        return total, own
    raise ValueError(f"Unknown op {call.op!r}")


def forward_collectives(model: ModelShape, parallel: ServingParallelism,
                        scenario: ForwardScenario, *, native_call_names=None) -> list[CollectiveCall]:
    """List the distinct collectives of one forward pass for ``scenario``."""
    if model.model_type == "step5":
        parallel.validate_step5_moe()
        scenario.validate(parallel)
        calls = _step5_collectives(model, parallel, scenario, into_output=native_call_names is not None)
        if native_call_names is None:
            return calls
        names = tuple(native_call_names)
        if isinstance(native_call_names, str) or not names or len(set(names)) != len(names):
            raise ValueError("native_call_names requires nonempty distinct call names")
        unknown = set(names) - {call.name for call in calls}
        if unknown:
            raise ValueError(f"Unknown or unreached native call names: {sorted(unknown)}")
        return [call for call in calls if call.name in names]
    if native_call_names is not None:
        raise ValueError("native_call_names is supported only for Step5")
    parallel.validate_sequence_parallel_moe()
    tp, dp = parallel.tensor_parallel_size, parallel.data_parallel_size
    scenario.validate(parallel)
    if scenario.active_dp_rails is not None:
        raise ValueError("DP rail isolation applies only to non-sequence-parallel Step5")

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


def _step5_collectives(model: ModelShape, parallel: ServingParallelism,
                      scenario: ForwardScenario, *, into_output=False) -> list[CollectiveCall]:
    tp, dp = parallel.tensor_parallel_size, parallel.data_parallel_size
    hidden, act = model.hidden_size, parallel.activation_dtype
    n_moe = len(model.moe_layer_ids())
    n_dense = model.num_hidden_layers - n_moe
    tokens = tuple(n for n in scenario.tokens_per_dp_rank for _ in range(tp))
    dp_tokens = tuple(n if scenario.active_dp_rails is None or rank % tp in scenario.active_dp_rails
                      else 0 for rank, n in enumerate(tokens))
    calls = [
        CollectiveCall("embedding_all_reduce", "all_reduce", "tp", 0, (hidden,), (act,),
                       tokens, 1, "vocab_parallel_embedding.py:VocabParallelEmbedding.forward"),
        CollectiveCall("attention_all_reduce", "all_reduce", "tp", 0, (hidden,), (act,),
                       tokens, model.num_hidden_layers, "step5/nvidia/model.py:Step5Attention.o_proj"),
    ]
    if n_dense:
        calls.append(CollectiveCall(
            "dense_down_proj_all_reduce", "all_reduce", "tp", 0, (hidden,), (act,),
            tokens, n_dense, "step5/nvidia/model.py:Step5MLP.down_proj"))
    if n_moe:
        calls.append(CollectiveCall(
            "shared_expert_down_proj_all_reduce", "all_reduce", "tp", 0, (hidden,), (act,),
            tokens, n_moe, "step5/nvidia/model.py:Step5MoE.share_expert.down_proj"))
        k = model.num_experts_per_tok
        n_fp8 = len(model.fp8_moe_layers)
        if n_fp8:
            if model.fp8_group_size is None:
                raise ValueError("FP8 dispatch requires the activation group size")
            calls.append(CollectiveCall(
                "moe_dispatch_all_gatherv_fp8", "all_gatherv", "dp", 0,
                (hidden, k, k, _ceil_div(hidden, model.fp8_group_size)),
                ("float8_e4m3fn", TOPK_WEIGHTS_DTYPE, TOPK_IDS_DTYPE, "float32"),
                dp_tokens, n_fp8, "naive_dp_ep.py:prepare; all2all.py:AgRsAll2AllManager.dispatch"))
        if n_moe - n_fp8:
            calls.append(CollectiveCall(
                "moe_dispatch_all_gatherv_bf16", "all_gatherv", "dp", 0, (hidden, k, k),
                (act, TOPK_WEIGHTS_DTYPE, TOPK_IDS_DTYPE), dp_tokens, n_moe - n_fp8,
                "naive_dp_ep.py:prepare; all2all.py:AgRsAll2AllManager.dispatch"))
        calls += [
            CollectiveCall("moe_combine_reduce_scatterv", "reduce_scatterv_into_output" if into_output else "reduce_scatterv", "dp", 0,
                           (hidden,), (act,), dp_tokens, n_moe,
                            "all2all.py:AgRsAll2AllManager.combine_into_output" if into_output else "all2all.py:AgRsAll2AllManager.combine"),
            CollectiveCall("routed_expert_all_reduce", "all_reduce", "tp", 0,
                           (hidden,), (act,), tokens, n_moe,
                           "moe_runner.py:_maybe_reduce_final_output"),
        ]
    calls = [replace(call, captured=scenario.cuda_graph) for call in calls]
    logit_rows = tuple(n for n in scenario.logit_rows_per_dp_rank for _ in range(tp))
    padded_vocab = _ceil_div(model.vocab_size, VOCAB_PADDING_SIZE) * VOCAB_PADDING_SIZE
    if padded_vocab % tp:
        raise ValueError("Padded vocab is not divisible by TP")
    if any(logit_rows):
        calls.append(CollectiveCall(
            "logits_all_gather", "all_gather", "tp", -1, (padded_vocab // tp,), (act,),
            logit_rows, 1, "logits_processor.py:LogitsProcessor._gather_logits"))
    calls.append(CollectiveCall(
        "dp_batch_sync_all_reduce", "cpu_all_reduce", "dp", 0, (dp,), (DP_SYNC_DTYPE,),
        (STEP5_DP_SYNC_ROWS,) * parallel.world_size, 1, "v1/worker/gpu/dp_utils.py:67-77"))
    return calls


def scenarios_from_grid(parallel: ServingParallelism, grid: dict) -> list[ForwardScenario]:
    """Build step scenarios from a grid spec.

    ``grid["eager"]`` lists ``[tokens, requests]`` points; each point runs with
    every DP rank busy (``uniform``) and with only DP rank 0 busy while the
    others run vLLM's one-token dummy batch (``single_active``,
    gpu_worker.py:1210). ``grid["cuda_graph_batch_sizes"]`` lists decode batch
    sizes, which vLLM pads to the same size on every DP rank.

    ``grid["scenarios"]`` optionally adds explicit ForwardScenario records with
    ``name``, ``tokens_per_dp_rank``, ``logit_rows_per_dp_rank``, ``cuda_graph``,
    and optionally ``useful_tokens_per_dp_rank`` and ``active_dp_rails``. Wire
    counts stay positive even on idle ranks: one in eager, capture size in graph.
    Explicit-only grids may omit the two legacy grid keys.
    """
    dp = parallel.data_parallel_size
    scenarios = []
    for tokens, requests in grid.get("eager", []):
        if not 0 < requests <= tokens:
            raise ValueError(f"Invalid eager point tokens={tokens} requests={requests}")
        scenarios.append(ForwardScenario(
            f"eager_uniform_t{tokens}_r{requests}", (tokens,) * dp, (requests,) * dp, False))
        scenarios.append(ForwardScenario(
            f"eager_single_active_t{tokens}_r{requests}", (tokens,) + (1,) * (dp - 1),
            (requests,) + (0,) * (dp - 1), False,
            useful_tokens_per_dp_rank=(tokens,) + (0,) * (dp - 1)))
    for batch in grid.get("cuda_graph_batch_sizes", []):
        scenarios.append(ForwardScenario(
            f"graph_decode_b{batch}", (batch,) * dp, (batch,) * dp, True))
    for entry in grid.get("scenarios", []):
        values = dict(entry)
        for field in ("tokens_per_dp_rank", "logit_rows_per_dp_rank",
                      "useful_tokens_per_dp_rank", "active_dp_rails"):
            if values.get(field) is not None:
                values[field] = tuple(values[field])
        scenarios.append(ForwardScenario(**values))
    for scenario in scenarios:
        scenario.validate(parallel)
    return scenarios
