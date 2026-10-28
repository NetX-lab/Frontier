"""Pure planner for the DeepSeek-V4 vLLM forward collectives measured by G5h.

Native references are to commit 2cf0a69:

* ``models/deepseek_v4/nvidia/model.py:804-812,962-967,1156-1157,1196-1197``:
  SP admission, per-layer attention gather/scatter, local shard and final mHC gather.
* ``models/common/ops/sequence_parallel.py:23-39``: custom collectives first;
  reduce-scatter pads the unpadded input to a multiple of TP.
* ``distributed/device_communicators/custom_all_reduce.py:399-500``: custom
  and MNNVL eligibility depends on bytes, dtype and contiguity.
* ``distributed/device_communicators/all2all.py``: AgRsAll2AllManager groups
  hidden, float32 top-k weights and int32 ids in one dispatch, then combines
  with reduce_scatterv. ``forward_context.py:102-113`` repeats each lane's
  ceil(T / TP) rows for every TP rank in that lane.
* ``model_executor/layers/logits_processor.py:118-129`` and
  ``platforms/interface.py:1102-1106``: eager logits gather on dimension -1.

Every V4 layer is MoE. G5h uses the default allgather_reducescatter backend;
DP host synchronization is explicitly priced as zero and has no planned call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

from frontier.profiling.collectives.vllm_forward_collectives import (
    CollectiveCall,
    ForwardScenario,
    ServingParallelism,
    TOPK_IDS_DTYPE,
    TOPK_WEIGHTS_DTYPE,
    VOCAB_PADDING_SIZE,
    ceil_div,
)

CALL_NAMES = (
    "embedding_all_reduce", "sp_attention_all_gather", "sp_attention_reduce_scatter",
    "sp_final_mhc_all_gather", "moe_dispatch_all_gatherv",
    "moe_combine_reduce_scatterv", "logits_all_gather",
)
GRID_KEYS = ("graph_tokens_per_lane", "eager_lane_tokens", "logits_rows_per_lane", "calls")


@dataclass(frozen=True)
class DeepseekV4Shape:
    hidden_size: int
    num_hidden_layers: int
    num_experts_per_tok: int
    vocab_size: int
    hc_mult: int

    @classmethod
    def from_hf_config(cls, config: dict) -> "DeepseekV4Shape":
        if config.get("n_routed_experts") is None:
            raise ValueError("Config has no routed experts; this path covers MoE models")
        return cls(
            hidden_size=int(config["hidden_size"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_experts_per_tok=int(config["num_experts_per_tok"]),
            vocab_size=int(config["vocab_size"]),
            hc_mult=int(config["hc_mult"]),
        )


def dsv4_forward_collectives(model: DeepseekV4Shape, parallel: ServingParallelism,
                             scenario: ForwardScenario) -> list[CollectiveCall]:
    """Distinct collective calls, in the G5h §3 call-list order, of one step."""
    parallel.validate_sequence_parallel_moe()
    tp, dp = parallel.tensor_parallel_size, parallel.data_parallel_size
    if len(scenario.tokens_per_dp_rank) != dp or len(scenario.logit_rows_per_dp_rank) != dp:
        raise ValueError("Scenario must give one token and logits count per DP rank")
    if min(scenario.tokens_per_dp_rank) <= 0:
        raise ValueError("Every DP rank runs at least one token (idle ranks run a dummy batch)")
    if scenario.cuda_graph and len(set(scenario.tokens_per_dp_rank)) != 1:
        raise ValueError("CUDA graph steps are DP-padded to a uniform size")

    def per_rank(values):
        return tuple(value for value in values for _ in range(tp))

    tokens = per_rank(scenario.tokens_per_dp_rank)
    sp_rows = per_rank(ceil_div(n, tp) for n in scenario.tokens_per_dp_rank)
    logit_rows = per_rank(scenario.logit_rows_per_dp_rank)
    padded_vocab = ceil_div(model.vocab_size, VOCAB_PADDING_SIZE) * VOCAB_PADDING_SIZE
    if padded_vocab % tp:
        raise ValueError("Padded vocab is not divisible by TP")
    act, hidden, k = parallel.activation_dtype, model.hidden_size, model.num_experts_per_tok
    layers = model.num_hidden_layers
    calls = [
        CollectiveCall("embedding_all_reduce", "all_reduce", "tp", 0, (hidden,), (act,),
                       tokens, 1, "vocab_parallel_embedding.py:506"),
        CollectiveCall("sp_attention_all_gather", "sp_all_gather", "tp", 0,
                       (hidden,), (act,), sp_rows, layers, "deepseek_v4/nvidia/model.py:962-963"),
        CollectiveCall("sp_attention_reduce_scatter", "sp_reduce_scatter", "tp", 0,
                       (hidden,), (act,), tokens, layers, "deepseek_v4/nvidia/model.py:966-967"),
        # Native [r, hc_mult, hidden] is contiguous. The bytes-equal 2-D tensor
        # takes the same custom/NCCL path, which depends on bytes and dtype.
        CollectiveCall("sp_final_mhc_all_gather", "sp_all_gather", "tp", 0,
                       (model.hc_mult * hidden,), (act,), sp_rows, 1,
                       "deepseek_v4/nvidia/model.py:1196-1197"),
        CollectiveCall("moe_dispatch_all_gatherv", "all_gatherv", "ep", 0,
                       (hidden, k, k), (act, TOPK_WEIGHTS_DTYPE, TOPK_IDS_DTYPE),
                       sp_rows, layers, "all2all.py AgRsAll2AllManager.dispatch"),
        CollectiveCall("moe_combine_reduce_scatterv", "reduce_scatterv", "ep", 0,
                       (hidden,), (act,), sp_rows, layers,
                       "all2all.py AgRsAll2AllManager.combine"),
    ]
    calls = [replace(call, captured=scenario.cuda_graph) for call in calls]
    if any(logit_rows):
        calls.append(CollectiveCall(
            "logits_all_gather", "all_gather", "tp", -1, (padded_vocab // tp,), (act,),
            logit_rows, 1, "logits_processor.py:126"))
    return calls


def dsv4_plan(model: DeepseekV4Shape, parallel: ServingParallelism,
              grid: dict) -> list[tuple[ForwardScenario, CollectiveCall]]:
    """Select G5h graph, ordered eager-lane and eager-logits measurements."""
    unknown = set(grid) - set(GRID_KEYS)
    if unknown:
        raise ValueError(f"Unknown grid keys: {sorted(unknown)}")
    selected = grid.get("calls")
    if selected is not None and set(selected) - set(CALL_NAMES):
        raise ValueError(f"Unknown calls: {sorted(set(selected) - set(CALL_NAMES))}")
    dp = parallel.data_parallel_size
    plan = []

    def append(scenario, keep):
        plan.extend((scenario, call) for call in
                    dsv4_forward_collectives(model, parallel, scenario)
                    if keep(call) and (selected is None or call.name in selected))

    for tokens in grid.get("graph_tokens_per_lane", []):
        append(ForwardScenario(f"graph_t{tokens}", (tokens,) * dp, (0,) * dp, True),
               lambda call: True)
    for lanes in product(grid.get("eager_lane_tokens", []), repeat=dp):
        uniform = len(set(lanes)) == 1
        # Mixed TP lanes have different costs; a maximum over all ranks would
        # mix those lanes. EP calls depend on the entire ordered lane vector.
        append(ForwardScenario("eager_t" + "_".join(map(str, lanes)), lanes, (0,) * dp, False),
               lambda call: uniform or call.group == "ep")
    for rows in grid.get("logits_rows_per_lane", []):
        append(ForwardScenario(f"logits_b{rows}", (rows,) * dp, (rows,) * dp, False),
               lambda call: call.name == "logits_all_gather")
    return plan
