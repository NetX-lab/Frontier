"""Validated V4 standalone and routed operator models at measured coordinates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pandas as pd

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.config.precision_type import PrecisionType
from frontier.execution_time_predictor.deepseek_v4 import validate_measured_identity
from frontier.execution_time_predictor.profiling_metadata import (
    CHECKPOINT_MXFP4_BACKEND_IDENTITIES, LINEAR_BACKEND_PRECISIONS,
)
from frontier.moe_ep_workload import materialize_balanced_ep_workload, materialize_balanced_rank_classes
from frontier.operators.deepseek_v4_axes import (
    DEEPSEEK_V4_OPERATOR_AXES, select_deepseek_v4_operator_tp,
)
from frontier.operators.families import DEEPSEEK_V4_MHC_FAMILY
from frontier.operators.binding import bind_operator_query
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN, matches_resolved_layer_contract,
    validate_typed_operator_contracts,
)
from frontier.profiling.moe.moe_rank_plan import CLASS_COLUMNS
from frontier.types import MeasurementType

OPERATOR_MODEL_PREFIX = "dsv4_operator__"
GROUPED_GEMM = "moe_grouped_gemm"
GATE_OPERATORS = ("moe_gating_linear", "moe_gating_routing_topk")
MOE_COORDINATE_COLUMNS = ("expert_parallel_size", "num_tokens", *CLASS_COLUMNS)


def _reject(operator, message):
    raise DeepSeekV4CoverageError(f"V4 {operator}: {message}")


def _positive_integer(value, operator, column, *, allow_zero=False):
    if (isinstance(value, bool) or not math.isfinite(float(value))
            or int(value) != value or value < (0 if allow_zero else 1)):
        _reject(operator, f"{column} requires an integer {'non-negative' if allow_zero else 'positive'} value")
    return int(value)


def _read_rows(path, model, replica, measurement):
    if measurement is not MeasurementType.KERNEL_ONLY:
        _reject(path, "operator tables require KERNEL_ONLY")
    try:
        rows = pd.read_csv(path, float_precision="round_trip")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DeepSeekV4CoverageError(f"V4 measured table cannot be read: {path}: {exc}") from exc
    for column in (TYPED_OPERATOR_CONTRACTS_COLUMN, "runtime_fingerprint",
                   "num_tokens", "num_tensor_parallel_workers"):
        if column not in rows or rows[column].isna().any():
            _reject(path, f"missing {column}")
    for _, group in rows.groupby("runtime_fingerprint"):
        validate_measured_identity(group, model, replica, measurement)
    return rows


def _layer_contract(model, replica, operator, ep):
    return model.get_model_architecture_profile().resolve_layer_contract(
        model, layer_id=0, operator_name=operator,
        attention_tp_size=replica.attn_tensor_parallel_size,
        attention_dp_size=replica.attn_dp, num_pipeline_stages=replica.num_pipeline_stages,
        moe_tp_size=replica.moe_tensor_parallel_size, expert_parallel_size=ep,
    )


def _contracts(row, model, replica, *, ep):
    profile = model.get_model_architecture_profile()
    try:
        contracts = validate_typed_operator_contracts(
            row[TYPED_OPERATOR_CONTRACTS_COLUMN], architecture_profile=profile, model_config=model,
        )
    except ValueError as exc:
        raise DeepSeekV4CoverageError(f"V4 typed operator contract rejected: {exc}") from exc
    if not contracts:
        _reject("typed_operator_contracts", "requires at least one operator")
    for name, metadata in contracts.items():
        binding = bind_operator_query(name)
        family, operator = binding.family, binding.operator
        if family is DEEPSEEK_V4_MHC_FAMILY:
            expected = {
                "operator_family_id": family.family_id,
                "tensor_parallel_mode": operator.tp_mode.value,
                "selected_tensor_parallel_size": select_deepseek_v4_operator_tp(
                    operator.tp_mode, replica.attn_tensor_parallel_size),
                "layer_kind": None,
            }
            for column, value in expected.items():
                if metadata[column] != value:
                    _reject(name, f"{column}={metadata[column]!r} disagrees with deployment/registry ({value!r})")
        else:
            contract = _layer_contract(model, replica, name, ep)
            if not matches_resolved_layer_contract(contracts, contract, operator_name=name):
                _reject(name, "typed layer contract disagrees with deployment/registry")
    return contracts


def _targets(row, name):
    target = f"time_stats.{name}.median"
    count = f"time_stats.{name}.kernel_count"
    if target not in row or not math.isfinite(float(row[target])) or row[target] <= 0:
        _reject(name, f"requires finite positive {target}")
    if count not in row:
        _reject(name, f"missing {count}")
    _positive_integer(row[count], name, count)
    return {target: row[target], count: row[count]}


def load_dsv4_operator_dataframe(path, model, replica, measurement=MeasurementType.KERNEL_ONLY):
    """Normalize each admitted operator using its typed row axis and selected TP."""
    rows = _read_rows(path, model, replica, measurement)
    normalized = []
    for _, row in rows.iterrows():
        contracts = _contracts(row, model, replica, ep=replica.moe_expert_parallel_size)
        for name, metadata in contracts.items():
            if name not in DEEPSEEK_V4_OPERATOR_AXES:
                _reject(name, "has no standalone row axis")
            axis = DEEPSEEK_V4_OPERATOR_AXES[name]
            if metadata.get("row_axis") != axis:
                _reject(name, f"row_axis disagrees with registry ({axis})")
            # The union's shared rows keep their axis in the per-operator contract.
            if pd.notna(row.get("row_axis")) and row.row_axis != axis:
                _reject(name, f"row_axis={row.row_axis!r} disagrees with typed axis {axis}")
            tp = metadata["selected_tensor_parallel_size"]
            if name.startswith("share_expert_") and name != "share_expert_act":
                if "linear_backends" not in row or pd.isna(row.linear_backends):
                    _reject(name, "missing projection backend identity")
                backends = json.loads(row.linear_backends).get(name, [])
                if not backends or any(LINEAR_BACKEND_PRECISIONS.get(b) is not PrecisionType.FP8 for b in backends):
                    _reject(name, "requires the qualified FP8 DeepGEMM projection backend")
            _positive_integer(row.num_tokens, name, axis)
            normalized.append({**row.to_dict(), "operator": name, "row_axis": axis,
                               "selected_tensor_parallel_size": tp, **_targets(row, name)})
    return pd.DataFrame(normalized)


def load_dsv4_moe_dataframe(path, model, replica, measurement=MeasurementType.KERNEL_ONLY):
    """Admit balanced MARLIN rows and the producer's shared rank-class algebra."""
    from frontier.execution_time_predictor.dsv4_predictor import validate_balanced_routing

    rows = _read_rows(path, model, replica, measurement)
    # MoE rows retain the routing identity columns, without attention's moe_routing blob.
    validate_balanced_routing(rows, require_descriptor=False)
    for _, row in rows.iterrows():
        for column in ("expert_parallel_size", "ep_rank", "gating_num_tokens", *CLASS_COLUMNS):
            if column not in row or pd.isna(row[column]):
                _reject(GROUPED_GEMM, f"missing {column}")
        ep = _positive_integer(row.expert_parallel_size, GROUPED_GEMM, "expert_parallel_size")
        _contracts(row, model, replica, ep=ep)
        backend = row.get("moe_grouped_gemm_backend")
        if backend not in CHECKPOINT_MXFP4_BACKEND_IDENTITIES:
            _reject(GROUPED_GEMM, f"unqualified checkpoint-expert backend {backend!r}")
        for column, expected in CHECKPOINT_MXFP4_BACKEND_IDENTITIES[backend].items():
            if row.get(column) != expected:
                _reject(GROUPED_GEMM, f"{column} requires {expected!r}, got {row.get(column)!r}")
        for column, expected in {
            "num_experts": model.num_experts, "router_topk": model.num_experts_per_tok,
            "hidden_dim": model.embedding_dim,
            "expert_hidden_dim": _layer_contract(model, replica, GROUPED_GEMM, ep).effective_ffn_width,
            "num_tensor_parallel_workers": replica.moe_tensor_parallel_size,
        }.items():
            if row.get(column) != expected:
                _reject(GROUPED_GEMM, f"{column} disagrees with model/deployment ({expected})")
        rank = _positive_integer(row.ep_rank, GROUPED_GEMM, "ep_rank", allow_zero=True)
        population = _positive_integer(row.num_tokens, GROUPED_GEMM, "num_tokens")
        classes = materialize_balanced_rank_classes(
            routing_token_count=population, router_topk=model.num_experts_per_tok,
            total_expert_num=model.num_experts, moe_expert_parallel_size=ep,
        )
        if rank >= ep or tuple(row[c] for c in CLASS_COLUMNS) != classes[rank]:
            _reject(GROUPED_GEMM, "rank-class coordinate disagrees with balanced routing")
        _positive_integer(row.gating_num_tokens, GATE_OPERATORS, "gating_num_tokens")
        for name in (*GATE_OPERATORS, GROUPED_GEMM):
            _targets(row, name)
    return rows


def operator_model_name(operator, row_axis, tp):
    return OPERATOR_MODEL_PREFIX + "__".join(map(str, (operator, row_axis, tp)))


def moe_model_name(operator):
    return OPERATOR_MODEL_PREFIX + operator


def train_dsv4_operator_models(operators, moe, train):
    models = {}
    for (name, axis, tp), rows in operators.groupby(["operator", "row_axis", "selected_tensor_parallel_size"]):
        key = operator_model_name(name, axis, tp)
        models[key] = train(model_name=key, df=rows, feature_cols=["num_tokens"],
                            target_col=f"time_stats.{name}.median")
    for name in (*GATE_OPERATORS, GROUPED_GEMM):
        columns = list(MOE_COORDINATE_COLUMNS) if name == GROUPED_GEMM else ["gating_num_tokens"]
        key = moe_model_name(name)
        models[key] = train(model_name=key, df=moe, feature_cols=columns,
                            target_col=f"time_stats.{name}.median")
    return models


@dataclass(frozen=True)
class OperatorPrediction:
    time_ms: float
    kernel_count: float


class DeepSeekV4OperatorPredictor:
    def __init__(self, models):
        self._models = {key: value for key, value in models.items() if key.startswith(OPERATOR_MODEL_PREFIX)}

    def _exact(self, name, key, coordinates):
        model = self._models.get(key)
        if model is None or coordinates not in model._frontier_exact_lookup:
            _reject(name, f"missing measured coordinate {coordinates}, model {key}")
        count = model._frontier_kernel_count_model
        if coordinates not in count._frontier_exact_lookup:
            _reject(name, f"missing measured kernel count at {coordinates}")
        return OperatorPrediction(model._frontier_exact_lookup[coordinates], count._frontier_exact_lookup[coordinates])

    def operator(self, name, rows, tp):
        return self._exact(name, operator_model_name(name, DEEPSEEK_V4_OPERATOR_AXES[name], tp), (float(rows),))

    def gate(self, name, local_rows):
        return self._exact(name, moe_model_name(name), (float(local_rows),))

    def grouped_gemm(self, lane):
        population = sum(lane.global_token_counts) // lane.router_topk
        classes = materialize_balanced_rank_classes(
            routing_token_count=population, router_topk=lane.router_topk,
            total_expert_num=lane.total_expert_num, moe_expert_parallel_size=lane.moe_expert_parallel_size,
        )
        rank_class = classes[lane.ep_id]
        measured_lane = materialize_balanced_ep_workload(
            routing_token_count=population, router_topk=lane.router_topk,
            total_expert_num=lane.total_expert_num,
            moe_expert_parallel_size=lane.moe_expert_parallel_size,
        ).lane(lane.ep_id)
        if lane.global_token_counts != measured_lane.global_token_counts:
            _reject(GROUPED_GEMM, "lane is not the measured balanced rank class")
        coordinate = (float(lane.moe_expert_parallel_size), float(population), *map(float, rank_class))
        return self._exact(GROUPED_GEMM, moe_model_name(GROUPED_GEMM), coordinate)


def operator_tensor_parallel_size(model, replica, name):
    binding = bind_operator_query(name)
    if binding.family_id == DEEPSEEK_V4_MHC_FAMILY.family_id:
        return select_deepseek_v4_operator_tp(binding.operator.tp_mode, replica.attn_tensor_parallel_size)
    return _layer_contract(model, replica, name, replica.moe_expert_parallel_size).tensor_parallel_size


def operator_rows(name, *, sp_local_rows, descriptor_rows, logits_rows):
    return {"sp_local_rows": sp_local_rows, "descriptor_rows": descriptor_rows,
            "logits_rows": logits_rows}[DEEPSEEK_V4_OPERATOR_AXES[name]]
