"""Shared utility for splitting profiling results into sharded and replicated rows."""

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Dict, List, Optional, Set, Tuple


def split_replicated_result(
    result: Dict,
    replicated_op_names: Set[str],
    unpadded_n_embd: Optional[int] = None,
    unpadded_n_expanded_embd: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Split a profiling result into sharded-only and replicated-only rows.

    The sharded row keeps the original TP size and padded dimensions.
    The replicated row forces num_tensor_parallel_workers=1 and uses
    unpadded dimensions (if provided).

    Args:
        result: Original profiling result dict with "time_stats" key.
        replicated_op_names: Set of op names considered replicated.
        unpadded_n_embd: Original embedding dim (for linear_op dimension correction).
        unpadded_n_expanded_embd: Original MLP hidden dim (for linear_op dimension correction).

    Returns:
        (sharded_row, replicated_row) — both are new dicts (shallow copy).
    """
    time_stats = result["time_stats"]

    replicated_stats = {k: v for k, v in time_stats.items() if k in replicated_op_names}
    sharded_stats = {k: v for k, v in time_stats.items() if k not in replicated_op_names}

    sharded_row = {**result, "time_stats": sharded_stats}

    replicated_row = {**result, "time_stats": replicated_stats}

    typed_contracts = result.get("typed_operator_contracts")
    if typed_contracts is not None:
        if not isinstance(typed_contracts, Mapping):
            raise TypeError(
                "typed_operator_contracts must be a mapping when present, "
                f"got {type(typed_contracts).__name__}"
            )
        sharded_row["typed_operator_contracts"] = deepcopy(
            {
                operator_name: typed_contracts[operator_name]
                for operator_name in sharded_stats
                if operator_name in typed_contracts
            }
        )
        replicated_row["typed_operator_contracts"] = deepcopy(
            {
                operator_name: typed_contracts[operator_name]
                for operator_name in replicated_stats
                if operator_name in typed_contracts
            }
        )

    replicated_row["num_tensor_parallel_workers"] = 1
    if unpadded_n_embd is not None:
        replicated_row["padded_n_embd"] = unpadded_n_embd
    if unpadded_n_expanded_embd is not None:
        replicated_row["padded_n_expanded_embd"] = unpadded_n_expanded_embd

    return sharded_row, replicated_row


def _compatible_tp1_rows(first: Dict, later: Dict) -> bool:
    first_contracts = first.get("typed_operator_contracts", {})
    later_contracts = later.get("typed_operator_contracts", {})
    overlap = first["time_stats"].keys() & later["time_stats"].keys()
    if any(first_contracts.get(name) != later_contracts.get(name) for name in overlap):
        return False

    new_operators = later["time_stats"].keys() - first["time_stats"].keys()
    observed_rows = (first, later) if new_operators else (later,)
    contracts = [
        row.get("typed_operator_contracts", {}).get(name, {})
        for row in observed_rows for name in row["time_stats"]
    ]
    replicated_non_layer = bool(contracts) and all(
        contract.get("tensor_parallel_mode") == "replicated"
        and contract.get("tensor_parallel_sizes") == [1]
        and contract.get("selected_tensor_parallel_size") == 1
        and contract.get("expert_parallel_mode") == "off"
        and contract.get("selected_expert_parallel_size") is None
        and contract.get("layer_kind") is None
        and contract.get("dimension_source") is None
        and contract.get("effective_ffn_width") is None
        and contract.get("selected_padded_ffn_width") is None
        for contract in contracts
    ) and len({contract.get("profile_id") for contract in contracts}) == 1
    if new_operators and not replicated_non_layer:
        return False

    ignored_fields = {"time_stats", "typed_operator_contracts", "linear_backends"}
    if replicated_non_layer:
        # Non-layer replicated contracts do not consume the FFN shape context.
        ignored_fields.update(("n_expanded_embd", "padded_n_expanded_embd"))
    return all(
        first[field] == later[field]
        for field in (first.keys() & later.keys()) - ignored_fields
    )


def _union_tp1_observations(first: Dict, later: Dict) -> Dict:
    new_operators = [name for name in later["time_stats"] if name not in first["time_stats"]]
    if not new_operators:
        return first

    combined = deepcopy({**later, **first})
    for name in new_operators:
        combined["time_stats"][name] = deepcopy(later["time_stats"][name])
        combined["typed_operator_contracts"][name] = deepcopy(
            later["typed_operator_contracts"][name]
        )

    if "linear_backends" in first or "linear_backends" in later:
        first_backends = first.get("linear_backends", {})
        later_backends = later.get("linear_backends", {})
        if isinstance(first_backends, str):
            first_backends = json.loads(first_backends)
        if isinstance(later_backends, str):
            later_backends = json.loads(later_backends)
        backends = {
            **first_backends,
            **{name: later_backends[name] for name in new_operators if name in later_backends},
        }
        source = first.get("linear_backends", later.get("linear_backends"))
        combined["linear_backends"] = (
            json.dumps(backends, sort_keys=True) if isinstance(source, str) else backends
        )
    return combined


def deduplicate_tp1_rows(
    results: List[Dict],
    tp1_key_fields: Tuple[str, ...] = ("num_tokens",),
) -> List[Dict]:
    """Keep first TP1 observations and union compatible replicated operators.

    This function operates on in-memory results from a single profile_model() call.
    quant_signature is NOT available at this stage (added post-hoc to DataFrame).
    Measurement types and conflicting row contexts remain separate. Only typed,
    non-layer replicated operators may augment an earlier observation; their
    unused FFN shape context does not change ownership of any measured operator.

    Args:
        results: List of result dicts (mix of sharded and replicated rows).
        tp1_key_fields: Fields that form the dedup key for TP=1 rows.
            Linear op: ("num_tokens", "model_arch") — model_arch in result dict.
            MoE: ("num_tokens",) — model_arch not in result dict; EP excluded
            because replicated ops are EP-agnostic.

    Returns:
        Deduplicated list.
    """
    key_fields = tuple(dict.fromkeys((*tp1_key_fields, "measurement_type")))
    tp1_rows: Dict[Tuple, List[int]] = {}
    deduped: List[Dict] = []
    for row in results:
        if row["num_tensor_parallel_workers"] != 1:
            deduped.append(row)
            continue
        key = tuple(row.get(field) for field in key_fields)
        candidates = tp1_rows.setdefault(key, [])
        for index in candidates:
            if _compatible_tp1_rows(deduped[index], row):
                deduped[index] = _union_tp1_observations(deduped[index], row)
                break
        else:
            candidates.append(len(deduped))
            deduped.append(row)
    return deduped
