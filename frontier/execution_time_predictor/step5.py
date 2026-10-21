"""Measured Step attention rows and fitting through the shared model cache.

CPU fixtures exercise this contract without admitting native profiling results.
Physical owners and workload semantics remain in ``frontier.attention.step5``.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from numbers import Real
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.ops import AttentionPhase
from frontier.attention.profiling_mapping import (
    STEP5_FEATURE_COLUMNS, STEP5_NUMERIC_COLUMNS, validate_attention_profiling_dataframe,
)
from frontier.attention.step5 import (
    Step5AttentionWorkload, describe_step5_layer, describe_step5_stage_operator_tensors,
    resolve_step5_model_precision,
)
from frontier.attention.step5_inputs import validate_step5_input_row
from frontier.config.precision_type import PrecisionType
from frontier.operators.typed_contracts import (
    parse_typed_operator_contracts,
    validate_typed_operator_metadata,
)
from frontier.profiling.step5.rows import operator_contracts
from frontier.profiling.moe.routed_layer_groups import resolve_routed_layer_groups
from frontier.types import MeasurementType


# Graph code and actual measurement family share one policy table.
STEP5_GRAPH_MODES = {
    "none": (0, MeasurementType.KERNEL_ONLY),
    "piecewise": (1, MeasurementType.CUDA_EVENT),
    "full": (2, MeasurementType.CUDA_EVENT),
}
STEP5_DURATION_STATISTICS = {
    MeasurementType.CUDA_EVENT: "mean",
    MeasurementType.KERNEL_ONLY: "median",
}
STEP5_VARIANTS = {
    name: index for index, name in enumerate(STEP5_ATTENTION_FAMILY.supported_variants)
}
STEP5_PHASES = {phase.value: index for index, phase in enumerate(AttentionPhase)}
STEP5_IDENTITY_COLUMNS = (
    "measurement_type", "attention_backend", "model_architecture_profile",
    "quant_signature", "model_dtype", "norm_dtype", "residual_dtype",
    "runtime_stack_signature", "device", "num_tensor_parallel_workers",
    "block_size", "cache_dtype", "physical_scope", "weight_source", "timing_basis",
)
STEP5_OPERATOR_NAMES = tuple(op.name for op in STEP5_ATTENTION_FAMILY.predictor_ops())
STEP5_SCOPE_NAMES = tuple(op.name for op in STEP5_ATTENTION_FAMILY.profiling_ops())
STEP5_STAGE_BOUNDARY_NAMES = ("input_cast", "final_residual")
_TIME_STATS = ("min", "max", "mean", "median", "std", "count")
_TIMING_BASES = {
    MeasurementType.CUDA_EVENT: "cuda_event_native_runtime_latency",
    MeasurementType.KERNEL_ONLY: "correlated_kernel_plus_memcpy_duration_sum",
}


def step5_measurement_type(graph_mode: str) -> MeasurementType:
    """Select actual graph latency or eager device activity without relabelling."""
    try:
        return STEP5_GRAPH_MODES[graph_mode][1]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported Step graph_mode={graph_mode!r}") from exc


def _lengths(value, name: str, minimum: int) -> tuple[int, ...]:
    values = json.loads(value) if isinstance(value, str) else value
    if not isinstance(values, (tuple, list)) or not values or any(
        type(item) is not int or item < minimum for item in values
    ):
        raise ValueError(f"Step {name} requires integer lengths >= {minimum}")
    return tuple(values)


def _row_workload(row) -> Step5AttentionWorkload:
    queries = _lengths(row["query_lens"], "query_lens", 1)
    contexts = _lengths(row["context_lens"], "context_lens", 0)
    if len(queries) != len(contexts):
        raise ValueError("Step query_lens and context_lens must have equal length")
    return Step5AttentionWorkload(
        phase=row["phase"], query_lengths=queries,
        sequence_lengths=tuple(q + c for q, c in zip(queries, contexts)),
        physical_query_rows=int(row["valid_tokens"] + row["padded_tokens"]),
        block_size_tokens=int(row["block_size"]),
        block_table_columns=int(row["block_table_columns"]),
        max_model_len_tokens=int(row["max_model_len"]),
        region_tokens=int(row["region_block_size"]), topk_regions=int(row["topk_regions"]),
        graph_mode=row["graph_mode"],
        has_attention_metadata=bool(row["has_attention_metadata"]),
    )


def step5_row_features(row) -> dict[str, float]:
    """Materialize numeric producer geometry and the real query/context shape."""
    features = {}
    for name, minimum in STEP5_NUMERIC_COLUMNS.items():
        value = row[name]
        if (not isinstance(value, Real) or isinstance(value, bool)
                or not math.isfinite(value) or value < minimum
                or (name != "rope_theta" and value != int(value))):
            raise ValueError(f"Invalid Step numeric feature {name}={value!r}")
        features[name] = float(value)
    if features["has_attention_metadata"] not in (0., 1.):
        raise ValueError("Step has_attention_metadata must be 0 or 1")
    workload = _row_workload(row)
    kind = validate_step5_input_row(row, workload)
    queries = workload.query_lengths
    contexts = tuple(length - query for length, query in zip(workload.sequence_lengths, queries))
    if (len(queries) != row["batch_size"] or sum(queries) != row["valid_tokens"]
            or row["batch_num_tokens"] != row["valid_tokens"]
            or row["batch_num_prefill_tokens"] + row["batch_num_decode_tokens"] != sum(queries)
            or row["score_regions"] != workload.score_regions):
        raise ValueError("Step query/context shape disagrees with producer token/region metadata")
    if (workload.phase == "prefill" and row["batch_num_decode_tokens"] != 0
            or workload.phase == "decode" and row["batch_num_prefill_tokens"] != 0):
        raise ValueError("Step phase disagrees with prefill/decode token counts")
    if workload.graph_mode == "full" and workload.phase != "decode":
        raise ValueError("Step full graph rows require decode phase")
    try:
        features.update(
            attention_variant_code=float(STEP5_VARIANTS[row["attention_variant"]]),
            phase_code=float(STEP5_PHASES[row["phase"]]),
            graph_mode_code=float(STEP5_GRAPH_MODES[row["graph_mode"]][0]),
            native_workload_code=float(kind.code),
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported Step variant/phase/graph label: {exc}") from exc
    mean_query = sum(queries) / len(queries)
    features.update(
        min_query_len=float(min(queries)), max_query_len=float(max(queries)),
        query_len_cv=math.sqrt(sum((q - mean_query) ** 2 for q in queries) / len(queries)) / mean_query,
        min_context_len=float(min(contexts)), max_context_len=float(max(contexts)),
        mean_context_len=sum(contexts) / len(contexts),
        max_sequence_len=float(max(workload.sequence_lengths)),
    )
    return features


def _validate_targets(frame: pd.DataFrame) -> None:
    """Reject missing actual launch counts before any estimator is fitted."""
    for _, row in frame.iterrows():
        active = set(json.loads(row["active_operators"]))
        counts = []
        for name in (*STEP5_SCOPE_NAMES, *STEP5_OPERATOR_NAMES):
            for stat in _TIME_STATS:
                column = f"time_stats.{name}.{stat}"
                value = row[column]
                if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Step timing coverage requires finite nonnegative {column}")
                if stat == "count":
                    if value < 2 or value != int(value):
                        raise ValueError(f"Step timing sample count must be integral and >= 2: {column}")
                    counts.append(value)
                else:
                    if stat != "std" and (name in active or name == "step5_layer_e2e") and value <= 0:
                        raise ValueError(f"Step active timing coverage requires positive {column}")
                    if name not in active and name != "step5_layer_e2e" and value != 0:
                        raise ValueError(f"Step inactive physical scope must be exactly zero: {column}")
            column = f"time_stats.{name}.kernel_count"
            value = row.get(column)
            if name in STEP5_OPERATOR_NAMES or name in active and value is not None and not pd.isna(value):
                if (not isinstance(value, Real) or isinstance(value, bool)
                        or not math.isfinite(value) or value < 1 or value != int(value)):
                    raise ValueError(
                        f"Step measured kernel coverage missing/invalid {column} at layer "
                        f"{row['global_layer_id']}; sample count and aggregate counts are not launch counts"
                    )
            elif value is not None and not pd.isna(value) and value != 0:
                raise ValueError(f"Step inactive scope has nonzero kernel count: {column}")
        if len(set(counts)) != 1:
            raise ValueError("Step timing statistics require matching sample counts")


def load_step5_attention_rows(path, replica_config, scheduler_config, measurement_type) -> pd.DataFrame:
    """Select and validate producer rows before shared-cache attention fitting."""
    measurement = (measurement_type if isinstance(measurement_type, MeasurementType)
                   else MeasurementType.from_string(measurement_type))
    if measurement not in _TIMING_BASES:
        raise ValueError(f"Step measured attention does not support {measurement.value}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Step measured attention coverage file does not exist: {path}")
    frame = pd.read_csv(path).drop_duplicates()
    schema = replace(STEP5_ATTENTION_FAMILY, execution_enabled=True)
    validate_attention_profiling_dataframe(frame, schema, measurement_type=measurement)
    required = {
        *STEP5_IDENTITY_COLUMNS, *STEP5_NUMERIC_COLUMNS,
        "query_lens", "context_lens", "phase", "native_workload", "active_operators", "typed_operator_contracts",
        *(f"time_stats.step5_layer_e2e.{stat}" for stat in _TIME_STATS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Step measured attention coverage is missing columns: {missing}")
    if frame["timing_basis"].ne(_TIMING_BASES[measurement]).any():
        raise ValueError("Step timing_basis conflicts with actual measurement_type; do not relabel methods")
    selected_policy = frame["graph_mode"].map(step5_measurement_type).eq(measurement)
    frame = frame.loc[selected_policy].copy()
    config = replica_config.model_config
    expected = {
        "model_architecture_profile": config.get_model_architecture_profile().profile_id,
        "quant_signature": config.get_quant_signature(), "device": replica_config.device,
        "num_tensor_parallel_workers": replica_config.attn_tensor_parallel_size,
        "block_size": scheduler_config.block_size, "cache_dtype": "auto",
    }
    for name, value in expected.items():
        frame = frame.loc[frame[name].eq(value)].copy()
    if frame.empty:
        raise ValueError(f"Step measured attention coverage is unavailable for {measurement.value}: {expected}")
    identity_columns = (
        *(name for name in STEP5_IDENTITY_COLUMNS if name not in ("attention_backend", "physical_scope")),
        *(("evidence_kind",) if "evidence_kind" in frame else ()),
    )
    validate_attention_profiling_dataframe(
        frame, schema, measurement_type=measurement, identity_columns=identity_columns,
    )
    backend_by_variant = {}
    for variant, rows in frame.groupby("attention_variant", sort=True):
        validate_attention_profiling_dataframe(
            rows, schema, measurement_type=measurement, identity_columns=("attention_backend",),
        )
        backend_by_variant[variant] = rows.iloc[0]["attention_backend"]
    scope_by_kind = {}
    for kind, rows in frame.groupby("native_workload", sort=True, dropna=False):
        validate_attention_profiling_dataframe(
            rows, schema, measurement_type=measurement, identity_columns=("physical_scope",),
        )
        scope_by_kind[kind] = rows.iloc[0]["physical_scope"]
    metadata_by_layer = {}
    for _, row in frame.iterrows():
        features = step5_row_features(row)
        layer_id = int(row["global_layer_id"])
        if layer_id not in metadata_by_layer:
            metadata_by_layer[layer_id] = describe_step5_layer(
                config, layer_id, attention_tp_size=replica_config.attn_tensor_parallel_size,
                moe_tp_size=1, expert_parallel_size=1,
            )
        metadata = metadata_by_layer[layer_id]
        geometry = asdict(metadata.geometry)
        geometry.update(n_q_head=metadata.geometry.num_q_heads, n_kv_head=metadata.geometry.num_kv_heads,
                        head_size=metadata.geometry.head_dim, rope_theta=metadata.rope_theta,
                        rotary_dim=metadata.rotary_dim, sliding_window=config.sliding_window,
                        region_block_size=config.sparse_config["region_block_size"],
                        topk_regions=config.sparse_config["topk"])
        for name in STEP5_NUMERIC_COLUMNS.keys() & geometry.keys():
            if features[name] != geometry[name]:
                raise ValueError(f"Step source geometry disagrees with {name} at layer {layer_id}")
        if row["attention_variant"] != metadata.attention_spec.variant_id:
            raise ValueError(f"Step attention_variant disagrees with layer {layer_id}")
        if PrecisionType.from_torch_dtype(row["model_dtype"]) is not metadata.model_precision:
            raise ValueError("Step model_dtype conflicts with requested model precision")
        if PrecisionType.from_torch_dtype(row["norm_dtype"]) is not metadata.norm_precision:
            raise ValueError("Step norm_dtype conflicts with source metadata")
        if PrecisionType.from_torch_dtype(row["residual_dtype"]) is not metadata.residual_precision:
            raise ValueError("Step residual_dtype conflicts with source metadata")
        if row["max_model_len"] != scheduler_config.max_model_len:
            raise ValueError("Step measured max_model_len differs from requested serving context")
        scopes = metadata.attention_scopes(_row_workload(row))
        if json.loads(row["fusion"]) != [[name, list(work)] for name, work in scopes]:
            raise ValueError("Step fusion scopes differ from source physical ownership")
        active = json.loads(row["active_operators"])
        if len(active) != len(set(active)) or set(active) != {name for name, _ in scopes}:
            raise ValueError("Step active_operators differ from source physical scopes")
        contracts = parse_typed_operator_contracts(row["typed_operator_contracts"])
        expected_contracts = {
            name: contract for name, contract in operator_contracts(
                config, layer_id, replica_config.attn_tensor_parallel_size,
            ).items() if name in STEP5_SCOPE_NAMES
        }
        if set(contracts) != set(expected_contracts):
            raise ValueError("Step typed scopes must cover exactly the registered attention operators")
        for name, contract in contracts.items():
            validate_typed_operator_metadata(
                contract, operator_name=name, expected_metadata=expected_contracts[name],
                architecture_profile=config.get_model_architecture_profile(),
            )
    derived_precision = frame["model_dtype"].map(lambda dtype: PrecisionType.from_torch_dtype(dtype).name)
    if "profiling_precision" in frame and any(
        PrecisionType.from_string(value).name != derived
        for value, derived in zip(frame["profiling_precision"], derived_precision)
    ):
        raise ValueError("Step profiling_precision conflicts with explicit model_dtype")
    frame["profiling_precision"] = derived_precision
    _validate_targets(frame)
    features = pd.DataFrame([step5_row_features(row) for _, row in frame.iterrows()], index=frame.index)
    for name in STEP5_FEATURE_COLUMNS:
        frame[name] = features[name]
    frame.attrs["measurement_identity"] = {
        name: (frame.iloc[0][name].item() if hasattr(frame.iloc[0][name], "item")
               else frame.iloc[0][name]) for name in identity_columns
    }
    frame.attrs["measurement_identity"]["attention_backend"] = backend_by_variant
    frame.attrs["measurement_identity"]["physical_scope"] = scope_by_kind
    return frame


def load_step5_attention_input_union(
    manager, cluster_type, replica_config, scheduler_config, predictor_config, path, measurement_type,
):
    """Load each declared role's actual compatible attention input before one fit."""
    declared = getattr(manager, "_cluster_configs", None)
    if not declared:
        return load_step5_attention_rows(path, replica_config, scheduler_config, measurement_type), predictor_config

    from frontier.attention.model_binding import resolve_runtime_attention_family
    from frontier.config import global_vars
    from frontier.execution_time_predictor.cache_io import dataset_fingerprint
    from frontier.execution_time_predictor.measurement_input_paths import resolve_measurement_input_paths
    from frontier.execution_time_predictor.step5_cost import step5_measurements_for_role

    graph_mode = str(global_vars.get_decode_cuda_graph_mode()).lower()
    sources = {}
    for role in {*declared, cluster_type}:
        if measurement_type not in step5_measurements_for_role(role, graph_mode):
            continue
        if role == cluster_type:
            # The existing component API supplies this role's resolved input explicitly.
            replica, scheduler, config, resolved = replica_config, scheduler_config, predictor_config, path
        else:
            cluster = declared[role]
            replica, scheduler, config = (cluster.replica_config, cluster.replica_scheduler_config,
                                          cluster.execution_time_predictor_config)
            if resolve_runtime_attention_family(replica.model_config).family_id != STEP5_ATTENTION_FAMILY.family_id:
                continue
            resolved = resolve_measurement_input_paths(
                config, measurement_type, device=replica.device,
                model=replica.model_config.get_name(), network_device=replica.network_device,
            ).attention
        sources[role] = (replica, scheduler, config, resolved)
    if not sources:
        raise ValueError(f"No declared Step attention role requires {measurement_type.value}")

    topologies = {}
    for role in sorted(sources, key=str):
        replica, scheduler, _, _ = sources[role]
        physical = {
            "device": replica.device, "model_name": replica.model_config.get_name(),
            "quant_signature": replica.model_config.get_quant_signature(),
            "attention_tp": replica.attn_tensor_parallel_size, "block_size": scheduler.block_size,
            **({"kv_cache_dtype": replica.kv_cache_dtype} if hasattr(replica, "kv_cache_dtype") else {}),
        }
        if topologies:
            reference = next(iter(topologies.values()))
            different = sorted(name for name in reference.keys() | physical.keys()
                               if reference.get(name) != physical.get(name))
            if different:
                raise ValueError(f"incompatible Step attention input topology for {role}: {different}")
        topologies[role] = physical

    frames, inputs = [], []
    identity = None
    canonical_config = None
    for role in sorted(sources, key=str):
        replica, scheduler, config, resolved = sources[role]
        frame = load_step5_attention_rows(resolved, replica, scheduler, measurement_type)
        actual = frame.attrs["measurement_identity"]
        if identity is None:
            identity = {**actual, **{name: dict(actual[name]) for name in ("attention_backend", "physical_scope")}}
            canonical_config = config
        else:
            for name in identity.keys() | actual.keys():
                if name in ("attention_backend", "physical_scope"):
                    continue
                if name not in identity or name not in actual or identity[name] != actual[name]:
                    raise ValueError(f"incompatible Step attention input identity: {name} for {role}")
            for name in ("attention_backend", "physical_scope"):
                for label, value in actual[name].items():
                    previous = identity[name].get(label)
                    if previous is not None and previous != value:
                        raise ValueError(f"incompatible Step attention input identity: {name}/{label}")
                    identity[name][label] = value
        frames.append(frame)
        inputs.append({
            "role": str(role), "path": str(Path(resolved).resolve()),
            "sha256": dataset_fingerprint(resolved),
            "topology": {name: value for name, value in topologies[role].items() if name != "quant_signature"},
            "shape_policy": {name: getattr(scheduler, name) for name in (
                "block_size", "max_model_len", "max_tokens_in_batch", "batch_size_cap",
                "long_prefill_token_threshold", "enable_chunked_prefill",
            ) if hasattr(scheduler, name)},
            "attention_measurements": [method.value for method in step5_measurements_for_role(role, graph_mode)],
        })
    union = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    # Bind declared sources/policies into the existing dataframe cache hash, not a second cache.
    union["step5_attention_inputs"] = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    union.attrs["measurement_identity"] = {**identity, "attention_inputs": inputs}
    return union, canonical_config


def set_step5_model_coverage(model, frame: pd.DataFrame) -> None:
    """Persist only the rows and physical domains that actually trained this operator."""
    model._frontier_step5_coverage = frozenset(
        tuple(float(row[name]) for name in STEP5_FEATURE_COLUMNS) for _, row in frame.iterrows()
    )
    model._frontier_feature_bounds = {
        name: (float(frame[name].min()), float(frame[name].max())) for name in STEP5_FEATURE_COLUMNS
    }
    model._frontier_step5_domains = {
        name: tuple(sorted(float(value) for value in frame[name].unique()))
        for name in ("global_layer_id", "attention_variant_code", "phase_code", "graph_mode_code",
                     "has_attention_metadata", "native_workload_code")
    }


def train_step5_attention_models(manager, df, predictor_config, training_context) -> dict[str, BaseEstimator]:
    """Fit measured durations and actual counts in the existing atomic cache chain."""
    _validate_targets(df)
    context = dict(training_context)
    identity = dict(context.get("measurement_identity", {}))
    for name, value in df.attrs["measurement_identity"].items():
        if name in identity and identity[name] != value:
            raise ValueError(f"Step training context conflicts with source measurement identity: {name}")
        identity[name] = value
    identity["feature_columns"] = list(STEP5_FEATURE_COLUMNS)
    statistic = STEP5_DURATION_STATISTICS[MeasurementType(identity["measurement_type"])]
    identity["duration_statistic"] = statistic
    context["measurement_identity"] = identity
    context["step5_training"] = True
    models = {}
    for name in STEP5_OPERATOR_NAMES:
        selected = df
        targets = [f"time_stats.{name}.{statistic}", f"time_stats.{name}.kernel_count"]
        # Repeated duration measurements use the shared exact-lookup aggregation.
        if (selected.groupby(list(STEP5_FEATURE_COLUMNS))[targets[1]].nunique() > 1).any():
            raise ValueError(f"Conflicting Step measured kernel counts for physical scope {name}")
        pair = []
        for model_name, target in zip((name, f"{name}_kernel_count"), targets):
            pair.append(manager._train_single_model(
                model_name=model_name, df=selected, feature_cols=list(STEP5_FEATURE_COLUMNS),
                target_col=target, execution_time_predictor_config=predictor_config,
                training_context=context, persist_exact_lookup=True, count_kernels=False,
            ))
            models[model_name] = pair[-1]
        pair[1]._frontier_operator_name = name
        pair[0]._frontier_kernel_count_model = pair[1]
    return models


def step5_operator_training_rows(operator_name, frame, training_context, measurement_type):
    """Select actual outside-root measurements and their physical precision domains.

    The producer's profiling_precision labels its input dtype, not FP8 expert
    weights or the FP32 router. Preserve that label while selecting estimator
    precision from explicit routed-group metadata and the source projection.
    """
    config = training_context["step5_generic_compute"]
    stage_boundary = operator_name in STEP5_STAGE_BOUNDARY_NAMES
    required = {"profiling_precision", "measurement_type", "model_architecture_profile",
                "quant_signature", "typed_operator_contracts"}
    if stage_boundary:
        required.update(("num_tokens", "num_tensor_parallel_workers", "n_embd"))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Step {operator_name} coverage is missing source metadata: {missing}")
    for name, expected in (("measurement_type", measurement_type.value),
                           ("model_architecture_profile", "step5"),
                           ("quant_signature", config.get_quant_signature())):
        if frame[name].ne(expected).any():
            raise ValueError(f"Step {operator_name} source {name} mismatch; do not relabel measurements")
    target = f"time_stats.{operator_name}.median"
    if stage_boundary:
        present = frame["typed_operator_contracts"].map(
            lambda value: operator_name in parse_typed_operator_contracts(value))
        for column in (target, f"time_stats.{operator_name}.kernel_count"):
            if column in frame:
                present |= frame[column].notna()
        selected = frame.loc[present].copy()
        if selected.empty:
            raise ValueError(f"Step {operator_name} requires positive finite duration coverage: {target}")
        if selected["num_tensor_parallel_workers"].ne(1).any():
            raise ValueError(f"Step {operator_name} requires replicated non-layer TP1 measurements")
        if selected["n_embd"].ne(config.embedding_dim).any():
            raise ValueError(f"Step {operator_name} source hidden width conflicts with model metadata")
        if not selected["num_tokens"].map(lambda value: isinstance(value, Real)
                and not isinstance(value, bool) and math.isfinite(value)
                and value > 0 and value == int(value)).all():
            raise ValueError(f"Step {operator_name} requires positive integral num_tokens")
        stage_metadata = describe_step5_stage_operator_tensors(
            config, operator_name, physical_rows=1, attention_tp_size=1,
        )
        from frontier.operators.binding import bind_operator_query

        binding = bind_operator_query(operator_name)
        for value in selected["typed_operator_contracts"]:
            contract = parse_typed_operator_contracts(value).get(operator_name)
            if contract is not None:
                validate_typed_operator_metadata(
                    contract, operator_name=operator_name,
                    expected_metadata={"profile_id": "step5", "operator_family_id": binding.family_id,
                                       "layer_kind": None, "tensor_parallel_mode": binding.operator.tp_mode.value,
                                       "selected_tensor_parallel_size": 1},
                    architecture_profile=config.get_model_architecture_profile(),
                )
    else:
        selected = frame.loc[frame["typed_operator_contracts"].map(
            lambda value: operator_name in parse_typed_operator_contracts(value)
        )].copy()
    if selected.empty:
        raise ValueError(f"Step {operator_name} has no measured physical scope coverage")
    values = pd.to_numeric(selected[target], errors="coerce") if target in selected else pd.Series(dtype=float)
    if len(values) != len(selected) or not values.map(
        lambda x: not isinstance(x, bool) and math.isfinite(x) and x > 0
    ).all():
        raise ValueError(f"Step {operator_name} requires positive finite duration coverage: {target}")
    column = f"time_stats.{operator_name}.kernel_count"
    counts = pd.to_numeric(selected[column], errors="coerce") if column in selected else pd.Series(dtype=float)
    if len(counts) != len(selected) or not counts.map(
        lambda x: not isinstance(x, bool) and math.isfinite(x) and x >= 1 and x == int(x)
    ).all():
        raise ValueError(f"Step {operator_name} requires actual positive integral kernel coverage: {column}")
    source_precision = selected["profiling_precision"].map(PrecisionType.from_string)
    model_precision = resolve_step5_model_precision(config)
    if not source_precision.eq(model_precision).all():
        raise ValueError(f"Step {operator_name} source profiling_precision conflicts with model input dtype")
    selected["source_profiling_precision"] = selected["profiling_precision"]
    if operator_name == "moe_grouped_gemm":
        columns = {"moe_layer_ids", "moe_routed_precision", "moe_activation", "moe_quantization_mode"}
        if columns - set(selected.columns):
            raise ValueError(f"Step routed precision coverage requires {sorted(columns)}")
        groups = {group.layer_range: group for group in resolve_routed_layer_groups(config)}
        for _, row in selected.iterrows():
            group = groups.get(row["moe_layer_ids"])
            if (group is None or PrecisionType.from_string(row["moe_routed_precision"]) is not group.precision
                    or row["moe_activation"] != group.activation
                    or row["moe_quantization_mode"] != group.precision.value):
                raise ValueError("Step routed precision/activation/exclusion metadata conflicts with source layer group")
        selected["profiling_precision"] = selected["moe_routed_precision"].map(
            lambda value: PrecisionType.from_string(value).name
        )
    elif stage_boundary:
        selected["profiling_precision"] = stage_metadata["dtype"]
    else:
        precision = model_precision
        routed = config.get_moe_layer_ids()
        layer_id = routed[0] if operator_name.startswith(("share_expert_", "moe_")) else next(
            layer for layer in range(config.num_layers) if layer not in routed
        )
        metadata = describe_step5_layer(config, layer_id, attention_tp_size=1,
                                       moe_tp_size=1, expert_parallel_size=1)
        projections = [p for p in metadata.projections if p.operator_name == operator_name]
        if projections:
            precisions = {p.weight_precision for p in projections}
            if len(precisions) != 1:
                raise ValueError(f"Step {operator_name} has heterogeneous projection precision")
            precision = precisions.pop()
        selected["profiling_precision"] = precision.name
    result = []
    for precision, rows in selected.groupby("profiling_precision", sort=True):
        context = dict(training_context)
        identity = dict(context.get("measurement_identity", {}))
        for name in ("measurement_type", "model_architecture_profile", "quant_signature",
                     "source_profiling_precision", "runtime_stack_signature", "evidence_kind"):
            if name in rows:
                values = rows[name].drop_duplicates().tolist()
                if len(values) != 1 or pd.isna(values[0]):
                    raise ValueError(f"Step {operator_name} mixes incompatible source identity {name}")
                if name in identity and identity[name] != values[0]:
                    raise ValueError(f"Step {operator_name} context conflicts with source {name}")
                identity[name] = values[0]
        identity["operator_name"] = operator_name
        identity["physical_scope"] = operator_name
        identity["operator_precision"] = precision
        if stage_boundary:
            context["step5_stage_boundary"] = operator_name
            identity.update(scope="stage", tensor_parallel_mode=binding.operator.tp_mode.value, tensor_parallel_size=1,
                            tensor_dtype=stage_metadata["tensor_dtype"], feature_columns=["num_tokens"])
        if "moe_layer_ids" in rows:
            identity["moe_layer_ids"] = sorted(rows["moe_layer_ids"].unique().tolist())
        context["measurement_identity"] = identity
        result.append((rows.copy(), context))
    return result
