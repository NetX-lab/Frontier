"""Validation helpers for PP stage-boundary overhead profiling outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from frontier.profiling.other_overhead.schema import (
    PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS,
    PP_PRODUCER_SEND_PATH_NUMERIC_COLUMNS,
    PP_PRODUCER_SEND_PATH_REQUIRED_COLUMNS,
    PP_RECEIVER_HEAD_IDENTITY_COLUMNS,
    PP_RECEIVER_HEAD_NUMERIC_COLUMNS,
    PP_RECEIVER_HEAD_REQUIRED_COLUMNS,
    PP_STAGE_BOUNDARY_IDENTITY_COLUMNS,
    PP_STAGE_BOUNDARY_NUMERIC_COLUMNS,
    PP_STAGE_BOUNDARY_REQUIRED_COLUMNS,
)


def validate_pp_stage_boundary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate PP stage-boundary overhead dataframe and return normalized copy."""
    if df.empty:
        raise ValueError("PP stage-boundary profiling dataframe must not be empty.")

    validated = df.copy()

    missing_columns = [
        column
        for column in PP_STAGE_BOUNDARY_REQUIRED_COLUMNS
        if column not in validated.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required PP stage-boundary columns: "
            f"{missing_columns}. Required columns: {list(PP_STAGE_BOUNDARY_REQUIRED_COLUMNS)}"
        )

    numeric_frame = validated.loc[:, PP_STAGE_BOUNDARY_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid_numeric_columns = [
        column
        for column in PP_STAGE_BOUNDARY_NUMERIC_COLUMNS
        if numeric_frame[column].isna().any()
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "PP stage-boundary profiling dataframe has non-numeric values in columns: "
            f"{invalid_numeric_columns}"
        )

    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise ValueError(
            "PP stage-boundary profiling dataframe contains non-finite numeric values."
        )

    validated.loc[:, PP_STAGE_BOUNDARY_NUMERIC_COLUMNS] = numeric_frame

    int_columns = (
        "batch_size",
        "tensor_parallel_degree",
        "num_prefill_tokens",
        "num_decode_tokens",
        "producer_pp_rank",
        "consumer_pp_rank",
        "pp_world_size",
        "activation_bytes_per_rank",
    )
    for column in int_columns:
        if (validated[column] < 0).any():
            raise ValueError(f"{column} must be non-negative for all PP stage-boundary rows.")
        if not (validated[column] == validated[column].astype(int)).all():
            raise ValueError(f"{column} must be an integer for all PP stage-boundary rows.")
        validated[column] = validated[column].astype(int)

    if (validated["batch_size"] <= 0).any():
        raise ValueError("batch_size must be > 0 for all PP stage-boundary rows.")
    if (validated["tensor_parallel_degree"] <= 0).any():
        raise ValueError(
            "tensor_parallel_degree must be > 0 for all PP stage-boundary rows."
        )
    if (validated["pp_world_size"] <= 1).any():
        raise ValueError("pp_world_size must be > 1 for all PP stage-boundary rows.")
    if (validated["consumer_pp_rank"] != validated["producer_pp_rank"] + 1).any():
        raise ValueError(
            "consumer_pp_rank must equal producer_pp_rank + 1 for all rows."
        )
    if (validated["pp_world_size"] <= validated["consumer_pp_rank"]).any():
        raise ValueError(
            "consumer_pp_rank must be strictly smaller than pp_world_size for all rows."
        )
    if ((validated["num_prefill_tokens"] + validated["num_decode_tokens"]) <= 0).any():
        raise ValueError(
            "num_prefill_tokens + num_decode_tokens must be > 0 for all rows."
        )

    non_negative_columns = (
        "boundary_critical_path_ms",
        "producer_send_duration_ms",
        "consumer_recv_duration_ms",
        "consumer_preprocess_duration_ms",
        "existing_pp_send_recv_wire_ms",
        "pp_stage_boundary_overhead_ms",
    )
    for column in non_negative_columns:
        if (validated[column] < 0).any():
            raise ValueError(f"{column} must be non-negative for all rows.")

    model_name_series = validated["model_name"].astype(str).str.strip()
    if model_name_series.eq("").any():
        raise ValueError("model_name must be non-empty for all PP stage-boundary rows.")
    validated["model_name"] = model_name_series

    duplicate_rows = validated.duplicated(
        subset=list(PP_STAGE_BOUNDARY_IDENTITY_COLUMNS),
        keep=False,
    )
    if duplicate_rows.any():
        duplicated_entries = validated.loc[
            duplicate_rows,
            list(PP_STAGE_BOUNDARY_IDENTITY_COLUMNS),
        ].to_dict("records")
        raise ValueError(
            "Duplicate PP stage-boundary rows found for identity columns "
            f"{PP_STAGE_BOUNDARY_IDENTITY_COLUMNS}: {duplicated_entries}"
        )

    return validated


def validate_pp_receiver_head_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate PP receiver-head profiling dataframe and return normalized copy."""
    if df.empty:
        raise ValueError("PP receiver-head profiling dataframe must not be empty.")

    validated = df.copy()

    missing_columns = [
        column
        for column in PP_RECEIVER_HEAD_REQUIRED_COLUMNS
        if column not in validated.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required PP receiver-head columns: "
            f"{missing_columns}. Required columns: {list(PP_RECEIVER_HEAD_REQUIRED_COLUMNS)}"
        )

    numeric_frame = validated.loc[:, PP_RECEIVER_HEAD_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid_numeric_columns = [
        column
        for column in PP_RECEIVER_HEAD_NUMERIC_COLUMNS
        if numeric_frame[column].isna().any()
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "PP receiver-head profiling dataframe has non-numeric values in columns: "
            f"{invalid_numeric_columns}"
        )

    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise ValueError(
            "PP receiver-head profiling dataframe contains non-finite numeric values."
        )

    validated.loc[:, PP_RECEIVER_HEAD_NUMERIC_COLUMNS] = numeric_frame

    int_columns = (
        "batch_size",
        "tensor_parallel_degree",
        "num_prefill_tokens",
        "num_decode_tokens",
        "consumer_pp_rank",
        "pp_world_size",
        "activation_bytes_per_rank",
        "producer_pp_rank",
        "sample_count",
    )
    for column in int_columns:
        if (validated[column] < 0).any():
            raise ValueError(f"{column} must be non-negative for all PP receiver-head rows.")
        if not (validated[column] == validated[column].astype(int)).all():
            raise ValueError(f"{column} must be an integer for all PP receiver-head rows.")
        validated[column] = validated[column].astype(int)

    if (validated["batch_size"] <= 0).any():
        raise ValueError("batch_size must be > 0 for all PP receiver-head rows.")
    if (validated["tensor_parallel_degree"] <= 0).any():
        raise ValueError("tensor_parallel_degree must be > 0 for all PP receiver-head rows.")
    if (validated["pp_world_size"] <= 1).any():
        raise ValueError("pp_world_size must be > 1 for all PP receiver-head rows.")
    if (validated["consumer_pp_rank"] <= 0).any():
        raise ValueError("consumer_pp_rank must be > 0 for all PP receiver-head rows.")
    if (validated["consumer_pp_rank"] >= validated["pp_world_size"]).any():
        raise ValueError(
            "consumer_pp_rank must be strictly smaller than pp_world_size for all rows."
        )
    if (validated["producer_pp_rank"] != validated["consumer_pp_rank"] - 1).any():
        raise ValueError(
            "producer_pp_rank must equal consumer_pp_rank - 1 for all rows."
        )
    if (validated["sample_count"] <= 0).any():
        raise ValueError("sample_count must be > 0 for all PP receiver-head rows.")

    phase_label_series = validated["phase_label"].astype(str).str.strip()
    if phase_label_series.eq("").any():
        raise ValueError("phase_label must be non-empty for all PP receiver-head rows.")
    if not phase_label_series.isin({"decode", "prefill", "mixed"}).all():
        raise ValueError(
            "phase_label must be one of {'decode', 'prefill', 'mixed'} for all "
            "PP receiver-head rows."
        )
    validated["phase_label"] = phase_label_series

    model_name_series = validated["model_name"].astype(str).str.strip()
    if model_name_series.eq("").any():
        raise ValueError("model_name must be non-empty for all PP receiver-head rows.")
    validated["model_name"] = model_name_series

    precision_series = validated["profiling_precision"].astype(str).str.strip()
    if precision_series.eq("").any():
        raise ValueError("profiling_precision must be non-empty for all PP receiver-head rows.")
    validated["profiling_precision"] = precision_series

    source_series = validated["other_overhead_source"].astype(str).str.strip()
    if source_series.eq("").any():
        raise ValueError(
            "other_overhead_source must be non-empty for all PP receiver-head rows."
        )
    validated["other_overhead_source"] = source_series

    non_negative_columns = (
        "pp_receiver_head_runtime_ms",
        "consumer_ready_to_preprocess_start_ms",
        "consumer_preprocess_duration_ms",
        "consumer_pre_forward_gap_ms",
        "handoff_complete_to_consumer_ready_ms",
        "producer_send_duration_ms",
        "consumer_recv_duration_ms",
    )
    for column in non_negative_columns:
        if (validated[column] < 0).any():
            raise ValueError(f"{column} must be non-negative for all PP receiver-head rows.")

    expected_runtime = (
        validated["consumer_ready_to_preprocess_start_ms"]
        + validated["consumer_preprocess_duration_ms"]
        + validated["consumer_pre_forward_gap_ms"]
    )
    if not np.allclose(
        validated["pp_receiver_head_runtime_ms"].to_numpy(dtype=float),
        expected_runtime.to_numpy(dtype=float),
    ):
        raise ValueError(
            "pp_receiver_head_runtime_ms must equal "
            "consumer_ready_to_preprocess_start_ms + "
            "consumer_preprocess_duration_ms + consumer_pre_forward_gap_ms."
        )

    duplicate_rows = validated.duplicated(
        subset=list(PP_RECEIVER_HEAD_IDENTITY_COLUMNS),
        keep=False,
    )
    if duplicate_rows.any():
        duplicated_entries = validated.loc[
            duplicate_rows,
            list(PP_RECEIVER_HEAD_IDENTITY_COLUMNS),
        ].to_dict("records")
        raise ValueError(
            "Duplicate PP receiver-head rows found for identity columns "
            f"{PP_RECEIVER_HEAD_IDENTITY_COLUMNS}: {duplicated_entries}"
        )

    return validated


def validate_pp_producer_send_path_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate PP producer send-path profiling dataframe and return normalized copy."""
    if df.empty:
        raise ValueError("PP producer send-path profiling dataframe must not be empty.")

    validated = df.copy()

    missing_columns = [
        column
        for column in PP_PRODUCER_SEND_PATH_REQUIRED_COLUMNS
        if column not in validated.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required PP producer send-path columns: "
            f"{missing_columns}. Required columns: {list(PP_PRODUCER_SEND_PATH_REQUIRED_COLUMNS)}"
        )

    numeric_frame = validated.loc[:, PP_PRODUCER_SEND_PATH_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid_numeric_columns = [
        column
        for column in PP_PRODUCER_SEND_PATH_NUMERIC_COLUMNS
        if numeric_frame[column].isna().any()
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "PP producer send-path profiling dataframe has non-numeric values in columns: "
            f"{invalid_numeric_columns}"
        )

    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise ValueError(
            "PP producer send-path profiling dataframe contains non-finite numeric values."
        )

    validated.loc[:, PP_PRODUCER_SEND_PATH_NUMERIC_COLUMNS] = numeric_frame

    int_columns = (
        "batch_size",
        "tensor_parallel_degree",
        "num_prefill_tokens",
        "num_decode_tokens",
        "producer_pp_rank",
        "pp_world_size",
        "activation_bytes_per_rank",
        "consumer_pp_rank",
        "sample_count",
    )
    for column in int_columns:
        if (validated[column] < 0).any():
            raise ValueError(
                f"{column} must be non-negative for all PP producer send-path rows."
            )
        if not (validated[column] == validated[column].astype(int)).all():
            raise ValueError(
                f"{column} must be an integer for all PP producer send-path rows."
            )
        validated[column] = validated[column].astype(int)

    if (validated["batch_size"] <= 0).any():
        raise ValueError("batch_size must be > 0 for all PP producer send-path rows.")
    if (validated["tensor_parallel_degree"] <= 0).any():
        raise ValueError(
            "tensor_parallel_degree must be > 0 for all PP producer send-path rows."
        )
    if (validated["pp_world_size"] <= 1).any():
        raise ValueError("pp_world_size must be > 1 for all PP producer send-path rows.")
    if (validated["producer_pp_rank"] >= validated["pp_world_size"] - 1).any():
        raise ValueError(
            "producer_pp_rank must be strictly smaller than pp_world_size - 1 for all rows."
        )
    if (validated["consumer_pp_rank"] != validated["producer_pp_rank"] + 1).any():
        raise ValueError(
            "consumer_pp_rank must equal producer_pp_rank + 1 for all PP producer send-path rows."
        )
    if (validated["sample_count"] <= 0).any():
        raise ValueError("sample_count must be > 0 for all PP producer send-path rows.")

    phase_label_series = validated["phase_label"].astype(str).str.strip()
    if phase_label_series.eq("").any():
        raise ValueError(
            "phase_label must be non-empty for all PP producer send-path rows."
        )
    if not phase_label_series.isin({"decode", "prefill", "mixed"}).all():
        raise ValueError(
            "phase_label must be one of {'decode', 'prefill', 'mixed'} for all "
            "PP producer send-path rows."
        )
    validated["phase_label"] = phase_label_series

    model_name_series = validated["model_name"].astype(str).str.strip()
    if model_name_series.eq("").any():
        raise ValueError(
            "model_name must be non-empty for all PP producer send-path rows."
        )
    validated["model_name"] = model_name_series

    precision_series = validated["profiling_precision"].astype(str).str.strip()
    if precision_series.eq("").any():
        raise ValueError(
            "profiling_precision must be non-empty for all PP producer send-path rows."
        )
    validated["profiling_precision"] = precision_series

    source_series = validated["other_overhead_source"].astype(str).str.strip()
    if source_series.eq("").any():
        raise ValueError(
            "other_overhead_source must be non-empty for all PP producer send-path rows."
        )
    validated["other_overhead_source"] = source_series

    non_negative_columns = (
        "pp_producer_send_path_runtime_ms",
        "producer_send_duration_ms",
        "existing_pp_send_recv_wire_ms",
    )
    for column in non_negative_columns:
        if (validated[column] < 0).any():
            raise ValueError(
                f"{column} must be non-negative for all PP producer send-path rows."
            )

    expected_runtime = (
        validated["producer_send_duration_ms"]
        - validated["existing_pp_send_recv_wire_ms"]
    )
    if not np.allclose(
        validated["pp_producer_send_path_runtime_ms"].to_numpy(dtype=float),
        expected_runtime.to_numpy(dtype=float),
    ):
        raise ValueError(
            "pp_producer_send_path_runtime_ms must equal "
            "producer_send_duration_ms - existing_pp_send_recv_wire_ms."
        )

    duplicate_rows = validated.duplicated(
        subset=list(PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS),
        keep=False,
    )
    if duplicate_rows.any():
        duplicated_entries = validated.loc[
            duplicate_rows,
            list(PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS),
        ].to_dict("records")
        raise ValueError(
            "Duplicate PP producer send-path rows found for identity columns "
            f"{PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS}: {duplicated_entries}"
        )

    return validated
