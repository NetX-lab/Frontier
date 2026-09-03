"""Shared file-level admission for profiling metadata identities."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_model_architecture_profile(
    frame: pd.DataFrame,
    *,
    file_path: str,
    expected_profile: str,
) -> str:
    """Validate the CSV-level architecture profile against runtime identity."""

    column = "model_architecture_profile"
    if column not in frame.columns:
        raise ValueError(
            f"{column} column is missing from '{file_path}'. "
            "Run the profiling metadata migration before loading this file."
        )
    values = frame[column].dropna().unique().tolist()
    if not values:
        raise ValueError(f"{column} column is empty in '{file_path}'")
    if len(values) > 1:
        raise ValueError(
            f"Multiple {column} values found in '{file_path}': {values}. "
            "Profiling data should have consistent architecture profile."
        )
    actual_profile = str(values[0])
    if actual_profile != str(expected_profile):
        raise ValueError(
            f"{column} mismatch: expected '{expected_profile}' but profiling data has "
            f"'{actual_profile}'. File: '{file_path}'"
        )
    return actual_profile


def infer_single_runtime_profile(manager: Any) -> str | None:
    """Return a manager-wide profile only when its cluster configs agree."""

    cluster_configs = getattr(manager, "_cluster_configs", None)
    if not isinstance(cluster_configs, dict):
        return None
    profiles = set()
    for cluster_config in cluster_configs.values():
        replica_config = getattr(cluster_config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        getter = getattr(model_config, "get_model_architecture_profile", None)
        if callable(getter):
            profiles.add(str(getter().profile_id))
    if len(profiles) == 1:
        return next(iter(profiles))
    return None


def infer_single_runtime_model_config(manager: Any) -> Any | None:
    """Return a representative model config when all clusters share one profile."""

    cluster_configs = getattr(manager, "_cluster_configs", None)
    if not isinstance(cluster_configs, dict):
        return None
    configs = []
    for cluster_config in cluster_configs.values():
        replica_config = getattr(cluster_config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if model_config is not None:
            configs.append(model_config)
    if configs:
        profile_ids = set()
        for config in configs:
            getter = getattr(config, "get_model_architecture_profile", None)
            if not callable(getter):
                return None
            profile_ids.add(str(getter().profile_id))
        if len(profile_ids) == 1:
            return configs[0]
    return None


__all__ = [
    "infer_single_runtime_model_config",
    "infer_single_runtime_profile",
    "validate_model_architecture_profile",
]
