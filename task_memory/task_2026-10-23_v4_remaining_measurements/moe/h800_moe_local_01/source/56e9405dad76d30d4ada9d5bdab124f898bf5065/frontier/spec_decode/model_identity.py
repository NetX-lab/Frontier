from __future__ import annotations

import os
from typing import Mapping


def _strip_model_name(raw_model_name: str) -> str:
    normalized = str(raw_model_name).strip()
    if not normalized:
        return ""
    return normalized.rstrip("/")


def normalize_model_alias(raw_model_name: str) -> str:
    normalized = _strip_model_name(raw_model_name)
    if not normalized:
        return ""
    if os.path.exists(normalized):
        return os.path.realpath(os.path.normpath(normalized))
    return normalized


def resolve_canonical_model_key(
    raw_model_name: str,
    *,
    explicit_alias_map: Mapping[str, str] | None = None,
) -> str:
    normalized_model_name = normalize_model_alias(raw_model_name)
    if not normalized_model_name:
        return ""

    if explicit_alias_map:
        for raw_alias, raw_canonical_key in explicit_alias_map.items():
            normalized_alias = normalize_model_alias(raw_alias)
            canonical_key = _strip_model_name(raw_canonical_key)
            if not normalized_alias or not canonical_key:
                continue
            if normalized_alias == normalized_model_name:
                return canonical_key

    return normalized_model_name
