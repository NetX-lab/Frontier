"""
Ray worker setup hook for linear-op profiling.

Disables Ray datasets serializers to avoid importing pyarrow, which can
trigger libstdc++ ABI conflicts in some environments.
"""

from __future__ import annotations

_RAY_DATASET_SERIALIZERS_DISABLED = False


def disable_ray_datasets_serializers() -> None:
    """
    Disable Ray datasets serializers to avoid pyarrow import.

    This is safe for linear-op profiling because Ray datasets are not used.
    """
    global _RAY_DATASET_SERIALIZERS_DISABLED
    if _RAY_DATASET_SERIALIZERS_DISABLED:
        return

    try:
        from ray.util import serialization_addons
        from ray._common.pydantic_compat import register_pydantic_serializers
    except Exception as exc:
        raise RuntimeError(
            "Failed to patch Ray serialization. "
            "Ray must be installed and importable to use profiling in Ray mode."
        ) from exc

    def _apply_without_datasets(serialization_context):
        register_pydantic_serializers(serialization_context)
        serialization_addons.register_starlette_serializer(serialization_context)
        # Skip datasets serializers to prevent pyarrow import.

    serialization_addons.apply = _apply_without_datasets
    _RAY_DATASET_SERIALIZERS_DISABLED = True
