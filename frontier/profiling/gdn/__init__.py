"""CPU-safe GDN profiling input contracts.

GPU runtime wrappers are intentionally absent from this early semantic stage.
"""

from frontier.profiling.gdn.inputs import (
    GDNProfileInput,
    build_profile_inputs,
    get_required_gdn_profiling_columns,
)

__all__ = [
    "GDNProfileInput",
    "build_profile_inputs",
    "get_required_gdn_profiling_columns",
]
