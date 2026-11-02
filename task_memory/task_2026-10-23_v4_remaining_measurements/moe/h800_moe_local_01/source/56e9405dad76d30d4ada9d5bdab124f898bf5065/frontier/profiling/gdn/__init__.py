"""CPU-safe GDN profiling input contracts.

GPU runtime wrappers are intentionally absent from this early semantic stage.
"""

from frontier.profiling.gdn.inputs import (
    GDNProfileInput,
    build_profile_inputs,
    get_required_gdn_profiling_columns,
)

# The vLLM wrapper is intentionally not imported here.  Importing this package
# must remain safe in the CPU release environment; construct the wrapper only
# from the dedicated profiling entry point.

__all__ = [
    "GDNProfileInput",
    "build_profile_inputs",
    "get_required_gdn_profiling_columns",
]
