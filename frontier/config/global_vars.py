"""
Global variables for the Vidur simulator.

This module maintains global state variables that are used throughout the simulation.
All variables are in uppercase following Python conventions for global constants.

IMPORTANT: IS_MOE is determined SOLELY by the model architecture (from JSON config),
NOT by parallelism settings like moe_expert_parallel_size or total_expert_num.
A MoE model remains a MoE model regardless of how it's parallelized.
"""

from typing import Optional

# Global simulation configuration variables
SIMULATION_MODE: str = None
SYS_ARCH: str = None
USE_CUDA_GRAPH: bool = False
CUDAGRAPH_CAPTURE_SIZES: Optional[list[int]] = None
DECODE_CUDA_GRAPH_MODE: str = "none"
ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC: bool = False
ENABLE_MONOLITHIC_MOE_STAGE_AGGREGATION: bool = False
QUANTIZATION_MANAGER = None

# Global MoE model indicator - determined by model architecture, NOT parallelism config
# This is set once during configuration loading and should not be modified afterwards
IS_MOE: Optional[bool] = None
_IS_MOE_INITIALIZED: bool = False


def set_global_vars(simulation_mode: str, sys_arch: str):
    """
    Set global variables from the simulation configuration.
    
    Args:
        simulation_mode: The simulation mode ('online' or 'offline')
        sys_arch: The system architecture ('co-location' or 'pd-af-disaggregation')
    """
    global SIMULATION_MODE, SYS_ARCH
    SIMULATION_MODE = simulation_mode
    SYS_ARCH = sys_arch


def set_cuda_graph_config(
    use_cuda_graph: bool,
    cudagraph_capture_sizes: Optional[list[int]],
    decode_cuda_graph_mode: str = "none",
    allow_spec_decode_cuda_graph_diagnostic: bool = False,
) -> None:
    """
    Set global CUDA Graph configuration.

    Args:
        use_cuda_graph: Whether CUDA Graph is enabled.
        cudagraph_capture_sizes: Capture sizes for CUDA Graph (if any).
        decode_cuda_graph_mode: Decode-only CUDA Graph mode for non-AFD paths.
        allow_spec_decode_cuda_graph_diagnostic: Opt-in switch for comparison-only
            speculative decode CUDA graph diagnostics.
    """
    global USE_CUDA_GRAPH, CUDAGRAPH_CAPTURE_SIZES, DECODE_CUDA_GRAPH_MODE
    global ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC
    USE_CUDA_GRAPH = use_cuda_graph
    CUDAGRAPH_CAPTURE_SIZES = cudagraph_capture_sizes
    DECODE_CUDA_GRAPH_MODE = decode_cuda_graph_mode
    ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC = (
        allow_spec_decode_cuda_graph_diagnostic
    )


def get_use_cuda_graph() -> bool:
    """Get whether CUDA Graph is enabled."""
    return USE_CUDA_GRAPH


def get_cudagraph_capture_sizes() -> Optional[list[int]]:
    """Get CUDA Graph capture sizes."""
    return CUDAGRAPH_CAPTURE_SIZES


def get_decode_cuda_graph_mode() -> str:
    """Get decode-only CUDA Graph mode for non-AFD paths."""
    return DECODE_CUDA_GRAPH_MODE


def get_allow_spec_decode_cuda_graph_diagnostic() -> bool:
    """Get the speculative decode CUDA graph diagnostic opt-in flag."""
    return ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC


def set_monolithic_moe_stage_aggregation(enabled: bool) -> None:
    """Set the MONOLITHIC MoE stage-level aggregation opt-in flag."""
    global ENABLE_MONOLITHIC_MOE_STAGE_AGGREGATION
    ENABLE_MONOLITHIC_MOE_STAGE_AGGREGATION = bool(enabled)


def get_monolithic_moe_stage_aggregation() -> bool:
    """Get whether MONOLITHIC MoE should use stage-level aggregation."""
    return ENABLE_MONOLITHIC_MOE_STAGE_AGGREGATION


def set_is_moe(is_moe: bool) -> None:
    """
    Set the global IS_MOE flag based on model architecture.
    
    This should be called ONCE during configuration loading, using the value
    from model_config.is_moe which is determined by the model's JSON configuration
    (specifically, whether num_experts > 1).
    
    IMPORTANT: This flag should NOT be influenced by parallelism settings like
    moe_expert_parallel_size or total_expert_num. A MoE model is a MoE model
    regardless of how it's parallelized.
    
    Args:
        is_moe: Whether the model is a Mixture of Experts model (from model_config.is_moe)
    
    Raises:
        RuntimeError: If IS_MOE has already been initialized with a different value
    """
    global IS_MOE, _IS_MOE_INITIALIZED
    
    if _IS_MOE_INITIALIZED:
        if IS_MOE != is_moe:
            raise RuntimeError(
                f"IS_MOE already initialized to {IS_MOE}, cannot change to {is_moe}. "
                "This indicates inconsistent model configurations across clusters."
            )
        return  # Already set to same value, no-op
    
    IS_MOE = is_moe
    _IS_MOE_INITIALIZED = True


def set_quantization_manager(manager) -> None:
    """
    Set the global QuantizationManager instance.

    Raises:
        RuntimeError: If an existing manager is already set to a different instance.
    """
    global QUANTIZATION_MANAGER
    if QUANTIZATION_MANAGER is not None and QUANTIZATION_MANAGER is not manager:
        raise RuntimeError("QuantizationManager already initialized with a different instance.")
    QUANTIZATION_MANAGER = manager


def get_quantization_manager():
    """Get the global QuantizationManager instance."""
    return QUANTIZATION_MANAGER


def get_is_moe() -> Optional[bool]:
    """
    Get the global IS_MOE flag.
    
    Returns:
        True if the model is MoE, False if dense, None if not yet initialized.
    """
    return IS_MOE


def is_moe_model() -> bool:
    """
    Check if the current model is a Mixture of Experts model.
    
    This is the recommended way to check for MoE models throughout the codebase.
    It returns False if IS_MOE has not been initialized (conservative default).
    
    Returns:
        True if the model is MoE, False otherwise (including if not initialized).
    """
    return IS_MOE is True


def reset_global_vars() -> None:
    """
    Reset all global variables to their initial state.
    
    This is primarily useful for testing to ensure clean state between test runs.
    Should NOT be called during normal simulation execution.
    """
    global SIMULATION_MODE, SYS_ARCH, USE_CUDA_GRAPH, CUDAGRAPH_CAPTURE_SIZES
    global DECODE_CUDA_GRAPH_MODE, ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC
    global ENABLE_MONOLITHIC_MOE_STAGE_AGGREGATION
    global IS_MOE, _IS_MOE_INITIALIZED, QUANTIZATION_MANAGER
    SIMULATION_MODE = None
    SYS_ARCH = None
    USE_CUDA_GRAPH = False
    CUDAGRAPH_CAPTURE_SIZES = None
    DECODE_CUDA_GRAPH_MODE = "none"
    ALLOW_SPEC_DECODE_CUDA_GRAPH_DIAGNOSTIC = False
    ENABLE_MONOLITHIC_MOE_STAGE_AGGREGATION = False
    IS_MOE = None
    _IS_MOE_INITIALIZED = False
    QUANTIZATION_MANAGER = None


def get_simulation_mode() -> str:
    """Get the current simulation mode."""
    return SIMULATION_MODE


def get_sys_arch() -> str:
    """Get the current system architecture."""
    return SYS_ARCH


def is_disaggregated_mode() -> bool:
    """Check if the current system architecture is disaggregated."""
    return SYS_ARCH in ["pd-disaggregation", "pd-af-disaggregation"]


def is_online_mode() -> bool:
    """Check if the current simulation mode is online."""
    return SIMULATION_MODE == "online"
