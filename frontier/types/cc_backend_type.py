"""CC Backend type enumeration for collective communication backends."""

from frontier.types.base_int_enum import BaseIntEnum


class CCBackendType(BaseIntEnum):
    """Enumeration of CC Backend types."""

    VIDUR = 1  # ML-based predictor using sklearn models trained on profiling data
    ANALYTICAL = 2  # Simple analytical model using bandwidth/latency formulas
    COLLECTIVE_SIM = 3  # Topology-aware discrete-event simulator backend
    AICONFIGURATOR = 4  # Internal-only backend excluded from the public release surface
    ASTRA_SIM_ANALYTICAL = 5  # Lightweight ASTRA-Sim analytical backend
