"""
CC Backend implementations.

This subpackage contains concrete implementations of the BaseCCBackend interface:
    - VidurCCBackend: ML-based predictor using sklearn models trained on profiling data
    - AnalyticalCCBackend: Simple analytical model using bandwidth/latency formulas
    - CollectiveSimCCBackend: Topology-aware predictor using collective-sim
    - AstraSimAnalyticalCCBackend: Lightweight ASTRA-Sim analytical predictor
"""

# Backend implementations
from frontier.cc_backend.backends.vidur_cc_backend import VidurCCBackend
from frontier.cc_backend.backends.analytical_cc_backend import AnalyticalCCBackend
from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend
from frontier.cc_backend.backends.astra_sim_analytical_cc_backend import (
    AstraSimAnalyticalCCBackend,
)

__all__ = [
    "VidurCCBackend",
    "AnalyticalCCBackend",
    "CollectiveSimCCBackend",
    "AstraSimAnalyticalCCBackend",
]
