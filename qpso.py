"""
Quantum-Behaved Particle Swarm Optimization (QPSO) Module Proxy.

Re-exports canonical QPSO, Volatility-Adaptive QPSO (VA-QPSO), and replan helpers
from src.planner.qpso for root-level accessibility.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.qpso import (
    DEFAULT_BETA_MAX,
    DEFAULT_BETA_MIN,
    default_budget,
    fixed_beta_qpso,
    replan,
    va_qpso,
)
from src.planner.qpso_encoding import (
    compute_distance_matrix,
    decode_order,
    pick_mutually_reachable_stops,
    tour_length,
)

__all__ = [
    "DEFAULT_BETA_MAX",
    "DEFAULT_BETA_MIN",
    "default_budget",
    "fixed_beta_qpso",
    "va_qpso",
    "replan",
    "decode_order",
    "compute_distance_matrix",
    "pick_mutually_reachable_stops",
    "tour_length",
]
