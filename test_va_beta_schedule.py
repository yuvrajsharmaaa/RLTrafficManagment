"""
Unit tests for va_qpso's beta schedule (src/planner/qpso.py):

    beta_floor(V) = beta_min + 0.25 * V
    beta(t, V)    = beta_floor(V) + (beta_max - beta_floor(V)) * (1 - t / T_max)

Pure arithmetic -- no SUMO network needed.
"""

import numpy as np

from src.planner.qpso import DEFAULT_BETA_MAX, DEFAULT_BETA_MIN, va_beta, va_beta_floor

T_MAX = 200
VOLATILITIES = np.linspace(0.0, 1.0, 11)


def test_va_beta_decreases_monotonically_in_t_for_fixed_v():
    for v in VOLATILITIES:
        betas = np.array([va_beta(t, T_MAX, v) for t in range(T_MAX + 1)])
        assert np.all(np.diff(betas) < 0), f"beta not strictly decreasing in t at V={v}"


def test_va_beta_at_t_max_increases_with_v():
    end_betas = np.array([va_beta(T_MAX, T_MAX, v) for v in VOLATILITIES])
    assert np.all(np.diff(end_betas) > 0)
    assert np.allclose(end_betas, [va_beta_floor(v) for v in VOLATILITIES])


def test_va_beta_endpoints_match_fixed_beta_range():
    # Starts at the same beta_max as fixed_beta_qpso for every V; at V = 0 it
    # ends at beta_min (identical to fixed_beta_qpso), at V = 1 at 0.75.
    for v in VOLATILITIES:
        assert np.isclose(va_beta(0, T_MAX, v), DEFAULT_BETA_MAX)
    assert np.isclose(va_beta(T_MAX, T_MAX, 0.0), DEFAULT_BETA_MIN)
    assert np.isclose(va_beta(T_MAX, T_MAX, 1.0), 0.75)


if __name__ == "__main__":
    test_va_beta_decreases_monotonically_in_t_for_fixed_v()
    test_va_beta_at_t_max_increases_with_v()
    test_va_beta_endpoints_match_fixed_beta_range()
    print("OK: va_beta schedule tests passed.")
