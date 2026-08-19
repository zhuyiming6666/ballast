"""Security-mode model for the adaptive consistency family Pi(tau_e, theta).

This module deliberately separates theorem inputs from workload measurements:

* ``eta`` is the protocol/transport rate limit used by the exposure theorem;
* the matching epoch-boundary construction computes the analytical lower bound;
* payment values are used only to measure how much honest traffic stays on the
  bilateral fast path for a chosen value cap ``theta``.

The resulting CSVs numerically illustrate the claimed upper/lower-bound
relationship.  They are not an independent distributed-systems experiment.
"""

from __future__ import annotations

import math

import numpy as np


def eta(window_s: float, r_min_s: float, counterparties: int,
        inflight_per_counterparty: int = 1) -> int:
    """Maximum honored draws in ``window_s`` under the stated rate limits."""
    if window_s <= 0:
        return 0
    if r_min_s <= 0 or counterparties <= 0 or inflight_per_counterparty <= 0:
        raise ValueError("rate limits must be positive")
    rounds = int(math.floor(window_s / r_min_s + 1e-12))
    return rounds * counterparties * inflight_per_counterparty


def exposure_bound(tau_e_s: float, theta: float, delta_s: float,
                   r_min_s: float, counterparties: int,
                   inflight_per_counterparty: int = 1) -> float:
    """Theorem upper bound E(tau_e, theta) = theta * eta(tau_e + delta)."""
    if math.isinf(tau_e_s) or math.isinf(theta):
        return math.inf
    return theta * eta(tau_e_s + delta_s, r_min_s, counterparties,
                       inflight_per_counterparty)


def simulate_greedy_epoch_attack(tau_e_s: float, theta: float, r_min_s: float,
                                 counterparties: int,
                                 inflight_per_counterparty: int = 1) -> float:
    """Event-level greedy attacker at an epoch boundary.

    Each counterparty exposes ``inflight_per_counterparty`` honor slots.  The
    adversary fills every slot at time 0 and refills it after ``r_min_s`` until
    the checkpoint closes the epoch.  Iterating the concrete events makes this
    measurement independent of the closed-form ``eta`` implementation used by
    :func:`exposure_bound`; equality is therefore a check, not an assignment.
    """
    if math.isinf(tau_e_s) or math.isinf(theta):
        return math.inf
    if tau_e_s <= 0:
        return 0.0
    if r_min_s <= 0 or counterparties <= 0 or inflight_per_counterparty <= 0:
        raise ValueError("rate limits must be positive")
    exposure = 0.0
    # The protocol convention honors slots at r_min, 2*r_min, ... <= tau_e,
    # matching eta(floor(window/r_min)); there is no unbounded time-zero slot.
    for _counterparty in range(counterparties):
        for _slot in range(inflight_per_counterparty):
            event_time = r_min_s
            while event_time <= tau_e_s + 1e-12:
                exposure += theta
                event_time += r_min_s
    return exposure


# Compatibility name used by older scripts/artifacts.
optimal_epoch_attack = simulate_greedy_epoch_attack


def fast_path_stats(values, theta: float) -> dict[str, float]:
    """Count and value fractions of honest draws that do not require confirm."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"fast_count_fraction": 0.0, "fast_value_fraction": 0.0}
    mask = arr <= theta
    return {
        "fast_count_fraction": float(np.mean(mask)),
        "fast_value_fraction": float(arr[mask].sum() / arr.sum()),
    }


def latency_samples(values, theta: float, peer_rtt_s: float = 0.2,
                    confirm_rtt_s: float = 0.4) -> np.ndarray:
    """Honest draw latency: one bilateral RTT, plus confirm above theta."""
    arr = np.asarray(values, dtype=float)
    return peer_rtt_s + (arr > theta).astype(float) * confirm_rtt_s
