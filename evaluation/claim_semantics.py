"""Event simulation for claim-triggered availability under freeze vs escrow."""

from __future__ import annotations

import heapq
import random

import numpy as np


def simulate_claims(*, semantics: str, n_channels: int, claim_rate_h: float,
                    seed: int, horizon_h: float = 24.0,
                    challenge_h: float = 1.0, honest_rate_per_channel_h: float = 2.0,
                    honest_hold_h: float = 0.1,
                    bond: float = 1.0, claim_fraction_per_channel: float = 0.5,
                    deposit_fraction: float = 0.25,
                    attacker_budget: float | None = None) -> dict[str, float]:
    """Simulate malicious immediate claims mixed with honest draw arrivals.

    ``freeze`` models the old session-wide stop. ``escrow`` moves each admitted
    claim into a bounded escrow and leaves unrelated headroom usable.  Claims
    expire after the challenge window; this isolates the blast-radius effect
    rather than assuming permanent loss of collateral.
    """
    if semantics not in {"freeze", "escrow"}:
        raise ValueError("semantics must be freeze or escrow")
    if n_channels <= 0 or bond <= 0:
        raise ValueError("n_channels and bond must be positive")

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    dt_h = 1.0 / 60.0
    steps = int(round(horizon_h / dt_h))
    claim_amount = bond * claim_fraction_per_channel / n_channels
    draw_amount = 0.25 * bond / n_channels
    deposit = deposit_fraction * claim_amount
    # One claimant is given liquid capital equal to one attempted claim.  A
    # positive deposit therefore limits simultaneous griefing attempts; zero
    # deposit deliberately leaves them unbounded.
    attacker_budget = claim_amount if attacker_budget is None else attacker_budget
    active = []  # heap of (expiry, sequence, amount, channel, deposit)
    honest_active = []  # heap of (expiry, sequence, amount)
    sequence = 0
    escrow = 0.0
    honest_outstanding = 0.0
    attacker_locked = 0.0
    max_escrow = 0.0
    max_attacker_locked = 0.0
    admitted_claims = refused_claims = refused_draws = honest_draws = 0
    deposit_limited_claims = capacity_limited_claims = 0
    affected_channel_hours = 0.0
    invariant_violations = 0

    for step in range(steps):
        now = step * dt_h
        while active and active[0][0] <= now:
            _expiry, _seq, amount, _channel, locked = heapq.heappop(active)
            escrow -= amount
            attacker_locked -= locked
        while honest_active and honest_active[0][0] <= now:
            _expiry, _seq, amount = heapq.heappop(honest_active)
            honest_outstanding -= amount
        if abs(escrow) < 1e-12:
            escrow = 0.0

        for _ in range(rng.poisson(claim_rate_h * dt_h)):
            channel = py_rng.randrange(n_channels)
            if deposit > 0 and attacker_locked + deposit > attacker_budget + 1e-12:
                refused_claims += 1
                deposit_limited_claims += 1
            elif escrow + honest_outstanding + claim_amount <= bond + 1e-12:
                sequence += 1
                heapq.heappush(active, (now + challenge_h, sequence,
                                        claim_amount, channel, deposit))
                escrow += claim_amount
                attacker_locked += deposit
                admitted_claims += 1
            else:
                refused_claims += 1
                capacity_limited_claims += 1

        honest_now = int(rng.poisson(honest_rate_per_channel_h * n_channels * dt_h))
        honest_draws += honest_now
        if semantics == "freeze":
            if active:
                refused_draws += honest_now
            else:
                for _ in range(honest_now):
                    if escrow + honest_outstanding + draw_amount <= bond + 1e-12:
                        sequence += 1
                        heapq.heappush(honest_active, (now + honest_hold_h,
                                                      sequence, draw_amount))
                        honest_outstanding += draw_amount
                    else:
                        refused_draws += 1
            affected = n_channels if active else 0
        else:
            for _ in range(honest_now):
                if escrow + honest_outstanding + draw_amount <= bond + 1e-12:
                    sequence += 1
                    heapq.heappush(honest_active, (now + honest_hold_h,
                                                  sequence, draw_amount))
                    honest_outstanding += draw_amount
                else:
                    refused_draws += 1
            affected = len({entry[3] for entry in active})

        affected_channel_hours += affected * dt_h
        max_escrow = max(max_escrow, escrow)
        max_attacker_locked = max(max_attacker_locked, attacker_locked)
        if (escrow < -1e-9 or honest_outstanding < -1e-9 or
                escrow + honest_outstanding > bond + 1e-9 or
                attacker_locked > attacker_budget + 1e-9):
            invariant_violations += 1

    return {
        "semantics": semantics,
        "n_channels": n_channels,
        "claim_rate_h": claim_rate_h,
        "seed": seed,
        "honest_draws": honest_draws,
        "refused_draws": refused_draws,
        "refusal_rate": refused_draws / honest_draws if honest_draws else 0.0,
        "affected_channel_hours": affected_channel_hours,
        "admitted_claims": admitted_claims,
        "refused_claims": refused_claims,
        "deposit_limited_claims": deposit_limited_claims,
        "capacity_limited_claims": capacity_limited_claims,
        "deposit_fraction": deposit_fraction,
        "attacker_budget": attacker_budget,
        "max_attacker_deposit_locked": max_attacker_locked,
        "max_escrow_fraction": max_escrow / bond,
        "invariant_violations": invariant_violations,
        "max_griefing_net_per_claim": claim_amount - deposit,
    }
