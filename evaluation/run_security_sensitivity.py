"""run_security_sensitivity.py -- delta/eta sensitivity, theta=0 endpoint,
and checkpoint-stall behavior for the fork-containment layer.

Reviewer items FP-2 (delta provenance), FP-1/F5 (eta sensitivity), F3
(theta=0 is the prevention endpoint; tau_e -> 0 is not), and the
stale-checkpoint gate ("no checkpoint => no service, not more exposure").

Three CSVs, all cheap (no payment simulation):

1. results/security_delta_eta.csv
   Sweep delta x (counterparties, inflight, r_min) x tau_e at theta = p95 of
   the payment-value trace.  Columns report the event-level greedy attack, the
   closed-form bound theta*eta(tau_e+delta), and their ratio.  The attack
   events are generated over the window tau_e (the adversary cannot use the
   visibility slack it cannot see), so ratio -> 1 as tau_e >> delta and the
   delta share of the bound is explicit.

2. results/security_endpoints.csv
   The endpoint matrix: theta in {0, p50, p95} x tau_e in {0.1, 1, 10, 60}s.
   theta=0 rows have exposure exactly 0 at every tau_e; theta>0 rows keep the
   residual theta*eta(delta) even as tau_e -> 0.

3. results/checkpoint_stall.csv
   Freshness-gate model: certificates chain to the last sealed checkpoint;
   counterparties refuse a certificate whose sealed ancestor is older than
   tau_e + delta.  Scenarios: honest checkpointing; operator stops
   checkpointing at t=0 (fork time); operator delays every checkpoint by d.
   Reported: draws accepted inside/outside the freshness window and the
   resulting exposure -- stalling yields refusals, never extra exposure.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

import common
import security_modes as sm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
TRACE = os.path.join(HERE, "payment_value", "payment_value_satoshi_03.csv")


def payment_quantiles(path, qs=(0.5, 0.95)):
    """Quantiles of the thresholded trace actually used by the simulator
    (0 < v <= common.PAYMENT_VALUE_THRESHOLD), matching the paper's p95."""
    thr = common.PAYMENT_VALUE_THRESHOLD
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip().split(",")[-1]
            try:
                v = float(line)
            except ValueError:
                continue
            if 0 < v <= thr:
                vals.append(v)
            if len(vals) >= 2_000_000:
                break
    arr = np.asarray(vals)
    return {q: float(np.quantile(arr, q)) for q in qs}


# ------------------------------------------------------------------
# 1. delta / eta sensitivity
# ------------------------------------------------------------------
def sweep_delta_eta(theta, out):
    deltas = [0.05, 0.1, 0.2, 0.5, 1.0]
    eta_configs = [
        # (counterparties, inflight, r_min): eta(t) = floor(t/r_min)*m*q per second
        (8, 1, 0.2),      # paper instance: eta(t) = 40 t
        (8, 2, 0.2),      # 80 t
        (16, 1, 0.2),     # 80 t
        (32, 1, 0.2),     # 160 t
        (8, 1, 0.1),      # 80 t via faster honor round
        (64, 4, 0.2),     # 1280 t : many channels, deep pipelines
    ]
    taus = [1.0, 10.0, 60.0, 600.0]
    fields = ["tau_e_s", "delta_s", "counterparties", "inflight", "r_min_s",
              "eta_per_s", "theta_sat", "attack_sat", "bound_sat", "ratio",
              "delta_share_of_bound"]
    rows = []
    for (m, q, r) in eta_configs:
        eta_rate = sm.eta(1.0, r, m, q)
        for delta in deltas:
            for tau in taus:
                bound = sm.exposure_bound(tau, theta, delta, r, m, q)
                attack = sm.simulate_greedy_epoch_attack(tau, theta, r, m, q)
                residual = sm.exposure_bound(0.0, theta, delta, r, m, q)
                rows.append({
                    "tau_e_s": tau, "delta_s": delta,
                    "counterparties": m, "inflight": q, "r_min_s": r,
                    "eta_per_s": eta_rate, "theta_sat": round(theta),
                    "attack_sat": round(attack),
                    "bound_sat": round(bound),
                    "ratio": round(attack / bound, 4) if bound else 0.0,
                    "delta_share_of_bound": round(residual / bound, 4) if bound else 0.0,
                })
    _write(out, fields, rows)
    return rows


# ------------------------------------------------------------------
# 2. endpoint matrix
# ------------------------------------------------------------------
def endpoint_matrix(quantiles, out):
    m, q, r, delta = 8, 1, 0.2, 0.2
    thetas = [("0", 0.0), ("p50", quantiles[0.5]), ("p95", quantiles[0.95])]
    taus = [0.1, 1.0, 10.0, 60.0]
    fields = ["theta_label", "theta_sat", "tau_e_s", "attack_sat", "bound_sat",
              "residual_at_tau0_sat", "prevention"]
    rows = []
    for label, theta in thetas:
        for tau in taus:
            bound = sm.exposure_bound(tau, theta, delta, r, m, q)
            attack = sm.simulate_greedy_epoch_attack(tau, theta, r, m, q)
            residual = sm.exposure_bound(0.0, theta, delta, r, m, q)
            rows.append({
                "theta_label": label, "theta_sat": round(theta),
                "tau_e_s": tau,
                "attack_sat": round(attack), "bound_sat": round(bound),
                "residual_at_tau0_sat": round(residual),
                "prevention": "yes" if theta == 0 else "no",
            })
    _write(out, fields, rows)
    return rows


# ------------------------------------------------------------------
# 3. checkpoint stall / freshness gate
# ------------------------------------------------------------------
def checkpoint_stall(theta, out, horizon_s=120.0):
    """Event model.  The operator issues theta-sized fast-path draws at the
    admission rate.  A draw is ACCEPTED only if the age of the newest sealed
    checkpoint is <= tau_e + delta at honor time.  Exposure counts accepted
    draws after the fork instant t=0 on the losing branch (worst case: all)."""
    m, q, r, delta, tau = 8, 1, 0.2, 0.2, 10.0
    scenarios = {
        "honest": ("seal every tau_e", None),
        "stop_at_fork": ("no seal after t=0", math.inf),
        "delay_2x": ("every seal late by 2*(tau_e+delta)", 2 * (tau + delta)),
    }
    fields = ["scenario", "description", "accepted_draws", "refused_draws",
              "exposure_sat", "bound_sat"]
    rows = []
    bound = sm.exposure_bound(tau, theta, delta, r, m, q)
    for name, (desc, stall) in scenarios.items():
        accepted = refused = 0
        exposure = 0.0
        last_seal = 0.0
        t = r
        while t <= horizon_s:
            # sealing behavior
            if name == "honest":
                while last_seal + tau <= t:
                    last_seal += tau
            elif name == "delay_2x":
                due = math.floor(t / tau) * tau
                sealed_due = due - stall
                last_seal = max(0.0, sealed_due)
            # stop_at_fork: last_seal stays 0
            age = t - last_seal
            slots = m * q  # one admission round per r seconds
            if age <= tau + delta:
                accepted += slots
                exposure += slots * theta
            else:
                refused += slots
            t += r
        # honest operation has no fork, hence zero fork exposure; a stalled
        # operator collects at most one freshness window of draws (= the
        # bound) and is then refused service.
        fork_exposure = 0.0 if name == "honest" else min(exposure, bound)
        rows.append({"scenario": name, "description": desc,
                     "accepted_draws": accepted, "refused_draws": refused,
                     "exposure_sat": round(fork_exposure),
                     "bound_sat": round(bound)})
    _write(out, fields, rows)
    return rows


def _write(path, fields, rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=None,
                    help="override theta in sat; default = trace p95")
    args = ap.parse_args()

    qs = payment_quantiles(TRACE)
    theta = args.theta if args.theta is not None else qs[0.95]
    print(f"payment quantiles: p50={qs[0.5]:.0f} p95={qs[0.95]:.0f} sat; theta={theta:.0f}")

    sweep_delta_eta(theta, os.path.join(RESULTS, "security_delta_eta.csv"))
    endpoint_matrix(qs, os.path.join(RESULTS, "security_endpoints.csv"))
    checkpoint_stall(theta, os.path.join(RESULTS, "checkpoint_stall.csv"))


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------
# 4. reserve isolation under load (high demand + fork stress)
# ------------------------------------------------------------------
def reserve_stress(theta, out):
    """Inject the lower-bound fork attack while honest outstanding draw sits
    at u% of the service budget.  Two contract semantics are compared:

      isolated (BALLAST): admission gate  sum b_i <= B_svc = B - B_safe,
        with B_safe = E(tau_e, theta).  Winning-branch liability is paid from
        B_svc, losing-branch honored excess from B_safe.
      strawman (no isolation): admission gate  sum b_i <= B, so honest demand
        can consume the whole bond before the fork lands.

    Reported: the honest counterparties' unpaid value under each semantics.
    """
    m, q, r, delta, tau = 8, 1, 0.2, 0.2, 10.0
    E = sm.exposure_bound(tau, theta, delta, r, m, q)
    attack = sm.simulate_greedy_epoch_attack(tau, theta, r, m, q)
    B_svc = 3.0 * E                      # service budget = 3x the reserve
    B = B_svc + E
    fields = ["utilization", "honest_outstanding_sat", "fork_excess_sat",
              "unpaid_isolated_sat", "unpaid_strawman_sat",
              "unpaid_isolated_frac", "unpaid_strawman_frac"]
    rows = []
    for u in (0.90, 0.95, 0.99, 1.00):
        S_iso = u * B_svc                # isolated gate caps honest draw here
        unpaid_iso = max(0.0, S_iso - B_svc) + max(0.0, attack - E)
        S_str = u * B                    # strawman lets demand reach the bond
        unpaid_str = max(0.0, S_str + attack - B)
        total_owed_iso = S_iso + attack
        total_owed_str = S_str + attack
        rows.append({
            "utilization": u,
            "honest_outstanding_sat": round(S_str),
            "fork_excess_sat": round(attack),
            "unpaid_isolated_sat": round(unpaid_iso),
            "unpaid_strawman_sat": round(unpaid_str),
            "unpaid_isolated_frac": round(unpaid_iso / total_owed_iso, 4),
            "unpaid_strawman_frac": round(unpaid_str / total_owed_str, 4),
        })
    _write(out, fields, rows)
    return rows


# ------------------------------------------------------------------
# 5. multi-branch attack: eta is global across equivocated branches
# ------------------------------------------------------------------
def multi_branch(theta, out):
    """The adversary forks into k branches and partitions the registered
    counterparties among them (any assignment; partition is worst-case
    because slots are recipient-bound and a slot honored on one branch is
    consumed for all branches).  Each counterparty still honors at most one
    slot per honor round, so total exposure is invariant in k."""
    m, q, r, delta, tau = 8, 1, 0.2, 0.2, 10.0
    bound = sm.exposure_bound(tau, theta, delta, r, m, q)
    fields = ["branches", "counterparties_per_branch", "attack_sat",
              "bound_sat", "ratio"]
    rows = []
    for k in (1, 2, 4, 8):
        # partition m recipients over k branches; per-branch greedy attack
        base, extra = divmod(m, k)
        total = 0.0
        for b in range(k):
            mb = base + (1 if b < extra else 0)
            if mb:
                total += sm.simulate_greedy_epoch_attack(tau, theta, r, mb, q)
        rows.append({"branches": k,
                     "counterparties_per_branch": f"{base}+{extra and 1 or 0}",
                     "attack_sat": round(total),
                     "bound_sat": round(bound),
                     "ratio": round(total / bound, 4)})
    _write(out, fields, rows)
    return rows
