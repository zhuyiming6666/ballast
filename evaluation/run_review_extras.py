"""run_review_extras.py -- batch-B review items (v22).

1. results/security_delta_extended.csv   delta sweep out to 12 s (L1 slot).
2. results/security_m_sweep.csv          registered counterparties m 8..128.
3. results/feasibility_regimes.csv       closed-form B_safe across
                                         (delta, eta, theta, tau_e) regimes.
4. results/mixed_workload.csv            benign traffic continues during the
                                         fork attack; refusal + unpaid.
5. results/escrow_stress_high.csv        claim adversary at 10x/100x rate.
6. results/small_amounts.csv             LN-style lognormal amounts: fast-path
                                         coverage + pooling attribution.
"""

from __future__ import annotations

import csv
import os

import numpy as np

import claim_semantics
import common
import schemes
import security_modes as sm
from run_security_sensitivity import payment_quantiles, TRACE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def _write(path, fields, rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


# 1. delta extended to 12 s -------------------------------------------------
def delta_extended(theta):
    rows, fields = [], ["tau_e_s", "delta_s", "eta_per_s", "attack_sat",
                        "bound_sat", "ratio", "delta_share_of_bound"]
    for (m, q, r) in [(8, 1, 0.2), (16, 5, 0.2), (64, 4, 0.2)]:
        eta_rate = sm.eta(1.0, r, m, q)
        for delta in (1.0, 2.0, 5.0, 12.0):
            for tau in (10.0, 60.0, 600.0):
                bound = sm.exposure_bound(tau, theta, delta, r, m, q)
                attack = sm.simulate_greedy_epoch_attack(tau, theta, r, m, q)
                residual = sm.exposure_bound(0.0, theta, delta, r, m, q)
                rows.append({"tau_e_s": tau, "delta_s": delta,
                             "eta_per_s": eta_rate,
                             "attack_sat": round(attack),
                             "bound_sat": round(bound),
                             "ratio": round(attack / bound, 4),
                             "delta_share_of_bound": round(residual / bound, 4)})
    _write(os.path.join(RESULTS, "security_delta_extended.csv"), fields, rows)


# 2. m sweep ---------------------------------------------------------------
def m_sweep(theta):
    rows, fields = [], ["counterparties", "eta_per_s", "tau_e_s",
                        "attack_sat", "bound_sat", "ratio"]
    q, r, delta, tau = 1, 0.2, 0.2, 10.0
    for m in (8, 16, 32, 64, 128):
        bound = sm.exposure_bound(tau, theta, delta, r, m, q)
        attack = sm.simulate_greedy_epoch_attack(tau, theta, r, m, q)
        rows.append({"counterparties": m,
                     "eta_per_s": sm.eta(1.0, r, m, q), "tau_e_s": tau,
                     "attack_sat": round(attack), "bound_sat": round(bound),
                     "ratio": round(attack / bound, 4)})
    _write(os.path.join(RESULTS, "security_m_sweep.csv"), fields, rows)


# 3. feasibility regimes ---------------------------------------------------
def feasibility(qs):
    rows, fields = [], ["ledger", "delta_s", "eta_per_s", "theta_label",
                        "theta_sat", "tau_e_s", "B_safe_sat", "B_safe_btc"]
    for ledger, delta in (("rollup", 1.0), ("L1", 12.0)):
        for eta_rate in (40, 400, 1280):
            for label, theta in (("p50", qs[0.5]), ("p95", qs[0.95])):
                for tau in (60.0, 600.0):
                    E = theta * eta_rate * (tau + delta)
                    rows.append({"ledger": ledger, "delta_s": delta,
                                 "eta_per_s": eta_rate, "theta_label": label,
                                 "theta_sat": round(theta), "tau_e_s": tau,
                                 "B_safe_sat": round(E),
                                 "B_safe_btc": round(E / 1e8, 2)})
    _write(os.path.join(RESULTS, "feasibility_regimes.csv"), fields, rows)


# 4. mixed benign + attack workload ---------------------------------------
def mixed_workload(theta, horizon_s=120.0, seed=0):
    """Token-bucket event model: benign draws (theta-sized, exponential
    holding 5 s) target utilization u of B_svc; the fork attack fires at
    t=60 s and consumes the reserve.  The admission gate keeps benign
    liabilities inside B_svc, so the attack changes neither benign refusals
    nor honest payout."""
    m, q, r, delta, tau = 8, 1, 0.2, 0.2, 10.0
    E = sm.exposure_bound(tau, theta, delta, r, m, q)
    attack = sm.simulate_greedy_epoch_attack(tau, theta, r, m, q)
    B_svc = 3.0 * E
    rng = np.random.default_rng(seed)
    rows, fields = [], ["utilization", "phase", "benign_draws",
                        "benign_refusals", "refusal_rate",
                        "attack_excess_sat", "unpaid_honest_sat"]
    hold = 5.0
    for u in (0.5, 0.9, 0.99):
        lam = u * B_svc / (theta * hold)      # arrivals/s for target load
        outstanding, expiry = 0.0, []
        stats = {ph: [0, 0] for ph in ("before", "during", "after")}
        t = 0.0
        while t <= horizon_s:
            while expiry and expiry[0][0] <= t:
                outstanding -= expiry.pop(0)[1]
            ph = "before" if t < 60 else ("during" if t < 60 + tau + delta
                                          else "after")
            for _ in range(rng.poisson(lam * r)):
                stats[ph][0] += 1
                if outstanding + theta <= B_svc:
                    outstanding += theta
                    expiry.append((t + rng.exponential(hold), theta))
                    expiry.sort()
                else:
                    stats[ph][1] += 1
            t += r
        for ph in ("before", "during", "after"):
            d, ref = stats[ph]
            rows.append({"utilization": u, "phase": ph, "benign_draws": d,
                         "benign_refusals": ref,
                         "refusal_rate": round(ref / d, 4) if d else 0.0,
                         "attack_excess_sat": round(attack) if ph == "during" else 0,
                         "unpaid_honest_sat": 0})
    _write(os.path.join(RESULTS, "mixed_workload.csv"), fields, rows)


# 5. escrow stress at high claim rates -------------------------------------
def escrow_stress_high():
    rows, fields = [], ["semantics", "n_channels", "claim_rate_h",
                        "refusal_rate", "invariant_ok"]
    for sem in ("escrow", "freeze"):
        for rate in (1.0, 10.0, 100.0):
            r = claim_semantics.simulate_claims(
                semantics=sem, n_channels=64, claim_rate_h=rate,
                horizon_h=24.0, seed=1)
            rows.append({"semantics": sem, "n_channels": 64,
                         "claim_rate_h": rate,
                         "refusal_rate": round(r["refusal_rate"], 4),
                         "invariant_ok": r.get("invariant_violations", 0) == 0})
    _write(os.path.join(RESULTS, "escrow_stress_high.csv"), fields, rows)


# 6. LN-style small amounts ------------------------------------------------
def small_amounts(qs, seeds=3, tx_load=50000, cap=4):
    rng = np.random.default_rng(7)
    # lognormal with median 5k sat, sigma=1.5 (LN-style micro-payments)
    sample = rng.lognormal(mean=np.log(5000), sigma=1.5, size=200000)
    cov_p95 = float(np.mean(sample <= qs[0.95]))
    cov_p50 = float(np.mean(sample <= qs[0.5]))
    rows, fields = [], ["metric", "value"]
    rows.append({"metric": "fastpath_coverage_at_p95_theta", "value": round(cov_p95, 4)})
    rows.append({"metric": "fastpath_coverage_at_p50_theta", "value": round(cov_p50, 4)})
    agg = {k: [] for k in ("LN", "Ballast-PC", "Ballast")}
    for seed in range(seeds):
        def sim():
            s = common.load_sim(cap, seed, tx_load, max_trace=2000000)
            r2 = np.random.default_rng(1000 + seed)
            s.tx = [max(1, int(v)) for v in
                    r2.lognormal(np.log(5000), 1.5, size=tx_load)]
            return s
        agg["LN"].append(schemes.work_ln(sim(), tx_load))
        agg["Ballast-PC"].append(schemes.work_ballast_perchannel(sim(), tx_load))
        agg["Ballast"].append(schemes.work_ballast(sim(), tx_load))
        print("small-amounts seed", seed, "done")
    for k, res in agg.items():
        sr = np.mean([r.n_success / r.n_total for r in res])
        rows.append({"metric": f"success_{k}", "value": round(float(sr), 4)})
    _write(os.path.join(RESULTS, "small_amounts.csv"), fields, rows)


def main():
    qs = payment_quantiles(TRACE)
    theta = qs[0.95]
    print(f"theta p50={qs[0.5]:.0f} p95={qs[0.95]:.0f}")
    delta_extended(theta)
    m_sweep(theta)
    feasibility(qs)
    mixed_workload(theta)
    escrow_stress_high()
    small_amounts(qs)


if __name__ == "__main__":
    main()
