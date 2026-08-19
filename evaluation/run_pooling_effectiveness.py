"""run_pooling_effectiveness.py -- the paper's sec 3.4 star-topology experiment.

Small-scale, payment-level validation of the pooling claim on the running
example, without the full-network machinery and without headline numbers:

  * one operator with n channels in a star; per-round outbound demand on each
    channel is i.i.d. truncated Normal(mu0=5, sigma0=3) (the intro example),
    inbound reverse flow has the same law (zero drift), so depletion is
    episodic, not persistent;
  * every variant locks identical capital per channel (funding c = 12, the
    99% per-channel peak of the example; operator side starts at c/2);
  * LN         : no bond; an outbound request beyond the balance fails.
    Ballast-PC : skim phi of each channel into a per-channel bond; a channel
                 may draw only against its own skim.
    Ballast    : the same skim pooled into one node bond; any channel draws
                 against shared headroom.  Repayment is same-channel, as in
                 the protocol.
  * n sweeps {4, 8, 10, 16, 32, 64}; we also size the minimal pooled bond
    reaching service level alpha per n and fit the log-log slope of its
    safety margin (theory: 1/2).

Writes results/pooling_effectiveness.csv (success per variant per n) and
results/pooling_margin_fit.csv (bond sizing + slope).  Runs in seconds.
"""

from __future__ import annotations

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

MU0, SIGMA0 = 5.0, 3.0
FUNDING = 12          # per-channel funding = 99% per-channel peak of the example
PHI = 0.30
ROUNDS = 20000
SEEDS = 10


def demand(rng, size):
    return np.clip(rng.normal(MU0, SIGMA0, size), 0.0, None)


def run_variant(n, seed, pooled, use_bond):
    rng = np.random.default_rng(seed)
    bal = np.full(n, FUNDING / 2.0)
    skim = bal * PHI
    bal -= skim
    if use_bond:
        bond = skim.sum() if pooled else skim.copy()
    outstanding = np.zeros(n)
    ok = tot = 0
    for _ in range(ROUNDS):
        out = demand(rng, n)
        inflow = demand(rng, n)
        for i in range(n):
            x = out[i]
            if x <= 0:
                continue
            tot += 1
            short = x - bal[i]
            if short <= 1e-12:
                bal[i] -= x
                ok += 1
            elif use_bond:
                head = (bond - outstanding.sum()) if pooled else (bond[i] - outstanding[i])
                if head >= short:
                    outstanding[i] += short
                    bal[i] = 0.0
                    ok += 1
                # else refuse
            # LN: refuse
        # inbound reverse flow: repay same-channel first, then rebuild balance
        for i in range(n):
            y = inflow[i]
            if y <= 0:
                continue
            rep = min(y, outstanding[i])
            outstanding[i] -= rep
            y -= rep
            bal[i] = min(bal[i] + y, FUNDING)  # capacity-capped
    return ok / max(1, tot)


def pooled_bond_for_alpha(n, alpha=0.99, samples=200000, seed=0):
    """alpha-quantile of aggregate demand D = sum_i X_i (sizing view)."""
    rng = np.random.default_rng(seed)
    D = demand(rng, (samples, n)).sum(axis=1)
    return float(np.quantile(D, alpha))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ns = [4, 8, 10, 16, 32, 64]

    # payment-level success comparison
    path1 = os.path.join(RESULTS, "pooling_effectiveness.csv")
    rows = []
    for n in ns:
        for name, (pooled, use_bond) in {
                "LN": (False, False),
                "Ballast-PC": (False, True),
                "Ballast": (True, True)}.items():
            vals = [run_variant(n, s, pooled, use_bond) for s in range(SEEDS)]
            m = float(np.mean(vals))
            ci = 1.96 * float(np.std(vals, ddof=1)) / len(vals) ** 0.5
            rows.append({"n": n, "variant": name,
                         "success_ratio": round(m, 5), "ci95": round(ci, 5),
                         "rounds": ROUNDS, "seeds": SEEDS})
            print(rows[-1])
    with open(path1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", path1)

    # sizing view: pooled bond vs sum of per-channel peaks, log-log margin slope
    path2 = os.path.join(RESULTS, "pooling_margin_fit.csv")
    z99 = 2.326
    p0 = MU0 + z99 * SIGMA0
    rows2 = []
    margins = []
    for n in ns:
        bstar = pooled_bond_for_alpha(n)
        margin = bstar - n * MU0
        margins.append(margin)
        rows2.append({"n": n, "pooled_bond_p99": round(bstar, 2),
                      "sum_per_channel_peaks": round(n * p0, 2),
                      "safety_margin": round(margin, 2),
                      "gain": round(n * p0 / bstar, 3)})
        print(rows2[-1])
    slope = float(np.polyfit(np.log(ns), np.log(margins), 1)[0])
    for r in rows2:
        r["loglog_margin_slope"] = round(slope, 3)
    with open(path2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0]))
        w.writeheader(); w.writerows(rows2)
    print("wrote", path2, "slope", round(slope, 3))


if __name__ == "__main__":
    main()
