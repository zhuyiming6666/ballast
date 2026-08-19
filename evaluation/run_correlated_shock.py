"""run_correlated_shock.py -- reviewer P0-2: correlated-demand stress test.

The multiplexing claim (sec 3.1 / Thm 4) assumes weakly correlated demand;
Appendix B gives the theory decay g(rho) but the paper had no *measured*
degradation curve. This script injects a one-factor Gaussian shock into the
running example's star model and measures, per average pairwise correlation
rho_bar in [0, 1]:

  * the pooled bond quantile B*(rho) and the multiplexing gain
    g = sum_i p_i / B* (sizing view), against the closed-form margin ratio
    sqrt((1 + (n-1) rho) / n) (Appendix B);
  * payment-level success of Ballast (pooled skim) versus Ballast-PC
    (per-channel skim) at the fixed default skim phi = 0.30, i.e. how much
    of the pooling margin survives a correlated shock.

Demand model: X_i = clip(mu0 + sigma0 * (sqrt(rho) Z + sqrt(1-rho) eps_i), 0)
with Z, eps_i iid standard normal, so corr(X_i, X_j) = rho before clipping.
Same funding, skim, repayment semantics as run_pooling_effectiveness.py.

Writes results/correlated_shock.csv. Runs in about a minute.
"""

from __future__ import annotations

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

MU0, SIGMA0 = 5.0, 3.0
FUNDING = 12
PHI = 0.30
N = 10
ROUNDS = 20000
SEEDS = 10
ALPHA = 0.99
RHOS = [0.0, 0.05, 0.1, 0.2, 0.37, 0.5, 0.7, 1.0]


def demand(rng, rounds, n, rho):
    z = rng.normal(size=(rounds, 1))
    eps = rng.normal(size=(rounds, n))
    x = MU0 + SIGMA0 * (np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps)
    return np.clip(x, 0.0, None)


def run_variant(n, seed, rho, pooled):
    rng = np.random.default_rng(seed)
    out_all = demand(rng, ROUNDS, n, rho)
    in_all = demand(rng, ROUNDS, n, rho)
    bal = np.full(n, FUNDING / 2.0)
    skim = bal * PHI
    bal -= skim
    bond = skim.sum() if pooled else skim.copy()
    outstanding = np.zeros(n)
    ok = tot = 0
    for t in range(ROUNDS):
        out, inflow = out_all[t], in_all[t]
        for i in range(n):
            x = out[i]
            if x <= 0:
                continue
            tot += 1
            short = x - bal[i]
            if short <= 1e-12:
                bal[i] -= x
                ok += 1
            else:
                head = (bond - outstanding.sum()) if pooled else (bond[i] - outstanding[i])
                if head >= short:
                    outstanding[i] += short
                    bal[i] = 0.0
                    ok += 1
        for i in range(n):
            y = inflow[i]
            if y <= 0:
                continue
            rep = min(y, outstanding[i])
            outstanding[i] -= rep
            y -= rep
            bal[i] = min(bal[i] + y, FUNDING)
    return ok / max(1, tot)


def sizing(rho, n=N, samples=400000, seed=0):
    rng = np.random.default_rng(seed)
    x = demand(rng, samples, n, rho)
    p_i = np.quantile(x[:, 0], ALPHA)
    b_star = np.quantile(x.sum(axis=1), ALPHA)
    mu_n = n * x.mean() / 1.0  # aggregate mean estimate: sum of means
    mu_n = x.sum(axis=1).mean()
    margin_pooled = b_star - mu_n
    margin_per = n * (p_i - x[:, 0].mean())
    theory_ratio = np.sqrt((1.0 + (n - 1) * rho) / n)
    return float(n * p_i), float(b_star), float(margin_pooled / margin_per), float(theory_ratio)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "correlated_shock.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "sum_peaks", "pooled_bond", "margin_ratio", "theory_ratio",
                    "gain", "ballast_mean", "ballast_ci", "pc_mean", "pc_ci", "edge_pts"])
        for rho in RHOS:
            sum_p, b_star, mratio, tratio = sizing(rho)
            gain = sum_p / b_star
            pooled = [run_variant(N, 1000 + s, rho, True) for s in range(SEEDS)]
            per = [run_variant(N, 1000 + s, rho, False) for s in range(SEEDS)]
            pm, pcm = float(np.mean(pooled)), float(np.mean(per))
            pci = 1.96 * np.std(pooled, ddof=1) / np.sqrt(SEEDS)
            pcci = 1.96 * np.std(per, ddof=1) / np.sqrt(SEEDS)
            w.writerow([rho, round(sum_p, 2), round(b_star, 2), round(mratio, 3),
                        round(tratio, 3), round(gain, 3), round(100 * pm, 2),
                        round(100 * pci, 2), round(100 * pcm, 2), round(100 * pcci, 2),
                        round(100 * (pm - pcm), 2)])
            print(f"rho={rho:.2f} gain={gain:.3f} margin_ratio={mratio:.3f} "
                  f"(theory {tratio:.3f}) ballast={100*pm:.2f}% pc={100*pcm:.2f}% "
                  f"edge={100*(pm-pcm):.2f}pp")
    print("wrote", path)


if __name__ == "__main__":
    main()
