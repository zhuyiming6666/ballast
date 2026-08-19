"""run_throttled_hub.py -- reviewer P0-3: throughput cost of admission throttling.

Sec 7 (feasible operating regimes) proposes throttling a large hub's admission
registration (e.g. a 500-channel hub to eta = 20 draws/s) to keep the safety
reserve B_safe = theta * eta * (tau_e + delta) affordable, but the paper had no
measurement of what throttling costs in end-to-end success. This script
quantifies it on an event-level 500-channel hub:

  * n = 500 channels, funding 12, operator side c/2, skim phi = 0.30 pooled;
  * each second, each channel independently carries a payment event with
    probability P_EVENT (hub-wide ~100 events/s), size truncated N(5, 3^2);
    matched reverse flow retires draws same-channel (zero drift);
  * a payment that needs a draw consumes one fast-path admission slot from a
    per-second token bucket of size eta; when the bucket is empty the draw is
    refused and the payment fails (conservative: no queueing/retry);
  * eta sweeps {unlimited, 160, 80, 40, 20, 10, 5}/s.

Reports success rate and the share of failures attributable to throttling
(draws refused with pool headroom available). Writes
results/throttled_hub.csv. Runs in a few minutes.
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
N = 500
P_EVENT = 0.2          # per-channel per-second event probability (~100/s hub-wide)
SECONDS = 3600
SEEDS = 5
ETAS = [None, 160, 80, 40, 20, 10, 5]  # None = unthrottled


def run(seed, eta):
    rng = np.random.default_rng(seed)
    bal = np.full(N, FUNDING / 2.0)
    skim = bal * PHI
    bal -= skim
    bond = skim.sum()
    outstanding = np.zeros(N)
    ok = tot = throttled = 0
    for _ in range(SECONDS):
        budget = np.inf if eta is None else eta
        active = np.flatnonzero(rng.random(N) < P_EVENT)
        sizes = np.clip(rng.normal(MU0, SIGMA0, active.size), 0.0, None)
        inflow_ch = np.flatnonzero(rng.random(N) < P_EVENT)
        inflows = np.clip(rng.normal(MU0, SIGMA0, inflow_ch.size), 0.0, None)
        for i, x in zip(active, sizes):
            if x <= 0:
                continue
            tot += 1
            short = x - bal[i]
            if short <= 1e-12:
                bal[i] -= x
                ok += 1
                continue
            head = bond - outstanding.sum()
            if head >= short:
                if budget >= 1:
                    budget -= 1
                    outstanding[i] += short
                    bal[i] = 0.0
                    ok += 1
                else:
                    throttled += 1
            # else: pool exhausted -> capacity failure
        for i, y in zip(inflow_ch, inflows):
            rep = min(y, outstanding[i])
            outstanding[i] -= rep
            bal[i] = min(bal[i] + (y - rep), FUNDING)
    return ok / max(1, tot), throttled / max(1, tot)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "throttled_hub.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eta_per_s", "success_mean", "success_ci", "throttle_loss_pts"])
        for eta in ETAS:
            res = [run(2000 + s, eta) for s in range(SEEDS)]
            succ = [r[0] for r in res]
            thr = [r[1] for r in res]
            m = float(np.mean(succ))
            ci = 1.96 * np.std(succ, ddof=1) / np.sqrt(SEEDS)
            tl = float(np.mean(thr))
            w.writerow(["inf" if eta is None else eta, round(100 * m, 2),
                        round(100 * ci, 2), round(100 * tl, 2)])
            print(f"eta={'inf' if eta is None else eta:>4} success={100*m:.2f}% "
                  f"+/-{100*ci:.2f} throttle-loss={100*tl:.2f}pp")
    print("wrote", path)


if __name__ == "__main__":
    main()
