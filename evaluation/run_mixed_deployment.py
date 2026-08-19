"""run_mixed_deployment.py -- partial / incremental adoption (review W5).

Only the top-K% nodes by degree adopt BALLAST (skim their side of each
channel into a node bond and draw against it); everyone else runs plain LN
semantics.  K = 0 is LN, K = 100 is full BALLAST.

Writes results/mixed_deployment.csv.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

import common
import schemes

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def work_ballast_partial(sim, tx_load, adopters, bond_fraction=0.30):
    """schemes.work_ballast restricted to an adopter set: only adopters skim
    and only adopters may draw; repay is same-channel, as in the paper."""
    phi = bond_fraction
    bond = {}
    outstanding = collections.defaultdict(int)
    total_outstanding = collections.defaultdict(int)
    for ch in list(sim.within.keys()):
        a, b = ch
        ba, bb = sim.within[ch]
        sa = int(ba * phi) if a in adopters else 0
        sb = int(bb * phi) if b in adopters else 0
        sim.within[ch] = (ba - sa, bb - sb)
        bond[a] = bond.get(a, 0) + sa
        bond[b] = bond.get(b, 0) + sb
    for n in sim.nodes:
        bond.setdefault(n, 0)

    res = schemes.RunResult()

    def headroom(u):
        return bond[u] - total_outstanding[u]

    for i in range(tx_load):
        t1, t2 = sim.sample_pair("uniform", None)
        path = schemes._route(sim, t1, t2)
        amt = sim.tx[i]
        flag = True
        planned = collections.defaultdict(int)
        draws = []
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            z0 = sim.get_within(u, v)[0]
            if z0 >= amt:
                continue
            zmax = amt - z0
            if u in adopters and headroom(u) - planned[u] >= zmax:
                planned[u] += zmax
                draws.append((u, v, zmax))
            else:
                flag = False
                break
        if flag:
            for (u, v, zmax) in draws:
                outstanding[(u, v)] += zmax
                total_outstanding[u] += zmax
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 + zmax, z1)
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                z0, z1 = sim.get_within(u, v)
                sim.update_within(u, v, z0 - amt, z1 + amt)
                reverse = (v, u)
                repaid = min(amt, outstanding[reverse])
                if repaid:
                    outstanding[reverse] -= repaid
                    total_outstanding[v] -= repaid
            res.n_success += 1
            res.success_volume += amt
        res.n_total += 1
        res.total_volume += amt
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--cap", type=int, default=4)
    args = ap.parse_args()
    seeds = 3 if not args.paper else 10
    tx_load = 50000
    max_trace = 2000000

    rows = []
    for K in (0, 1, 5, 10, 25, 100):
        srs = []
        for seed in range(seeds):
            sim = common.load_sim(args.cap, seed, tx_load, max_trace=max_trace)
            deg = collections.Counter()
            for (a, b) in sim.within:
                deg[a] += 1
                deg[b] += 1
            ranked = [n for n, _ in deg.most_common()]
            k = int(len(ranked) * K / 100)
            adopters = set(ranked[:k]) if K < 100 else set(sim.nodes)
            r = work_ballast_partial(sim, tx_load, adopters)
            srs.append(r.n_success / r.n_total)
        rows.append({"adoption_pct": K,
                     "adopters": k if K < 100 else len(ranked),
                     "success": round(float(np.mean(srs)), 4),
                     "ci95": round(1.96 * float(np.std(srs)) / max(1, len(srs)) ** 0.5, 4),
                     "seeds": seeds})
        print(rows[-1])
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "mixed_deployment.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
