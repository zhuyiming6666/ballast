"""run_capital_utilization.py -- utilization-side capital efficiency (Q2).

The theory answers "how small can the bond be" (sizing).  This experiment
answers "how much of it is used" (utilization), the operator-facing view a
reviewer will ask about.

Metrics, at capacity scale 4 on the main traces:
  * capital turnover  = successful payment value / locked capital
    (comparable across ALL schemes; locked capital is equal by construction);
  * bond utilization  = time-average and peak of  sum_u outstanding_u(t)
    divided by sum_u bond_u  (BALLAST and BALLAST-PC only; sampled every
    `sample_every` payments);
  * draw-assisted share = fraction of successful payments that needed at
    least one draw.

The bond is sized for peaks, so the honest reading pairs the average with the
peak: a low average with a high peak is the multiplexing story working, not
idle waste.

Writes results/capital_utilization.csv.
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


def work_ballast_sampled(sim, tx_load, bond_fraction=0.30, pooled=True,
                         sample_every=100):
    """work_ballast / work_ballast_perchannel with periodic utilization
    sampling.  Mirrors schemes.py semantics (same-channel repay)."""
    phi = bond_fraction
    bond = {}
    outstanding = collections.defaultdict(int)
    total_outstanding = collections.defaultdict(int)
    per_channel_bond = {}

    for ch in list(sim.within.keys()):
        a, b = ch
        ba, bb = sim.within[ch]
        sa, sb = int(ba * phi), int(bb * phi)
        sim.within[ch] = (ba - sa, bb - sb)
        bond[a] = bond.get(a, 0) + sa
        bond[b] = bond.get(b, 0) + sb
        per_channel_bond[(a, b)] = sa
        per_channel_bond[(b, a)] = sb
    for n in sim.nodes:
        bond.setdefault(n, 0)
    total_bond = sum(bond.values())

    def headroom(u, v):
        if pooled:
            return bond[u] - total_outstanding[u]
        return per_channel_bond.get((u, v), 0) - outstanding[(u, v)]

    res = schemes.RunResult()
    samples = []
    peak_util, util_sum, util_cnt = {}, {}, {}
    draw_assisted = 0

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
            key = u if pooled else (u, v)
            if headroom(u, v) - planned[key] >= zmax:
                planned[key] += zmax
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
            if draws:
                draw_assisted += 1
        res.n_total += 1
        res.total_volume += amt

        if i % sample_every == 0:
            samples.append(sum(total_outstanding.values()) / total_bond)
            for u, o in total_outstanding.items():
                if o > 0 and bond.get(u, 0) > 0:
                    peak_util[u] = max(peak_util.get(u, 0.0), o / bond[u])
                    util_sum[u] = util_sum.get(u, 0.0) + o / bond[u]
                    util_cnt[u] = util_cnt.get(u, 0) + 1

    res.util_avg = float(np.mean(samples))
    res.util_peak = float(np.max(samples))
    # per-active-node utilization: nodes that ever held outstanding draw
    peaks = np.array(list(peak_util.values())) if peak_util else np.array([0.0])
    res.node_peak_mean = float(np.mean(peaks))
    res.node_peak_p95 = float(np.quantile(peaks, 0.95))
    res.node_peak_over80 = float(np.mean(peaks > 0.8))
    means = np.array([util_sum[u] / util_cnt[u] for u in util_sum]) if util_sum else np.array([0.0])
    res.node_avg_mean = float(np.mean(means))
    res.active_nodes = len(peak_util)
    res.draw_assisted = draw_assisted
    res.total_bond = total_bond
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--tx-load", type=int, default=None)
    ap.add_argument("--max-trace", type=int, default=None)
    ap.add_argument("--phi", type=float, default=0.30)
    args = ap.parse_args()

    repeat = args.repeat or (10 if args.paper else 2)
    tx_load = args.tx_load or (50000 if args.paper else 2000)
    max_trace = args.max_trace or (None if args.paper else 200000)

    rows = []
    agg = collections.defaultdict(list)
    for seed in range(repeat):
        def sim():
            return common.load_sim(args.cap, seed, tx_load, max_trace=max_trace)
        locked = sum(sum(v) for v in sim().within.values())
        runs = {
            "LN": schemes.work_ln(sim(), tx_load),
            "Horcrux": schemes.work_horcrux(sim(), tx_load),
            "Ballast-PC": work_ballast_sampled(sim(), tx_load, args.phi, pooled=False),
            "Ballast": work_ballast_sampled(sim(), tx_load, args.phi, pooled=True),
        }
        for name, r in runs.items():
            agg[(name, "turnover")].append(r.success_volume / locked)
            agg[(name, "success")].append(r.n_success / max(1, r.n_total))
            if hasattr(r, "util_avg"):
                agg[(name, "node_peak_mean")].append(r.node_peak_mean)
                agg[(name, "node_peak_p95")].append(r.node_peak_p95)
                agg[(name, "node_peak_over80")].append(r.node_peak_over80)
                agg[(name, "node_avg_mean")].append(r.node_avg_mean)
                agg[(name, "active_nodes")].append(r.active_nodes)
                agg[(name, "draw_share")].append(r.draw_assisted / max(1, r.n_success))
        print(f"seed {seed} done")

    ln_turnover = float(np.mean(agg[("LN", "turnover")]))
    fields = ["scheme", "success_ratio", "turnover_vs_LN",
              "node_peak_util_mean", "node_peak_util_p95", "nodes_peak_over80",
              "node_avg_util_mean", "active_nodes", "draw_assisted_share",
              "seeds", "tx_load", "cap"]
    out = []
    for name in ("LN", "Horcrux", "Ballast-PC", "Ballast"):
        def g(k, digits=4):
            return round(float(np.mean(agg[(name, k)])), digits) if agg[(name, k)] else ""
        row = {"scheme": name,
               "success_ratio": g("success"),
               "turnover_vs_LN": round(float(np.mean(agg[(name, "turnover")])) / ln_turnover, 3),
               "node_peak_util_mean": g("node_peak_mean"),
               "node_peak_util_p95": g("node_peak_p95"),
               "nodes_peak_over80": g("node_peak_over80"),
               "node_avg_util_mean": g("node_avg_mean"),
               "active_nodes": g("active_nodes", 0),
               "draw_assisted_share": g("draw_share"),
               "seeds": repeat, "tx_load": tx_load, "cap": args.cap}
        out.append(row)
        print(row)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "capital_utilization.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print("wrote", path)


if __name__ == "__main__":
    main()
