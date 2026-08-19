"""Measure cross-channel shortfall correlation from the payment simulation.

The synthetic rho sweep tests the theorem over controlled dependence.  This
script supplies the missing trace-measured row: it bins directional Ballast
shortfalls by payment index, forms one demand series per channel, and reports
pairwise correlations for operators with at least four active channels.
"""

import argparse
import csv
import os
import statistics

import numpy as np

import common
import schemes

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def one_seed(seed, cap, tx_load, max_trace, phi, bins):
    sim = common.load_sim(cap, seed, tx_load, max_trace=max_trace)
    result = schemes.work_ballast(sim, tx_load, bond_fraction=phi)
    by_node = {}
    for index, operator, counterparty, amount in result.channel_demand_events:
        by_node.setdefault(operator, {}).setdefault(counterparty, np.zeros(bins))
        cell = min(bins - 1, index * bins // tx_load)
        by_node[operator][counterparty][cell] += amount

    correlations = []
    node_rows = []
    for operator, channels in by_node.items():
        active = [series for series in channels.values() if np.count_nonzero(series) >= 2]
        if len(active) < 4:
            continue
        matrix = np.asarray(active, dtype=float)
        corr = np.corrcoef(matrix)
        values = corr[np.triu_indices_from(corr, k=1)]
        values = values[np.isfinite(values)]
        if values.size:
            correlations.extend(values.tolist())
            node_rows.append((operator, len(active), float(np.mean(values))))
    return result.success_ratio, result.refusals, correlations, node_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--tx-load", type=int, default=50000)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--max-trace", type=int, default=None)
    ap.add_argument("--phi", type=float, default=.30)
    ap.add_argument("--bins", type=int, default=100)
    args = ap.parse_args()

    raw = []
    summaries = []
    for seed in range(args.repeat):
        success, refusals, correlations, nodes = one_seed(
            seed, args.cap, args.tx_load, args.max_trace, args.phi, args.bins)
        for operator, n_channels, rho in nodes:
            raw.append({"seed": seed, "operator": operator,
                        "active_channels": n_channels, "mean_pairwise_rho": rho})
        summaries.append({
            "seed": seed, "success_ratio": success, "refusals": refusals,
            "pair_count": len(correlations), "node_count": len(nodes),
            "mean_pairwise_rho": float(np.mean(correlations)) if correlations else float("nan"),
            "median_pairwise_rho": float(np.median(correlations)) if correlations else float("nan"),
        })
        print(f"seed={seed} nodes={len(nodes)} pairs={len(correlations)} "
              f"rho={summaries[-1]['mean_pairwise_rho']:.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    for name, rows in (("e2_measured_correlation_raw.csv", raw),
                       ("e2_measured_correlation.csv", summaries)):
        with open(os.path.join(RESULTS, name), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    rhos = [r["mean_pairwise_rho"] for r in summaries]
    ci = 1.96 * statistics.stdev(rhos) / len(rhos) ** .5 if len(rhos) > 1 else 0.0
    print(f"trace-measured rho={statistics.mean(rhos):.4f} +/- {ci:.4f} (95% CI)")


if __name__ == "__main__":
    main()
