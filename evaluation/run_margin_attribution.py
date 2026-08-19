"""run_margin_attribution.py -- why does the Horcrux margin narrow on 2026?

Reviewer consensus E: the margin over Horcrux is 6.49pp on the 2021 graph and
1.64pp on the 2026 RGS view, and the paper reports but does not attribute the
narrowing.  This experiment decomposes the 2021->2026 difference with
controlled graph swaps, holding the payment trace, seeds, capacity scale, and
phi fixed:

  G2021      : ln_677167 as-is                       (baseline)
  G2026      : ln_20260804_rgs as-is                 (narrowed margin)
  H_capacity : 2026 topology, capacities resampled from the 2021 empirical
               capacity distribution (isolates the capacity-distribution effect)
  H_topology : 2021 topology, capacities resampled from the 2026 distribution
               (the symmetric swap)
  H_scale    : 2021 graph restricted to a random node-induced subgraph whose
               largest connected component matches the 2026 node count
               (isolates the scale effect; degree structure is not controlled)

For each variant we run LN, Horcrux, Ballast, and Ballast-PC and report the
Ballast-Horcrux margin and the pooling attribution (Ballast - Ballast-PC).
Whichever swap moves the margin toward 1.64pp carries the attribution.

Writes results/margin_attribution.csv and the derived edgelists under
network/snapshots/derived/ for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import networkx as nx

import common
import schemes

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
SNAP = os.path.join(HERE, "network", "snapshots")
DERIVED = os.path.join(SNAP, "derived")

G2021 = os.path.join(SNAP, "ln_677167.edgelist")
G2026 = os.path.join(SNAP, "ln_20260804_rgs.edgelist")


def read_edges(path):
    edges = []
    with open(path) as f:
        for line in f:
            a, b, cap = line.split()
            edges.append((int(a), int(b), int(cap)))
    return edges


def write_edges(path, edges):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for a, b, cap in edges:
            f.write(f"{a} {b} {cap}\n")


def resample_capacities(target_edges, source_edges, seed):
    """Keep target topology; draw each capacity i.i.d. from the source
    empirical capacity distribution (seeded, sorted for determinism)."""
    rng = random.Random(seed)
    pool = sorted(cap for _, _, cap in source_edges)
    return [(a, b, pool[rng.randrange(len(pool))]) for a, b, _ in target_edges]


def scale_matched_subgraph(edges, target_nodes, seed):
    """Random node-induced subgraph of the 2021 graph whose largest connected
    component has ~target_nodes nodes (binary search on the keep fraction)."""
    G = nx.Graph()
    for a, b, cap in edges:
        G.add_edge(a, b, cap=cap)
    nodes = sorted(G.nodes())
    rng = random.Random(seed)
    lo, hi = 0.1, 1.0
    best = None
    for _ in range(12):
        frac = (lo + hi) / 2
        keep = set(rng.sample(nodes, int(len(nodes) * frac)))
        H = G.subgraph(keep)
        if H.number_of_edges() == 0:
            lo = frac
            continue
        comp = max(nx.connected_components(H), key=len)
        if len(comp) < target_nodes:
            lo = frac
        else:
            hi = frac
            best = H.subgraph(comp).copy()
        rng = random.Random(seed)  # deterministic resample per iteration
    if best is None:
        best = G
    return [(a, b, best[a][b]["cap"]) for a, b in best.edges()]


def build_variants(seed):
    e21 = read_edges(G2021)
    e26 = read_edges(G2026)
    n26 = len({x for a, b, _ in e26 for x in (a, b)})
    variants = {
        "G2021": G2021,
        "G2026": G2026,
    }
    hc = os.path.join(DERIVED, "h_capacity_2026topo_2021caps.edgelist")
    write_edges(hc, resample_capacities(e26, e21, seed))
    variants["H_capacity"] = hc
    ht = os.path.join(DERIVED, "h_topology_2021topo_2026caps.edgelist")
    write_edges(ht, resample_capacities(e21, e26, seed))
    variants["H_topology"] = ht
    hs = os.path.join(DERIVED, f"h_scale_2021sub_{n26}n.edgelist")
    write_edges(hs, scale_matched_subgraph(e21, n26, seed))
    variants["H_scale"] = hs
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--tx-load", type=int, default=None)
    ap.add_argument("--max-trace", type=int, default=None)
    ap.add_argument("--phi", type=float, default=0.30)
    ap.add_argument("--graph-seed", type=int, default=0)
    args = ap.parse_args()

    if args.paper:
        repeat = args.repeat or 10
        tx_load = args.tx_load or 50000
        max_trace = args.max_trace
    else:
        repeat = args.repeat or 2
        tx_load = args.tx_load or 2000
        max_trace = args.max_trace or 200000

    variants = build_variants(args.graph_seed)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "margin_attribution.csv")
    fields = ["variant", "scheme", "success_ratio", "ci95",
              "margin_vs_horcrux_pp", "pooling_attribution_pp",
              "nodes", "edges", "seeds", "tx_load"]
    rows = []

    for vname, vpath in variants.items():
        edges = read_edges(vpath)
        nnodes = len({x for a, b, _ in edges for x in (a, b)})
        per_scheme = {k: [] for k in ("LN", "Horcrux", "Ballast", "Ballast-PC")}
        for seed in range(repeat):
            def sim():
                return common.load_sim(args.cap, seed, tx_load,
                                       max_trace=max_trace, network_file=vpath)
            per_scheme["LN"].append(schemes.work_ln(sim(), tx_load))
            per_scheme["Horcrux"].append(schemes.work_horcrux(sim(), tx_load))
            per_scheme["Ballast"].append(
                schemes.work_ballast(sim(), tx_load, bond_fraction=args.phi))
            per_scheme["Ballast-PC"].append(
                schemes.work_ballast_perchannel(sim(), tx_load,
                                                bond_fraction=args.phi))
        means = {}
        for k, results in per_scheme.items():
            ratios = [r.n_success / max(1, r.n_total) for r in results]
            m = float(np.mean(ratios))
            ci = 1.96 * float(np.std(ratios, ddof=1)) / max(1, len(ratios)) ** 0.5 \
                if len(ratios) > 1 else 0.0
            means[k] = (m, ci)
        margin = (means["Ballast"][0] - means["Horcrux"][0]) * 100
        pooling = (means["Ballast"][0] - means["Ballast-PC"][0]) * 100
        for k, (m, ci) in means.items():
            rows.append({"variant": vname, "scheme": k,
                         "success_ratio": round(m, 6), "ci95": round(ci, 6),
                         "margin_vs_horcrux_pp": round(margin, 3),
                         "pooling_attribution_pp": round(pooling, 3),
                         "nodes": nnodes, "edges": len(edges),
                         "seeds": repeat, "tx_load": tx_load})
        print(vname, {k: round(v[0], 4) for k, v in means.items()},
              "margin", round(margin, 2), "pp")

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
