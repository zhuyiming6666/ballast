"""Audit the 2021/2023/2026 Lightning snapshots used by the paper.

The three public datasets expose different fields.  The 2021 baseline and the
2026 LDK RGS v2 snapshot contain funding capacity and can therefore be used by
the capacity-constrained simulator.  The 2023 geolocated archive contains
topology and directional routing policy, but no funding amount; it is used for
an independently reproducible topology/policy regime check and is never fed a
fabricated capacity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics

import networkx as nx


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_DATASETS = os.path.join(ROOT, "datasets")
DEFAULT_RESULTS = os.path.join(ROOT, "results", "dataset_regimes.csv")


def percentile(values, q):
    values = sorted(values)
    if not values:
        return ""
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    weight = index - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def graph_stats(graph, seed=20260808, samples=256):
    simple = nx.Graph(graph)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    component = max(nx.connected_components(simple), key=len)
    lcc = simple.subgraph(component).copy()
    degrees = [degree for _, degree in lcc.degree()]
    rng = random.Random(seed)
    nodes = list(lcc.nodes())
    distances = []
    # One BFS serves several targets, keeping the audit cheap and deterministic.
    sources = rng.sample(nodes, min(samples, len(nodes)))
    for source in sources:
        lengths = nx.single_source_shortest_path_length(lcc, source)
        target = rng.choice(nodes)
        distances.append(lengths[target])
    return {
        "nodes": simple.number_of_nodes(),
        "pairs": simple.number_of_edges(),
        "lcc_nodes": lcc.number_of_nodes(),
        "lcc_edges": lcc.number_of_edges(),
        "median_degree": percentile(degrees, .50),
        "p90_degree": percentile(degrees, .90),
        "mean_sampled_distance": statistics.mean(distances),
        "distance_samples": len(distances),
    }


def load_edgelist(path):
    graph = nx.Graph()
    capacities = []
    with open(path) as handle:
        for line in handle:
            source, target, capacity = line.split()
            capacity = int(capacity)
            graph.add_edge(source, target, capacity=capacity)
            capacities.append(capacity)
    return graph, capacities


def row_2021():
    path = os.path.join(HERE, "network", "lightning_simplified_component.edgelist")
    graph, capacities = load_edgelist(path)
    return {
        "year": 2021,
        "snapshot": "block-677167",
        "source_kind": "archival channel graph",
        **graph_stats(graph),
        "directional_policies": "",
        "median_capacity_sat": percentile(capacities, .50),
        "median_base_fee_msat": "",
        "median_fee_ppm": "",
        "median_cltv_delta": "",
        "capacity_observed": True,
        "capacity_experiment": True,
        "caveat": "baseline largest connected component",
    }


def row_2023(datasets):
    path = os.path.join(datasets, "20230716.gml.geo")
    graph = nx.read_gml(path)
    fees_base = []
    fees_ppm = []
    cltv = []
    for _, _, data in graph.edges(data=True):
        if "fee_base_msat" in data:
            fees_base.append(float(data["fee_base_msat"]))
        if "fee_proportional_millionths" in data:
            fees_ppm.append(float(data["fee_proportional_millionths"]))
        if "cltv_expiry_delta" in data:
            cltv.append(float(data["cltv_expiry_delta"]))
    stats = graph_stats(graph)
    return {
        "year": 2023,
        "snapshot": "2023-07-16-geolocated",
        "source_kind": "Scientific Data topology/policy archive",
        **stats,
        "directional_policies": graph.number_of_edges(),
        "median_capacity_sat": "",
        "median_base_fee_msat": percentile(fees_base, .50),
        "median_fee_ppm": percentile(fees_ppm, .50),
        "median_cltv_delta": percentile(cltv, .50),
        "capacity_observed": False,
        "capacity_experiment": False,
        "caveat": "funding capacity absent; topology/policy experiment only",
    }


def row_2026(datasets):
    metadata_path = os.path.join(datasets, "rgs_snapshot_20260804_metadata.json")
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    graph_path = os.path.join(HERE, "network", "snapshots", "ln_20260804_rgs.edgelist")
    graph, capacities = load_edgelist(graph_path)
    fees_base = []
    fees_ppm = []
    cltv = []
    directional = 0
    with open(os.path.join(datasets, "rgs_channels_20260804.csv")) as handle:
        for row in csv.DictReader(handle):
            for prefix in ("dir0", "dir1"):
                if row[f"{prefix}_disabled"] == "True":
                    continue
                directional += 1
                fees_base.append(float(row[f"{prefix}_fee_base_msat"]))
                fees_ppm.append(float(row[f"{prefix}_fee_ppm"]))
                cltv.append(float(row[f"{prefix}_cltv_delta"]))
    return {
        "year": 2026,
        "snapshot": metadata["latest_seen_utc"][:10] + "-LDK-RGS-v2",
        "source_kind": "LDK Rapid Gossip Sync",
        **graph_stats(graph),
        "directional_policies": directional,
        "median_capacity_sat": percentile(capacities, .50),
        "median_base_fee_msat": percentile(fees_base, .50),
        "median_fee_ppm": percentile(fees_ppm, .50),
        "median_cltv_delta": percentile(cltv, .50),
        "capacity_observed": True,
        "capacity_experiment": True,
        "caveat": "RGS prunes stale state; current routing view, not archival census",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows = [row_2021(), row_2023(args.datasets), row_2026(args.datasets)]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['year']}: nodes={row['nodes']:,}, pairs={row['pairs']:,}, "
            f"LCC={row['lcc_nodes']:,}/{row['lcc_edges']:,}, "
            f"capacity_experiment={row['capacity_experiment']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
