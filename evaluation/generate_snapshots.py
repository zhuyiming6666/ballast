"""Generate temporal Lightning snapshots from the raw channel history.

The source data bundled with the related-work artifact contains channel open
and close block heights.  For each cutoff, this script keeps channels open at
that block, merges parallel channels, selects the largest connected component,
relabels public keys to stable integers, and writes a compact edge list.

No network download is required.  Defaults produce snapshots at blocks
600000, 640000, and 677167 (the paper baseline cutoff).
"""

import argparse
import json
import os

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE_DIR = os.path.abspath(os.path.join(
    HERE, "..", "..", "一种支付通道网络再平衡机制的设计与实现",
    "code", "Shaduf-main", "evaluation", "network"))
DEFAULT_INPUTS = [
    os.path.join(DEFAULT_SOURCE_DIR, "channel_1_600000.json", "channel_1_600000.json"),
    os.path.join(DEFAULT_SOURCE_DIR, "channel_600001_677167.json", "channel_600001_677167.json"),
]
DEFAULT_OUT = os.path.join(HERE, "network", "snapshots")


def load_channels(paths):
    channels = []
    for path in paths:
        with open(path) as f:
            channels.extend(json.load(f))
    return channels


def graph_at(channels, cutoff):
    graph = nx.Graph()
    for ch in channels:
        opened = ch.get("open", {}).get("block")
        closed = ch.get("close", {}).get("block")
        cap = int(ch.get("satoshis") or 0)
        nodes = ch.get("nodes") or []
        if opened is None or opened > cutoff or cap <= 0 or len(nodes) != 2:
            continue
        if closed is not None and closed <= cutoff:
            continue
        a, b = nodes
        if a == b:
            continue
        if graph.has_edge(a, b):
            graph[a][b]["capacity"] += cap
        else:
            graph.add_edge(a, b, capacity=cap)
    if graph.number_of_nodes() == 0:
        raise ValueError(f"empty snapshot at block {cutoff}")
    component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def write_snapshot(graph, cutoff, out_dir):
    nodes = {node: i for i, node in enumerate(sorted(graph.nodes()))}
    relabeled = nx.relabel_nodes(graph, nodes)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ln_{cutoff}.edgelist")
    nx.write_edgelist(relabeled, path, data=["capacity"])
    print(f"block={cutoff}: nodes={relabeled.number_of_nodes()} "
          f"edges={relabeled.number_of_edges()} -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="generate temporal LN snapshots")
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--cutoffs", type=int, nargs="+",
                    default=[600000, 640000, 677167])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    missing = [path for path in args.inputs if not os.path.isfile(path)]
    if missing:
        expected = [
            os.path.join(args.out, f"ln_{cutoff}.edgelist")
            for cutoff in args.cutoffs
        ]
        if all(os.path.isfile(path) for path in expected):
            print("raw channel-history JSON is not bundled; using the checked-in "
                  "derived snapshots:")
            for path in expected:
                print(f"  {path}")
            return
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            "missing raw channel-history inputs and no complete derived snapshot "
            f"set is available:\n  {formatted}\n"
            "Pass --inputs with the Shaduf channel-history JSON files."
        )
    channels = load_channels(args.inputs)
    for cutoff in args.cutoffs:
        write_snapshot(graph_at(channels, cutoff), cutoff, args.out)


if __name__ == "__main__":
    main()
