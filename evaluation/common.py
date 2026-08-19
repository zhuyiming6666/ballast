"""
common.py -- shared simulation infrastructure for the Ballast evaluation.

This factors out the code that shaduf.py and horcrux.py duplicate verbatim
(topology loading, the on-chain transaction-value proxy, the per-channel balance
store, and source/destination sampling) so that every scheme -- LN, Shaduf,
Horcrux, Revive, and Ballast -- runs on *exactly* the same graph, the same
balances, and the same payment sequence for a given seed.  Only the per-payment
routing/rebalancing logic differs between schemes; that lives in schemes.py.

Topology  : ./network/lightning_simplified_component.edgelist   (real LN snapshot,
            same file the baselines use; symlinked from the Horcrux repo)
Workload  : ./payment_value/payment_value_satoshi_03.csv         (on-chain
            tx-value proxy, March 2021; same file the baselines use)

Both paths and the threshold/value conventions are identical to
github-shaduf/evaluation/shaduf.py so that our LN/Shaduf/Horcrux numbers
reproduce the baselines' numbers at zero coordination latency.
"""

import os
import random
import collections

import networkx as nx
import numpy as np

# --- paths and workload conventions (identical to the baselines) -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
NETWORK_FILE = os.path.join(_HERE, "network", "lightning_simplified_component.edgelist")
PAYMENT_VALUE_FILE = os.path.join(_HERE, "payment_value", "payment_value_satoshi_03.csv")

PAYMENT_VALUE_THRESHOLD = 466359   # satoshi; baselines keep 0 < v <= threshold
DEFAULT_TX_LOAD = 50000            # payments per run (baseline default)

# memoized parse of the payment-value trace, keyed by (threshold, max_trace);
# the file parse is identical across calls, only the per-seed shuffle differs.
_TRACE_CACHE = {}


def tuple_sort(a):
    return tuple(sorted(a))


def tuple_trans(a, b):
    """(common, min(other), max(other)) for two channels sharing one endpoint."""
    a = set(a)
    b = set(b)
    common = list(a & b)[0]
    other = list(a ^ b)
    return (common, min(other), max(other))


class Sim:
    """Holds one fully-initialized simulation instance.

    Replaces the module-level globals (G, within_data, tx_8, ...) the baselines
    use, so multiple schemes/seeds can run in one process without cross-talk.
    The channel-balance API mirrors the baselines' get_within / update_within /
    get_total_amount exactly.
    """

    def __init__(self):
        self.G = None
        self.within = {}          # sorted-tuple channel -> (bal_low, bal_high)
        self.tx = []              # shuffled payment-value trace
        self.nodes = []
        self.route_mode = "shortest"
        self._route_cache = {}
        self._pareto_hot = None

    # -- channel balance store (mirrors baselines) --------------------------
    def get_within(self, a, b):
        ch = tuple_sort((a, b))
        return self.within[ch] if ch[0] == a else self.within[ch][::-1]

    def update_within(self, a, b, bal_a, bal_b):
        ch = tuple_sort((a, b))
        self.within[ch] = (bal_a, bal_b) if ch[0] == a else (bal_b, bal_a)
        if bal_a < 0 or bal_b < 0:
            raise ValueError("negative channel balance in update_within")

    def get_total_amount(self, a):
        return sum(self.get_within(a, nb)[0] for nb in self.G[a])

    # -- payment endpoint sampling (mirrors baselines) ----------------------
    def sample_pair(self, mode, skew_param):
        nodes = self.nodes
        n = len(nodes)
        if mode == "uniform":
            while True:
                t1 = random.choice(nodes)
                t2 = random.choice(nodes)
                if t1 != t2:
                    return t1, t2
        elif mode == "pareto80":
            # Degree-ranked 80/20 sender and receiver hotspots.  The sender
            # and receiver draws are independent, so this stresses both ends.
            if self._pareto_hot is None:
                self._pareto_hot = sorted(
                    nodes, key=self.G.degree, reverse=True)[:max(1, n // 5)]
            hot = self._pareto_hot
            while True:
                t1 = random.choice(hot if random.random() < .8 else nodes)
                t2 = random.choice(hot if random.random() < .8 else nodes)
                if t1 != t2:
                    return t1, t2
        elif mode == "cyclic":
            i = random.randrange(n)
            return nodes[i], nodes[(i + 1) % n]
        else:  # skew: heavy senders, exponential index into node list
            while True:
                t1 = n
                while t1 >= n:
                    t1 = int(np.random.exponential(n / skew_param))
                t2 = random.choice(nodes)
                if t1 != t2:
                    return nodes[t1], t2

    def route(self, source, target):
        """Route using shortest path or a deterministic choice among k=3 paths."""
        if self.route_mode == "shortest":
            return nx.shortest_path(self.G, source, target)
        if self.route_mode != "k3":
            raise ValueError(f"unknown route mode: {self.route_mode}")
        key = (source, target)
        paths = self._route_cache.get(key)
        if paths is None:
            paths = []
            for path in nx.shortest_simple_paths(self.G, source, target):
                paths.append(path)
                if len(paths) == 3:
                    break
            # Avoid unbounded growth on long runs while retaining hot pairs.
            if len(self._route_cache) < 20000:
                self._route_cache[key] = paths
        return random.choice(paths)


def load_sim(channel_rate, seed, tx_load=DEFAULT_TX_LOAD,
             threshold=PAYMENT_VALUE_THRESHOLD, max_trace=None,
             network_file=None):
    """Build a Sim: load the LN topology with each channel split 50/50 at
    `channel_rate`x its listed capacity, and load+shuffle the payment trace.

    This is the baselines' initialize() with the globals replaced by a Sim
    object and an optional `max_trace` cap so smoke runs don't read all 9.3M
    rows of the trace (paper-scale runs leave it None).
    """
    sim = Sim()
    sim.G = nx.Graph()
    random.seed(seed)
    np.random.seed(seed)

    topology = network_file or NETWORK_FILE
    with open(topology) as f:
        for line in f:
            a, b, cap = line.split()
            a, b = int(a), int(b)
            cap = int(int(cap) * channel_rate)
            sim.G.add_edge(a, b)
            cap += cap % 2
            half = cap // 2
            if a < b:
                sim.within[(a, b)] = (half, half)
            else:
                sim.within[(b, a)] = (half, half)

    # we only need tx_load samples; cap the read so the 9.3M-row trace does
    # not have to be fully materialized for small runs.  The *parse* of the
    # 9.3M-row trace is identical across every call with the same
    # (threshold, max_trace), so we memoize it once; only the per-seed shuffle
    # below differs.  This is a pure speedup -- the resulting tx list is
    # bit-identical to re-reading the file every call.
    cap_read = max_trace if max_trace is not None else None
    cache_key = (threshold, cap_read)
    cached = _TRACE_CACHE.get(cache_key)
    if cached is None:
        cached = []
        with open(PAYMENT_VALUE_FILE) as f:
            for line in f:
                v = int(float(line))
                if 0 < v <= threshold:
                    cached.append(v)
                    if cap_read is not None and len(cached) >= cap_read:
                        break
        _TRACE_CACHE[cache_key] = cached
    sim.tx = list(cached)
    random.shuffle(sim.tx)
    if len(sim.tx) < tx_load:
        # tile the trace if a small cap was used; keeps runs deterministic
        reps = (tx_load // max(1, len(sim.tx))) + 1
        sim.tx = (sim.tx * reps)[:tx_load]

    sim.nodes = list(sim.G.nodes())
    return sim
