"""run_adaptive_q1.py -- the deployable-sizing companion to Q1.

Reviewer consensus D: the headline Q1 number (fixed phi=0.30 skim) is an
equal-capital mechanism comparison, not the performance of an online sizing
controller.  This experiment drives the SAME payment traces with an online
adaptive bond controller and reports the deployable success rate next to the
fixed-skim upper bound.

Controller (safety-first, mirrors the Q3 sizing loop but now in-line with the
payment simulation):
  * warm-up: every node starts with a small skim phi_init of each channel;
  * every `update_period` payments, each active node re-estimates its target
    bond as the alpha-quantile of its outstanding-draw samples in a sliding
    window;
  * top-ups (target > bond) move funds from the node's channel balances into
    the bond, pro-rata; they activate after `topup_delay` payments (0 models
    the immediate upper bound; >0 models chain/coordination latency);
  * withdrawals (target < bond) are capped at slack above current outstanding
    exposure and are delayed by `withdraw_delay` payments (the Delta discipline
    of the paper).

Capital is conserved per node at every step: bond + channel balances is
invariant, so this remains an equal-locked-capital comparison.

Writes results/adaptive_q1.csv with one row per (capacity, variant, seed-agg):
  variant in {fixed_phi, adaptive_immediate, adaptive_delayed_<D>}.

Smoke by default; --paper uses tx_load=50000, repeat=10, capacity scale 4.
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


def work_ballast_adaptive(sim, tx_load, alpha=0.99, phi_init=0.05,
                          window=64, update_period=200,
                          topup_delay=0, withdraw_delay=1000,
                          phi_cap=0.90):
    """Ballast with an online per-node bond controller (see module docstring)."""
    bond = {}
    channel_total = {}          # node -> current total operator-side channel balance
    outstanding = collections.defaultdict(int)
    total_outstanding = collections.defaultdict(int)
    demand_window = collections.defaultdict(lambda: collections.deque(maxlen=window))
    pending_topups = []         # (activation_index, node, amount)
    pending_withdrawals = []    # (activation_index, node, amount)
    stats = {"topups": 0, "withdrawals": 0, "topup_volume": 0}

    # initial small skim
    for ch in list(sim.within.keys()):
        a, b = ch
        ba, bb = sim.within[ch]
        sa, sb = int(ba * phi_init), int(bb * phi_init)
        sim.within[ch] = (ba - sa, bb - sb)
        bond[a] = bond.get(a, 0) + sa
        bond[b] = bond.get(b, 0) + sb
    for n in sim.nodes:
        bond.setdefault(n, 0)

    # per-node channel index for pro-rata moves
    node_channels = collections.defaultdict(list)   # node -> [(key, side)]
    for (a, b) in sim.within.keys():
        node_channels[a].append(((a, b), 0))
        node_channels[b].append(((a, b), 1))

    def node_channel_balance(u):
        return sum(sim.within[k][s] for k, s in node_channels[u])

    def move_to_bond(u, amount):
        """Move up to `amount` from u's channel balances into u's bond, pro-rata."""
        total = node_channel_balance(u)
        if total <= 0 or amount <= 0:
            return 0
        amount = min(amount, total)
        moved = 0
        for k, s in node_channels[u]:
            ba, bb = sim.within[k]
            bal = ba if s == 0 else bb
            take = min(bal, int(round(amount * (bal / total))) if total else 0)
            if take <= 0:
                continue
            if s == 0:
                sim.within[k] = (ba - take, bb)
            else:
                sim.within[k] = (ba, bb - take)
            moved += take
        bond[u] += moved
        return moved

    def move_to_channels(u, amount):
        """Return up to `amount` from u's bond back into channels, pro-rata by capacity headroom."""
        amount = min(amount, bond[u] - total_outstanding[u])
        if amount <= 0:
            return 0
        chans = node_channels[u]
        if not chans:
            return 0
        share = amount // len(chans)
        moved = 0
        for k, s in chans:
            ba, bb = sim.within[k]
            if s == 0:
                sim.within[k] = (ba + share, bb)
            else:
                sim.within[k] = (ba, bb + share)
            moved += share
        bond[u] -= moved
        return moved

    def headroom(u):
        return bond[u] - total_outstanding[u]

    res = schemes.RunResult()

    for i in range(tx_load):
        # activate matured top-ups / withdrawals
        while pending_topups and pending_topups[0][0] <= i:
            _, u, amt = pending_topups.pop(0)
            moved = move_to_bond(u, amt)
            if moved:
                stats["topups"] += 1
                stats["topup_volume"] += moved
        while pending_withdrawals and pending_withdrawals[0][0] <= i:
            _, u, amt = pending_withdrawals.pop(0)
            if move_to_channels(u, amt):
                stats["withdrawals"] += 1

        t1, t2 = sim.sample_pair("uniform", None)
        path = schemes._route(sim, t1, t2)
        amt = sim.tx[i]

        flag, refused = True, False
        planned = collections.defaultdict(int)
        draws = []
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            z0 = sim.get_within(u, v)[0]
            if z0 >= amt:
                continue
            zmax = amt - z0
            if headroom(u) - planned[u] >= zmax:
                planned[u] += zmax
                draws.append((u, v, zmax))
            else:
                flag = False
                refused = True
                # record unmet demand so the controller can react
                demand_window[u].append(total_outstanding[u] + planned[u] + zmax)
                break

        if flag:
            for (u, v, zmax) in draws:
                outstanding[(u, v)] += zmax
                total_outstanding[u] += zmax
                demand_window[u].append(total_outstanding[u])
                res.draw_events.append((u, zmax, total_outstanding[u]))
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
                res.rebal_depths.append(len(draws))
        else:
            if refused:
                res.refusals += 1
        res.n_total += 1
        res.total_volume += amt

        # periodic re-estimation
        if i and i % update_period == 0:
            for u, dw in demand_window.items():
                if len(dw) < 8:
                    continue
                target = int(np.quantile(np.asarray(dw, dtype=float), alpha))
                cap_limit = int(phi_cap * (bond[u] + node_channel_balance(u)))
                target = min(target, cap_limit)
                if target > bond[u]:
                    pending_topups.append((i + topup_delay, u, target - bond[u]))
                elif target < bond[u]:
                    slack = bond[u] - max(target, total_outstanding[u])
                    if slack > 0:
                        pending_withdrawals.append((i + withdraw_delay, u, slack))

    res.node_bond = dict(bond)
    res.controller_stats = stats
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--capacities", type=int, nargs="+", default=None)
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--tx-load", type=int, default=None)
    ap.add_argument("--max-trace", type=int, default=None)
    ap.add_argument("--phi", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--delays", type=int, nargs="+", default=[0, 1000],
                    help="top-up activation delays in payments; 0 = immediate")
    args = ap.parse_args()

    if args.paper:
        caps = args.capacities or [4]
        repeat = args.repeat or 10
        tx_load = args.tx_load or 50000
        max_trace = args.max_trace
    else:
        caps = args.capacities or [4]
        repeat = args.repeat or 2
        tx_load = args.tx_load or 2000
        max_trace = args.max_trace or 200000

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "adaptive_q1.csv")
    fields = ["capacity", "variant", "success_ratio", "success_ratio_ci95",
              "refusals", "topups", "topup_volume", "seeds", "tx_load"]
    rows = []

    for cap in caps:
        variants = {"fixed_phi": None}
        for d in args.delays:
            name = "adaptive_immediate" if d == 0 else f"adaptive_delayed_{d}"
            variants[name] = d
        for name, delay in variants.items():
            ratios, refus, tps, tpv = [], [], [], []
            for seed in range(repeat):
                sim = common.load_sim(cap, seed, tx_load, max_trace=max_trace)
                if name == "fixed_phi":
                    r = schemes.work_ballast(sim, tx_load, bond_fraction=args.phi)
                    st = {"topups": 0, "topup_volume": 0}
                else:
                    r = work_ballast_adaptive(sim, tx_load, alpha=args.alpha,
                                              topup_delay=delay)
                    st = r.controller_stats
                ratios.append(r.n_success / max(1, r.n_total))
                refus.append(r.refusals)
                tps.append(st["topups"])
                tpv.append(st["topup_volume"])
            m = float(np.mean(ratios))
            ci = 1.96 * float(np.std(ratios, ddof=1)) / max(1, len(ratios)) ** 0.5 \
                if len(ratios) > 1 else 0.0
            rows.append({"capacity": cap, "variant": name,
                         "success_ratio": round(m, 6),
                         "success_ratio_ci95": round(ci, 6),
                         "refusals": float(np.mean(refus)),
                         "topups": float(np.mean(tps)),
                         "topup_volume": float(np.mean(tpv)),
                         "seeds": repeat, "tx_load": tx_load})
            print(rows[-1])

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
