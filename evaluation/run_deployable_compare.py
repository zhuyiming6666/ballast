"""run_deployable_compare.py -- deployable Ballast vs deployable Horcrux.

The Q1 headline compares idealized mechanisms.  The adaptive-sizing experiment
(run_adaptive_q1.py) charges Ballast its deployability cost (online bond
sizing) but leaves Horcrux idealized.  This experiment charges EACH mechanism
its own binding deployability cost, on the same traces and seeds:

  * Ballast   : online adaptive sizing (immediate and delayed top-up), plus
                the coordination model with its constant per-draw latency
                (2 RTT, no third-party liveness: p^0).
  * Horcrux   : the idealized run, degraded by the Coordination-Cost Model of
                the paper's appendix: per-op latency c_H * L (coins move along
                the path) and path-party liveness p^k, applied to the recorded
                per-payment rebalance depths.

Grid: coin-shift latency L in {0, 1, 5, 25} RTT units (0 to ~5 s at 0.2 s RTT)
and liveness p in {1.0, 0.99, 0.95}; background trigger rate lambda_r = 0.05.
L=0, p=1 reproduces the idealized numbers, so the table contains both the
Q1 corner and the deployable region.

Writes results/deployable_compare.csv.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import common
import schemes
import coordination
from run_adaptive_q1 import work_ballast_adaptive

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--tx-load", type=int, default=None)
    ap.add_argument("--max-trace", type=int, default=None)
    ap.add_argument("--phi", type=float, default=0.30)
    ap.add_argument("--lambda-r", type=float, default=0.05)
    args = ap.parse_args()

    if args.paper:
        repeat = args.repeat or 10
        tx_load = args.tx_load or 50000
        max_trace = args.max_trace
    else:
        repeat = args.repeat or 2
        tx_load = args.tx_load or 2000
        max_trace = args.max_trace or 200000

    coords = coordination.default_schemes()
    # Shaduf's default model treats liveness as node-local (p^0), but each
    # shift needs BOTH bound channels' counterparties online and co-signing;
    # the source-channel counterparty is a third party.  Add a strict variant
    # charging p^k, reported alongside the default as a sensitivity row.
    import dataclasses
    coords["Shaduf-strict"] = dataclasses.replace(coords["Shaduf"],
                                                  name="Shaduf-strict",
                                                  p_exp_per_depth=1)
    Ls = [0.0, 1.0, 5.0, 25.0]
    ps = [1.0, 0.99, 0.95]

    # one simulation set per seed, reused across the (L, p) grid
    runs = []
    for seed in range(repeat):
        def sim():
            return common.load_sim(args.cap, seed, tx_load, max_trace=max_trace)
        runs.append({
            "ballast_fixed": schemes.work_ballast(sim(), tx_load,
                                                  bond_fraction=args.phi),
            "ballast_adaptive": work_ballast_adaptive(sim(), tx_load,
                                                      topup_delay=0),
            "ballast_adaptive_delayed": work_ballast_adaptive(sim(), tx_load,
                                                              topup_delay=1000),
            "horcrux": schemes.work_horcrux(sim(), tx_load),
            "shaduf": schemes.work_shaduf(sim(), tx_load),
            "revive": schemes.work_revive(sim(), tx_load),
        })
        print(f"seed {seed} done")

    fields = ["L_rtt", "p", "variant", "effective_success", "ci95"]
    rows = []
    variants = [
        ("ballast_fixed", "Ballast"), ("ballast_adaptive", "Ballast"),
        ("ballast_adaptive_delayed", "Ballast"), ("horcrux", "Horcrux"),
        ("shaduf", "Shaduf"), ("shaduf", "Shaduf-strict"),
        ("revive", "Revive"),
    ]
    for L in Ls:
        for p in ps:
            for key, cname in variants:
                vals = [coordination.effective_success_ratio(
                            r[key], coords[cname], args.lambda_r, L, p)
                        for r in runs]
                m = float(np.mean(vals))
                ci = 1.96 * float(np.std(vals, ddof=1)) / max(1, len(vals)) ** 0.5 \
                    if len(vals) > 1 else 0.0
                label = key if cname not in ("Shaduf-strict",) else "shaduf_strict"
                rows.append({"L_rtt": L, "p": p, "variant": label,
                             "effective_success": round(m, 5),
                             "ci95": round(ci, 5)})
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "deployable_compare.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)
    # console summary at the deployable corner
    for L, p in [(0.0, 1.0), (5.0, 0.99), (25.0, 0.95)]:
        sel = {r["variant"]: r["effective_success"] for r in rows
               if r["L_rtt"] == L and r["p"] == p}
        print(f"L={L} p={p}: {sel}")


if __name__ == "__main__":
    main()
