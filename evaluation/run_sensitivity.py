"""
run_sensitivity.py -- the demoted, honest Q2 figure (paper fig:latency).

The original Q2 swept coin-shift latency L through the Coordination-Cost Model
with one hand-chosen set of constants and reported the resulting curve as a
headline.  We demote it: the physical x-axis (coin-shift latency L) is kept, but
instead of one curve per scheme we report a *band* obtained by sweeping the
model's free constants over plausible ranges:

    T_chain in [300, 900] s        (on-chain confirmation, ~5-15 min)
    lambda_r in [0.02, 0.10] / RTT (background rebalance trigger rate)
    p        in [0.90, 0.99]       (counterparty liveness)

For each scheme and each L we report the central curve (default constants) and
the min/max envelope across the constant grid.  Ballast is flat (it does not
move coins, so its tau is constant in L); the coin-movers decay from their L=0
value toward LN.  The takeaway is directional and robust to the constants: the
baselines' zero-latency numbers are an upper bound that coordination cost erodes,
while Ballast's do not move.

Writes results/q2_sensitivity.csv.
"""

import argparse
import csv
import itertools
import os

import common
import schemes
import coordination

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def base_runs(cap, tx_load, phi, max_trace):
    sims = lambda: common.load_sim(cap, 0, tx_load, max_trace=max_trace)
    return {
        "LN":      schemes.work_ln(sims(), tx_load),
        "Shaduf":  schemes.work_shaduf(sims(), tx_load),
        "Horcrux": schemes.work_horcrux(sims(), tx_load),
        "Revive":  schemes.work_revive(sims(), tx_load),
        "Ballast": schemes.work_ballast(sims(), tx_load, bond_fraction=phi),
    }


def main():
    ap = argparse.ArgumentParser(description="Q2 coordination-cost sensitivity band")
    ap.add_argument("--tx_load", type=int, default=5000)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--phi", type=float, default=0.30)
    ap.add_argument("--max_trace", type=int, default=200000)
    ap.add_argument("--latencies", type=float, nargs="+",
                    default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 600])
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    if args.paper:
        args.tx_load = 50000
        args.max_trace = None

    runs = base_runs(args.cap, args.tx_load, args.phi, args.max_trace)

    # constant grid for the band
    T_chain_grid = [300.0, 600.0, 900.0]
    lambda_grid = [0.02, 0.05, 0.10]
    p_grid = [0.90, 0.95, 0.99]
    # central = the middle of each grid
    central = (600.0, 0.05, 0.95)

    rows = []
    for L in args.latencies:
        for name, r in runs.items():
            if name == "LN":
                # LN never rebalances; its success ratio is flat in L.
                s_central = r.success_ratio
                rows.append([L, name, s_central, s_central, s_central, r.success_ratio])
                continue
            vals = []
            s_central = None
            for T_chain, lam, p in itertools.product(T_chain_grid, lambda_grid, p_grid):
                coords = coordination.default_schemes(T_chain=T_chain)
                coord = coords[name]
                pp = p if name != "Ballast" else 1.0
                S = coordination.effective_success_ratio(r, coord, lam, L, p=pp)
                vals.append(S)
                if (T_chain, lam, p) == central:
                    s_central = S
            rows.append([L, name, s_central, min(vals), max(vals), r.success_ratio])
        line = "  ".join(
            f"{n}={[row[2] for row in rows if row[0]==L and row[1]==n][0]:.3f}"
            for n in runs)
        print(f"[Q2-sens] L={L:>6.1f}  {line}")

    path = os.path.join(RESULTS, "q2_sensitivity.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "scheme", "S_central", "S_low", "S_high", "base_success"])
        w.writerows(rows)
    print(f"  -> wrote {path}")


if __name__ == "__main__":
    main()
