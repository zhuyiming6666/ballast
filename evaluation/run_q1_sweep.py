"""run_q1_sweep.py -- Q1 capacity sweep with per-seed rows (for CI bands).

Same experiment as run_experiments.py Q1, but writes one row per
(capacity, scheme, seed) so the figure can carry 95% confidence bands.
Writes results/q1_sweep_seeds.csv.
"""

import argparse
import csv
import os

import common
import schemes

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--capacities", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 25])
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--tx-load", type=int, default=None)
    ap.add_argument("--max-trace", type=int, default=None)
    ap.add_argument("--phi", type=float, default=0.30)
    args = ap.parse_args()

    repeat = args.repeat or (10 if args.paper else 2)
    tx_load = args.tx_load or (50000 if args.paper else 2000)
    max_trace = args.max_trace or (None if args.paper else 200000)

    builders = {
        "LN": lambda s: schemes.work_ln(s, tx_load),
        "Revive": lambda s: schemes.work_revive(s, tx_load),
        "Shaduf": lambda s: schemes.work_shaduf(s, tx_load),
        "Horcrux": lambda s: schemes.work_horcrux(s, tx_load),
        "Ballast": lambda s: schemes.work_ballast(s, tx_load, bond_fraction=args.phi),
        "Ballast-PC": lambda s: schemes.work_ballast_perchannel(
            s, tx_load, bond_fraction=args.phi),
    }

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "q1_sweep_seeds.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["capacity", "scheme", "seed",
                                          "success_ratio", "tx_load"])
        w.writeheader()
        for cap in args.capacities:
            for seed in range(repeat):
                for name, build in builders.items():
                    sim = common.load_sim(cap, seed, tx_load,
                                          max_trace=max_trace)
                    r = build(sim)
                    w.writerow({"capacity": cap, "scheme": name, "seed": seed,
                                "success_ratio": r.n_success / max(1, r.n_total),
                                "tx_load": tx_load})
                    f.flush()
            print(f"capacity {cap} done")
    print("wrote", path)


if __name__ == "__main__":
    main()
