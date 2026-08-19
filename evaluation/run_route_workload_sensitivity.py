"""E1 routing/workload sensitivity for the matched-capital comparison."""

import argparse
import csv
import os
import statistics

import common
import schemes

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def run_cell(route, workload, seed, args):
    def sim():
        s = common.load_sim(args.cap, seed, args.tx_load, max_trace=args.max_trace)
        s.route_mode = route
        return s
    runners = {
        "LN": lambda: schemes.work_ln(sim(), args.tx_load, mode=workload),
        "Ballast-PC": lambda: schemes.work_ballast_perchannel(
            sim(), args.tx_load, bond_fraction=args.phi, mode=workload),
        "Ballast": lambda: schemes.work_ballast(
            sim(), args.tx_load, bond_fraction=args.phi, mode=workload),
    }
    return {name: fn().success_ratio for name, fn in runners.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--tx-load", type=int, default=20000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--max-trace", type=int, default=500000)
    ap.add_argument("--phi", type=float, default=.30)
    args = ap.parse_args()
    raw = []
    for route in ("shortest", "k3"):
        for workload in ("uniform", "pareto80", "cyclic"):
            for seed in range(args.repeat):
                for scheme, value in run_cell(route, workload, seed, args).items():
                    raw.append({"route": route, "workload": workload, "seed": seed,
                                "scheme": scheme, "success_ratio": value})
            print(f"done route={route} workload={workload}")
    summary = []
    for route in ("shortest", "k3"):
        for workload in ("uniform", "pareto80", "cyclic"):
            for scheme in ("LN", "Ballast-PC", "Ballast"):
                vals = [r["success_ratio"] for r in raw if r["route"] == route
                        and r["workload"] == workload and r["scheme"] == scheme]
                ci = 1.96 * statistics.stdev(vals) / len(vals) ** .5
                summary.append({"route": route, "workload": workload,
                                "scheme": scheme, "success_ratio": statistics.mean(vals),
                                "success_ci95": ci, "repeat": args.repeat,
                                "tx_load": args.tx_load})
    os.makedirs(RESULTS, exist_ok=True)
    for name, rows in (("e1_route_workload_raw.csv", raw),
                       ("e1_route_workload.csv", summary)):
        with open(os.path.join(RESULTS, name), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
