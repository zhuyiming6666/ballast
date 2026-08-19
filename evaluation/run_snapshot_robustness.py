"""Cross-snapshot robustness experiment for the Ballast pooling conclusion.

Runs the matched-capital ablation on each temporal topology produced by
generate_snapshots.py.  The payment trace, seeds, and capacity scale are held
fixed.  Writes results/snapshot_robustness.csv.
"""

import argparse
import csv
import glob
import multiprocessing as mp
import os
import statistics

import common
import schemes

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DEFAULT_GLOB = os.path.join(HERE, "network", "snapshots", "ln_*.edgelist")


def run_once(path, cap, seed, tx_load, max_trace, phi):
    def sim():
        return common.load_sim(cap, seed, tx_load, max_trace=max_trace,
                               network_file=path)
    return {
        "LN": schemes.work_ln(sim(), tx_load),
        "Shaduf": schemes.work_shaduf(sim(), tx_load),
        "Horcrux": schemes.work_horcrux(sim(), tx_load),
        "Ballast-perch": schemes.work_ballast_perchannel(
            sim(), tx_load, bond_fraction=phi),
        "Ballast": schemes.work_ballast(sim(), tx_load, bond_fraction=phi),
    }


def run_task(task):
    path, cap, seed, tx_load, max_trace, phi = task
    runs = run_once(path, cap, seed, tx_load, max_trace, phi)
    return path, seed, {name: result.success_ratio for name, result in runs.items()}


def main():
    ap = argparse.ArgumentParser(description="cross-snapshot robustness")
    ap.add_argument("--snapshots", nargs="+", default=sorted(glob.glob(DEFAULT_GLOB)))
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--tx_load", type=int, default=10000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max_trace", type=int, default=500000)
    ap.add_argument("--phi", type=float, default=0.30)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    if args.paper:
        args.tx_load = 50000
        args.repeat = 10
        args.max_trace = None
    if not args.snapshots:
        raise SystemExit("no snapshots; run generate_snapshots.py first")

    rows = []
    aggregates = {path: {} for path in args.snapshots}
    raw_rows = []
    tasks = [(path, args.cap, seed, args.tx_load, args.max_trace, args.phi)
             for path in args.snapshots for seed in range(args.repeat)]
    pool = mp.Pool(args.jobs) if args.jobs > 1 else None
    iterator = pool.imap_unordered(run_task, tasks) if pool else map(run_task, tasks)
    for path, seed, metrics in iterator:
        for name, value in metrics.items():
            aggregates[path].setdefault(name, []).append(value)
            raw_rows.append([os.path.splitext(os.path.basename(path))[0],
                             args.cap, args.phi, seed, name, value,
                             args.tx_load])
    if pool:
        pool.close()
        pool.join()

    for path in args.snapshots:
        aggregate = aggregates[path]
        snapshot = os.path.splitext(os.path.basename(path))[0]
        for name, values in aggregate.items():
            ci95 = (1.96 * statistics.stdev(values) / len(values) ** .5
                    if len(values) > 1 else 0.0)
            rows.append([snapshot, args.cap, args.phi, name,
                         statistics.mean(values), ci95,
                         args.repeat, args.tx_load])
        pooled = statistics.mean(aggregate["Ballast"])
        perch = statistics.mean(aggregate["Ballast-perch"])
        print(f"[{snapshot}] pooled={pooled:.4f} perch={perch:.4f} "
              f"gain={pooled-perch:+.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "snapshot_robustness.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snapshot", "capacity", "phi", "scheme", "success_ratio",
                    "success_ci95", "repeat", "tx_load"])
        w.writerows(rows)
    raw_out = os.path.join(RESULTS, "snapshot_robustness_raw.csv")
    with open(raw_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snapshot", "capacity", "phi", "seed", "scheme",
                    "success_ratio", "tx_load"])
        w.writerows(sorted(raw_rows, key=lambda row: (row[0], row[3], row[4])))
    print(f"wrote {out}\nwrote {raw_out}")


if __name__ == "__main__":
    main()
