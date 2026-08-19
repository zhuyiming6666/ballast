"""Evaluate the v8 Ballast-Q epoch-checkpoint trade-off.

Writes ../results/epoch_sweep.csv with a fully dynamic pooled Ballast upper
bound, the fixed per-channel control, and Ballast-Q at several epoch lengths.
All variants use identical capital, topology, payment trace, and seeds.
"""

import argparse
import csv
import multiprocessing as mp
import os
import time

import common
import schemes

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _run_one(task):
    seed, cap, tx_load, max_trace, phi, epochs = task

    def sim():
        return common.load_sim(cap, seed, tx_load, max_trace=max_trace)

    rows = []
    pooled = schemes.work_ballast(sim(), tx_load, bond_fraction=phi)
    perch = schemes.work_ballast_perchannel(sim(), tx_load, bond_fraction=phi)
    rows.append(("Ballast-dynamic", 0, pooled.success_ratio, pooled.refusals, 0))
    rows.append(("Ballast-PC", -1, perch.success_ratio, perch.refusals, 0))
    for epoch in epochs:
        run = schemes.work_ballast_epoch(
            sim(), tx_load, bond_fraction=phi, epoch_size=epoch
        )
        rows.append(("Ballast-Q", epoch, run.success_ratio, run.refusals,
                     run.checkpoint_events))
    return seed, rows


def run(args):
    tasks = [
        (seed, args.cap, args.tx_load, args.max_trace, args.phi, tuple(args.epochs))
        for seed in range(args.repeat)
    ]
    sums = {}
    t0 = time.time()
    pool = mp.Pool(args.jobs) if args.jobs > 1 else None
    iterator = pool.imap_unordered(_run_one, tasks) if pool else map(_run_one, tasks)
    for done, (_, rows) in enumerate(iterator, 1):
        for scheme, epoch, success, refusals, checkpoints in rows:
            acc = sums.setdefault((scheme, epoch), [0.0, 0.0, 0.0])
            acc[0] += success
            acc[1] += refusals
            acc[2] += checkpoints
        print(f"epoch sweep seed {done}/{args.repeat} [{time.time()-t0:.1f}s]")
    if pool:
        pool.close()
        pool.join()

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "epoch_sweep.csv")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scheme", "epoch_payments", "success_ratio", "refusals",
            "checkpoints", "repeat", "tx_load", "phi", "capacity"
        ])
        order = [("Ballast-PC", -1), ("Ballast-dynamic", 0)] + [
            ("Ballast-Q", e) for e in args.epochs
        ]
        for key in order:
            acc = sums[key]
            writer.writerow([
                key[0], key[1], acc[0] / args.repeat, acc[1] / args.repeat,
                acc[2] / args.repeat, args.repeat, args.tx_load, args.phi, args.cap
            ])
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Ballast-Q epoch-length sweep")
    ap.add_argument("--epochs", type=int, nargs="+", default=[100, 500, 1000, 5000])
    ap.add_argument("--tx_load", type=int, default=5000)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--phi", type=float, default=0.30)
    ap.add_argument("--max_trace", type=int, default=200000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    if args.paper:
        args.tx_load = 50000
        args.repeat = 10
        args.max_trace = None
    run(args)


if __name__ == "__main__":
    main()
