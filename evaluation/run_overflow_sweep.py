"""Ballast-Q+ overflow-reserve sweep for the v8 paper."""

import argparse
import csv
import multiprocessing as mp
import os

import common
import schemes

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _one(task):
    seed, cap, tx_load, max_trace, phi, epoch, gammas = task
    rows = []
    for gamma in gammas:
        sim = common.load_sim(cap, seed, tx_load, max_trace=max_trace)
        run = schemes.work_ballast_epoch(
            sim, tx_load, bond_fraction=phi, epoch_size=epoch,
            overflow_fraction=gamma
        )
        rows.append((gamma, run.success_ratio, run.refusals,
                     run.slow_path_events, run.checkpoint_events))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Ballast-Q+ overflow sweep")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--epoch", type=int, default=100)
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

    tasks = [(s, args.cap, args.tx_load, args.max_trace, args.phi,
              args.epoch, tuple(args.gammas)) for s in range(args.repeat)]
    sums = {g: [0.0, 0.0, 0.0, 0.0] for g in args.gammas}
    pool = mp.Pool(args.jobs) if args.jobs > 1 else None
    iterator = pool.imap_unordered(_one, tasks) if pool else map(_one, tasks)
    for done, rows in enumerate(iterator, 1):
        for gamma, success, refusals, slow, checkpoints in rows:
            acc = sums[gamma]
            acc[0] += success; acc[1] += refusals
            acc[2] += slow; acc[3] += checkpoints
        print(f"overflow sweep seed {done}/{args.repeat}")
    if pool:
        pool.close(); pool.join()

    out = os.path.join(RESULTS, "overflow_sweep.csv")
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["overflow_fraction", "success_ratio", "refusals",
                    "slow_path_payments", "slow_path_rate_all_payments",
                    "checkpoints", "epoch_payments", "repeat", "tx_load"])
        for gamma in args.gammas:
            a = sums[gamma]
            slow = a[2] / args.repeat
            w.writerow([gamma, a[0] / args.repeat, a[1] / args.repeat,
                        slow, slow / args.tx_load, a[3] / args.repeat,
                        args.epoch, args.repeat, args.tx_load])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
