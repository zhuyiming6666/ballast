"""
run_phi_sweep.py -- sensitivity of the equal-capital skim fraction phi (W4/C3).

The equal-capital discipline skims a fraction phi of every channel's balance
into the node bond and runs the channel at 1-phi.  phi=0.30 is the paper
default; this sweep shows the pooling conclusion is not an artifact of that
choice.  Only Ballast and Ballast-PC depend on phi -- LN / Shaduf / Horcrux
ignore it -- so the sweep runs the two phi-dependent schemes per phi value and
the phi-independent references once.

Writes results/phi_sweep.csv:
    phi,scheme,success_ratio,refusals,mean_depth
(reference rows carry phi=-1).  Parallel + incremental like run_ablation.py:
each finished phi value immediately rewrites the CSV, so a kill keeps every
completed phi.  Fast by default; pass --paper for the heavy config.
"""

import argparse
import csv
import os
import time
import multiprocessing as mp

import common
import schemes

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

REF_SCHEMES = ["LN", "Shaduf", "Horcrux"]          # phi-independent references
PHI_SCHEMES = ["Ballast-perch", "Ballast"]         # phi-dependent


def _write(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def _run_one(task):
    """One (phi, seed) cell.  phi=None runs the reference schemes instead."""
    phi, seed, cap, tx_load, max_trace = task
    def sim():
        return common.load_sim(cap, seed, tx_load, max_trace=max_trace)
    if phi is None:
        runs = {
            "LN":      schemes.work_ln(sim(), tx_load),
            "Shaduf":  schemes.work_shaduf(sim(), tx_load),
            "Horcrux": schemes.work_horcrux(sim(), tx_load),
        }
    else:
        runs = {
            "Ballast-perch": schemes.work_ballast_perchannel(sim(), tx_load, bond_fraction=phi),
            "Ballast":       schemes.work_ballast(sim(), tx_load, bond_fraction=phi),
        }
    metrics = {}
    for name, r in runs.items():
        md = (sum(r.rebal_depths) / len(r.rebal_depths)) if r.rebal_depths else 0.0
        metrics[name] = (r.success_ratio, r.refusals, md)
    return phi, seed, metrics


def run(args):
    path = os.path.join(RESULTS, "phi_sweep.csv")
    fields = ["phi", "scheme", "success_ratio", "refusals", "mean_depth"]

    keys = [None] + list(args.phis)                 # None = reference block
    tasks = [(phi, seed, args.cap, args.tx_load, args.max_trace)
             for phi in keys for seed in range(args.repeat)]

    agg = {k: {} for k in keys}                     # key -> name -> [Ssr,Sref,Smd]
    done = {k: 0 for k in keys}
    t0 = time.time()

    def flush():
        rows = []
        for k in keys:
            if done[k] < args.repeat:
                continue
            order = REF_SCHEMES if k is None else PHI_SCHEMES
            for name in order:
                a = agg[k].get(name)
                if a:
                    rows.append([(-1 if k is None else k), name,
                                 a[0] / args.repeat, a[1] / args.repeat,
                                 a[2] / args.repeat])
        _write(path, fields, rows)

    print(f"phi sweep: {len(tasks)} cells (cap={args.cap}, repeat={args.repeat}, "
          f"phis={list(args.phis)}) on {args.jobs} worker(s)")
    pool = mp.Pool(processes=args.jobs) if args.jobs > 1 else None
    it = pool.imap_unordered(_run_one, tasks) if pool else map(_run_one, tasks)

    for k, seed, metrics in it:
        d = agg[k]
        for name, (sr, ref, md) in metrics.items():
            a = d.setdefault(name, [0.0, 0.0, 0.0])
            a[0] += sr; a[1] += ref; a[2] += md
        done[k] += 1
        if done[k] == args.repeat:
            flush()
            m = {n: agg[k][n][0] / args.repeat for n in agg[k]}
            tag = "ref " if k is None else f"phi={k:.2f}"
            print(f"[phi] {tag}  " +
                  "  ".join(f"{n}={v:.4f}" for n, v in m.items()) +
                  f"  [{time.time()-t0:.0f}s]")

    if pool:
        pool.close(); pool.join()
    flush()
    print(f"  -> wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="phi (skim fraction) sensitivity sweep")
    ap.add_argument("--tx_load", type=int, default=5000)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--phis", type=float, nargs="+",
                    default=[0.10, 0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--max_trace", type=int, default=200000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    if args.paper:
        args.tx_load = 50000
        args.repeat = 10
        args.max_trace = None
    t0 = time.time()
    run(args)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
