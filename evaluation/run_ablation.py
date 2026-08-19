"""
run_ablation.py -- the two evidence pieces that attribute Ballast's edge.

E1  Pooling attribution at L=0 (results/e1_pooling.csv)
    Schemes on the SAME graph/balances/payments, all at zero coordination
    latency:
        LN                     -- no rebalancing (floor)
        Shaduf                 -- strongest reproduced coin-mover (node-local)
        Horcrux                -- coin-mover (single-upstream-hop artifact)
        Revive                 -- cycle-based, multi-party LP rebalancing
        Ballast(per-channel)   -- logical draw, reserve stranded per channel
        Ballast(pooled)        -- logical draw, reserve pooled per node
    Ballast(pooled) - Ballast(per-channel) is the PURE statistical-multiplexing
    gain: identical capital, identical mechanics, the only delta is pooling.

E-lat  Direct rebalance latency (results/e_latency.csv)
    Per-op completion time derived from each scheme's RECORDED rebalance depths
    times real network constants (rtt, T_chain) -- latency as a first-class
    measured metric, not a swept knob.

Parallel + incremental: the (cap, seed) cells are independent, so they run on a
process pool (one task per cell).  e1_pooling.csv is rewritten (sorted, valid)
every time a capacity finishes all its seeds -- a kill mid-run keeps every
completed capacity.  Fast by default (smoke-scale); pass --paper for the heavy
config.  Tune worker count with --jobs (default: all cores).
"""

import argparse
import csv
import os
import statistics
import time
import multiprocessing as mp

import common
import schemes
import coordination

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# fixed output order (a kill-resistant CSV is easier to diff if rows are stable)
SCHEME_ORDER = ["LN", "Shaduf", "Horcrux", "Revive", "Ballast-perch", "Ballast"]
LAT_ORDER = ["Ballast", "Horcrux", "Shaduf", "Revive"]


def _write(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def _build(cap, seed, tx_load, max_trace, phi):
    """Run every scheme on a fresh load of the SAME (cap, seed) instance."""
    def sim():
        return common.load_sim(cap, seed, tx_load, max_trace=max_trace)
    return {
        "LN":            schemes.work_ln(sim(), tx_load),
        "Shaduf":        schemes.work_shaduf(sim(), tx_load),
        "Horcrux":       schemes.work_horcrux(sim(), tx_load),
        "Revive":        schemes.work_revive(sim(), tx_load),
        "Ballast-perch": schemes.work_ballast_perchannel(sim(), tx_load, bond_fraction=phi),
        "Ballast":       schemes.work_ballast(sim(), tx_load, bond_fraction=phi),
    }


def _run_one(task):
    """Pool worker: one (cap, seed) cell -> compact per-scheme metrics.

    Returns (cap, seed, metrics, depths).  metrics[name] = (success_ratio,
    refusals, mean_depth).  depths is the full per-scheme rebal_depths list ONLY
    for the cell that feeds the latency table (largest cap, seed 0); else None.
    """
    cap, seed, tx_load, max_trace, phi, want_depths = task
    runs = _build(cap, seed, tx_load, max_trace, phi)
    metrics, depths = {}, {}
    for name, r in runs.items():
        md = (sum(r.rebal_depths) / len(r.rebal_depths)) if r.rebal_depths else 0.0
        metrics[name] = (r.success_ratio, r.refusals, md)
        if want_depths:
            depths[name] = list(r.rebal_depths)
    return cap, seed, metrics, (depths if want_depths else None)


class _Depths:
    """Lightweight stand-in carrying only what latency_table reads."""
    def __init__(self, ks):
        self.rebal_depths = ks


def run(args):
    e1_path = os.path.join(RESULTS, "e1_pooling.csv")
    fields = ["capacity", "scheme", "success_ratio", "success_ci95",
              "refusals", "mean_depth", "repeat", "tx_load", "phi"]
    raw_path = os.path.join(RESULTS, "e1_pooling_raw.csv")
    raw_fields = ["capacity", "seed", "scheme", "success_ratio", "refusals",
                  "mean_depth", "tx_load", "phi"]
    last_cap = args.capacities[-1]

    tasks = []
    for cap in args.capacities:
        for seed in range(args.repeat):
            want = (cap == last_cap and seed == 0)
            tasks.append((cap, seed, args.tx_load, args.max_trace, args.phi, want))

    agg = {cap: {} for cap in args.capacities}
    raw_rows = []
    done_seeds = {cap: 0 for cap in args.capacities}
    lat_depths = {}
    t0 = time.time()

    def flush_csv():
        rows = []
        for cap in args.capacities:
            if done_seeds[cap] < args.repeat:
                continue
            for name in SCHEME_ORDER:
                a = agg[cap].get(name)
                if a is None:
                    continue
                success = a["success"]
                ci95 = (1.96 * statistics.stdev(success) / len(success) ** .5
                        if len(success) > 1 else 0.0)
                rows.append([cap, name, statistics.mean(success), ci95,
                             statistics.mean(a["refusals"]),
                             statistics.mean(a["depth"]), args.repeat,
                             args.tx_load, args.phi])
        _write(e1_path, fields, rows)
        _write(raw_path, raw_fields, sorted(raw_rows,
               key=lambda row: (row[0], row[1], SCHEME_ORDER.index(row[2]))))

    os.makedirs(RESULTS, exist_ok=True)
    print(f"E1: {len(tasks)} cells ({len(args.capacities)} caps x {args.repeat} "
          f"seeds) on {args.jobs} worker(s)")

    if args.jobs == 1:
        results_iter = map(_run_one, tasks)
        pool = None
    else:
        pool = mp.Pool(processes=args.jobs)
        results_iter = pool.imap_unordered(_run_one, tasks)

    for cap, seed, metrics, depths in results_iter:
        d = agg[cap]
        for name, (sr, ref, md) in metrics.items():
            a = d.setdefault(name, {"success": [], "refusals": [], "depth": []})
            a["success"].append(sr)
            a["refusals"].append(ref)
            a["depth"].append(md)
            raw_rows.append([cap, seed, name, sr, ref, md,
                             args.tx_load, args.phi])
        done_seeds[cap] += 1
        if depths:
            lat_depths = depths
        if done_seeds[cap] == args.repeat:
            flush_csv()                      # incremental: this cap is now durable
            m = {n: statistics.mean(agg[cap][n]["success"]) for n in agg[cap]}
            gap = m.get("Ballast", 0) - m.get("Ballast-perch", 0)
            ncaps = sum(1 for c in args.capacities if done_seeds[c] == args.repeat)
            print(f"[E1] cap={cap:>2}  LN={m.get('LN',0):.4f}  "
                  f"Shaduf={m.get('Shaduf',0):.4f}  "
                  f"Horcrux={m.get('Horcrux',0):.4f}  "
                  f"Revive={m.get('Revive',0):.4f}  "
                  f"perch={m.get('Ballast-perch',0):.4f}  "
                  f"pooled={m.get('Ballast',0):.4f}  pool-gain=+{gap:.4f}  "
                  f"[{ncaps}/{len(args.capacities)} caps, {time.time()-t0:.0f}s]")

    if pool is not None:
        pool.close()
        pool.join()
    flush_csv()
    print(f"  -> wrote {e1_path}\n  -> wrote {raw_path}")

    # ---- E-lat : direct latency table (depths recorded at largest cap, seed 0)
    if lat_depths:
        results = {n: _Depths(lat_depths[n]) for n in LAT_ORDER if n in lat_depths}
        table = coordination.latency_table(results)
        lat_path = os.path.join(RESULTS, "e_latency.csv")
        lat_fields = ["scheme", "n_rebal", "median_s", "p99_s", "max_s", "onchain_frac"]
        lrows = [[t["scheme"], t["n_rebal"], t["median_s"], t["p99_s"],
                  t["max_s"], t["onchain_frac"]] for t in table]
        print(f"\n  rebalance latency (rtt=0.2s, T_chain=600s), cap={last_cap}, seed0:")
        for t in table:
            print(f"[E-lat] {t['scheme']:8s} n={t['n_rebal']:>6}  "
                  f"median={t['median_s']:.3f}s  p99={t['p99_s']:.3f}s  "
                  f"max={t['max_s']:.1f}s  on-chain={t['onchain_frac']*100:.1f}%")
        _write(lat_path, lat_fields, lrows)
        print(f"  -> wrote {lat_path}")


def main():
    ap = argparse.ArgumentParser(description="Ballast pooling-attribution + latency")
    ap.add_argument("--tx_load", type=int, default=5000)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--phi", type=float, default=0.30)
    ap.add_argument("--capacities", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--max_trace", type=int, default=200000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="worker processes (default: all cores; lower if RAM-bound)")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    if args.paper:
        args.tx_load = 50000
        args.repeat = 10
        args.capacities = list(range(1, 26))
        args.max_trace = None
    t0 = time.time()
    run(args)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
