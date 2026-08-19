"""
run_experiments.py -- drive the Ballast evaluation and write results/*.csv.

Maps one-to-one onto the paper's evaluation (ballast-7.tex):

  Q1  Payment performance at zero coordination latency  -> results/q1_perf.csv
      (Table tab:perf: success ratio / volume / locked capital per scheme)
  Q2  Robustness to coordination latency                -> results/q2_latency.csv
      (Figure fig:latency: effective success ratio vs coordination stress)
  Q3  Capital efficiency vs the service level alpha      -> results/q3_theta.csv
      (multiplexing gain and held-out exceedance diagnostic; the historical
       CSV field name refusal_rate is retained for compatibility)

All schemes share the same topology, balances, and payment sequence per seed
(common.Sim), so Q1's LN/Shaduf/Horcrux numbers reproduce the baselines at
tau=0, and Q2 is purely the Coordination-Cost Model applied on top.

Defaults are a fast SMOKE configuration so the pipeline runs end-to-end in
minutes; pass --paper for the baseline-scale configuration (tx_load=50000,
repeat=10, capacities 1..25), which is heavy.
"""

import argparse
import csv
import os
import time

import common
import schemes
import coordination
import sizing

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _avg(rows):
    return sum(rows) / len(rows) if rows else 0.0


# --------------------------------------------------------------------------
# Q1 : payment performance at zero coordination latency
# --------------------------------------------------------------------------
def run_q1(args):
    path = os.path.join(RESULTS, "q1_perf.csv")
    fields = ["capacity", "scheme", "success_ratio", "success_volume",
              "total_volume", "locked_capital"]
    rows = []
    for cap in args.capacities:
        agg = {}
        for seed in range(args.repeat):
            sim = common.load_sim(cap, seed, tx_load=args.tx_load,
                                  max_trace=args.max_trace)
            locked = sum(sum(v) for v in sim.within.values())
            builders = {
                "LN":      lambda: schemes.work_ln(common.load_sim(cap, seed, args.tx_load, max_trace=args.max_trace), args.tx_load),
                "Revive":  lambda: schemes.work_revive(common.load_sim(cap, seed, args.tx_load, max_trace=args.max_trace), args.tx_load),
                "Shaduf":  lambda: schemes.work_shaduf(common.load_sim(cap, seed, args.tx_load, max_trace=args.max_trace), args.tx_load),
                "Horcrux": lambda: schemes.work_horcrux(common.load_sim(cap, seed, args.tx_load, max_trace=args.max_trace), args.tx_load),
                "Ballast": lambda: schemes.work_ballast(common.load_sim(cap, seed, args.tx_load, max_trace=args.max_trace), args.tx_load, bond_fraction=args.phi),
            }
            runs = {name: fn() for name, fn in builders.items() if name not in args.skip}
            for name, r in runs.items():
                a = agg.setdefault(name, [0, 0, 0, 0])
                a[0] += r.success_ratio
                a[1] += r.success_volume
                a[2] += r.total_volume
                a[3] += locked
        for name, a in agg.items():
            rows.append([cap, name, a[0] / args.repeat, a[1] / args.repeat,
                         a[2] / args.repeat, a[3] / args.repeat])
            print(f"[Q1] cap={cap:>3} {name:8s} success={a[0]/args.repeat:.4f}")
    _write(path, fields, rows)
    return path


# --------------------------------------------------------------------------
# Q2 : robustness to coordination latency (THE experiment baselines omit)
# --------------------------------------------------------------------------
def run_q2(args):
    path = os.path.join(RESULTS, "q2_latency.csv")
    fields = ["L", "scheme", "lambda_r", "p",
              "effective_success_ratio", "base_success_ratio"]
    coords = coordination.default_schemes()
    cap = args.q2_capacity
    lam = args.lambda_r

    # one base run per scheme at L=0; the model then sweeps coin-shift latency L.
    _base_builders = {
        "Revive":  lambda: schemes.work_revive(common.load_sim(cap, 0, args.tx_load, max_trace=args.max_trace), args.tx_load),
        "Shaduf":  lambda: schemes.work_shaduf(common.load_sim(cap, 0, args.tx_load, max_trace=args.max_trace), args.tx_load),
        "Horcrux": lambda: schemes.work_horcrux(common.load_sim(cap, 0, args.tx_load, max_trace=args.max_trace), args.tx_load),
        "Ballast": lambda: schemes.work_ballast(common.load_sim(cap, 0, args.tx_load, max_trace=args.max_trace), args.tx_load, bond_fraction=args.phi),
    }
    base = {name: fn() for name, fn in _base_builders.items() if name not in args.skip}
    rows = []
    for L in args.latencies:
        for name, r in base.items():
            coord = coords[name]
            p = args.p if name != "Ballast" else 1.0  # Ballast is p-independent
            S = coordination.effective_success_ratio(r, coord, lam, L, p=p)
            rows.append([L, name, lam, p, S, r.success_ratio])
        print(f"[Q2] L={L:>7.1f}  "
              + "  ".join(f"{n}={coordination.effective_success_ratio(base[n], coords[n], lam, L, p=(args.p if n!='Ballast' else 1.0)):.4f}"
                          for n in base))
    _write(path, fields, rows)

    # annihilation latency per scheme (Corollary cor:annihilation)
    ann_path = os.path.join(RESULTS, "q2_annihilation.csv")
    ann_fields = ["scheme", "gain_G", "annihilation_latency_L"]
    ln_base = schemes.work_ln(common.load_sim(cap, 0, args.tx_load, max_trace=args.max_trace), args.tx_load)
    ln_S = ln_base.success_ratio
    ann_rows = []
    for name, r in base.items():
        G = r.success_ratio / ln_S if ln_S else float("inf")
        p = args.p if name != "Ballast" else 1.0
        L_star = coordination.annihilation_latency(r, coords[name], G, lam, p=p)
        ann_rows.append([name, G, L_star])
        print(f"[Q2] {name:8s} gain G={G:.3f}  annihilation L={L_star}")
    _write(ann_path, ann_fields, ann_rows)
    return path


# --------------------------------------------------------------------------
# Q3 : capital efficiency vs the service level alpha
# --------------------------------------------------------------------------
def run_q3(args):
    path = os.path.join(RESULTS, "q3_theta.csv")
    fields = ["alpha", "mean_g", "median_g", "refusal_rate",
              "counterparty_loss", "n_active_nodes"]
    cap = args.q2_capacity
    sim = common.load_sim(cap, 0, args.tx_load, max_trace=args.max_trace)
    r = schemes.work_ballast(sim, args.tx_load, bond_fraction=args.phi)
    table = sizing.capital_efficiency_table(r, sim, alphas=args.alphas)
    rows = [[t["alpha"], t["mean_g"], t["median_g"], t["refusal_rate"],
             t["counterparty_loss"], t["n_active_nodes"]] for t in table]
    for t in table:
        print(f"[Q3] alpha={t['alpha']:.2f}  g={t['mean_g']:.2f}  "
              f"heldout_exceedance={t['refusal_rate']:.4f}  loss={t['counterparty_loss']}  "
              f"nodes={t['n_active_nodes']}")
    print(f"[Q3] synchronization index = {sizing.correlation_regime(r):.4f}")
    _write(path, fields, rows)
    return path


def _write(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)
    print(f"  -> wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Ballast evaluation driver")
    ap.add_argument("--which", default="all",
                    help="comma list of q1,q2,q3 (default all)")
    ap.add_argument("--skip", default="",
                    help="comma list of scheme names to skip, e.g. Revive")
    ap.add_argument("--tx_load", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--phi", type=float, default=0.30, help="Ballast bond skim fraction")
    ap.add_argument("--capacities", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--q2_capacity", type=int, default=4)
    ap.add_argument("--lambda_r", type=float, default=0.05,
                    help="background rebalance trigger rate (per round trip)")
    ap.add_argument("--latencies", type=float, nargs="+",
                    default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 600],
                    help="coin-shift latency L sweep (round-trip units; on-chain ~600)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.99, 0.95, 0.90])
    ap.add_argument("--p", type=float, default=0.95, help="counterparty liveness prob")
    ap.add_argument("--max_trace", type=int, default=200000,
                    help="cap rows read from the payment trace (None=all)")
    ap.add_argument("--paper", action="store_true",
                    help="baseline-scale config (heavy): tx_load=50000, repeat=10, cap 1..25")
    args = ap.parse_args()

    if args.paper:
        args.tx_load = 50000
        args.repeat = 10
        args.capacities = list(range(1, 26))
        args.max_trace = None

    args.skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if args.skip:
        print(f"skipping schemes: {sorted(args.skip)}")
    which = [w.strip() for w in args.which.split(",")]
    t0 = time.time()
    if "all" in which or "q1" in which:
        run_q1(args)
    if "all" in which or "q2" in which:
        run_q2(args)
    if "all" in which or "q3" in which:
        run_q3(args)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
