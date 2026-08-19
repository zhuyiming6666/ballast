"""E2 sizing closure on real simulator draw diagnostics.

Reports the exact cohort/estimator decomposition requested by the paper:
all active operators, operators with >=50 draws, and high-degree hubs; each is
evaluated with a static prefix quantile and a safety-first adaptive loop.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

import common
import schemes
import sizing


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def evaluate_sequence(seq, alpha, estimator, train_frac=.5,
                      window=64, update_period=16):
    cut = max(1, int(len(seq) * train_frac))
    train = list(seq[:cut])
    test = list(seq[cut:])
    if not test:
        return None
    bond = float(np.quantile(train, alpha))
    exceed = 0
    refused_after_control = 0
    history = train[-window:]
    topups = 0
    for pos, value in enumerate(test, 1):
        if value > bond:
            exceed += 1
            if estimator == "adaptive":
                # Safety-first controller: an observed surge tops up now.
                bond = float(value)
                topups += 1
            else:
                refused_after_control += 1
        history.append(value)
        history = history[-window:]
        if estimator == "adaptive" and pos % update_period == 0:
            target = float(np.quantile(history, alpha))
            if target > bond:
                bond = target
                topups += 1
            # A lower target is a delayed withdrawal and is intentionally not
            # applied inside this holdout horizon.
    return exceed, refused_after_control, len(test), topups, bond


def cohort_nodes(result, sim):
    demand = sizing.per_node_demand_ordered(result)
    active = {n for n, seq in demand.items() if len(seq) >= 2}
    ge50 = {n for n, seq in demand.items() if len(seq) >= 50}
    ranked = sorted(active, key=lambda n: sim.G.degree[n], reverse=True)
    hub_count = max(1, int(np.ceil(.10 * len(ranked)))) if ranked else 0
    hubs = set(ranked[:hub_count])
    return demand, {"all": active, "ge50": ge50, "top10pct-degree": hubs}


def run_seed(seed, args):
    sim = common.load_sim(args.cap, seed, args.tx_load, max_trace=args.max_trace)
    result = schemes.work_ballast(sim, args.tx_load, bond_fraction=args.phi)
    demand, cohorts = cohort_nodes(result, sim)
    peaks = sizing.per_node_channel_peaks(result, sim)
    rows = []
    for cohort, nodes in cohorts.items():
        for estimator in ("static", "adaptive"):
            total_exceed = total_refused = total_test = total_topups = 0
            gains = []
            used = 0
            for node in nodes:
                outcome = evaluate_sequence(demand[node], args.alpha, estimator,
                                            window=args.window,
                                            update_period=args.update_period)
                if outcome is None:
                    continue
                exceed, refused, n_test, topups, final_bond = outcome
                total_exceed += exceed
                total_refused += refused
                total_test += n_test
                total_topups += topups
                used += 1
                if final_bond > 0 and peaks.get(node, 0) > 0:
                    gains.append(peaks[node] / final_bond)
            rows.append({
                "seed": seed, "cohort": cohort, "estimator": estimator,
                "nodes": used, "holdout_samples": total_test,
                "heldout_exceedance": total_exceed / total_test if total_test else float("nan"),
                "post_control_refusal": total_refused / total_test if total_test else float("nan"),
                "topup_trigger_rate": total_topups / total_test if total_test else 0.0,
                "mean_multiplexing_gain": float(np.mean(gains)) if gains else float("nan"),
                "topups": total_topups,
                "success_ratio_source_run": result.success_ratio,
            })
    return rows


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--tx-load", type=int, default=20000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-trace", type=int, default=500000)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--phi", type=float, default=.30)
    parser.add_argument("--alpha", type=float, default=.99)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--update-period", type=int, default=16)
    args = parser.parse_args()
    if args.paper:
        args.tx_load = 50000
        args.repeat = 10
        args.max_trace = None
    raw = []
    for seed in range(args.repeat):
        rows = run_seed(seed, args)
        raw.extend(rows)
        print(f"seed {seed}: source success={rows[0]['success_ratio_source_run']:.4f}")
    raw_path = os.path.join(RESULTS, "e2_sizing_closure_raw.csv")
    write(raw_path, raw)

    summary = []
    for cohort in ("all", "ge50", "top10pct-degree"):
        for estimator in ("static", "adaptive"):
            cells = [r for r in raw if r["cohort"] == cohort
                     and r["estimator"] == estimator
                     and np.isfinite(r["heldout_exceedance"])]
            vals = np.asarray([r["heldout_exceedance"] for r in cells])
            served = np.asarray([r["post_control_refusal"] for r in cells])
            trigger = np.asarray([r["topup_trigger_rate"] for r in cells])
            gains = np.asarray([r["mean_multiplexing_gain"] for r in cells])
            summary.append({
                "cohort": cohort, "estimator": estimator,
                "heldout_exceedance_mean": float(np.mean(vals)) if len(vals) else float("nan"),
                "heldout_exceedance_ci95": (float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
                                             if len(vals) > 1 else 0.0),
                "post_control_refusal_mean": float(np.mean(served)) if len(served) else float("nan"),
                "topup_trigger_rate_mean": float(np.mean(trigger)) if len(trigger) else float("nan"),
                "mean_multiplexing_gain": float(np.nanmean(gains)) if len(gains) else float("nan"),
                "mean_nodes": float(np.mean([r["nodes"] for r in cells])) if cells else 0,
                "seeds": len(cells),
            })
    out = os.path.join(RESULTS, "e2_sizing_closure.csv")
    write(out, summary)
    print(f"wrote {raw_path}\nwrote {out}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
