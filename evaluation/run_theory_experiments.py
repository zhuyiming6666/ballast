"""Run the BALLAST-R experiments that directly correspond to new theory.

E2: correlation degradation and tight sqrt(n) pooled safety margin.
E3: Pi(tau_e, theta) exposure upper/lower bounds and availability costs.
E4: session freeze versus amount-scoped claim escrow.

The defaults are reproducible smoke/validation settings. ``--paper`` raises
Monte-Carlo samples and seeds but does not change the model or reported fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os

import numpy as np

import common
from claim_semantics import simulate_claims
from security_modes import (exposure_bound, fast_path_stats, latency_samples,
                            simulate_greedy_epoch_attack)


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGURES = os.path.abspath(os.path.join(HERE, "..", "figures"))


def write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    os.makedirs(RESULTS, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {name}")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def correlated_lognormal(samples, n, rho, rng):
    """Unit-mean lognormal demands coupled by a Gaussian common factor."""
    common_factor = rng.normal(size=(samples, 1))
    idiosyncratic = rng.normal(size=(samples, n))
    z = math.sqrt(rho) * common_factor + math.sqrt(1.0 - rho) * idiosyncratic
    sigma = 1.0
    return np.exp(sigma * z - 0.5 * sigma * sigma)


def correlated_gamma(samples, n, rho, rng, shape=8.0):
    """Unit-mean Gamma demands with exact pairwise correlation ``rho``.

    Each channel is the sum of one shared Gamma component and one independent
    component.  Both have the same scale, so their sum keeps a fixed marginal
    Gamma(shape, 1/shape) distribution while the shared variance fraction is
    exactly rho.
    """
    common = (rng.gamma(shape * rho, 1.0 / shape, size=(samples, 1))
              if rho > 0 else np.zeros((samples, 1)))
    independent = (rng.gamma(shape * (1.0 - rho), 1.0 / shape,
                             size=(samples, n))
                   if rho < 1 else np.zeros((samples, n)))
    return common + independent


def run_e2(args):
    rng = np.random.default_rng(args.seed)
    alpha = 0.99
    scaling_rows = []
    for distribution, generator in (
        ("gamma-light-tail", correlated_gamma),
        ("lognormal-heavy-tail", correlated_lognormal),
    ):
        for rho in args.rhos:
            margins = []
            gains = []
            block = []
            for n in args.ns:
                x = generator(args.samples, n, rho, rng)
                aggregate = x.sum(axis=1)
                pooled_q = float(np.quantile(aggregate, alpha))
                pooled_margin = pooled_q - float(np.mean(aggregate))
                per_channel_q = np.quantile(x, alpha, axis=0)
                separate = float(per_channel_q.sum())
                gain = separate / pooled_q
                margins.append(max(pooled_margin, 1e-12))
                gains.append(gain)
                block.append({
                    "distribution": distribution,
                    "rho": rho, "n": n, "alpha": alpha,
                    "samples": args.samples, "seed": args.seed,
                    "pooled_margin": pooled_margin,
                    "separate_reserve": separate, "pooled_reserve": pooled_q,
                    "multiplexing_gain": gain,
                })
            slope = float(np.polyfit(np.log(args.ns), np.log(margins), 1)[0])
            for row in block:
                row["loglog_margin_slope"] = slope
            scaling_rows.extend(block)
            print(f"E2 {distribution} rho={rho:.2f}: margin slope={slope:.3f}, "
                  f"gain(n={args.ns[-1]})={gains[-1]:.3f}x")
    write_csv("e2_correlation_scaling.csv", scaling_rows)


def load_payment_values(max_values, seed):
    values = []
    with open(common.PAYMENT_VALUE_FILE) as handle:
        for line in handle:
            value = int(float(line))
            if 0 < value <= common.PAYMENT_VALUE_THRESHOLD:
                values.append(value)
    if not values:
        raise RuntimeError("payment trace is empty")
    values = np.asarray(values, dtype=float)
    if max_values is not None and len(values) > max_values:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), size=max_values, replace=False)]
    return values


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_e3(args):
    values = load_payment_values(args.max_values, args.seed)
    trace_hash = file_sha256(common.PAYMENT_VALUE_FILE)
    quantiles = [("p50", .50), ("p80", .80), ("p95", .95), ("p99", .99)]
    thetas = [("zero", 0.0)]
    thetas.extend((name, float(np.quantile(values, q))) for name, q in quantiles)
    thetas.append(("inf", math.inf))
    rows = []
    latency_rows = []
    for tau in args.taus:
        tau_label = "inf" if math.isinf(tau) else f"{tau:g}"
        for theta_label, theta in thetas:
            upper = exposure_bound(tau, theta, args.delta, args.r_min,
                                   args.counterparties)
            measured = simulate_greedy_epoch_attack(
                tau, theta, args.r_min, args.counterparties)
            fast = fast_path_stats(values, theta)
            ratio = (measured / upper if math.isfinite(upper) and upper > 0
                     else (1.0 if measured == upper == 0 else math.nan))
            rows.append({
                "tau_e_s": tau_label, "theta_quantile": theta_label,
                "theta_satoshi": theta,
                "simulated_attack_exposure": measured,
                "theorem_upper_bound": upper, "lower_upper_ratio": ratio,
                "payment_values_sampled": len(values),
                "payment_sampling_seed": args.seed,
                "payment_trace_sha256": trace_hash,
                "checkpoint_gas": args.checkpoint_gas,
                **fast,
                "checkpoints_per_day": (0.0 if math.isinf(tau) else
                                         (math.inf if tau == 0 else 86400.0 / tau)),
                "checkpoint_gas_per_day": (0.0 if math.isinf(tau) else
                    (math.inf if tau == 0 else args.checkpoint_gas * 86400.0 / tau)),
            })
            lat = latency_samples(values, theta, args.peer_rtt, args.confirm_rtt)
            for q in (0.50, 0.95, 0.99):
                latency_rows.append({
                    "tau_e_s": tau_label, "theta_quantile": theta_label,
                    "latency_quantile": q,
                    "latency_s": float(np.quantile(lat, q)),
                })
    write_csv("e3_safety_grid.csv", rows)
    write_csv("e3_latency_quantiles.csv", latency_rows)

    throughput = []
    committee_capacity = args.committee_workers / args.confirm_rtt
    cas_capacity = 1.0 / args.cas_latency
    for offered in (10, 100, 1000):
        for theta_label, theta in thetas[:-1]:
            slow_fraction = float(np.mean(values > theta))
            confirmed_load = offered * slow_fraction
            for backend, capacity in (("2-of-3", committee_capacity),
                                      ("onchain-CAS", cas_capacity)):
                throughput.append({
                    "offered_draws_s": offered, "theta_quantile": theta_label,
                    "backend": backend, "confirmed_load_s": confirmed_load,
                    "service_capacity_s": capacity,
                    "utilization": confirmed_load / capacity,
                    "stable": confirmed_load < capacity,
                })
    write_csv("e3_confirmation_throughput.csv", throughput)
    finite = [r for r in rows if math.isfinite(float(r["theorem_upper_bound"]))
              and float(r["theorem_upper_bound"]) > 0]
    print("E3 median lower/upper tightness ratio: "
          f"{np.median([float(r['lower_upper_ratio']) for r in finite]):.3f}")


def run_e4(args):
    raw = []
    for n in args.claim_channels:
        for rate in args.claim_rates:
            for deposit_fraction in args.deposit_fractions:
                for seed in range(args.claim_seeds):
                    for semantics in ("freeze", "escrow"):
                        raw.append(simulate_claims(
                            semantics=semantics, n_channels=n, claim_rate_h=rate,
                            deposit_fraction=deposit_fraction, seed=seed,
                            horizon_h=args.claim_horizon))
    write_csv("e4_claim_semantics_raw.csv", raw)

    summary = []
    keys = sorted({(r["n_channels"], r["claim_rate_h"], r["deposit_fraction"])
                   for r in raw})
    for n, rate, deposit_fraction in keys:
        by_semantics = {}
        for semantics in ("freeze", "escrow"):
            cells = [r for r in raw if r["n_channels"] == n
                     and r["claim_rate_h"] == rate
                     and r["deposit_fraction"] == deposit_fraction
                     and r["semantics"] == semantics]
            refusal_values = np.asarray([r["refusal_rate"] for r in cells])
            affected_values = np.asarray(
                [r["affected_channel_hours"] for r in cells])
            by_semantics[semantics] = {
                "refusal": float(np.mean(refusal_values)),
                "refusal_ci95": (float(1.96 * np.std(refusal_values, ddof=1)
                                       / np.sqrt(len(refusal_values)))
                                   if len(refusal_values) > 1 else 0.0),
                "affected": float(np.mean(affected_values)),
                "affected_ci95": (float(1.96 * np.std(affected_values, ddof=1)
                                        / np.sqrt(len(affected_values)))
                                    if len(affected_values) > 1 else 0.0),
                "violations": int(sum(r["invariant_violations"] for r in cells)),
                "admitted": float(np.mean([r["admitted_claims"] for r in cells])),
                "deposit_limited": float(np.mean(
                    [r["deposit_limited_claims"] for r in cells])),
            }
        freeze = by_semantics["freeze"]
        escrow = by_semantics["escrow"]
        summary.append({
            "n_channels": n, "claim_rate_h": rate,
            "deposit_fraction": deposit_fraction,
            "freeze_refusal_rate": freeze["refusal"],
            "freeze_refusal_ci95": freeze["refusal_ci95"],
            "escrow_refusal_rate": escrow["refusal"],
            "escrow_refusal_ci95": escrow["refusal_ci95"],
            "freeze_affected_channel_hours": freeze["affected"],
            "freeze_affected_channel_hours_ci95": freeze["affected_ci95"],
            "escrow_affected_channel_hours": escrow["affected"],
            "escrow_affected_channel_hours_ci95": escrow["affected_ci95"],
            "blast_radius_ratio": freeze["affected"] / max(escrow["affected"], 1e-12),
            "escrow_invariant_violations": escrow["violations"],
            "mean_admitted_claims": escrow["admitted"],
            "mean_deposit_limited_claims": escrow["deposit_limited"],
        })
    write_csv("e4_claim_semantics.csv", summary)
    print("E4 max escrow invariant violations:",
          max(r["escrow_invariant_violations"] for r in summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0, .25, .5, .75, 1])
    parser.add_argument("--ns", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--max-values", type=int, default=200000)
    parser.add_argument("--taus", type=float, nargs="+", default=[0, .1, 1, 10, 60, 600, math.inf])
    parser.add_argument("--delta", type=float, default=.2)
    parser.add_argument("--r-min", type=float, default=.2)
    parser.add_argument("--counterparties", type=int, default=8)
    parser.add_argument("--peer-rtt", type=float, default=.2)
    parser.add_argument("--confirm-rtt", type=float, default=.4)
    parser.add_argument("--committee-workers", type=int, default=64)
    parser.add_argument("--cas-latency", type=float, default=12.0)
    parser.add_argument("--checkpoint-gas", type=int, default=87645)
    parser.add_argument("--claim-channels", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--claim-rates", type=float, nargs="+", default=[.05, .25, 1.0])
    parser.add_argument("--deposit-fractions", type=float, nargs="+",
                        default=[0.0, .10, .25, .50])
    parser.add_argument("--claim-seeds", type=int, default=20)
    parser.add_argument("--claim-horizon", type=float, default=24.0)
    args = parser.parse_args()
    if args.paper:
        args.samples = 200000
        args.max_values = 1000000
        args.claim_seeds = 100
        args.claim_horizon = 168.0
    run_e2(args)
    run_e3(args)
    run_e4(args)


if __name__ == "__main__":
    main()
