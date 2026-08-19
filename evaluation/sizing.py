"""
sizing.py -- capital efficiency and the service-level dial (paper Q3, sec:ev-theta).

From a Ballast run's draw diagnostics we measure the statistical-multiplexing
gain and the capital/availability trade-off the service level alpha controls.

Definitions (paper sec:sizing, sec:ev-theta):
  * per-channel peak demand p_i : the largest shortfall ever drawn on channel i.
  * node aggregate demand        : the time series of a node's outstanding draw.
  * bond(alpha)                  : the alpha-quantile of the node's aggregate
                                   demand -- the bond that serves a fraction
                                   alpha of draws (Lemma lem:avail).
  * multiplexing gain  g = sum_i p_i / bond(alpha) : how much per-channel
                                   reserve one node-level bond replaces.  g >= 1
                                   because aggregate peak <= sum of per-channel
                                   peaks; g grows as channels peak less together.
  * held-out exceedance(alpha)   : fraction of later accepted-draw samples
                                   above a prefix-fitted bond.  It estimates
                                   refusal only under stationarity and is
                                   selection-biased because the source run
                                   itself has a finite bond.
  * counterparty loss            : identically 0 -- the bond fully backs every
                                   accepted draw at every alpha.
"""

import collections
import numpy as np


def per_node_demand(result):
    """node -> sorted list of outstanding-draw samples (one per draw event)."""
    d = collections.defaultdict(list)
    for node, _amt, outstanding_after in result.draw_events:
        d[node].append(outstanding_after)
    return d


def per_node_channel_peaks(result, sim):
    """node -> sum of per-channel peak shortfalls (the sum_i p_i for that node)."""
    s = collections.defaultdict(int)
    for (operator, _counterparty), peak in result.channel_peak_short.items():
        s[operator] += peak
    return s


def per_node_demand_ordered(result):
    """node -> list of outstanding-draw samples in TEMPORAL draw order (not
    sorted), so a train/holdout split respects time: size on the past, serve
    the future."""
    d = collections.defaultdict(list)
    for node, _amt, outstanding_after in result.draw_events:
        d[node].append(outstanding_after)
    return d


def capital_efficiency_table(result, sim, alphas=(1.0, 0.99, 0.95, 0.90),
                             min_draws=50, train_frac=0.5):
    """Return rows (alpha, mean_g, refusal_rate, counterparty_loss), sized
    OUT-OF-SAMPLE.

    The earlier in-sample version sized bond(alpha) as the alpha-quantile of a
    node's draw samples and then measured the refusal rate on the *same*
    samples; that makes refusal == 1-alpha a tautology (the empirical CDF
    evaluated at its own quantile).  Here we instead:

      * split each node's draws temporally into a TRAIN prefix (first
        `train_frac`) and a HELD-OUT suffix;
      * size  bond(alpha) = alpha-quantile of the TRAIN draws (the sizing
        decision an operator could actually make from history);
      * measure held-out exceedance = fraction of later accepted-draw samples
        exceeding that bond.

    The reported field retains its historical ``refusal_rate`` name for CSV
    compatibility.  It tracks refusal only if demand is stationary and the
    accepted-draw sample is representative; neither is guaranteed.  The
    multiplexing gain g = sum_i p_i / bond(alpha) uses directed-channel peaks.
    """
    demand = per_node_demand_ordered(result)
    chan_peaks = per_node_channel_peaks(result, sim)
    active = {n: v for n, v in demand.items() if len(v) >= min_draws}

    rows = []
    for alpha in alphas:
        gains, refusals = [], []
        for n, seq in active.items():
            cut = max(1, int(len(seq) * train_frac))
            train = np.array(seq[:cut])
            holdout = np.array(seq[cut:])
            if len(holdout) == 0:
                continue
            bond_a = float(np.quantile(train, alpha))
            if bond_a <= 0:
                continue
            sum_pi = chan_peaks.get(n, 0)
            if sum_pi > 0:
                gains.append(sum_pi / bond_a)
            refusals.append(float(np.mean(holdout > bond_a)))  # OUT-OF-SAMPLE
        mean_g = float(np.mean(gains)) if gains else float("nan")
        med_g = float(np.median(gains)) if gains else float("nan")
        refusal = float(np.mean(refusals)) if refusals else 0.0
        rows.append({
            "alpha": alpha,
            "mean_g": mean_g,
            "median_g": med_g,
            "refusal_rate": refusal,
            "counterparty_loss": 0.0,   # invariant: bond fully backs accepted draws
            "n_active_nodes": len(gains),
        })
    return rows


def correlation_regime(result):
    """Empirical proxy for how synchronized a node's channels peak: ratio of
    aggregate peak to sum-of-per-channel-peaks across active nodes.  Near 1 =>
    comonotone (gain collapses); small => well multiplexed.  (paper:
    'Correlation erodes the multiplexing gain'.)"""
    demand = per_node_demand(result)
    ratios = []
    for node, samples in demand.items():
        if len(samples) >= 5:
            agg_peak = max(samples)
            # crude per-channel proxy: average draw, scaled by channel count is
            # not tracked here; we report agg_peak vs total drawn as a coarse
            # synchronization index.
            ratios.append(agg_peak / (sum(samples) + 1e-9))
    return float(np.mean(ratios)) if ratios else float("nan")
