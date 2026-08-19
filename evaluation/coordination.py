"""
coordination.py -- the Coordination-Cost Model (paper Appendix, sec:cost-model).

The baselines (Shaduf, Horcrux, Revive) charge a coin shift / multi-party
negotiation NO time: rebalancing is an instantaneous atomic balance update, so
their reported success ratio is the value at coin-shift latency L = 0.  This
module turns that idealized number into a latency-aware effective success ratio.

Model (verbatim from the appendix):
  * a rebalancing op holds a single server for mean time tau; under trigger
    rate lambda_r the op is blocked with the Erlang-B (one-server) probability
        beta = (lambda_r * tau) / (1 + lambda_r * tau)                 (eq:block)
  * an end-to-end payment whose rebalancing has depth k and needs m extra
    online/cooperative parties (each up with prob p) is retained with
        R = (1 - beta)^k * p^m                                         (eq:retention)
  * effective success ratio
        S = ( #success_no_rebal + sum over rebal payments of R ) / #total

The swept variable is the **coin-shift latency L** -- exactly the quantity prior
work fixes at 0 (Fig.~fig:latency's x-axis).  The decisive asymmetry:

  * Schemes that MOVE COINS incur L on every shift, so their per-op latency is
        tau = tau_mult * L + chain_term
    and grows without bound as L grows.
  * Ballast does NOT move coins. Its certified logical draw uses one
    operator/counterparty RTT plus one certification-service RTT. The latency
    2r is CONSTANT in L, and
    Ballast's curve is flat in this variable -- the paper's central claim.

Per-scheme instantiation (table in sec:cost-model):
  scheme    moves coins   tau(L)               depth exp.   liveness
  Ballast   no            2r           (const) k            p^0
  Shaduf    yes           c_S L + psi T_chain  k            p^0  (local)
  Horcrux   yes           c_H L                k            p^k
  Revive    yes           L + T_chain          1            p^k
"""

from dataclasses import dataclass


@dataclass
class SchemeCoord:
    name: str
    moves_coins: bool    # if False, latency is constant in L (Ballast)
    tau_mult: float      # latency multiplier on the swept coin-shift latency L
    chain_term: float    # additive on-chain latency (units of r), amortized
    p_exp_per_depth: int # liveness exponent grows as p_exp_per_depth * k (Horcrux=1)
    r_const: float = 2.0 # counterparty RTT + certification-service RTT

    def tau(self, L):
        if not self.moves_coins:
            return self.r_const
        return self.tau_mult * L + self.chain_term

    def beta(self, lambda_r, L):
        a = lambda_r * self.tau(L)
        return a / (1.0 + a)


def default_schemes(T_chain=600.0, c_S=2.0, c_H=1.0, psi=0.02):
    """Realistic-ish defaults.

    T_chain: an on-chain confirmation in round-trip units (~10 min vs ~1 s RTT
        => hundreds-to-thousands).  psi: fraction of Shaduf shifts that force an
        on-chain rebind, so its amortized chain term is psi*T_chain.
    c_S, c_H: O(1) per-shift round counts (node-local resp. per-hop).
    """
    return {
        "Ballast": SchemeCoord("Ballast", moves_coins=False, tau_mult=0.0,
                               chain_term=0.0, p_exp_per_depth=0, r_const=2.0),
        "Shaduf":  SchemeCoord("Shaduf",  moves_coins=True,  tau_mult=c_S,
                               chain_term=psi * T_chain, p_exp_per_depth=0),
        "Horcrux": SchemeCoord("Horcrux", moves_coins=True,  tau_mult=c_H,
                               chain_term=0.0, p_exp_per_depth=1),
        "Revive":  SchemeCoord("Revive",  moves_coins=True,  tau_mult=1.0,
                               chain_term=T_chain, p_exp_per_depth=1),
    }


def effective_success_ratio(result, coord, lambda_r, L, p=1.0):
    """Apply the model to one RunResult at background load lambda_r and
    coin-shift latency L.  Returns S in [0, S_0]."""
    beta = coord.beta(lambda_r, L)
    one_minus_beta = max(0.0, 1.0 - beta)
    n_rebal = len(result.rebal_depths)
    retained = float(result.n_success - n_rebal)  # k=0 successes always retained
    for k in result.rebal_depths:
        retained += (one_minus_beta ** k) * (p ** (coord.p_exp_per_depth * k))
    return retained / result.n_total if result.n_total else 0.0


def latency_table(results, rtt=0.2, T_chain=600.0, psi=0.02):
    """Direct per-operation rebalancing latency, DERIVED -- not swept.

    Unlike effective_success_ratio (which routes a free latency knob L through a
    blocking model), this reports latency itself as a first-class metric, built
    from each scheme's RECORDED rebalance depths (schemes.py rebal_depths) times
    real network constants.  Nothing here is hand-tuned to a chosen operating
    point: the depths are measured, rtt/T_chain are cited constants.

      rtt     : off-chain round-trip time, seconds (~0.2 s on the public LN).
      T_chain : on-chain confirmation, seconds (~600 s = 10 min, 1-conf).
      psi     : fraction of Shaduf node-local shifts that force an on-chain rebind.

    Per-op completion time as a function of rebalance depth k:
      Ballast  : 2 * rtt                  -- peer + certification RTTs.
      Horcrux  : k * rtt                  -- coins shift along k path hops, off-chain.
      Shaduf   : (1-psi)*rtt + psi*T_chain-- node-local, occasional on-chain rebind.
      Revive   : k * rtt + T_chain        -- multi-party cycle + on-chain settlement.

    results: {scheme_name: RunResult}.  Returns one row per scheme with the
    median / p99 / max completion time (seconds) and the fraction of rebalances
    that touch chain.
    """
    import numpy as np

    def op(name, k):
        if name == "Ballast":
            return 2.0 * rtt, 0.0                   # peer + certification
        if name == "Horcrux":
            return k * rtt, 0.0                     # k hops shift coins, off-chain
        if name == "Shaduf":
            return (1.0 - psi) * rtt + psi * T_chain, psi
        if name == "Revive":
            return k * rtt + T_chain, 1.0           # cycle + settlement, always on-chain
        return rtt, 0.0

    rows = []
    for name, r in results.items():
        ks = r.rebal_depths
        if not ks:
            rows.append({"scheme": name, "n_rebal": 0, "median_s": 0.0,
                         "p99_s": 0.0, "max_s": 0.0, "onchain_frac": 0.0})
            continue
        lat = np.array([op(name, k)[0] for k in ks], dtype=float)
        onchain = float(np.mean([op(name, k)[1] for k in ks]))
        rows.append({
            "scheme": name,
            "n_rebal": len(ks),
            "median_s": float(np.median(lat)),
            "p99_s": float(np.percentile(lat, 99)),
            "max_s": float(np.max(lat)),
            "onchain_frac": onchain,
        })
    return rows


def annihilation_latency(result, coord, gain_G, lambda_r, p=1.0, grid=None):
    """Smallest coin-shift latency L at which the scheme's retention erases its
    advantage over plain LN: R(L) <= 1/G (Corollary cor:annihilation).  Returns
    None if it never annihilates on the grid (e.g. Ballast, flat in L)."""
    if grid is None:
        grid = [i * 0.5 for i in range(0, 4001)]  # 0 .. 2000 RTT units
    threshold = 1.0 / gain_G if gain_G > 0 else 0.0
    base_S = result.success_ratio
    for L in grid:
        S = effective_success_ratio(result, coord, lambda_r, L, p)
        R = S / base_S if base_S else 1.0
        if R <= threshold:
            return L
    return None
