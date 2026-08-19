"""run_deterrence_economics.py -- link the attack experiment to the
rational-deterrence theorem (reviewer item FP-6).

For each (tau_e, theta) cell of the safety-mode table we report:
  * the attack value E(tau_e, theta) the Byzantine lower-bound adversary can
    honor across branches (the bounded-loss guarantee for Byzantine operators);
  * the attacker's own irrevocable outlay: every honored draw is an HTLC the
    victims' counterparties fulfilled, i.e. the adversary must forward real
    value to collect, so the gross take equals E only after paying routing
    value through; the net extraction against the bond is what deterrence
    prices;
  * the deterrence condition of the theorem: a proven fork forfeits the whole
    bond, so a rational operator's profit is at most max(0, E - B).  The
    break-even bond is B* = E: for every B >= E the attack is weakly
    dominated.  We tabulate the attacker's net profit at B/E in {0.5, 0.9,
    1.0, 1.1} and the reserve the protocol actually enforces
    (B_safe = E, sec:model), under which net profit is <= 0 by construction.

Writes results/deterrence_economics.csv.  Pure computation; runs in seconds.
"""

from __future__ import annotations

import csv
import os

import common
import security_modes as sm
from run_security_sensitivity import payment_quantiles, TRACE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def main():
    qs = payment_quantiles(TRACE)
    theta = qs[0.95]
    m, q, r, delta = 8, 1, 0.2, 0.2            # paper instance: eta(t) = 40 t
    taus = [1.0, 10.0, 60.0, 600.0]
    bond_ratios = [0.5, 0.9, 1.0, 1.1]

    fields = ["tau_e_s", "theta_sat", "attack_value_sat", "bound_sat",
              "honored_draws", "attacker_outlay_note",
              "bond_over_E", "bond_sat", "attacker_net_sat", "deterred"]
    rows = []
    for tau in taus:
        attack = sm.simulate_greedy_epoch_attack(tau, theta, r, m, q)
        bound = sm.exposure_bound(tau, theta, delta, r, m, q)
        honored = int(round(attack / theta)) if theta else 0
        for br in bond_ratios:
            bond = br * bound
            net = max(0.0, attack - bond)      # Thm deterrence: profit <= max(0, E - B)
            rows.append({
                "tau_e_s": tau, "theta_sat": round(theta),
                "attack_value_sat": round(attack),
                "bound_sat": round(bound),
                "honored_draws": honored,
                "attacker_outlay_note": "each honored draw is an irrevocably "
                                        "forwarded HTLC of size theta",
                "bond_over_E": br,
                "bond_sat": round(bond),
                "attacker_net_sat": round(net),
                "deterred": "yes" if net <= 0 else "no",
            })
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "deterrence_economics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")
    for row in rows:
        if row["bond_over_E"] in (1.0, 0.9):
            print(row["tau_e_s"], "s  B/E=", row["bond_over_E"],
                  " net=", row["attacker_net_sat"], " deterred=", row["deterred"])


if __name__ == "__main__":
    main()
