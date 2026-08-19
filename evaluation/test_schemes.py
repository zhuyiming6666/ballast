"""Focused accounting regression tests for the evaluation simulator."""

import unittest

import networkx as nx

import schemes
import coordination
import claim_semantics
import security_modes


class FixedSim:
    """Minimal deterministic simulation accepted by ``work_ballast``."""

    def __init__(self, pairs, amounts):
        self.G = nx.Graph()
        self.G.add_edge("A", "B")
        self.G.add_edge("A", "C")
        self.nodes = ["A", "B", "C"]
        self.within = {
            ("A", "B"): (100, 100),
            ("A", "C"): (100, 100),
        }
        self.tx = amounts
        self._pairs = iter(pairs)

    def sample_pair(self, _mode, _skew_param):
        return next(self._pairs)

    def get_within(self, a, b):
        key = tuple(sorted((a, b)))
        value = self.within[key]
        return value if key[0] == a else value[::-1]

    def update_within(self, a, b, bal_a, bal_b):
        if bal_a < 0 or bal_b < 0:
            raise ValueError("negative balance")
        key = tuple(sorted((a, b)))
        self.within[key] = (bal_a, bal_b) if key[0] == a else (bal_b, bal_a)


class BallastAccountingTests(unittest.TestCase):
    def test_unrelated_inbound_flow_does_not_repay_another_channel(self):
        # With phi=0.5, A has a bond of 100.  A->B draws 30.  The subsequent
        # C->A payment is inbound on A-C and therefore cannot repay A's A-B
        # exposure.  A second A->B payment needs 80 but only 70 headroom remains.
        sim = FixedSim(
            pairs=[("A", "B"), ("C", "A"), ("A", "B")],
            amounts=[80, 80, 80],
        )
        result = schemes.work_ballast(sim, 3, bond_fraction=0.5)
        self.assertEqual(result.n_success, 2)
        self.assertEqual(result.refusals, 1)

    def test_same_channel_reverse_flow_retires_exposure(self):
        # The B->A reverse payment does restore the A-B channel, so the final
        # A->B draw can reuse the released bond headroom.
        sim = FixedSim(
            pairs=[("A", "B"), ("B", "A"), ("A", "B")],
            amounts=[80, 30, 100],
        )
        result = schemes.work_ballast(sim, 3, bond_fraction=0.5)
        self.assertEqual(result.n_success, 3)
        self.assertEqual(result.refusals, 0)

    def test_failed_shaduf_payment_rolls_back_speculative_shifts(self):
        sim = FixedSim(pairs=[("A", "C")], amounts=[50])
        sim.G.remove_edge("A", "C")
        del sim.within[("A", "C")]
        sim.G.add_edge("B", "C")
        sim.G.add_edge("A", "D")
        sim.nodes.append("D")
        sim.within[("A", "B")] = (10, 10)
        sim.within[("B", "C")] = (10, 10)
        sim.within[("A", "D")] = (100, 100)
        before = dict(sim.within)
        result = schemes.work_shaduf(sim, 1)
        self.assertEqual(result.n_success, 0)
        self.assertEqual(sim.within, before)

    def test_certified_draw_latency_counts_two_round_trips(self):
        result = schemes.RunResult()
        result.rebal_depths = [1, 3]
        row = coordination.latency_table({"Ballast": result}, rtt=0.2)[0]
        self.assertEqual(row["median_s"], 0.4)
        self.assertEqual(row["p99_s"], 0.4)
        self.assertEqual(row["onchain_frac"], 0.0)


class AdaptiveSecurityTests(unittest.TestCase):
    def test_exposure_upper_and_matching_lower_bound(self):
        upper = security_modes.exposure_bound(
            tau_e_s=10.0, theta=100.0, delta_s=0.2,
            r_min_s=0.2, counterparties=8)
        lower = security_modes.optimal_epoch_attack(
            tau_e_s=10.0, theta=100.0, r_min_s=0.2,
            counterparties=8)
        self.assertLessEqual(lower, upper)
        self.assertGreater(lower / upper, 0.95)

    def test_strong_endpoint_has_zero_exposure(self):
        self.assertEqual(security_modes.exposure_bound(
            0.0, 0.0, 0.2, 0.2, 8), 0.0)

    def test_escrow_invariant_and_blast_radius(self):
        freeze = claim_semantics.simulate_claims(
            semantics="freeze", n_channels=16, claim_rate_h=0.25, seed=7)
        escrow = claim_semantics.simulate_claims(
            semantics="escrow", n_channels=16, claim_rate_h=0.25, seed=7)
        self.assertEqual(escrow["invariant_violations"], 0)
        self.assertLess(escrow["affected_channel_hours"],
                        freeze["affected_channel_hours"])

    def test_claim_deposit_limits_concurrent_griefing(self):
        free = claim_semantics.simulate_claims(
            semantics="escrow", n_channels=16, claim_rate_h=20, seed=9,
            horizon_h=4, deposit_fraction=0.0)
        bonded = claim_semantics.simulate_claims(
            semantics="escrow", n_channels=16, claim_rate_h=20, seed=9,
            horizon_h=4, deposit_fraction=0.5)
        self.assertGreater(bonded["deposit_limited_claims"], 0)
        self.assertLess(bonded["admitted_claims"], free["admitted_claims"])


if __name__ == "__main__":
    unittest.main()
