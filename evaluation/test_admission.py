"""Regression tests for the protocol-enforced admission-bound model."""

import unittest

from admission import AdmissionPolicy, Slot


class AdmissionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.slot = Slot(
            session="sid", mode=0, epoch=7, number=3,
            channel="A-B", recipient="B", max_value=100, expiry=10.0,
        )
        self.policy = AdmissionPolicy([self.slot])

    def test_accepts_registered_recipient_slot_once(self):
        self.policy.accept(self.slot, amount=100, honor_time=9.0)
        self.assertEqual(self.policy.honored, 1)
        with self.assertRaisesRegex(ValueError, "slot replay"):
            self.policy.accept(self.slot, amount=1, honor_time=9.5)

    def test_rejects_cross_recipient_branch_reuse(self):
        forged = Slot(
            session="sid", mode=0, epoch=7, number=3,
            channel="A-C", recipient="C", max_value=100, expiry=10.0,
        )
        with self.assertRaisesRegex(ValueError, "unregistered slot"):
            self.policy.accept(forged, amount=100, honor_time=9.0)

    def test_rejects_value_and_expiry_violations(self):
        with self.assertRaisesRegex(ValueError, "value cap"):
            self.policy.accept(self.slot, amount=101, honor_time=9.0)
        with self.assertRaisesRegex(ValueError, "expired slot"):
            self.policy.accept(self.slot, amount=100, honor_time=10.1)


if __name__ == "__main__":
    unittest.main()
