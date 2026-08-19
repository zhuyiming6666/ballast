"""Recipient-bound fast-path admission slots for the containment model.

This lightweight validator makes the theorem input explicit: a bond session
pre-allocates each epoch slot to one mode, channel, and recipient. A recipient
accepts at most one certificate for that slot and never accepts more than its
registered value cap. It models the local enforcement required for ``eta``;
it is not a replacement for a production PCN transport implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    session: str
    mode: int
    epoch: int
    number: int
    channel: str
    recipient: str
    max_value: int
    expiry: float


class AdmissionPolicy:
    """Fixed session allocation with recipient-local replay protection."""

    def __init__(self, slots: list[Slot]):
        self._slots: dict[tuple[int, int], Slot] = {}
        self._seen: set[tuple[int, int]] = set()
        for slot in slots:
            if slot.max_value <= 0 or slot.expiry <= 0:
                raise ValueError("invalid slot limits")
            key = (slot.epoch, slot.number)
            if key in self._slots:
                raise ValueError("duplicate epoch slot")
            self._slots[key] = slot

    def accept(self, certificate: Slot, amount: int, honor_time: float) -> None:
        """Accept once iff the certificate exactly matches its allocation."""
        key = (certificate.epoch, certificate.number)
        registered = self._slots.get(key)
        if registered is None or registered != certificate:
            raise ValueError("unregistered slot")
        if key in self._seen:
            raise ValueError("slot replay")
        if amount <= 0 or amount > registered.max_value:
            raise ValueError("value cap")
        if honor_time > registered.expiry:
            raise ValueError("expired slot")
        self._seen.add(key)

    @property
    def honored(self) -> int:
        return len(self._seen)
