"""Tests for foundation signals."""

from astraturbo.foundation.signals import property_changed


class TestSignals:
    def test_signal_send_receive(self):
        received = []

        def handler(sender, **kwargs):
            received.append(kwargs)

        property_changed.connect(handler)
        property_changed.send("test_sender", node="obj", name="x", value=42)
        assert len(received) == 1
        assert received[0]["value"] == 42
        property_changed.disconnect(handler)
