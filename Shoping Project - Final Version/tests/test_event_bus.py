"""
Tests for EventBus
==================

Validates publish/subscribe mechanics, wildcard subscriptions,
event logging, and error handling.
"""

import pytest

from gazeshop.toolkit.event_bus import (
    EventBus,
    GazeEvent,
    GazeEventType,
    Modality,
    SpeechEvent,
    SpeechEventType,
)


class TestEventBus:
    """Core EventBus functionality."""

    def test_subscribe_and_emit(self):
        """Subscriber receives events of the registered type."""
        bus = EventBus()
        received = []

        bus.subscribe("SpeechEvent", lambda e: received.append(e))
        bus.emit(SpeechEvent(type=SpeechEventType.LISTENING))

        assert len(received) == 1
        assert received[0].type == SpeechEventType.LISTENING

    def test_subscriber_does_not_receive_other_types(self):
        """SpeechEvent subscriber does NOT receive GazeEvents."""
        bus = EventBus()
        received = []

        bus.subscribe("SpeechEvent", lambda e: received.append(e))
        bus.emit(GazeEvent(type=GazeEventType.LOCK, payload={"target_id": "item_1"}))

        assert len(received) == 0

    def test_wildcard_subscriber(self):
        """Wildcard '*' subscriber receives ALL events."""
        bus = EventBus()
        received = []

        bus.subscribe("*", lambda e: received.append(e))
        bus.emit(SpeechEvent(type=SpeechEventType.LISTENING))
        bus.emit(GazeEvent(type=GazeEventType.LOCK))

        assert len(received) == 2

    def test_multiple_subscribers(self):
        """Multiple subscribers for the same type all get called."""
        bus = EventBus()
        r1, r2 = [], []

        bus.subscribe("SpeechEvent", lambda e: r1.append(e))
        bus.subscribe("SpeechEvent", lambda e: r2.append(e))
        bus.emit(SpeechEvent(type=SpeechEventType.INTENT))

        assert len(r1) == 1
        assert len(r2) == 1

    def test_unsubscribe(self):
        """After unsubscribing, the callback no longer receives events."""
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)

        bus.subscribe("SpeechEvent", cb)
        bus.emit(SpeechEvent(type=SpeechEventType.LISTENING))
        assert len(received) == 1

        bus.unsubscribe("SpeechEvent", cb)
        bus.emit(SpeechEvent(type=SpeechEventType.STOPPED))
        assert len(received) == 1  # unchanged

    def test_unsubscribe_nonexistent_callback(self):
        """Unsubscribing a callback that was never registered does not crash."""
        bus = EventBus()
        bus.unsubscribe("SpeechEvent", lambda e: None)  # should not raise

    def test_subscriber_exception_does_not_block_others(self):
        """If one subscriber throws, other subscribers still get the event."""
        bus = EventBus()
        received = []

        def bad_cb(e):
            raise RuntimeError("Boom!")

        bus.subscribe("SpeechEvent", bad_cb)
        bus.subscribe("SpeechEvent", lambda e: received.append(e))

        bus.emit(SpeechEvent(type=SpeechEventType.INTENT))
        assert len(received) == 1  # second subscriber still worked


class TestEventBusLogging:
    """Event logging features."""

    def test_logging_records_events(self):
        """When logging is enabled, emitted events appear in the log."""
        bus = EventBus()
        bus.enable_logging()

        bus.emit(SpeechEvent(type=SpeechEventType.LISTENING))
        bus.emit(SpeechEvent(type=SpeechEventType.STOPPED))

        log = bus.get_log()
        assert len(log) == 2
        assert log[0]["type"] == "LISTENING"
        assert log[1]["type"] == "STOPPED"

    def test_logging_disabled_by_default(self):
        """No events recorded when logging is not enabled."""
        bus = EventBus()
        bus.emit(SpeechEvent(type=SpeechEventType.LISTENING))
        assert len(bus.get_log()) == 0

    def test_clear_log(self):
        """clear_log() empties the recorded event list."""
        bus = EventBus()
        bus.enable_logging()
        bus.emit(SpeechEvent(type=SpeechEventType.LISTENING))
        bus.clear_log()
        assert len(bus.get_log()) == 0


class TestSpeechEvent:
    """SpeechEvent data class."""

    def test_default_values(self):
        """SpeechEvent has sensible defaults."""
        event = SpeechEvent()
        assert event.modality == Modality.SPEECH
        assert event.type == SpeechEventType.INTENT
        assert event.payload == {}
        assert event.transcript == ""
        assert event.confidence == 0.0
        assert event.requires_confirmation is False

    def test_to_dict(self):
        """to_dict() serialises all fields correctly."""
        event = SpeechEvent(
            type=SpeechEventType.INTENT,
            payload={"intent": "ADD_TO_CART", "params": {}, "target_required": True},
            transcript="add this to cart",
            confidence=0.85,
        )
        d = event.to_dict()
        assert d["modality"] == "speech"
        assert d["type"] == "INTENT"
        assert d["payload"]["intent"] == "ADD_TO_CART"
        assert d["confidence"] == 0.85


class TestGazeEvent:
    """GazeEvent data class."""

    def test_default_values(self):
        event = GazeEvent()
        assert event.modality == Modality.GAZE
        assert event.type == GazeEventType.LOCK

    def test_to_dict(self):
        event = GazeEvent(
            type=GazeEventType.AMBIGUOUS,
            payload={"candidates": ["item_1", "item_2"]},
            confidence=0.5,
        )
        d = event.to_dict()
        assert d["type"] == "AMBIGUOUS"
        assert d["payload"]["candidates"] == ["item_1", "item_2"]
