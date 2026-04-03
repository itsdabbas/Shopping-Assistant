"""
Tests for SpeechAdapter
=======================

Tests the SpeechAdapter using the text-bypass mode (process_text)
which allows testing the full pipeline without microphone hardware.

These tests verify:
- Text-to-event pipeline (process_text)
- EventBus emission
- Dialog context switching
- Event field completeness for FusionEngine compatibility
"""

import pytest

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter


@pytest.fixture
def bus():
    """Shared EventBus for test isolation."""
    return EventBus()


@pytest.fixture
def adapter(bus):
    """SpeechAdapter in text-only mode (no microphone required)."""
    cfg = Config()
    return SpeechAdapter(event_bus=bus, config=cfg)


# ────────────────────────────────────────────────────────────────────
# Basic pipeline tests
# ────────────────────────────────────────────────────────────────────

class TestProcessText:
    """Tests using the process_text() bypass method."""

    def test_returns_speech_event(self, adapter):
        """process_text() returns a SpeechEvent."""
        event = adapter.process_text("add this to cart")
        assert isinstance(event, SpeechEvent)

    def test_recognized_intent(self, adapter):
        """Known phrases produce INTENT events."""
        event = adapter.process_text("add this to cart")
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == "ADD_TO_CART"

    def test_unrecognized_phrase(self, adapter):
        """Unknown phrases produce ERROR events."""
        event = adapter.process_text("lorem ipsum dolor sit amet")
        assert event.type == SpeechEventType.ERROR

    def test_transcript_preserved(self, adapter):
        """The raw transcript is stored in the event."""
        original = "show me the details"
        event = adapter.process_text(original)
        assert event.transcript == original

    def test_confidence_present(self, adapter):
        """Confidence score is a float between 0 and 1."""
        event = adapter.process_text("scroll down", asr_confidence=0.9)
        assert isinstance(event.confidence, float)
        assert 0.0 <= event.confidence <= 1.0

    def test_timestamp_present(self, adapter):
        """Timestamp is a positive float (epoch time)."""
        event = adapter.process_text("help")
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0


# ────────────────────────────────────────────────────────────────────
# EventBus emission tests
# ────────────────────────────────────────────────────────────────────

class TestEventEmission:
    """Verify events are emitted on the EventBus."""

    def test_event_emitted_on_bus(self, bus, adapter):
        """process_text() emits the event on the shared bus."""
        received = []
        bus.subscribe("SpeechEvent", lambda e: received.append(e))

        adapter.process_text("add this to cart")

        assert len(received) == 1
        assert received[0].payload["intent"] == "ADD_TO_CART"

    def test_multiple_events_emitted(self, bus, adapter):
        """Multiple calls emit multiple events."""
        received = []
        bus.subscribe("SpeechEvent", lambda e: received.append(e))

        adapter.process_text("scroll down")
        adapter.process_text("open cart")
        adapter.process_text("help")

        assert len(received) == 3

    def test_error_event_also_emitted(self, bus, adapter):
        """ERROR events are also broadcast on the bus."""
        received = []
        bus.subscribe("SpeechEvent", lambda e: received.append(e))

        adapter.process_text("gibberish xyzzy foobar")

        assert len(received) == 1
        assert received[0].type == SpeechEventType.ERROR


# ────────────────────────────────────────────────────────────────────
# Dialog context tests
# ────────────────────────────────────────────────────────────────────

class TestDialogContext:
    """Dialog context switching for REPAIR / CONFIRM parsing."""

    def test_dialog_inactive_by_default(self, adapter):
        """Dialog context is off by default."""
        # "yes" should NOT produce a CONFIRM event
        event = adapter.process_text("yes")
        assert event.type != SpeechEventType.CONFIRM

    def test_dialog_active_enables_confirm(self, adapter):
        """When dialog is active, 'yes' produces CONFIRM."""
        adapter.set_dialog_active(True)
        event = adapter.process_text("yes")
        assert event.type == SpeechEventType.CONFIRM
        assert event.payload["confirm"] is True

    def test_dialog_active_enables_deny(self, adapter):
        """When dialog is active, 'no' produces CONFIRM(confirm=False)."""
        adapter.set_dialog_active(True)
        event = adapter.process_text("no")
        assert event.type == SpeechEventType.CONFIRM
        assert event.payload["confirm"] is False

    def test_dialog_active_enables_repair(self, adapter):
        """When dialog is active, 'the left one' produces REPAIR."""
        adapter.set_dialog_active(True)
        event = adapter.process_text("the left one")
        assert event.type == SpeechEventType.REPAIR
        assert event.payload["repair_target"] == "left"

    def test_dialog_deactivated(self, adapter):
        """After deactivating dialog, CONFIRM patterns stop working."""
        adapter.set_dialog_active(True)
        adapter.set_dialog_active(False)
        event = adapter.process_text("yes")
        assert event.type != SpeechEventType.CONFIRM


# ────────────────────────────────────────────────────────────────────
# Fusion compatibility — required fields
# ────────────────────────────────────────────────────────────────────

class TestFusionCompatibility:
    """Verify all fields required by FusionEngine are present."""

    def test_intent_event_has_required_fields(self, adapter):
        """INTENT events contain all fields the FusionEngine needs."""
        event = adapter.process_text("add this to cart", asr_confidence=0.9)

        # Required top-level fields
        assert hasattr(event, "timestamp")
        assert hasattr(event, "type")
        assert hasattr(event, "transcript")
        assert hasattr(event, "confidence")
        assert hasattr(event, "requires_confirmation")

        # Required payload fields for INTENT
        assert "intent" in event.payload
        assert "target_required" in event.payload
        assert "params" in event.payload

        # Types
        assert isinstance(event.payload["intent"], str)
        assert isinstance(event.payload["target_required"], bool)
        assert isinstance(event.payload["params"], dict)

    def test_target_required_field_for_object_commands(self, adapter):
        """Object-bound commands set target_required=True."""
        event = adapter.process_text("add this to cart")
        assert event.payload["target_required"] is True

    def test_target_required_false_for_global_commands(self, adapter):
        """Global commands set target_required=False."""
        event = adapter.process_text("scroll down")
        assert event.payload["target_required"] is False

    def test_repair_event_has_repair_target(self, adapter):
        """REPAIR events contain repair_target field."""
        adapter.set_dialog_active(True)
        event = adapter.process_text("the right one")
        assert "repair_target" in event.payload
        assert event.payload["repair_target"] == "right"

    def test_confirm_event_has_confirm_field(self, adapter):
        """CONFIRM events contain confirm boolean field."""
        adapter.set_dialog_active(True)
        event = adapter.process_text("yes")
        assert "confirm" in event.payload
        assert isinstance(event.payload["confirm"], bool)

    def test_to_dict_serialization(self, adapter):
        """Events can be serialized for logging/transmission."""
        event = adapter.process_text("show details")
        d = event.to_dict()

        assert isinstance(d, dict)
        assert "timestamp" in d
        assert "modality" in d
        assert "type" in d
        assert "payload" in d
        assert "confidence" in d
        assert d["modality"] == "speech"


# ────────────────────────────────────────────────────────────────────
# All 13 intents via process_text
# ────────────────────────────────────────────────────────────────────

class TestAllIntentsViaAdapter:
    """Smoke-test all 13 intents through the full adapter pipeline."""

    @pytest.mark.parametrize("phrase,expected_intent", [
        ("add this to cart", "ADD_TO_CART"),
        ("show details", "SHOW_DETAILS"),
        ("find similar", "FIND_SIMILAR"),
        ("compare this", "COMPARE"),
        ("show alternatives", "SHOW_ALTERNATIVES"),
        ("pin this", "PIN_ITEM"),
        ("remove this", "REMOVE_ITEM"),
        ("scroll down", "SCROLL"),
        ("open cart", "OPEN_CART"),
        ("go back", "GO_BACK"),
        ("help", "HELP"),
        ("cancel", "CANCEL"),
        ("undo", "UNDO"),
    ])
    def test_intent_recognized(self, adapter, phrase, expected_intent):
        event = adapter.process_text(phrase)
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == expected_intent
