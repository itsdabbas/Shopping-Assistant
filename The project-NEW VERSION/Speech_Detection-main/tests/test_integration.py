"""
Integration Test: Speech + Gaze Stub + Fusion Compatibility
============================================================

This test simulates the FULL multimodal flow using:
- SpeechAdapter (real, text-bypass mode)
- GazeAdapterStub (simulated gaze events)
- EventBus (shared communication channel)

Purpose: Your friend building the Gaze module can run these tests to
verify their GazeAdapter is compatible with the Speech module.

The FusionEngine is NOT implemented here — instead we manually
implement the fusion logic to prove the interfaces are compatible.
"""

import time
import pytest

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import (
    EventBus,
    GazeEvent,
    GazeEventType,
    SpeechEvent,
    SpeechEventType,
)
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub


@pytest.fixture
def bus():
    bus = EventBus()
    bus.enable_logging()
    return bus


@pytest.fixture
def speech(bus):
    return SpeechAdapter(event_bus=bus, config=Config())


@pytest.fixture
def gaze(bus):
    stub = GazeAdapterStub(event_bus=bus)
    stub.start()
    return stub


# ────────────────────────────────────────────────────────────────────
# Test: Both adapters share the same EventBus
# ────────────────────────────────────────────────────────────────────

class TestSharedEventBus:
    """Both modalities publish to the same bus."""

    def test_speech_and_gaze_events_on_same_bus(self, bus, speech, gaze):
        """Both SpeechEvents and GazeEvents are received by a wildcard subscriber."""
        all_events = []
        bus.subscribe("*", lambda e: all_events.append(e))

        # Gaze locks on an item
        gaze.simulate_lock("item_10")

        # Speech command
        speech.process_text("add this to cart")

        assert len(all_events) == 2
        assert isinstance(all_events[0], GazeEvent)
        assert isinstance(all_events[1], SpeechEvent)

    def test_events_have_compatible_timestamps(self, bus, speech, gaze):
        """Both event types use time.time() for timestamps."""
        gaze_event = gaze.simulate_lock("item_5")
        speech_event = speech.process_text("show details")

        # Both should be recent epoch timestamps
        now = time.time()
        assert abs(gaze_event.timestamp - now) < 2.0
        assert abs(speech_event.timestamp - now) < 2.0

    def test_modality_field_distinguishes_sources(self, bus, speech, gaze):
        """Events can be distinguished by their modality field."""
        gaze_event = gaze.simulate_lock("item_1")
        speech_event = speech.process_text("help")

        assert gaze_event.modality.value == "gaze"
        assert speech_event.modality.value == "speech"


# ────────────────────────────────────────────────────────────────────
# Test: Simulated Fusion Logic
# ────────────────────────────────────────────────────────────────────

class TestSimulatedFusion:
    """Manually implement fusion logic to verify interface compatibility.

    This is what the FusionEngine will do — we test the interfaces
    here to make sure Speech + Gaze outputs are fusible.
    """

    def test_locked_gaze_plus_object_command(self, bus, speech, gaze):
        """Object-bound command + locked gaze → fusible."""
        # Step 1: User looks at an item (gaze locks)
        gaze_event = gaze.simulate_lock("item_42")
        assert gaze.state == "locked"
        assert gaze.current_target == "item_42"

        # Step 2: User says "add this to cart"
        speech_event = speech.process_text("add this to cart")
        assert speech_event.payload["intent"] == "ADD_TO_CART"
        assert speech_event.payload["target_required"] is True

        # Step 3: Fusion check (manual)
        time_delta = speech_event.timestamp - gaze_event.timestamp
        assert time_delta <= 2.0  # Within fusion time window

        # ✅ Result: Can bind intent to gaze target
        fused_target = gaze_event.payload["target_id"]
        fused_intent = speech_event.payload["intent"]
        assert fused_target == "item_42"
        assert fused_intent == "ADD_TO_CART"

    def test_global_command_no_gaze_needed(self, bus, speech, gaze):
        """Global command works regardless of gaze state."""
        # Gaze is unlocked
        assert gaze.state == "unlocked"

        # User says "scroll down"
        speech_event = speech.process_text("scroll down")
        assert speech_event.payload["target_required"] is False

        # Fusion: global command → execute immediately, no gaze check
        assert speech_event.payload["intent"] == "SCROLL"
        assert speech_event.payload["params"]["direction"] == "down"

    def test_ambiguous_gaze_triggers_disambiguation(self, bus, speech, gaze):
        """Object command + ambiguous gaze → disambiguation needed."""
        # Gaze is ambiguous between two items
        gaze_event = gaze.simulate_ambiguous(["item_3", "item_4"])
        assert gaze.state == "ambiguous"

        # User says "show details"
        speech_event = speech.process_text("show details")
        assert speech_event.payload["target_required"] is True

        # Fusion: ambiguous → must ask "left or right?"
        candidates = gaze_event.payload["candidates"]
        assert candidates == ["item_3", "item_4"]

        # User repairs with "the left one"
        speech.set_dialog_active(True)
        repair_event = speech.process_text("the left one")
        assert repair_event.type == SpeechEventType.REPAIR
        assert repair_event.payload["repair_target"] == "left"

        # Resolve: "left" → first candidate
        resolved = candidates[0]  # "item_3"
        assert resolved == "item_3"

    def test_unlocked_gaze_rejects_object_command(self, bus, speech, gaze):
        """Object command without gaze target → should prompt user."""
        # No gaze lock
        assert gaze.state == "unlocked"

        # User says "buy this"
        speech_event = speech.process_text("buy this")
        assert speech_event.payload["target_required"] is True

        # Fusion: no gaze target → prompt to look at an item
        # (FusionEngine would call DM.prompt())
        assert gaze.current_target is None

    def test_gaze_expired(self, bus, speech, gaze):
        """Object command with an old gaze lock → time window expired."""
        # Lock gaze, then simulate time passing
        gaze_event = gaze.simulate_lock("item_7")

        # Manually move gaze timestamp back 5 seconds
        gaze_event_copy_timestamp = gaze_event.timestamp - 5.0

        # User says "add this"
        speech_event = speech.process_text("add this to cart")

        # Check time delta
        time_delta = speech_event.timestamp - gaze_event_copy_timestamp
        assert time_delta > 2.0  # Expired!

    def test_gaze_lock_then_unlock_then_command(self, bus, speech, gaze):
        """Lock → Unlock → Command = no target available."""
        gaze.simulate_lock("item_1")
        gaze.simulate_unlock()

        assert gaze.state == "unlocked"
        assert gaze.current_target is None

        speech_event = speech.process_text("remove this")
        assert speech_event.payload["target_required"] is True
        # FusionEngine would see unlocked state → prompt user


# ────────────────────────────────────────────────────────────────────
# Test: Event Log captures full flow
# ────────────────────────────────────────────────────────────────────

class TestEventLogging:
    """EventBus logging captures the complete interaction sequence."""

    def test_full_flow_logged(self, bus, speech, gaze):
        """All events from a complete interaction are logged."""
        gaze.simulate_lock("item_99")
        speech.process_text("add this to cart")
        gaze.simulate_unlock()

        log = bus.get_log()
        assert len(log) == 3

        # First: gaze LOCK
        assert log[0]["modality"] == "gaze"
        assert log[0]["type"] == "LOCK"
        assert log[0]["payload"]["target_id"] == "item_99"

        # Second: speech INTENT
        assert log[1]["modality"] == "speech"
        assert log[1]["type"] == "INTENT"
        assert log[1]["payload"]["intent"] == "ADD_TO_CART"

        # Third: gaze UNLOCK
        assert log[2]["modality"] == "gaze"
        assert log[2]["type"] == "UNLOCK"
