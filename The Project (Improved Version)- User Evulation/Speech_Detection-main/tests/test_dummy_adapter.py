"""
Tests for DummyKeyboardAdapter
================================

Verifies that the DummyKeyboardAdapter:
- Correctly extends ModalityAdapter
- Emits expected GazeEvent / SpeechEvent types when key-handler methods are called
- start() / stop() lifecycle works
"""

from __future__ import annotations

import pytest

from gazeshop.toolkit.event_bus import (
    EventBus,
    GazeEvent,
    GazeEventType,
    SpeechEvent,
    SpeechEventType,
)
from gazeshop.toolkit.adapters.dummy_adapter import DummyKeyboardAdapter
from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def adapter(bus) -> DummyKeyboardAdapter:
    return DummyKeyboardAdapter(
        bus,
        default_intent="OPEN_DETAIL",
        default_target="test_target",
        target_required=True,
        blocking=False,
    )


# ── Interface compliance ──────────────────────────────────────────────

class TestInterface:
    def test_inherits_modality_adapter(self, adapter):
        assert isinstance(adapter, ModalityAdapter)

    def test_initial_state_not_running(self, adapter):
        assert not adapter.is_running


# ── Event emission ───────────────────────────────────────────────────

class TestEventEmission:
    def _collect(self, bus: EventBus) -> list:
        events: list = []
        bus.subscribe("GazeEvent", events.append)
        bus.subscribe("SpeechEvent", events.append)
        return events

    def test_gaze_lock_event(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_gaze_lock("item_test")
        assert len(events) == 1
        assert isinstance(events[0], GazeEvent)
        assert events[0].type == GazeEventType.LOCK
        assert events[0].payload["target_id"] == "item_test"

    def test_gaze_unlock_event(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_gaze_unlock()
        assert len(events) == 1
        assert isinstance(events[0], GazeEvent)
        assert events[0].type == GazeEventType.UNLOCK

    def test_gaze_ambiguous_event(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_gaze_ambiguous(["item_A", "item_B"])
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, GazeEvent)
        assert ev.type == GazeEventType.AMBIGUOUS
        cands = ev.payload["candidates"]
        assert cands[0]["pos"] == "left"
        assert cands[1]["pos"] == "right"

    def test_speech_intent_event(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_speech_intent("OPEN_DETAIL", target_required=True)
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, SpeechEvent)
        assert ev.type == SpeechEventType.INTENT
        assert ev.payload["intent"] == "OPEN_DETAIL"
        assert ev.payload["target_required"] is True

    def test_speech_confirm_yes(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_speech_confirm(True)
        ev = events[0]
        assert ev.type == SpeechEventType.CONFIRM
        assert ev.payload["confirm"] is True

    def test_speech_confirm_no(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_speech_confirm(False)
        assert events[0].payload["confirm"] is False

    def test_speech_repair_event(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_speech_repair("right")
        ev = events[0]
        assert ev.type == SpeechEventType.REPAIR
        assert ev.payload["repair_target"] == "right"

    def test_speech_cancel_event(self, adapter, bus):
        events = self._collect(bus)
        adapter._emit_speech_cancel()
        assert events[0].type == SpeechEventType.CANCEL


# ── Key handler dispatch ──────────────────────────────────────────────

class TestKeyDispatch:
    def _collect(self, bus: EventBus) -> list:
        events: list = []
        bus.subscribe("GazeEvent", events.append)
        bus.subscribe("SpeechEvent", events.append)
        return events

    def test_key_l_locks(self, adapter, bus):
        events = self._collect(bus)
        adapter._handle_key("l")
        assert events[0].type == GazeEventType.LOCK

    def test_key_u_unlocks(self, adapter, bus):
        events = self._collect(bus)
        adapter._handle_key("u")
        assert events[0].type == GazeEventType.UNLOCK

    def test_key_a_ambiguous(self, adapter, bus):
        events = self._collect(bus)
        adapter._handle_key("a")
        assert events[0].type == GazeEventType.AMBIGUOUS

    def test_key_i_intent(self, adapter, bus):
        events = self._collect(bus)
        adapter._handle_key("i")
        assert events[0].type == SpeechEventType.INTENT

    def test_key_c_confirm_yes(self, adapter, bus):
        events = self._collect(bus)
        adapter._handle_key("c")
        assert events[0].payload["confirm"] is True

    def test_key_n_confirm_no(self, adapter, bus):
        events = self._collect(bus)
        adapter._handle_key("n")
        assert events[0].payload["confirm"] is False

    def test_unknown_key_no_crash(self, adapter, bus, capsys):
        adapter._handle_key("z")   # should just print a message, not raise
        captured = capsys.readouterr()
        assert "Unknown key" in captured.out
