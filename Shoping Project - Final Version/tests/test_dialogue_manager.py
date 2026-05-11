"""
Tests for DialogueManager — Revised
=====================================

Tests the current EventBus-driven DialogueManager (state machine).
The DM subscribes to IntentNeedsX events and emits MultimodalCommandEvent / PromptEvent.
No callback constructor arguments are used (matches actual implementation).
"""

from __future__ import annotations

import time
import pytest

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType, GazeEvent, GazeEventType
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager
from gazeshop.toolkit.events import (
    MultimodalCommandEvent,
    PromptEvent,
    DisambiguationPromptEvent,
    ConfirmationPromptEvent,
    ActionCancelledEvent,
    IntentNeedsTargetEvent,
    IntentNeedsDisambiguationEvent,
    IntentNeedsConfirmationEvent,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def config() -> Config:
    cfg = Config(DISAMBIGUATION_TIMEOUT_S=2.0, MAX_REPAIR_ATTEMPTS=2)
    return cfg


@pytest.fixture
def dm(bus, config) -> DialogueManager:
    return DialogueManager(event_bus=bus, config=config)


@pytest.fixture
def commands(bus) -> list:
    collected: list = []
    bus.subscribe("MultimodalCommandEvent", collected.append)
    return collected


@pytest.fixture
def prompts(bus) -> list:
    collected: list = []
    bus.subscribe("PromptEvent", collected.append)
    bus.subscribe("DisambiguationPromptEvent", collected.append)
    bus.subscribe("ConfirmationPromptEvent", collected.append)
    bus.subscribe("ActionCancelledEvent", collected.append)
    return collected


# ── Helpers ───────────────────────────────────────────────────────────

def _emit_needs_target(bus: EventBus, intent: str = "OPEN_DETAIL", reason: str = "no_lock"):
    ev = IntentNeedsTargetEvent(reason=reason, intent=intent)
    bus.emit(ev)


def _emit_needs_disambig(bus: EventBus, intent: str = "OPEN_DETAIL"):
    ev = IntentNeedsDisambiguationEvent(
        intent=intent,
        candidates=[{"id": "item_A", "pos": "left"}, {"id": "item_B", "pos": "right"}],
    )
    bus.emit(ev)


def _emit_needs_confirm(bus: EventBus, intent: str = "OPEN_DETAIL", target: str = "item_X"):
    ev = IntentNeedsConfirmationEvent(intent=intent, target_id=target)
    bus.emit(ev)


def _speech_repair(bus: EventBus, pos: str):
    bus.emit(SpeechEvent(
        type=SpeechEventType.REPAIR,
        payload={"repair_target": pos},
        transcript=pos,
        confidence=1.0,
    ))


def _speech_confirm(bus: EventBus, value: bool):
    bus.emit(SpeechEvent(
        type=SpeechEventType.CONFIRM,
        payload={"confirm": value},
        transcript="yes" if value else "no",
        confidence=1.0,
    ))


def _speech_cancel(bus: EventBus):
    bus.emit(SpeechEvent(
        type=SpeechEventType.CANCEL,
        payload={},
        transcript="cancel",
        confidence=1.0,
    ))


# ── NeedsTarget flow ──────────────────────────────────────────────────

class TestNeedsTarget:
    def test_needs_target_emits_prompt(self, dm, bus, prompts):
        _emit_needs_target(bus)
        assert dm.state == "WAIT_TARGET"
        assert len(prompts) == 1
        assert isinstance(prompts[0], PromptEvent)
        assert "look at" in prompts[0].message.lower()

    def test_needs_target_lock_expired_prompt(self, dm, bus, prompts):
        _emit_needs_target(bus, reason="lock_expired")
        assert len(prompts) == 1
        assert "expired" in prompts[0].message.lower() or "look" in prompts[0].message.lower()

    def test_gaze_lock_resolves_wait_target(self, dm, bus, commands, prompts):
        _emit_needs_target(bus, intent="READ_ALOUD")
        assert dm.state == "WAIT_TARGET"

        # User now locks gaze
        bus.emit(GazeEvent(
            type=GazeEventType.LOCK,
            payload={"target_id": "exhibit_42"},
            confidence=1.0,
        ))
        assert dm.state == "IDLE"
        assert len(commands) == 1
        assert commands[0].intent == "READ_ALOUD"
        assert commands[0].target_id == "exhibit_42"


# ── Disambiguation flow ───────────────────────────────────────────────

class TestDisambiguation:
    def test_disambig_emits_prompt(self, dm, bus, prompts):
        _emit_needs_disambig(bus)
        assert dm.state == "WAIT_DISAMBIGUATION"
        assert len(prompts) == 1
        assert isinstance(prompts[0], DisambiguationPromptEvent)

    def test_repair_left_resolves(self, dm, bus, commands, prompts):
        _emit_needs_disambig(bus, intent="ZOOM_IN")
        _speech_repair(bus, "left")
        assert len(commands) == 1
        assert commands[0].target_id == "item_A"
        assert commands[0].intent == "ZOOM_IN"
        assert dm.state == "IDLE"

    def test_repair_right_resolves(self, dm, bus, commands, prompts):
        _emit_needs_disambig(bus)
        _speech_repair(bus, "right")
        assert len(commands) == 1
        assert commands[0].target_id == "item_B"

    def test_invalid_repair_reprompts(self, dm, bus, commands, prompts):
        _emit_needs_disambig(bus)
        _speech_repair(bus, "centre")   # not a valid position
        assert len(commands) == 0
        assert len(prompts) == 2        # initial + re-prompt
        assert dm.state == "WAIT_DISAMBIGUATION"

    def test_max_repair_attempts_cancels(self, dm, bus, commands, prompts):
        _emit_needs_disambig(bus)
        _speech_repair(bus, "centre")
        _speech_repair(bus, "centre")
        assert dm.state == "IDLE"
        assert len(commands) == 0
        cancels = [p for p in prompts if isinstance(p, ActionCancelledEvent)]
        assert len(cancels) == 1

    def test_cancel_during_disambiguation(self, dm, bus, commands, prompts):
        _emit_needs_disambig(bus)
        _speech_cancel(bus)
        assert dm.state == "IDLE"
        assert len(commands) == 0
        cancels = [p for p in prompts if isinstance(p, ActionCancelledEvent)]
        assert len(cancels) == 1


# ── Confirmation flow ─────────────────────────────────────────────────

class TestConfirmation:
    def test_confirm_emits_prompt(self, dm, bus, prompts):
        _emit_needs_confirm(bus)
        assert dm.state == "WAIT_CONFIRMATION"
        assert len(prompts) == 1
        assert isinstance(prompts[0], ConfirmationPromptEvent)

    def test_user_confirms_yes_executes(self, dm, bus, commands, prompts):
        _emit_needs_confirm(bus, intent="ADD_TO_CART", target="product_99")
        _speech_confirm(bus, True)
        assert len(commands) == 1
        assert commands[0].intent == "ADD_TO_CART"
        assert commands[0].target_id == "product_99"
        assert dm.state == "IDLE"

    def test_user_confirms_no_cancels(self, dm, bus, commands, prompts):
        _emit_needs_confirm(bus)
        _speech_confirm(bus, False)
        assert len(commands) == 0
        assert dm.state == "IDLE"
        cancels = [p for p in prompts if isinstance(p, ActionCancelledEvent)]
        assert len(cancels) == 1

    def test_cancel_during_confirmation(self, dm, bus, commands, prompts):
        _emit_needs_confirm(bus)
        _speech_cancel(bus)
        assert dm.state == "IDLE"
        assert len(commands) == 0


# ── Timeout ───────────────────────────────────────────────────────────

class TestTimeout:
    def test_disambiguation_timeout(self, dm, bus, commands, prompts):
        """After DISAMBIGUATION_TIMEOUT_S with no input, next speech event
        triggers auto-cancel (lazy timeout check on next event)."""
        _emit_needs_disambig(bus)
        assert dm.state == "WAIT_DISAMBIGUATION"

        # Simulate timeout by waiting > DISAMBIGUATION_TIMEOUT_S (set to 2s)
        time.sleep(2.2)

        # Any speech event triggers the timeout check
        _speech_repair(bus, "left")

        assert dm.state == "IDLE"
        cancels = [p for p in prompts if isinstance(p, ActionCancelledEvent)]
        assert len(cancels) == 1

    def test_confirmation_timeout(self, dm, bus, commands, prompts):
        _emit_needs_confirm(bus)
        assert dm.state == "WAIT_CONFIRMATION"

        time.sleep(2.2)
        _speech_confirm(bus, True)

        assert dm.state == "IDLE"
        cancels = [p for p in prompts if isinstance(p, ActionCancelledEvent)]
        assert len(cancels) == 1
