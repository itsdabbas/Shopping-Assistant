"""
Fusion + Dialog Integration Tests
=================================

Simulates interaction on EventBus between FusionEngine and DialogueManager.
"""

from __future__ import annotations

import time
import pytest

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, GazeEvent, GazeEventType, SpeechEvent, SpeechEventType
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager
from gazeshop.toolkit.events import (
    MultimodalCommandEvent, 
    PromptEvent, 
    DisambiguationPromptEvent, 
    ConfirmationPromptEvent,
    ActionCancelledEvent
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()

@pytest.fixture
def config() -> Config:
    return Config(
         FUSION_TIME_WINDOW_S=2.0,
         GAZE_DWELL_TO_LOCK_S=1.0,
         CONFIDENCE_THRESHOLD=0.60,
         MAX_REPAIR_ATTEMPTS=2,
         LOCK_TTL_S=4.0
    )

@pytest.fixture
def engine(bus, config) -> FusionEngine:
    return FusionEngine(event_bus=bus, config=config)

@pytest.fixture
def manager(bus, config) -> DialogueManager:
    return DialogueManager(event_bus=bus, config=config)

@pytest.fixture
def commands(bus) -> list[MultimodalCommandEvent]:
    collected: list[MultimodalCommandEvent] = []
    bus.subscribe("MultimodalCommandEvent", lambda e: collected.append(e))
    return collected

@pytest.fixture
def prompts(bus) -> list:
    collected: list = []
    bus.subscribe("PromptEvent", lambda e: collected.append(e))
    bus.subscribe("DisambiguationPromptEvent", lambda e: collected.append(e))
    bus.subscribe("ConfirmationPromptEvent", lambda e: collected.append(e))
    bus.subscribe("ActionCancelledEvent", lambda e: collected.append(e))
    return collected


# ── Helpers ───────────────────────────────────────────────────────────

def gaze_lock(bus: EventBus, target_id: str, ts: float | None = None) -> GazeEvent:
    ev = GazeEvent(
        timestamp=ts if ts is not None else time.time(),
        type=GazeEventType.LOCK,
        payload={"target_id": target_id},
        confidence=1.0,
    )
    bus.emit(ev)
    return ev

def gaze_ambiguous(bus: EventBus, candidates: list[dict]) -> GazeEvent:
    ev = GazeEvent(
        timestamp=time.time(),
        type=GazeEventType.AMBIGUOUS,
        payload={"candidates": candidates},
        confidence=0.4,
    )
    bus.emit(ev)
    return ev

def speech_intent(
    bus: EventBus,
    intent: str,
    target_required: bool = True,
    confidence: float = 0.85,
    requires_confirmation: bool = False,
) -> SpeechEvent:
    ev = SpeechEvent(
        timestamp=time.time(),
        type=SpeechEventType.INTENT,
        payload={"intent": intent, "target_required": target_required, "params": {}},
        transcript=intent,
        confidence=confidence,
        requires_confirmation=requires_confirmation,
    )
    bus.emit(ev)
    return ev

def speech_repair(bus: EventBus, pos: str) -> SpeechEvent:
    ev = SpeechEvent(
        timestamp=time.time(),
        type=SpeechEventType.REPAIR,
        payload={"repair_target": pos},
        transcript=pos,
        confidence=1.0,
    )
    bus.emit(ev)
    return ev

def speech_confirm(bus: EventBus) -> SpeechEvent:
    ev = SpeechEvent(
        timestamp=time.time(), type=SpeechEventType.CONFIRM, payload={}, transcript="yes", confidence=1.0
    )
    bus.emit(ev)
    return ev


# ── Tests ─────────────────────────────────────────────────────────────

def test_happy_path(engine, manager, bus, commands, prompts):
    """Rule 1: Lock + Obj Intent -> Command."""
    gaze_lock(bus, "item_99")
    speech_intent(bus, "ADD_TO_CART")
    
    assert len(commands) == 1
    assert commands[0].intent == "ADD_TO_CART"
    assert commands[0].target_id == "item_99"

def test_no_target_prompt(engine, manager, bus, commands, prompts):
    """Rule 2: Obj Intent with no lock -> PromptEvent."""
    speech_intent(bus, "ADD_TO_CART")
    assert len(commands) == 0
    assert len(prompts) == 1
    assert isinstance(prompts[0], PromptEvent)
    assert "Please look at an item" in prompts[0].message

def test_disambiguation_repair(engine, manager, bus, commands, prompts):
    """Rule 3/5: AMBIGUOUS -> DisambiguationPrompt -> REPAIR -> Command."""
    gaze_ambiguous(bus, [{"id": "item1", "pos": "left"}, {"id": "item2", "pos": "right"}])
    speech_intent(bus, "SHOW_DETAILS")
    
    assert len(prompts) == 1
    assert isinstance(prompts[0], DisambiguationPromptEvent)
    assert manager.state == "WAIT_DISAMBIGUATION"
    
    speech_repair(bus, "right")
    
    assert len(commands) == 1
    assert commands[0].target_id == "item2"
    assert manager.state == "IDLE"

def test_confirmation_flow(engine, manager, bus, commands, prompts):
    """Rule 4: Low conf -> ConfirmationPrompt -> CONFIRM -> Command."""
    gaze_lock(bus, "test_item")
    speech_intent(bus, "BUY", requires_confirmation=True)
    
    assert len(prompts) == 1
    assert isinstance(prompts[0], ConfirmationPromptEvent)
    assert manager.state == "WAIT_CONFIRMATION"
    
    speech_confirm(bus)
    
    assert len(commands) == 1
    assert commands[0].intent == "BUY"
    assert commands[0].target_id == "test_item"

def test_ttl_stale_target(engine, manager, bus, commands, prompts):
    """Rule: Stale Gaze Lock should fire TARGET EXPIRED and clear lock."""
    # lock from 5s ago
    gaze_lock(bus, "stale_item", ts=time.time() - 5.0)
    speech_intent(bus, "PIN_ITEM")
    
    assert len(commands) == 0
    assert len(prompts) == 1
    assert isinstance(prompts[0], PromptEvent)
    assert "Lock expired" in prompts[0].message
