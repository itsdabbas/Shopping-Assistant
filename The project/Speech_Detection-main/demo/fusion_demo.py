"""
Fusion Demo — Tam Multimodal Entegrasyon Simülasyonu (Refactored)
=================================================================

Çalıştır:
    cd "Speech_Detection-main"
    python demo/fusion_demo.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus
from gazeshop.toolkit.telemetry import TelemetryLogger
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager
from gazeshop.toolkit.events import (
    MultimodalCommandEvent, PromptEvent, DisambiguationPromptEvent, 
    ConfirmationPromptEvent, ActionCancelledEvent, TargetLockedEvent
)

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


def print_ui(tag: str, msg: str, color: str = "\033[93m") -> None:
    E = "\033[0m"
    print(f"  {color}[{tag}] {msg}{E}")


def build_scene():
    bus = EventBus()
    config = Config(FUSION_TIME_WINDOW_S=2.0, GAZE_DWELL_TO_LOCK_S=1.0, LOCK_TTL_S=4.0)

    # Adapters 
    gaze = GazeAdapterStub(event_bus=bus)
    speech = SpeechAdapter(event_bus=bus, config=config)

    # Core Logic
    engine = FusionEngine(event_bus=bus, config=config)
    dm = DialogueManager(event_bus=bus, config=config)
    telemetry = TelemetryLogger(event_bus=bus, config=config)

    # UI Emulators via Bus
    bus.subscribe("MultimodalCommandEvent", lambda e: print_ui("COMMAND EXECUTED", f"{e.intent} on {e.target_id} conf={e.confidence:.2f}", "\033[92m"))
    bus.subscribe("PromptEvent", lambda e: print_ui("PROMPT", e.message, "\033[95m"))
    bus.subscribe("DisambiguationPromptEvent", lambda e: print_ui("OVERLAY (A/B)", e.message, "\033[96m"))
    bus.subscribe("ConfirmationPromptEvent", lambda e: print_ui("OVERLAY (Y/N)", e.message, "\033[96m"))
    bus.subscribe("ActionCancelledEvent", lambda e: print_ui("CANCELLED", e.message, "\033[91m"))
    bus.subscribe("TargetLockedEvent", lambda e: print_ui("UI FOCUS", f"Bounding box on {e.target_id}", "\033[90m"))

    gaze.start()
    return bus, gaze, speech, engine, dm


def scenario_1(gaze, speech, dm):
    print("\n\nSenaryo 1: LOCK + Speech -> Komut")
    gaze.simulate_lock("shopping_item_42")
    speech.process_text("add to cart")
    time.sleep(0.1)

def scenario_2(gaze, speech, dm):
    print("\n\nSenaryo 2: No Target -> Prompt")
    gaze.simulate_unlock()
    speech.process_text("buy this")
    time.sleep(0.1)

def scenario_3(gaze, speech, dm):
    print("\n\nSenaryo 3: AMBIGUOUS -> Repair -> Komut")
    gaze.simulate_ambiguous(["item_1", "item_2"])
    speech.process_text("show details")
    time.sleep(0.1)
    # UI shows disambiguation. We send repair:
    speech.process_text("left")
    time.sleep(0.1)

def scenario_4(gaze, speech, dm):
    print("\n\nSenaryo 4: LOCK TTL (Zaman Aşımı)")
    gaze.simulate_lock("item_stale")
    print("  (Bekleniyor > 4 saniye...)")
    # Simulate time travel or just sleep 
    engine = dm.event_bus  # hack: access fusion engine state 
    gaze_adapter_ts = gaze.event_bus
    # We will simulate lock_ts explicitly
    bus = dm.event_bus
    from gazeshop.toolkit.event_bus import GazeEvent, GazeEventType
    bus.emit(GazeEvent(ts:=time.time() - 5.0, GazeEventType.LOCK, {"target_id": "stale"}, 1.0))
    time.sleep(0.1)
    speech.process_text("remove")
    
if __name__ == "__main__":
    bus, gaze, speech, engine, dm = build_scene()
    scenario_1(gaze, speech, dm)
    scenario_2(gaze, speech, dm)
    scenario_3(gaze, speech, dm)
    scenario_4(gaze, speech, dm)
    print("\nDemolar başarıyla simüle edildi!")
