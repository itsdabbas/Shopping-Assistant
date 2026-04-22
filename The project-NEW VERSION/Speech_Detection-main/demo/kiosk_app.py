"""
Kiosk App Demo — Reusing the MMUI Toolkit
=========================================

Demonstrates that the GazeShop MMUI toolkit is completely decoupled 
from "Shopping" logic. By simply providing a new intent vocabulary, 
the same late fusion + state machine loop powers a completely 
different use case (e.g. Museum Catalog Kiosk).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager


# ── Custom Domain Vocabulary ──────────────────────────────────────────

KIOSK_INTENTS = {
    "READ_ALOUD": {
        "patterns": [r"read this", r"audio guide", r"tell me about this"],
        "target_required": True,
    },
    "SUMMARIZE": {
        "patterns": [r"summarize", r"give me the short version", r"tl dr"],
        "target_required": True,
    },
    "NEXT_PAGE": {
        "patterns": [r"next", r"go forward", r"next page"],
        "target_required": False,  # Global navigation
    },
}

def print_kiosk_ui(msg: str) -> None:
    print(f"  \033[96;1m[KIOSK UI]\033[0m {msg}")

def build_kiosk_app():
    bus = EventBus()
    config = Config()
    
    # 1. Provide custom parser
    parser = IntentParser(config)
    parser._intents = KIOSK_INTENTS  # Override default shopping intents

    # 2. Instantiate Toolkit Modalities & Core
    gaze = GazeAdapterStub(event_bus=bus)
    speech = SpeechAdapter(event_bus=bus, config=config, intent_parser=parser)
    FusionEngine(event_bus=bus, config=config)
    DialogueManager(event_bus=bus, config=config)

    # 3. Kiosk-specific Fission / Screen reactions
    bus.subscribe("MultimodalCommandEvent", lambda e: print_kiosk_ui(
        f"🟢 EXECUTING {e.intent} on Artifact: {e.target_id or 'GLOBAL'}"
    ))
    bus.subscribe("PromptEvent", lambda e: print_kiosk_ui(
        f"🟠 {e.message}"
    ))
    bus.subscribe("TargetLockedEvent", lambda e: print_kiosk_ui(
        f"👀 Cursor locked on {e.target_id}"
    ))
    
    gaze.start()
    return gaze, speech

if __name__ == "__main__":
    print("\n\033[1m=== KIOSK APP SKELETON DEMO ===\033[0m")
    gaze, speech = build_kiosk_app()
    
    print("\n1) User looks at the 'Mona_Lisa' exhibit frame and asks for the audio guide:")
    gaze.simulate_lock("Mona_Lisa")
    time.sleep(0.1)
    speech.process_text("tell me about this")
    time.sleep(0.5)
    
    print("\n2) User looks away and says 'next page' (Global command):")
    gaze.simulate_unlock()
    time.sleep(0.1)
    speech.process_text("next page")
    time.sleep(0.5)
    
    print("\n3) User wants a summary but forgot to look at any exhibit:")
    speech.process_text("summarize")
    time.sleep(0.5)
    
    print("\n\033[92mDone.\033[0m")
