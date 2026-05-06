# -*- coding: utf-8 -*-
"""
Demo #2 -- Museum / Document Kiosk
====================================

Demonstrates the SAME MMUI Toolkit core powering a completely different
application: a museum exhibit / document kiosk.

Run from the Speech_Detection-main directory:
    python demo/demo_kiosk.py

What this demo proves (reusability)
-------------------------------------
* Identical toolkit setup (MMUIToolkit, adapters, fusion, dialogue).
* ONLY the intent vocabulary and action handlers differ.
* No changes to toolkit core were required.
* Three tasks shown:
    1. LOCK on "Mona_Lisa" + "tell me about this" → READ_ALOUD
    2. No lock + "next" (global)                  → NAVIGATE_NEXT
    3. LOCK + "summarize" → SUMMARIZE
    4. AMBIGUOUS + "zoom in" → Disambiguation → ZOOM_IN
    5. LOCK + "bookmark this" → PIN_EXHIBIT
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ensure UTF-8 output on Windows terminals
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.intent_parser import IntentParser
from apps.kiosk.intents import KIOSK_INTENTS, WHISPER_PROMPT
from apps.kiosk.actions import KIOSK_ACTION_HANDLERS

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


def hdr(text: str) -> None:
    print(f"\n\033[1;35m{'─' * 56}\033[0m")
    print(f"\033[1;35m  {text}\033[0m")
    print(f"\033[1;35m{'─' * 56}\033[0m")

def ui(tag: str, msg: str, color: str = "\033[93m") -> None:
    print(f"  {color}[{tag}]\033[0m {msg}")


def build_kiosk_toolkit() -> tuple[MMUIToolkit, GazeAdapterStub, SpeechAdapter]:
    """Wire up the kiosk demo — identical boilerplate, different app layer."""

    config = Config(
        FUSION_TIME_WINDOW_S=2.0,
        LOCK_TTL_S=5.0,
        MAX_REPAIR_ATTEMPTS=2,
        ENABLE_TELEMETRY=False,
        WHISPER_INITIAL_PROMPT=WHISPER_PROMPT,
    )

    # 1. Create toolkit (same as shopping — same 5 steps)
    tk = MMUIToolkit(config)

    # 2. Adapters
    parser  = IntentParser(config, custom_patterns=KIOSK_INTENTS)
    gaze    = GazeAdapterStub(event_bus=tk.bus)
    speech  = SpeechAdapter(event_bus=tk.bus, config=config, intent_parser=parser)

    # 3. Register adapters
    tk.register_adapter(gaze).register_adapter(speech)

    # 4. Register kiosk domain actions (different from shopping)
    for intent, handler in KIOSK_ACTION_HANDLERS.items():
        tk.register_action(intent, handler)

    # 5. Register UI feedback
    tk.register_feedback("PromptEvent",
        lambda e: ui("KIOSK PROMPT", e.message, "\033[35m"))
    tk.register_feedback("DisambiguationPromptEvent",
        lambda e: ui("KIOSK A/B", e.message, "\033[35m"))
    tk.register_feedback("ActionCancelledEvent",
        lambda e: ui("CANCELLED", e.message, "\033[91m"))
    tk.register_feedback("TargetLockedEvent",
        lambda e: ui("EXHIBIT FOCUS", f"[eye] Cursor -> {e.target_id}", "\033[90m"))

    return tk, gaze, speech


def run_scenarios(tk: MMUIToolkit, gaze: GazeAdapterStub, speech: SpeechAdapter) -> None:

    # ── Task 1: Look at exhibit + audio guide ────────────────────────
    hdr("Task 1 — Gaze exhibit + 'tell me about this'")
    gaze.simulate_lock("Mona_Lisa")
    speech.process_text("tell me about this")
    time.sleep(0.05)

    # ── Task 2: Global navigation (no gaze needed) ───────────────────
    hdr("Task 2 — Global command: 'next' (no gaze required)")
    gaze.simulate_unlock()
    speech.process_text("next")
    time.sleep(0.05)

    # ── Task 3: Look at exhibit + summarize ──────────────────────────
    hdr("Task 3 — Gaze exhibit + 'summarize'")
    gaze.simulate_lock("Starry_Night")
    speech.process_text("summarize")
    time.sleep(0.05)

    # ── Task 4: Ambiguous gaze → disambiguation ──────────────────────
    hdr("Task 4 — AMBIGUOUS → Disambiguation → ZOOM_IN")
    gaze.simulate_ambiguous(["exhibit_left", "exhibit_right"])
    speech.process_text("zoom in")
    time.sleep(0.05)
    speech.set_dialog_active(True)
    speech.process_text("left")
    time.sleep(0.05)

    # ── Task 5: Bookmark an exhibit ──────────────────────────────────
    hdr("Task 5 — Bookmark exhibit")
    gaze.simulate_lock("The_Scream")
    speech.process_text("bookmark this")
    time.sleep(0.05)

    # ── Task 6: No target → Prompt ───────────────────────────────────
    hdr("Task 6 — No target + 'open detail' → Prompt")
    gaze.simulate_unlock()
    speech.process_text("open detail")
    time.sleep(0.05)


if __name__ == "__main__":
    print("\n\033[1;35m" + "=" * 48)
    print("   MMUI Toolkit - Demo #2: Museum Kiosk")
    print("   (Same toolkit core - different domain)")
    print("=" * 48 + "\033[0m")

    tk, gaze, speech = build_kiosk_toolkit()
    tk.start()

    try:
        run_scenarios(tk, gaze, speech)
    finally:
        tk.stop()

    print("\n\033[1;35m[DONE] Kiosk demo complete.\033[0m\n")
