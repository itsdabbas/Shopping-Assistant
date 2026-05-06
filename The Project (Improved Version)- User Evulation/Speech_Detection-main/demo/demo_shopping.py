# -*- coding: utf-8 -*-
"""
Demo #1 -- Shopping Assistant
==============================

Demonstrates the MMUI Toolkit powering an e-commerce shopping assistant.

Run from the Speech_Detection-main directory:
    python demo/demo_shopping.py

What this demo shows
---------------------
* Toolkit facade (MMUIToolkit) wiring adapters + handlers in 5 lines.
* Shopping-specific intent vocabulary loaded from apps.shopping.intents.
* Full late-fusion + dialogue flow: lock → intent → command.
* Four scenarios:
    1. LOCK + "add to cart"         → ADD_TO_CART executed
    2. No lock + "buy this"         → PromptEvent (please look at item)
    3. AMBIGUOUS + "show details"   → Disambiguation → SHOW_DETAILS
    4. Stale lock (> TTL)           → Lock expired → PromptEvent
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
from apps.shopping.intents import SHOPPING_INTENTS, WHISPER_PROMPT
from apps.shopping.actions import SHOPPING_ACTION_HANDLERS

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

# ── ANSI helpers ─────────────────────────────────────────────────────

def hdr(text: str) -> None:
    print(f"\n\033[1;36m{'─' * 56}\033[0m")
    print(f"\033[1;36m  {text}\033[0m")
    print(f"\033[1;36m{'─' * 56}\033[0m")

def ui(tag: str, msg: str, color: str = "\033[93m") -> None:
    print(f"  {color}[{tag}]\033[0m {msg}")


def build_shopping_toolkit() -> tuple[MMUIToolkit, GazeAdapterStub, SpeechAdapter]:
    """Wire up the shopping demo using the MMUIToolkit facade."""

    config = Config(
        FUSION_TIME_WINDOW_S=2.0,
        LOCK_TTL_S=4.0,
        MAX_REPAIR_ATTEMPTS=2,
        ENABLE_TELEMETRY=False,
        WHISPER_INITIAL_PROMPT=WHISPER_PROMPT,
    )

    # 1. Create toolkit
    tk = MMUIToolkit(config)

    # 2. Create adapters (they share tk.bus automatically)
    parser  = IntentParser(config, custom_patterns=SHOPPING_INTENTS)
    gaze    = GazeAdapterStub(event_bus=tk.bus)
    speech  = SpeechAdapter(event_bus=tk.bus, config=config, intent_parser=parser)

    # 3. Register adapters
    tk.register_adapter(gaze).register_adapter(speech)

    # 4. Register domain actions
    for intent, handler in SHOPPING_ACTION_HANDLERS.items():
        tk.register_action(intent, handler)

    # 5. Register UI feedback handlers
    tk.register_feedback("PromptEvent",
        lambda e: ui("PROMPT", e.message, "\033[95m"))
    tk.register_feedback("DisambiguationPromptEvent",
        lambda e: ui("OVERLAY A/B", e.message, "\033[96m"))
    tk.register_feedback("ConfirmationPromptEvent",
        lambda e: ui("OVERLAY Y/N", e.message, "\033[96m"))
    tk.register_feedback("ActionCancelledEvent",
        lambda e: ui("CANCELLED", e.message, "\033[91m"))
    tk.register_feedback("TargetLockedEvent",
        lambda e: ui("UI FOCUS", f"Bounding box → {e.target_id}", "\033[90m"))

    return tk, gaze, speech


def run_scenarios(tk: MMUIToolkit, gaze: GazeAdapterStub, speech: SpeechAdapter) -> None:

    # ── Scenario 1: Happy path ───────────────────────────────────────
    hdr("Scenario 1 — LOCK + Speech → Command")
    gaze.simulate_lock("product_42")
    speech.process_text("add this to cart")
    time.sleep(0.05)

    # ── Scenario 2: No gaze target ───────────────────────────────────
    hdr("Scenario 2 — No Target → Prompt")
    gaze.simulate_unlock()
    speech.process_text("buy this")
    time.sleep(0.05)

    # ── Scenario 3: Ambiguous gaze → disambiguation ──────────────────
    hdr("Scenario 3 — AMBIGUOUS → Repair → Command")
    gaze.simulate_ambiguous(["product_left", "product_right"])
    speech.process_text("show details")
    time.sleep(0.05)
    speech.set_dialog_active(True)
    speech.process_text("the left one")
    time.sleep(0.05)

    # ── Scenario 4: Stale lock (> TTL) ──────────────────────────────
    hdr("Scenario 4 — Stale Lock → Lock Expired")
    from gazeshop.toolkit.event_bus import GazeEvent, GazeEventType
    stale_event = GazeEvent(
        timestamp=time.time() - 6.0,   # 6s ago — beyond TTL=4s
        type=GazeEventType.LOCK,
        payload={"target_id": "product_stale"},
        confidence=1.0,
    )
    tk.bus.emit(stale_event)
    time.sleep(0.05)
    speech.process_text("pin this")
    time.sleep(0.05)


if __name__ == "__main__":
    print("\n\033[1;32m" + "=" * 48)
    print("   MMUI Toolkit - Demo #1: Shopping App")
    print("=" * 48 + "\033[0m")

    tk, gaze, speech = build_shopping_toolkit()
    tk.start()

    try:
        run_scenarios(tk, gaze, speech)
    finally:
        tk.stop()

    print("\n\033[1;32m[DONE] Shopping demo complete.\033[0m\n")
