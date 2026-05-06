"""
Speech Module Interactive Demo
==============================

A command-line demo that lets you type text as if it were speech
and see how the Speech module processes it.

No microphone, ASR engine, or gaze hardware required.

Usage:
    python demo/speech_demo.py

Type phrases like "add this to cart", "scroll down", "help" etc.
Type "quit" or "exit" to stop.
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, GazeEvent
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.dialogue_manager import DialogueManager


# ── Color helpers for terminal output ──────────────────────────────

def _green(text):  return f"\033[92m{text}\033[0m"
def _yellow(text): return f"\033[93m{text}\033[0m"
def _red(text):    return f"\033[91m{text}\033[0m"
def _cyan(text):   return f"\033[96m{text}\033[0m"
def _bold(text):   return f"\033[1m{text}\033[0m"
def _dim(text):    return f"\033[2m{text}\033[0m"


def print_banner():
    print()
    print(_bold("=" * 60))
    print(_bold("   GazeShop — Speech Module Interactive Demo"))
    print(_bold("=" * 60))
    print()
    print(_dim("This demo simulates the Speech Modality pipeline."))
    print(_dim("Type a phrase as if you spoke it. The system will:"))
    print(_dim("  1. Parse the intent (regex-based)"))
    print(_dim("  2. Extract parameters/slots"))
    print(_dim("  3. Compute confidence score"))
    print(_dim("  4. Emit a SpeechEvent on the EventBus"))
    print()
    print(_cyan("Available commands:"))
    print(f"  {_bold('Object-bound')} (need gaze target):")
    print(f"    add this to cart, show details, find similar,")
    print(f"    compare this, show alternatives, pin this, remove this")
    print(f"  {_bold('Global')} (no target needed):")
    print(f"    scroll down, open cart, go back, help, cancel, undo")
    print()
    print(_cyan("Gaze simulation commands:"))
    print(f"  {_bold(':lock <id>')}       — Simulate gaze lock on item")
    print(f"  {_bold(':unlock')}          — Simulate gaze unlock")
    print(f"  {_bold(':ambiguous <a> <b>')} — Simulate ambiguous gaze")
    print(f"  {_bold(':dialog on/off')}   — Toggle dialog context")
    print(f"  {_bold(':log')}             — Show event log")
    print(f"  {_bold(':status')}          — Show current system state")
    print(f"  {_bold('quit / exit')}      — Exit")
    print()
    print(_bold("-" * 60))


def print_event(event):
    """Pretty-print a SpeechEvent."""
    if event.type.value == "INTENT":
        intent = event.payload.get("intent", "?")
        params = event.payload.get("params", {})
        target_req = event.payload.get("target_required", False)

        color = _green if not event.requires_confirmation else _yellow
        print()
        print(color(f"  ✓ Intent: {intent}"))
        print(f"    Target required: {target_req}")
        if params:
            print(f"    Parameters: {params}")
        print(f"    Confidence: {event.confidence:.2f}")
        if event.requires_confirmation:
            print(_yellow(f"    ⚠ Low confidence — confirmation would be required"))

    elif event.type.value == "CONFIRM":
        confirmed = event.payload.get("confirm", False)
        symbol = "✓ YES" if confirmed else "✗ NO"
        color = _green if confirmed else _red
        print(color(f"  {symbol} (confirmation response)"))

    elif event.type.value == "REPAIR":
        target = event.payload.get("repair_target", "?")
        print(_cyan(f"  🔧 Repair: target = \"{target}\""))

    elif event.type.value == "REPEAT":
        print(_cyan(f"  🔁 Repeat requested"))

    elif event.type.value == "CANCEL":
        print(_yellow(f"  ✗ Cancel"))

    elif event.type.value == "UNDO":
        print(_yellow(f"  ↩ Undo"))

    elif event.type.value == "ERROR":
        error = event.payload.get("error", "Unknown error")
        print(_red(f"  ✗ Error: {error}"))

    else:
        print(f"  Event: {event.type.value}")


def main():
    print_banner()

    # Setup
    config = Config()
    bus = EventBus()
    bus.enable_logging()

    speech = SpeechAdapter(event_bus=bus, config=config)
    gaze = GazeAdapterStub(event_bus=bus)
    gaze.start()

    actions_log = []

    dm = DialogueManager(
        event_bus=bus,
        config=config,
        on_prompt=lambda msg: print(_cyan(f"\n  💬 System: {msg}")),
        on_action=lambda intent, target, params: actions_log.append(
            {"intent": intent, "target": target, "params": params}
        ),
    )

    while True:
        try:
            user_input = input(f"\n{_bold('You')} 🎤 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(_dim("Goodbye!"))
            break

        # ── Meta commands ───────────────────────────────────────────

        if user_input.startswith(":lock"):
            parts = user_input.split()
            target_id = parts[1] if len(parts) > 1 else "item_1"
            gaze.simulate_lock(target_id)
            print(_green(f"  👁 Gaze LOCKED on: {target_id}"))
            continue

        if user_input == ":unlock":
            gaze.simulate_unlock()
            print(_yellow(f"  👁 Gaze UNLOCKED"))
            continue

        if user_input.startswith(":ambiguous"):
            parts = user_input.split()
            candidates = parts[1:] if len(parts) > 1 else ["item_1", "item_2"]
            gaze.simulate_ambiguous(candidates)
            print(_yellow(f"  👁 Gaze AMBIGUOUS: {candidates}"))
            continue

        if user_input.startswith(":dialog"):
            parts = user_input.split()
            on = parts[1].lower() in ("on", "true", "1") if len(parts) > 1 else True
            speech.set_dialog_active(on)
            status = "ON" if on else "OFF"
            print(_cyan(f"  Dialog context: {status}"))
            continue

        if user_input == ":log":
            log = bus.get_log()
            print(_bold(f"\n  Event Log ({len(log)} events):"))
            for i, entry in enumerate(log):
                print(f"    [{i}] {entry['modality']}.{entry['type']} "
                      f"t={entry['timestamp']:.2f}")
            continue

        if user_input == ":status":
            print(_bold("\n  System Status:"))
            print(f"    Gaze state: {gaze.state}")
            print(f"    Gaze target: {gaze.current_target}")
            print(f"    Dialog active: {dm.is_active}")
            print(f"    Events logged: {len(bus.get_log())}")
            print(f"    Actions executed: {len(actions_log)}")
            continue

        # ── Process as speech input ─────────────────────────────────
        event = speech.process_text(user_input, asr_confidence=0.85)
        print_event(event)

        # Show fusion hint
        if (event.type.value == "INTENT"
                and event.payload.get("target_required")):
            if gaze.state == "locked":
                print(_dim(f"    → Would fuse with gaze target: "
                          f"{gaze.current_target}"))
            elif gaze.state == "ambiguous":
                print(_dim(f"    → Gaze ambiguous — disambiguation needed"))
            else:
                print(_dim(f"    → No gaze target — would prompt user"))


if __name__ == "__main__":
    main()
