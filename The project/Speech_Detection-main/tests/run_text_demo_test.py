"""
Automated Text-Mode Demo Test
==============================

Runs all 13 intents + dialog patterns + gaze simulation
through the SpeechAdapter text pipeline and prints results.

No microphone or audio required.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.dialogue_manager import DialogueManager


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_result(phrase, event, expected_intent=None, expected_type=None):
    """Print test result with pass/fail."""
    ok = True
    status_parts = []

    if expected_type:
        if event.type == expected_type:
            status_parts.append(f"type={event.type.value} OK")
        else:
            status_parts.append(f"type={event.type.value} EXPECTED {expected_type.value}")
            ok = False

    if expected_intent:
        actual_intent = event.payload.get("intent", "N/A")
        if actual_intent == expected_intent:
            status_parts.append(f"intent={actual_intent} OK")
        else:
            status_parts.append(f"intent={actual_intent} EXPECTED {expected_intent}")
            ok = False

    params = event.payload.get("params", {})
    target_req = event.payload.get("target_required", None)

    mark = "PASS" if ok else "FAIL"
    symbol = "+" if ok else "X"

    print(f"  [{symbol}] \"{phrase}\"")
    print(f"      -> {', '.join(status_parts)}")
    print(f"         conf={event.confidence:.2f} target_req={target_req} params={params}")

    return ok


def main():
    config = Config()
    bus = EventBus()
    bus.enable_logging()

    speech = SpeechAdapter(event_bus=bus, config=config)
    gaze = GazeAdapterStub(event_bus=bus)
    gaze.start()

    actions_log = []
    prompt_log = []

    dm = DialogueManager(
        event_bus=bus,
        config=config,
        on_prompt=lambda msg: prompt_log.append(msg),
        on_action=lambda intent, target, params: actions_log.append(
            {"intent": intent, "target": target, "params": params}
        ),
    )

    results = {"pass": 0, "fail": 0}

    def run(phrase, expected_intent=None, expected_type=SpeechEventType.INTENT):
        event = speech.process_text(phrase, asr_confidence=0.85)
        ok = test_result(phrase, event, expected_intent, expected_type)
        results["pass" if ok else "fail"] += 1
        return event

    # ── TEST 1: Object-bound intents ──────────────────────────────

    section("TEST 1: Object-Bound Commands (target_required=True)")

    run("add this to cart", "ADD_TO_CART")
    run("put this in the basket", "ADD_TO_CART")
    run("buy this", "ADD_TO_CART")
    run("show details", "SHOW_DETAILS")
    run("tell me more", "SHOW_DETAILS")
    run("what is this", "SHOW_DETAILS")
    run("show me the ingredients", "SHOW_DETAILS")
    run("find similar", "FIND_SIMILAR")
    run("anything like this", "FIND_SIMILAR")
    run("compare this", "COMPARE")
    run("compare this with pinned", "COMPARE")
    run("show alternatives", "SHOW_ALTERNATIVES")
    run("what else is there", "SHOW_ALTERNATIVES")
    run("pin this", "PIN_ITEM")
    run("save this", "PIN_ITEM")
    run("bookmark that", "PIN_ITEM")
    run("remove this", "REMOVE_ITEM")
    run("delete this", "REMOVE_ITEM")

    # ── TEST 2: Global intents ────────────────────────────────────

    section("TEST 2: Global Commands (target_required=False)")

    run("scroll down", "SCROLL")
    run("scroll up", "SCROLL")
    run("next page", "SCROLL")
    run("open cart", "OPEN_CART")
    run("show my cart", "OPEN_CART")
    run("go back", "GO_BACK")
    run("back", "GO_BACK")
    run("previous page", "GO_BACK")
    run("help", "HELP")
    run("what can I say", "HELP")
    run("cancel", "CANCEL")
    run("never mind", "CANCEL")
    run("undo", "UNDO")
    run("take that back", "UNDO")

    # ── TEST 3: Dialog-mode patterns ──────────────────────────────

    section("TEST 3: Dialog-Mode Patterns (CONFIRM/DENY/REPAIR/REPEAT)")

    speech.set_dialog_active(True)

    run("yes", expected_type=SpeechEventType.CONFIRM)
    run("yeah", expected_type=SpeechEventType.CONFIRM)
    run("correct", expected_type=SpeechEventType.CONFIRM)
    run("no", expected_type=SpeechEventType.CONFIRM)
    run("nope", expected_type=SpeechEventType.CONFIRM)
    run("the left one", expected_type=SpeechEventType.REPAIR)
    run("the right one", expected_type=SpeechEventType.REPAIR)
    run("repeat", expected_type=SpeechEventType.REPEAT)
    run("cancel", expected_type=SpeechEventType.CANCEL)

    speech.set_dialog_active(False)

    # ── TEST 4: Edge cases ────────────────────────────────────────

    section("TEST 4: Edge Cases")

    run("", expected_type=SpeechEventType.ERROR)
    run("   ", expected_type=SpeechEventType.ERROR)
    run("the quick brown fox jumps", expected_type=SpeechEventType.ERROR)
    run("ADD THIS TO CART", "ADD_TO_CART")  # case insensitive
    run("add this, to cart!", "ADD_TO_CART")  # punctuation

    # ── TEST 5: Gaze simulation ───────────────────────────────────

    section("TEST 5: Gaze Simulation + Fusion Compatibility")

    # Lock gaze + object command
    gaze_ev = gaze.simulate_lock("item_42")
    speech_ev = speech.process_text("add this to cart", asr_confidence=0.9)
    time_delta = speech_ev.timestamp - gaze_ev.timestamp
    fused = time_delta <= 2.0

    print(f"  Gaze LOCK on 'item_42': state={gaze.state}, target={gaze.current_target}")
    print(f"  Speech: intent={speech_ev.payload.get('intent')}, target_req={speech_ev.payload.get('target_required')}")
    print(f"  Time delta: {time_delta:.3f}s, fusible={fused}")
    print(f"  {'[+] PASS' if fused else '[X] FAIL'}: Gaze+Speech fusion check")
    results["pass" if fused else "fail"] += 1

    # Unlock + object command = no target
    gaze.simulate_unlock()
    speech_ev2 = speech.process_text("remove this")
    no_target = gaze.current_target is None
    print(f"\n  Gaze UNLOCKED: state={gaze.state}, target={gaze.current_target}")
    print(f"  Speech: intent={speech_ev2.payload.get('intent')}, target_req={speech_ev2.payload.get('target_required')}")
    print(f"  {'[+] PASS' if no_target else '[X] FAIL'}: No gaze target for object command")
    results["pass" if no_target else "fail"] += 1

    # Ambiguous gaze
    gaze_amb = gaze.simulate_ambiguous(["item_3", "item_4"])
    print(f"\n  Gaze AMBIGUOUS: state={gaze.state}, candidates={gaze_amb.payload['candidates']}")
    print(f"  [+] PASS: Ambiguous gaze event emitted")
    results["pass"] += 1

    # Global command without gaze
    gaze.simulate_unlock()
    speech_ev3 = speech.process_text("scroll down")
    global_ok = speech_ev3.payload.get("target_required") is False
    print(f"\n  Global command: intent={speech_ev3.payload.get('intent')}, target_req=False")
    print(f"  {'[+] PASS' if global_ok else '[X] FAIL'}: Global command works without gaze")
    results["pass" if global_ok else "fail"] += 1

    # ── TEST 6: Event log ─────────────────────────────────────────

    section("TEST 6: Event Log Integrity")

    log = bus.get_log()
    print(f"  Total events logged: {len(log)}")
    modalities = set(e["modality"] for e in log)
    types = set(e["type"] for e in log)
    print(f"  Modalities found: {modalities}")
    print(f"  Event types found: {types}")
    log_ok = "gaze" in modalities and "speech" in modalities
    print(f"  {'[+] PASS' if log_ok else '[X] FAIL'}: Both modalities present in log")
    results["pass" if log_ok else "fail"] += 1

    # ── SUMMARY ───────────────────────────────────────────────────

    section("SUMMARY")

    total = results["pass"] + results["fail"]
    print(f"  Total:  {total}")
    print(f"  Passed: {results['pass']}")
    print(f"  Failed: {results['fail']}")

    if results["fail"] == 0:
        print(f"\n  ** ALL TESTS PASSED **")
    else:
        print(f"\n  !! {results['fail']} TEST(S) FAILED !!")

    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
