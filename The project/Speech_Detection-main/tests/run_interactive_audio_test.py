"""
Interactive Audio Test — Say a Command, See the Intent
=======================================================

This script records your voice, transcribes it via Vosk ASR,
and runs it through the IntentParser pipeline.

It loops 3 times so you can test different phrases:
  "add this to cart", "scroll down", "show details", etc.

Usage:
    .\.venv\Scripts\python.exe tests/run_interactive_audio_test.py
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.intent_parser import IntentParser


def record(duration_s, sample_rate):
    audio = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    # Show countdown
    for i in range(duration_s, 0, -1):
        print(f"    {i}...", end=" ", flush=True)
        time.sleep(1)
    sd.wait()
    print("DONE")
    return audio.flatten()


def transcribe(rec, audio):
    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.FinalResult())
    text = result.get("text", "").strip()
    words = result.get("result", [])
    conf = sum(w.get("conf", 0.0) for w in words) / len(words) if words else 0.5
    return text, conf


def main():
    SAMPLE_RATE = 16000
    MODEL_PATH = "models/vosk-model-small-en-us-0.15"
    RECORD_SECONDS = 4
    NUM_ROUNDS = 3

    print("\n" + "=" * 60)
    print("  GazeShop - Interactive Audio Test")
    print("=" * 60)

    # Init
    SetLogLevel(-1)
    model = Model(MODEL_PATH)
    config = Config()
    bus = EventBus()
    parser = IntentParser(config)

    suggestions = [
        "add this to cart",
        "scroll down",
        "show details",
        "help",
        "compare this",
        "open cart",
    ]

    results = []

    for i in range(NUM_ROUNDS):
        suggestion = suggestions[i % len(suggestions)]
        print(f"\n{'─' * 60}")
        print(f"  Round {i+1}/{NUM_ROUNDS}")
        print(f"  Suggestion: try saying \"{suggestion}\"")
        print(f"  (You have {RECORD_SECONDS} seconds)")
        print(f"{'─' * 60}")

        input("  Press ENTER when ready to record...")

        print(f"\n  🎤 RECORDING — SPEAK NOW!")
        audio = record(RECORD_SECONDS, SAMPLE_RATE)

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        print(f"  Audio RMS energy: {rms:.1f}")

        if rms < 50:
            print(f"  ⚠ Very low energy — probably silence")

        # Create fresh recognizer each time
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        transcript, asr_conf = transcribe(rec, audio)

        print(f"\n  📝 Transcript: \"{transcript}\"")
        print(f"     ASR Confidence: {asr_conf:.2f}")

        if transcript:
            event = parser.parse(transcript, asr_confidence=asr_conf)
            bus.emit(event)

            print(f"\n  🔍 Parsed Event:")
            print(f"     Type: {event.type.value}")
            if event.type == SpeechEventType.INTENT:
                intent = event.payload.get("intent")
                target_req = event.payload.get("target_required")
                params = event.payload.get("params", {})
                print(f"     Intent: {intent}")
                print(f"     Target Required: {target_req}")
                if params:
                    print(f"     Params: {params}")
                print(f"     Confidence: {event.confidence:.2f}")
                print(f"     Needs Confirmation: {event.requires_confirmation}")
                results.append(("PASS", transcript, intent))
            elif event.type == SpeechEventType.ERROR:
                print(f"     Error: {event.payload.get('error')}")
                results.append(("NO_MATCH", transcript, "N/A"))
        else:
            print(f"  ⚠ No speech detected")
            results.append(("SILENCE", "", "N/A"))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for idx, (status, transcript, intent) in enumerate(results):
        print(f"  Round {idx+1}: [{status}] \"{transcript}\" -> {intent}")

    passed = sum(1 for s, _, _ in results if s == "PASS")
    print(f"\n  Recognized: {passed}/{NUM_ROUNDS}")
    print(f"  Audio pipeline test complete!\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
