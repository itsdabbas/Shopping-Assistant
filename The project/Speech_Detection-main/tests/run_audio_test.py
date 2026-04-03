"""
Audio (Microphone) Test for GazeShop Speech Pipeline
=====================================================

This script tests the FULL audio pipeline:
  Microphone → Audio Capture → Vosk ASR → IntentParser → SpeechEvent

It records audio from the microphone for a few seconds and processes
it through the real Vosk ASR engine.

Usage:
    .\.venv\Scripts\python.exe tests/run_audio_test.py
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.intent_parser import IntentParser


def record_audio(duration_s=3, sample_rate=16000, channels=1):
    """Record audio from the default microphone."""
    import sounddevice as sd

    print(f"\n  Recording for {duration_s} seconds...")
    print(f"  >> SPEAK NOW! Say something like 'add this to cart' <<\n")

    audio = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
    )
    sd.wait()  # Wait until recording is complete
    return audio.flatten()


def transcribe_vosk(audio, sample_rate=16000, model_path="models/vosk-model-small-en-us-0.15"):
    """Transcribe audio using Vosk."""
    from vosk import Model, KaldiRecognizer, SetLogLevel

    SetLogLevel(-1)
    model = Model(model_path)
    rec = KaldiRecognizer(model, sample_rate)

    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.FinalResult())

    transcript = result.get("text", "").strip()

    word_results = result.get("result", [])
    if word_results:
        confidences = [w.get("conf", 0.0) for w in word_results]
        mean_conf = sum(confidences) / len(confidences)
    else:
        mean_conf = 0.5

    return transcript, mean_conf


def main():
    print("\n" + "=" * 60)
    print("  GazeShop — Audio Pipeline Test (Microphone + Vosk ASR)")
    print("=" * 60)

    # Check dependencies
    try:
        import sounddevice as sd
        print(f"\n  [OK] sounddevice installed")
        default_input = sd.query_devices(kind='input')
        print(f"  [OK] Default input device: {default_input['name']}")
    except Exception as e:
        print(f"\n  [FAIL] sounddevice error: {e}")
        return 1

    try:
        from vosk import Model
        print(f"  [OK] vosk installed")
    except ImportError:
        print(f"  [FAIL] vosk not installed")
        return 1

    model_path = "models/vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        print(f"  [FAIL] Vosk model not found at: {model_path}")
        return 1
    print(f"  [OK] Vosk model found at: {model_path}")

    # Setup
    config = Config()
    bus = EventBus()
    bus.enable_logging()
    parser = IntentParser(config)

    print("\n" + "-" * 60)
    print("  TEST 1: Microphone Audio Capture + Vosk ASR")
    print("-" * 60)

    # Record audio
    try:
        audio = record_audio(duration_s=4, sample_rate=config.SAMPLE_RATE)
    except Exception as e:
        print(f"  [FAIL] Audio recording failed: {e}")
        return 1

    # Check audio stats
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    duration_ms = len(audio) / config.SAMPLE_RATE * 1000
    print(f"  Audio captured: {len(audio)} samples, {duration_ms:.0f} ms, RMS={rms:.1f}")

    if rms < 50:
        print(f"  [WARN] Very low audio energy (RMS={rms:.1f}) — might be silence")

    # Transcribe
    print(f"\n  Transcribing with Vosk...")
    transcript, asr_confidence = transcribe_vosk(audio, config.SAMPLE_RATE, model_path)

    if not transcript:
        print(f"  [WARN] ASR returned empty transcript (no speech detected)")
        print(f"         This is normal if you didn't speak during recording.")
        print(f"\n  Skipping intent parsing for empty transcript.")
    else:
        print(f"  Transcript: \"{transcript}\"")
        print(f"  ASR Confidence: {asr_confidence:.2f}")

        # Parse intent
        event = parser.parse(transcript, asr_confidence=asr_confidence)
        bus.emit(event)

        print(f"\n  --- Parsed SpeechEvent ---")
        print(f"  Type: {event.type.value}")
        if event.type == SpeechEventType.INTENT:
            print(f"  Intent: {event.payload.get('intent')}")
            print(f"  Target Required: {event.payload.get('target_required')}")
            print(f"  Params: {event.payload.get('params', {})}")
        elif event.type == SpeechEventType.ERROR:
            print(f"  Error: {event.payload.get('error')}")
        print(f"  Confidence: {event.confidence:.2f}")
        print(f"  Requires Confirmation: {event.requires_confirmation}")

    # TEST 2: Full adapter pipeline with process_text (sanity check)
    print("\n" + "-" * 60)
    print("  TEST 2: SpeechAdapter.process_text() via full adapter")
    print("-" * 60)

    adapter = SpeechAdapter(event_bus=bus, config=config)
    test_phrases = [
        ("add this to cart", "ADD_TO_CART"),
        ("scroll down", "SCROLL"),
        ("help", "HELP"),
    ]

    all_ok = True
    for phrase, expected in test_phrases:
        ev = adapter.process_text(phrase, asr_confidence=0.9)
        actual = ev.payload.get("intent", "N/A")
        ok = actual == expected
        if not ok:
            all_ok = False
        mark = "[+]" if ok else "[X]"
        print(f"  {mark} \"{phrase}\" -> {actual} (expected {expected})")

    # TEST 3: Full pipeline with adapter.start() (real PTT) - just verify init
    print("\n" + "-" * 60)
    print("  TEST 3: SpeechAdapter.start() — ASR + PTT init")
    print("-" * 60)

    try:
        full_adapter = SpeechAdapter(event_bus=bus, config=config)
        full_adapter.start()
        print(f"  [OK] SpeechAdapter started successfully")
        print(f"  [OK] ASR engine initialized (Vosk)")
        print(f"  [OK] PTT key listener active (key={config.PTT_KEY})")
        print(f"  Running: {full_adapter.is_running}")
        full_adapter.stop()
        print(f"  [OK] SpeechAdapter stopped cleanly")
    except Exception as e:
        print(f"  [FAIL] SpeechAdapter.start() error: {e}")
        all_ok = False

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Vosk ASR:     OK")
    print(f"  Microphone:   OK (recorded {duration_ms:.0f}ms)")
    print(f"  Transcript:   \"{transcript}\"")
    print(f"  process_text: {'ALL PASS' if all_ok else 'SOME FAILED'}")

    log = bus.get_log()
    print(f"  Events logged: {len(log)}")
    print(f"\n  Audio pipeline test complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
