"""
benchmark_speech.py
===================
Standalone benchmark for the GazeShop speech recognition pipeline.

Usage (run from Speech_Detection-main/):
    python benchmark_speech.py

Expects .wav files under benchmark_clips/ named:
    INTENT_TAG_NO.wav   e.g. ADD_CART_01.wav, SHOW_DETAIL_02.wav

The numeric suffix (_01, _02, …) is stripped to derive the expected intent:
    ADD_CART_01.wav     → expected ADD_CART
    SHOW_DETAIL_02.wav  → expected SHOW_DETAIL
    FIND_SIMILAR_03.wav → expected FIND_SIMILAR

Writes results to benchmark_results.csv and prints overall accuracy.
"""

from __future__ import annotations

import csv
import sys
import wave
from pathlib import Path

import numpy as np

# Allow running directly from any working directory
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus

CLIPS_DIR = _HERE / "benchmark_clips"
RESULTS_CSV = _HERE / "benchmark_results.csv"


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (int16 samples, sample_rate) from a mono or stereo .wav file."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        # Take the left channel only
        audio = audio.reshape(-1, n_channels)[:, 0].copy()
    return audio, sample_rate


def expected_intent(stem: str) -> str:
    """Drop the trailing numeric suffix and rejoin with underscores.

    ADD_CART_01      → ADD_CART
    SHOW_DETAIL_02   → SHOW_DETAIL
    FIND_SIMILAR_03  → FIND_SIMILAR
    NAV_BACK_10      → NAV_BACK
    """
    parts = stem.split("_")
    return "_".join(parts[:-1])


def main() -> None:
    wav_files = sorted(CLIPS_DIR.glob("*.wav"))
    n_total = len(wav_files)

    if n_total == 0:
        print(f"No .wav files found under {CLIPS_DIR}")
        return

    # Build pipeline — use absolute model path so the script works from any cwd
    config = Config()
    config.VOSK_MODEL_PATH = str(_HERE / config.VOSK_MODEL_PATH)
    event_bus = EventBus()
    adapter = SpeechAdapter(event_bus, config)
    adapter._init_asr()

    rows: list[dict] = []
    n_correct = 0

    for wav_path in wav_files:
        stem = wav_path.stem
        exp = expected_intent(stem)

        transcript = ""
        asr_conf = 0.0
        try:
            audio, _ = load_wav(wav_path)
            transcript, asr_conf = adapter._transcribe_vosk(audio)
        except Exception as exc:
            print(f"  [WARN] {wav_path.name}: transcription failed — {exc}")

        event = adapter.parser.parse(transcript, asr_confidence=asr_conf)
        intent = event.payload.get("intent", "UNKNOWN")

        if intent == exp:
            n_correct += 1

        rows.append({
            "file": wav_path.name,
            "transcript": transcript,
            "intent": intent,
            "confidence": round(event.confidence, 4),
            "expected": exp,
        })

    # Write CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "transcript", "intent", "confidence", "expected"]
        )
        writer.writeheader()
        writer.writerows(rows)

    pct = round(n_correct / n_total * 100)
    print(f"Accuracy: {n_correct}/{n_total} = {pct}%")


if __name__ == "__main__":
    main()
