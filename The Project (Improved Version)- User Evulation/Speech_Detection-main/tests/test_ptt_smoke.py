"""
Toggle PTT Smoke Test
=====================
Drives begin_listening() / end_listening() programmatically (no real mic, no
real ASR) three times in a row.  Asserts:

  1. State returns to IDLE after every cycle.
  2. A STOPPED event is emitted after every cycle.
  3. The 2nd and 3rd begin_listening() calls succeed (state was IDLE) ->
     no freeze.

Run from repo root:
    python tests/test_ptt_smoke.py
    python -m pytest tests/test_ptt_smoke.py -v
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter, PipelineState

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_adapter() -> tuple[SpeechAdapter, list[SpeechEvent]]:
    """Return a SpeechAdapter with ASR and sounddevice fully stubbed out."""
    cfg = Config(
        ASR_ENGINE       = "vosk",
        VOSK_MODEL_PATH  = str(ROOT / "models" / "vosk-model-small-en-us-0.15"),
        PTT_KEY          = "",       # no pynput in test
        PTT_MODE         = "toggle",
        TAIL_SILENCE_MS  = 0,        # no sleep
        MIN_UTTERANCE_MS = 0,        # accept any length
        ENERGY_THRESHOLD = 0,        # bypass energy gate
        VAD_ENABLED      = False,
        SAMPLE_RATE      = 16000,
    )
    bus    = EventBus()
    events: list[SpeechEvent] = []
    bus.subscribe("SpeechEvent", events.append)

    adapter = SpeechAdapter(event_bus=bus, config=cfg)

    # ── Stub: bypass real ASR ────────────────────────────────────────────────
    def _fake_transcribe(audio: np.ndarray) -> tuple[str, float]:
        print(f"  [STUB ASR] {len(audio)} samples")
        return "add to cart", 0.95

    adapter._transcribe   = _fake_transcribe   # type: ignore[method-assign]
    adapter._asr_initialised = True            # skip _init_asr()

    # ── Stub: bypass real sounddevice ────────────────────────────────────────
    def _fake_start_audio_stream():
        # Inject 0.5 s of silent audio into the buffer
        adapter._audio_buffer.append(np.zeros(8000, dtype=np.int16))
        print("  [STUB SD] stream started, 8000 samples injected")

    def _fake_force_stop_stream():
        print("  [STUB SD] stream stopped")

    adapter._start_audio_stream = _fake_start_audio_stream   # type: ignore
    adapter._force_stop_stream  = _fake_force_stop_stream    # type: ignore

    return adapter, events


# ── test ──────────────────────────────────────────────────────────────────────

def test_toggle_ptt_three_cycles():
    """3 ON/OFF toggle cycles must all complete with IDLE + STOPPED emitted."""
    adapter, events = _make_adapter()
    adapter.start()

    try:
        for cycle in range(1, 4):
            print(f"\n=== Cycle #{cycle} ===")

            # --- must be IDLE before begin_listening ---
            assert adapter.pipeline_state == PipelineState.IDLE, (
                f"Cycle #{cycle}: expected IDLE before begin_listening, "
                f"got {adapter.pipeline_state}"
            )

            # --- Toggle ON ---
            adapter.begin_listening()
            assert adapter.pipeline_state == PipelineState.LISTENING, (
                f"Cycle #{cycle}: expected LISTENING after begin_listening"
            )
            print(f"  -> LISTENING confirmed")

            # Simulate user speaking for a moment
            time.sleep(0.05)

            # --- Toggle OFF ---
            adapter.end_listening()

            # Wait up to 20 s for worker to finish and return to IDLE
            deadline = time.time() + 20.0
            while adapter.pipeline_state != PipelineState.IDLE:
                assert time.time() < deadline, (
                    f"Cycle #{cycle}: pipeline did not return to IDLE within 20 s "
                    f"(state={adapter.pipeline_state})"
                )
                time.sleep(0.05)

            print(f"  -> IDLE confirmed after cycle {cycle}")

            # --- verify STOPPED event was emitted this cycle ---
            stopped_count = sum(
                1 for e in events if e.type == SpeechEventType.STOPPED
            )
            assert stopped_count >= cycle, (
                f"Cycle #{cycle}: expected at least {cycle} STOPPED event(s), "
                f"got {stopped_count}"
            )
            print(f"  -> STOPPED events total: {stopped_count}")

    finally:
        adapter.stop()

    print("\nAll 3 toggle cycles completed without freezing.")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    test_toggle_ptt_three_cycles()
