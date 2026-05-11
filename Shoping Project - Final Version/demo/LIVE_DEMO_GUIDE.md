# MMUI Toolkit — Live CV Demo Quick-Start Guide

Two OpenCV-based scripts that use **real webcam + real microphone** input.

For the full command reference, calibration guide, and troubleshooting, see
`demo/README.md`.

---

## Architecture

```
Webcam  ──► MediaPipe FaceMesh ──► RealGazeAdapter ──► GazeEvent (LOCK/UNLOCK/AMBIGUOUS)
                                                              │
Mic ─────► sounddevice (PTT) ──► ASR (Vosk/Whisper) ──► SpeechEvent (INTENT/…)
                                                              │
                                               EventBus ──────┤
                                                              │
                                         FusionEngine + DialogueManager
                                                              │
                                              MultimodalCommandEvent
                                                              │
                                              OpenCV Overlay (real-time)
```

---

## Prerequisites

```bash
python -m pip install -r requirements.txt
python -m pip install opencv-python mediapipe
```

The Vosk model is included in the repository at:

```
speech/models/vosk-model-small-en-us-0.15/
```

No manual download is needed if the repository is complete.

---

## Demo #1 — Shopping (`live_shopping_cv.py`)

```bash
# from project root
python demo/live_shopping_cv.py
```

- **12 products** in a 4×3 grid
- **Condition B:** look at a card for **1.2 s** → `LOCKED` → press **M**,
  speak a command, press **M** again to send
- **Condition A:** press **A** to switch to mouse-only; click the action
  buttons on each card
- Key commands: *"add to cart"*, *"add"*, *"cart"*, *"show details"*,
  *"find similar"*, *"compare"*, …

---

## Demo #2 — Museum Kiosk (`live_kiosk_cv.py`)

```bash
python demo/live_kiosk_cv.py
```

- **6 exhibits** in a 3×2 grid
- Same gaze + dwell + speech pipeline as the shopping demo
- Key commands: *"tell me about this"*, *"tell"*, *"about"*, *"this"*,
  *"bookmark"*, *"book"*, *"pin"*, *"save"*, *"summarize"*,
  *"next"*, *"go back"*, …

---

## Keys (both demos)

| Key | Action |
|-----|--------|
| **A** | Condition A — mouse-only baseline (gaze paused, click buttons shown) |
| **B** | Condition B — gaze + speech multimodal (default) |
| **M** | Push-to-talk toggle — press once to start recording, press again to send |
| **U** | Start or restart gaze calibration (Condition B only) |
| **SPACE** | Capture the current calibration point |
| **Q** | Quit |

---

## Overlay Elements

| Element | Description |
|---------|-------------|
| Gaze dot / crosshair | Mapped gaze position on screen |
| Dwell bar (bottom of card) | Progress toward the 1.2-second lock |
| `LOCKED` badge | Gaze lock confirmed |
| Fusion state badge (top-right) | `IDLE / LOCKED / NEEDS_TARGET / DISAMBIG / CONFIRM / COMMAND` |
| Red circle (mic indicator) | Active while PTT recording |
| Mode badge (below header) | `MODE: A Mouse-only` or `MODE: B Gaze+Speech` |
| Status bar (bottom) | Target, intent, transcript, confidence, last command |
| Event log (bottom-right) | Last five events |

---

## Switching to Whisper ASR

Edit the `cfg = Config(…)` block in the demo file:

```python
cfg = Config(
    ASR_ENGINE         = "whisper",
    WHISPER_MODEL_SIZE = "small",   # tiny / base / small / medium / large
    ...
)
```

Requires `pip install openai-whisper`.  Whisper provides higher accuracy
but is slower on CPU.
