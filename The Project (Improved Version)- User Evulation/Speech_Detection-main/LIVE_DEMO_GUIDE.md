# MMUI Toolkit — Live CV Demo Guide

Two OpenCV-based scripts that use **real camera + real microphone** input — no
simulation buttons.

---

## Architecture

```
Webcam  ──► MediaPipe FaceMesh ──► RealGazeAdapter ──► GazeEvent (LOCK/UNLOCK/AMBIGUOUS)
                                                              │
Mic ─────► sounddevice (PTT) ──► ASR (Vosk/Whisper) ──► SpeechEvent (INTENT/REPAIR/…)
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

```powershell
pip install opencv-python mediapipe sounddevice numpy vosk
# OR use Whisper instead:
pip install openai-whisper
```

Download the Vosk model (if using Vosk):
```powershell
# from repo root:
Invoke-WebRequest https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip `
    -OutFile models\vosk-model-small-en-us-0.15.zip
Expand-Archive models\vosk-model-small-en-us-0.15.zip -DestinationPath models\
```

---

## Demo #1 — Shopping

```powershell
# from repo root
python demo/live_shopping_cv.py
```

- **12 products** in a 4×3 grid
- Look at a card for **1.2 s** → LOCK
- **Hold M** to speak → release to process
- Commands: *"add this to cart"*, *"show details"*, *"find similar"*, *"compare"*, …

---

## Demo #2 — Museum Kiosk

```powershell
python demo/live_kiosk_cv.py
```

- **6 exhibits** in a 3×2 grid
- Same gaze + dwell + speech pipeline
- Commands: *"tell me about this"*, *"summarize"*, *"zoom in"*, *"bookmark this"*,
  *"next"*, *"go back"*, …

---

## Keys (both demos)

| Key | Action |
|-----|--------|
| **M** (hold) | Push-to-Talk — hold while speaking, release to process |
| **Q** | Quit |

---

## Overlay elements

| Element | Description |
|---------|-------------|
| Gaze dot / crosshair | Real-time nose-tip position from MediaPipe |
| Dwell bar (bottom of card) | Progress toward 1.2 s lock |
| `LOCKED` badge | Item successfully locked |
| Fusion state badge (top-right) | `IDLE / LOCKED / NEEDS_TARGET / DISAMBIG / CONFIRM / COMMAND` |
| Red circle (mic) | Turns red while recording |
| Status bar (bottom) | Target, Intent, Transcript, Confidence, last CMD |
| Event log (bottom-right) | Last 5 events |

---

## Switch to Whisper ASR

Edit the `cfg = Config(...)` block at the bottom of either script:

```python
cfg = Config(
    ASR_ENGINE         = "whisper",
    WHISPER_MODEL_SIZE = "small",   # tiny / base / small / medium
    ...
)
```

---

## Simulated demos (Streamlit, button-driven)

These remain available as before:

```powershell
streamlit run demo/live_shopping_app.py
streamlit run demo/live_kiosk_app.py
```
