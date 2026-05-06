# GazeShop — Live Demo Guide

This guide explains how to run the two live demos included with the GazeShop multimodal interaction toolkit.

---

## Demo Overview

The live demos show **gaze + speech multimodal interaction** in real time.  
A webcam tracks where the user is looking (eye-relative iris position via MediaPipe).  
A microphone captures push-to-talk speech commands.  
The toolkit fuses gaze target selection with the spoken intent to perform actions.

Two demo scenarios are provided:
- **Shopping** — browse and interact with a product grid
- **Museum Kiosk** — explore exhibit information cards

---

## Requirements

- Python 3.10 or later
- A working **webcam**
- A working **microphone**
- All Python packages listed in `requirements.txt`
- The **Vosk speech model** (required if using Vosk ASR, which is the default)

### Install Python packages

From the repository root:

```
python -m pip install -r requirements.txt
```

Additionally install OpenCV and MediaPipe (not in requirements.txt):

```
python -m pip install opencv-python mediapipe
```

### Install rapidfuzz (if missing)

```
python -m pip install rapidfuzz
```

### Vosk model

The default config expects the model at:

```
models/vosk-model-small-en-us-0.15/
```

If the folder is missing, download and unzip the model from https://alphacephei.com/vosk/models and place it at that path.

---

## How to Run the Demos

Run both commands from the **repository root** (the folder containing `gazeshop/` and `demo/`):

```
python demo/live_shopping_cv.py
python demo/live_kiosk_cv.py
```

---

## Live Shopping Demo

**Purpose:** Demonstrates gaze-guided product interaction in a simulated online shop.

- A 4×3 grid of 12 product cards is shown.
- The user **looks at a product card** for ~1.2 seconds to lock gaze on it.
- The user then **presses M** (push-to-talk), **speaks a command**, and **presses M again** to send it.
- Gaze target + spoken intent are **fused** by the toolkit to perform the action (e.g. add the looked-at product to cart).

---

## Live Kiosk Demo

**Purpose:** Demonstrates the same toolkit in a museum kiosk scenario.

- A 3×2 grid of 6 exhibit information cards is shown.
- The user **looks at an exhibit card** to select it.
- The user **speaks commands** such as show details, read summary, get help, or go back.
- The interaction pipeline (gaze + speech fusion, dialogue, confirmation) is identical to the shopping demo.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| **M** | Toggle push-to-talk — press once to start recording, press again to send |
| **U** | Start or restart gaze calibration |
| **SPACE** | Capture a calibration point (during calibration mode) |
| **Q** | Quit the demo |

---

## Gaze Calibration

Calibration maps eye movement to screen position. Run it once at the start of each session, or whenever gaze tracking feels inaccurate.

1. Start the demo.
2. Press **U** — calibration mode starts.
3. Keep your **head mostly still** throughout (move only your eyes).
4. A target dot appears on screen — **look directly at it**.
5. Press **SPACE** to capture that point. Hold your gaze steady while the progress ring fills.
6. Repeat for each of the 5 targets.
7. Calibration completes automatically after all 5 points are captured.
8. If gaze drifts later, press **U** again to recalibrate.

---

## Supported Speech Commands

The following commands are recognised by the current intent vocabulary.  
Press **M** to start recording, speak the command, press **M** again to process.

### Shopping demo

| Command | Action |
|---------|--------|
| `add to cart` | Add the locked item to cart |
| `add` / `cart` / `card` | Short form — also adds to cart |
| `to cart` / `to card` | Also recognised as add to cart |
| `two cart` / `two card` | Also recognised as add to cart |
| `show details` | Show details for the locked item |
| `find similar` | Find similar products |
| `compare items` | Compare the locked item |
| `go back` | Navigate back |
| `get help` | Show available commands |
| `yes` / `confirm` | Confirm a pending action |
| `no` / `cancel` | Cancel a pending action |

### Kiosk demo

| Command | Action |
|---------|--------|
| `show details` | Show exhibit details |
| `read summary` | Read a summary of the exhibit |
| `zoom in please` | Zoom in on the exhibit |
| `go back` | Navigate back |
| `next item` | Move to the next exhibit |
| `get help` | Show available commands |
| `yes` / `confirm` | Confirm a pending action |
| `no` / `cancel` | Cancel a pending action |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` for any package | Run `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError: rapidfuzz` | Run `python -m pip install rapidfuzz` |
| `ModuleNotFoundError: cv2` | Run `python -m pip install opencv-python` |
| `ModuleNotFoundError: mediapipe` | Run `python -m pip install mediapipe` |
| Camera does not open | Check that your webcam is connected and not in use by another app; check OS camera permissions |
| Microphone not working | Check microphone is set as the default input device in OS audio settings |
| Vosk model missing | Download `vosk-model-small-en-us-0.15` and place it under `models/` in the repo root |
| Speech not recognised | Speak clearly after pressing M; check microphone level; ensure the Vosk model path is correct |
| Gaze dot jumps or drifts | Press **U** to recalibrate; ensure your face is well-lit and fully visible to the camera |
| Gaze calibration rejected | Keep your head still and look directly at each target; retry with **U** |
