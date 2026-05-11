# demo/ — Live Demo Applications

This folder contains the two final runnable demonstrations of the GazeShop
multimodal interaction toolkit.

| File | Description |
|------|-------------|
| `live_shopping_cv.py` | Product shopping assistant — 12 items in a 4×3 grid |
| `live_kiosk_cv.py` | Museum kiosk — 6 exhibit cards in a 3×2 grid |
| `LIVE_DEMO_GUIDE.md` | Quick-start guide and architecture diagram |

Both demos use a **real webcam** for gaze tracking and a **real
microphone** for speech recognition.  No simulated input is required.

---

## Requirements

- Python 3.10 or later
- A working webcam
- A working microphone
- All packages from `requirements.txt` plus `opencv-python` and `mediapipe`
- The Vosk model at `speech/models/vosk-model-small-en-us-0.15/`

```bash
python -m pip install -r requirements.txt
python -m pip install opencv-python mediapipe
```

---

## Running the Demos

Run from the **project root** (the folder containing `README.md`):

```bash
python demo/live_shopping_cv.py
python demo/live_kiosk_cv.py
```

---

## Condition A and Condition B

Both demos support two interaction modes that can be switched at any time
by pressing a single key.

### Condition A — Mouse-only baseline (`A`)

- Gaze tracking is paused and the gaze dot is hidden.
- Each product or exhibit card displays four clickable action buttons.
- The user interacts entirely by clicking with the mouse.
- Speech is disabled in this mode.
- Intended as a controlled baseline for user-study comparison.

### Condition B — Gaze + speech multimodal (`B`)

- Gaze tracking is active.
- The user looks at a card for approximately 1.2 seconds to lock gaze on it.
  A progress bar at the bottom of the card shows dwell accumulation.
  The card is highlighted and shows a `LOCKED` badge when the lock is confirmed.
- The user presses **M** to begin recording, speaks a command, then presses
  **M** again to send the audio for recognition.
- The toolkit fuses the gaze-locked target with the spoken intent to perform
  the action.
- This is the full multimodal interaction mode.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `A` | Switch to Condition A (mouse-only baseline) |
| `B` | Switch to Condition B (gaze + speech multimodal) |
| `M` | Push-to-talk toggle — first press starts recording, second press sends |
| `U` | Start or restart gaze calibration (Condition B only) |
| `SPACE` | Capture the current calibration point |
| `Q` | Quit the demo |

---

## Gaze Calibration

Calibration maps the raw eye-relative iris feature to screen coordinates
using a 5-point affine transform.  Run it once at the beginning of each
session, or whenever gaze accuracy degrades.

1. Ensure Condition B is active (press **B** if needed).
2. Press **U** — calibration mode starts and the first target dot appears.
3. Keep your **head still** throughout the process (move only your eyes).
4. **Look directly at** the highlighted target dot.
5. Press **SPACE** — the system collects stable samples over approximately
   one second, then automatically advances to the next target.
6. Repeat for all five targets (corners + centre).
7. Calibration completes automatically.  A status banner confirms success.
8. Press **U** again at any time to recalibrate if gaze drifts.

---

## Shopping Demo — `live_shopping_cv.py`

**Scenario:** A simulated online shop with 12 product cards arranged in a
4×3 grid.

**Condition B interaction flow:**

1. Ensure Condition B is active.
2. Look at a product card — the card highlights as gaze hovers, and a
   progress bar fills toward the 1.2-second dwell lock.
3. When the card shows `LOCKED`, speak a command:
   - Press **M**, speak, press **M** again.
4. The toolkit fuses the locked product with the spoken intent and executes
   the action.  Results appear in the status bar and terminal.

**Condition A interaction flow:**

1. Press **A** to switch to mouse-only mode.
2. Each card shows four buttons: Add / Info / Similar / Compare.
3. Click any button to trigger the corresponding action.

### Recognised speech commands (shopping)

| Command | Alternatives | Intent |
|---------|--------------|--------|
| "add to cart" | add, cart, card, add cart | `ADD_TO_CART` |
| | put in cart, add this | `ADD_TO_CART` |
| | to cart, two cart, to card, two card | `ADD_TO_CART` |
| "show details" | details | `SHOW_DETAILS` |
| "find similar" | — | `FIND_SIMILAR` |
| "compare" | — | `COMPARE` |
| "show alternatives" | — | `SHOW_ALTERNATIVES` |
| "pin this" | save this, remember this | `PIN_ITEM` |
| "remove this" | delete this | `REMOVE_ITEM` |
| "scroll down / up" | — | `SCROLL` |
| "open cart" | show cart, view cart | `OPEN_CART` |
| "go back" | back | `GO_BACK` |
| "undo" | take that back | `UNDO` |
| "help" | — | `HELP` |
| "cancel" | never mind, stop | `CANCEL` |

---

## Kiosk Demo — `live_kiosk_cv.py`

**Scenario:** A museum kiosk with 6 exhibit information cards arranged in
a 3×2 grid.

**Condition B interaction flow:**

1. Ensure Condition B is active.
2. Look at an exhibit card until it shows `LOCKED`.
3. Press **M**, speak a command, press **M** again.

**Condition A interaction flow:**

1. Press **A** to switch to mouse-only mode.
2. Each card shows four buttons: Info / Sum / Zoom / Pin.
3. Click any button to trigger the corresponding action.

### Recognised speech commands (kiosk)

| Command | Alternatives | Intent |
|---------|--------------|--------|
| "tell me about this" | tell, me, about, this | `READ_ALOUD` |
| | me about, tell me, about this | `READ_ALOUD` |
| | tell about, tell this | `READ_ALOUD` |
| "summarize" | — | `SUMMARIZE` |
| "zoom in" | enlarge, bigger, closer | `ZOOM_IN` |
| "open detail" | show full info, learn more | `OPEN_DETAIL` |
| "bookmark this" | bookmark, book, mark, book mark | `PIN_EXHIBIT` |
| | pin, pin this, save, save this | `PIN_EXHIBIT` |
| "compare" | side by side | `COMPARE_ITEMS` |
| "next" | go forward, next exhibit | `NAVIGATE_NEXT` |
| "go back" | back, previous | `NAVIGATE_PREV` |
| "help" | what can I say | `HELP` |
| "cancel" | never mind, stop | `CANCEL` |

---

## UI Overlay Elements

| Element | Description |
|---------|-------------|
| Gaze dot / crosshair | Real-time mapped gaze position |
| Dwell progress bar | Fill toward the 1.2-second lock threshold |
| `LOCKED` badge | Gaze lock confirmed on this item |
| Fusion state badge (top-right) | `IDLE / LOCKED / NEEDS_TARGET / DISAMBIG / CONFIRM / COMMAND / CANCELLED` |
| Microphone indicator (red circle) | Turns red while PTT recording is active |
| Status bar (bottom) | Shows target, intent, transcript, confidence, last command |
| Event log (bottom-right) | Last five events with timestamps |
| Mode badge (below header) | Shows `MODE: A Mouse-only` or `MODE: B Gaze+Speech` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` for any package | Run `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError: cv2` | Run `python -m pip install opencv-python` |
| `ModuleNotFoundError: mediapipe` | Run `python -m pip install mediapipe` |
| Camera does not open | Check that the webcam is connected and not in use by another application; verify OS camera permissions |
| Microphone not working | Ensure the microphone is set as the default input device in OS audio settings |
| Vosk model missing | Verify that `speech/models/vosk-model-small-en-us-0.15/` exists relative to the project root |
| Speech command not recognised | Speak clearly after pressing **M**; pause briefly before speaking; check that the Vosk model path is correct |
| Gaze dot jumps or is inaccurate | Press **U** to recalibrate; ensure your face is well-lit and fully visible to the camera |
| Gaze calibration fails or says "poor" | Keep your head still, look directly at each target, and press **SPACE** only when looking at the dot |
| Command triggers no action | Check that a gaze lock (`LOCKED` badge) is visible before speaking; in Condition A, ensure you are clicking a button |
