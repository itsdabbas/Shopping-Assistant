# gaze/ — Original Gaze Development Module (Reference Material)

This folder contains the original, standalone gaze tracking prototype developed
early in the project.  It operates as an independent WebSocket server and
single-page browser application.

> **Important:** This folder is **not used** by the final live demos
> (`demo/live_shopping_cv.py` and `demo/live_kiosk_cv.py`).
> The production gaze modality is implemented in:
>
> ```
> fusion/gazeshop/toolkit/adapters/real_gaze_adapter.py
> ```
>
> This folder is retained as the original gaze research and development
> reference.  It documents the architecture explored before the toolkit
> refactor.

---

## Contents

| File | Description |
|------|-------------|
| `gaze_server.py` | Python WebSocket server — camera capture, MediaPipe landmark extraction, gaze-to-grid mapping, WebSocket streaming |
| `index.html` | Single-page browser UI — calibration overlay, webcam preview, product grid, speech integration |
| `fusion.py` | Standalone fusion engine prototype (gaze lock + speech intent) |
| `intents.py` | Early speech intent patterns |
| `event_bus.py` | Simple event dispatcher (prototype) |
| `dwell_tracker.py` | Dwell timing logic prototype |
| `SPEECH_FUSION_INTEGRATION.md` | WebSocket protocol specification for the old architecture |

---

## Gaze Concepts Used in the Final Toolkit

The following concepts from this prototype were carried forward into the
production toolkit, though the implementation was fully rewritten:

### Webcam-based iris tracking

Both the prototype and the final adapter use **MediaPipe FaceMesh** with
`refine_landmarks=True`.  Iris centre positions are read from landmarks
468 (left iris) and 473 (right iris) and averaged to obtain a stable
gaze estimate.

### Eye-relative gaze feature

Rather than using absolute pixel coordinates (which are sensitive to
head movement), the final adapter normalises the iris centre position
relative to the surrounding eye corner landmarks.  This produces a
feature that is stable under moderate head translation.

### Dwell-based target locking

A gaze point that remains within an item's bounding box for a configurable
duration (`DWELL_S`, default 1.2 seconds) triggers a `GazeEvent(LOCK)`.
An `AMBIGUOUS` event is emitted when the gaze falls near a boundary
shared by two items.

### Calibration

The final adapter includes a 5-point affine calibration procedure.
During calibration, the user looks at five targets positioned at the
corners and centre of the screen.  The recorded eye-feature-to-screen
mappings are used to build an affine transform that improves gaze
accuracy across the full display area.

**Controls in the live demos:**
1. Press **B** to ensure Condition B (gaze + speech) is active.
2. Press **U** to start calibration.
3. Look at each highlighted target.
4. Press **SPACE** to capture each point.
5. Repeat for all five targets.
6. Recalibrate with **U** if gaze drifts during a session.

---

## Why This Folder Is Kept

The WebSocket-based prototype validated several design decisions — in
particular, that gaze-lock events and speech-intent events can be fused
reliably using a time-window approach — before the architecture was
consolidated into the EventBus-based toolkit.  Keeping this folder
provides a historical record of the design evolution.
