# GazeShop — Gaze Modality Module

This module forms the **visual tracking (Gaze)** part of the multimodal GazeShop project. It is a single-page gaze interaction demo and backend server powered by Python + MediaPipe.

## What This Module Does

- Tracks face/iris landmarks with MediaPipe Face Landmarker
- Calibrates gaze mapping on a 9-point grid
- Streams gaze data to the browser over WebSocket
- Highlights and selects products with gaze dwell logic
- Shows live webcam preview with face detection overlays
- Exposes speech-fusion-ready event contracts for team integration

## Tech Stack

- Python: `mediapipe`, `opencv-python`, `websockets`, `numpy`
- Frontend: plain `HTML/CSS/JavaScript`
- Transport: WebSocket (`ws://127.0.0.1:8765` by default)

## Project Files

- `gaze_server.py` - Python gaze server (camera, calibration, streaming, fusion-ready events)
- `index.html` - Single-page UI (calibration overlay, webcam preview, product selection)
- `fusion.py` - Fusion engine for gaze/speech intent combination
- `intents.py` - Supported speech command catalog and parser
- `event_bus.py` - Simple event bus utility

## Setup

1. Open a terminal in the project folder:

```bash
cd /path/to/Gaze
```

2. Install dependencies:

```bash
pip install opencv-python mediapipe websockets numpy
```

3. Start the Python server:

```bash
python gaze_server.py
```

4. Open `index.html` in your browser.

## Runtime Flow

1. Browser connects to WebSocket server
2. Calibration overlay appears with red target points
3. For each target, press **Space** (or click **Capture Point**)
4. After all points are captured, tracking becomes active
5. Look at a card to trigger `looking`, hold gaze to trigger `selected`

## Controls

### In the Browser

- `Space` - capture current calibration point
- `R` - reset calibration

### In Terminal

- `Ctrl+C` - stop server

## Webcam Preview

- Top-left webcam box is persistent
- Green border/badge when face is detected
- Red border/badge when face is not detected
- Face box + iris markers are drawn by the Python server

## Configuration

Environment variables for server:

- `GAZE_WS_HOST` (default: `127.0.0.1`)
- `GAZE_WS_PORT` (default: `8765`)

Example:

```powershell
$env:GAZE_WS_PORT="8766"
python gaze_server.py
```

Then update frontend WebSocket URL to match.

## Speech Integration Readiness (Team Project)

This repo is speech-ready without requiring a local ASR engine in your part.

- On connect, server emits a `CAPABILITIES` event including command catalog
- Server accepts transcript commands over WS:

```json
{"cmd":"speech_transcript","transcript":"add this to cart","confidence":0.9}
```

- Server responds with:
  - `SPEECH_EVENT`
  - `FUSION_EVENT`

- Regular gaze stream also includes `gaze_event` payloads:
  - `LOCK`
  - `UNLOCK`
  - `AMBIGUOUS`

## Troubleshooting

### Port already in use

If you see WinError 10048, another process already uses the port.

- Stop the old process, or
- Start on a different port using `GAZE_WS_PORT`

### Black webcam preview

Current implementation streams preview frames from Python server, so browser webcam conflicts are minimized.  
If still black:

- Restart server
- Refresh page
- Ensure camera access is allowed and not blocked at OS level

### Poor selection accuracy

Re-run calibration carefully and keep head position stable.  
Selection quality depends heavily on calibration quality and lighting.

## Notes

- MediaPipe model file (`face_landmarker.task`) is auto-downloaded on first run
- Python 3.13 uses MediaPipe Tasks API (not legacy `mp.solutions`)
