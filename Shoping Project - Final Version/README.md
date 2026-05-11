# GazeShop: A Multimodal Gaze + Speech Interaction Toolkit

GazeShop is a reusable, domain-agnostic multimodal interaction toolkit that
combines real-time **gaze tracking** with **speech command recognition** to
produce a unified, context-aware user command.  The system is designed as a
clean, layered architecture: any application can be wired on top of the same
core toolkit simply by supplying an intent vocabulary and a set of action
handlers.

Two demonstration applications are included — a product shopping interface
and a museum kiosk — to show that the toolkit core is genuinely reusable
across different interaction domains.

---

## Project Motivation

This project was developed as part of a Multimodal User Interfaces (MMUI)
course assignment.  The emphasis is on **toolkit architecture**, not on a
single fixed application.  Key design goals are:

- **Active modalities**: gaze and speech operate independently; neither blocks
  the other.
- **Late fusion**: the two modality streams are combined by a timing-aware
  FusionEngine, not hardcoded together.
- **Extensibility**: adding a new application domain, a new intent vocabulary,
  or even a new input modality requires no changes to the toolkit core.
- **Dialogue recovery**: when the system is ambiguous or under-confident, a
  DialogueManager handles repair, confirmation, and cancellation automatically.

---

## Folder Structure

```
project-root/
│
├── gaze/                    Original gaze research module (reference material)
│
├── speech/
│   └── models/              Offline speech recognition model (Vosk)
│       └── vosk-model-small-en-us-0.15/
│
├── fusion/                  Reusable toolkit core + application layers
│   ├── gazeshop/
│   │   └── toolkit/         Core MMUI toolkit (no application-specific code)
│   │       ├── runtime.py   MMUIToolkit façade (developer entry point)
│   │       ├── config.py    All tunable parameters
│   │       ├── event_bus.py EventBus, GazeEvent, SpeechEvent
│   │       ├── events.py    Internal event dataclasses
│   │       ├── fusion_engine.py    Late-fusion rules
│   │       ├── dialogue_manager.py Dialogue state machine
│   │       ├── intent_parser.py    Regex + fuzzy intent recognition
│   │       ├── intents.py          IntentPattern base class, dialog patterns
│   │       ├── dwell.py            DwellTracker
│   │       ├── calibration.py      Gaze calibration lifecycle
│   │       ├── telemetry.py        JSONL event logging
│   │       └── adapters/
│   │           ├── base_adapter.py    ModalityAdapter abstract base class
│   │           ├── speech_adapter.py  Audio → ASR → intent pipeline
│   │           ├── real_gaze_adapter.py  Live webcam gaze tracker
│   │           ├── gaze_adapter.py    GazeAdapterStub (simulation)
│   │           └── dummy_adapter.py   Keyboard-driven test adapter
│   └── apps/
│       ├── shopping/        Shopping application vocabulary and handlers
│       │   ├── intents.py
│       │   └── actions.py
│       └── kiosk/           Museum kiosk vocabulary and handlers
│           ├── intents.py
│           └── actions.py
│
├── demo/
│   ├── live_shopping_cv.py  Shopping demo — real webcam + microphone
│   ├── live_kiosk_cv.py     Kiosk demo — real webcam + microphone
│   ├── LIVE_DEMO_GUIDE.md   Quick-start guide for running the demos
│   └── README.md            Detailed demo documentation
│
├── tests/                   Unit and integration test suite (183 tests)
│
├── conftest.py              Pytest path configuration
├── requirements.txt
└── README.md                This file
```

---

## Architecture

The toolkit is structured in six layers:

```
EventBus
  ↑ ↑
  │ │  Modality Adapters (SpeechAdapter, RealGazeAdapter, …)
  │ │  Each adapter operates in its own thread and emits typed events.
  │ │
  └─┴─► FusionEngine
          Combines gaze LOCK events with speech INTENT events within a
          configurable time window (FUSION_TIME_WINDOW_S).
          ↓
        DialogueManager
          Handles ambiguity (NEEDS_TARGET, DISAMBIG), confirmation
          (CONFIRM), and cancellation / repair flows.
          ↓
        MultimodalCommandEvent
          A single, resolved command: intent + target_id + params.
          ↓
        MMUIToolkit  (runtime.py)
          Developer façade.  Dispatches the command to the registered
          action handler for the matched intent.
          ↓
        Application Layer  (apps/shopping/, apps/kiosk/, …)
          Domain-specific action handlers and intent vocabulary.
```

**Happy-path event sequence:**

1. `RealGazeAdapter` detects a 1.2-second dwell → emits
   `GazeEvent(LOCK, target_id="product_3")`.
2. User presses **M**, speaks *"add to cart"*, presses **M** again.
3. `SpeechAdapter` runs Vosk ASR → `IntentParser` matches regex →
   emits `SpeechEvent(INTENT, intent="ADD_TO_CART")`.
4. `FusionEngine` finds a LOCK event within the time window →
   emits an internal `IntentReadyEvent`.
5. `DialogueManager` emits `MultimodalCommandEvent(intent="ADD_TO_CART",
   target_id="product_3")`.
6. `MMUIToolkit` calls `handle_add_to_cart(cmd)`.

---

## Condition A and Condition B

Both live demos support two interaction modes, switchable at runtime:

| Mode | Key | Description |
|------|-----|-------------|
| **Condition A** | `A` | Mouse-only baseline.  Gaze tracking is paused.  Each card shows clickable action buttons. |
| **Condition B** | `B` | Full gaze + speech multimodal interaction.  Gaze locks a target; speech provides the command. |

Condition A is provided as a controlled baseline for user-study comparison.
Condition B is the full MMUI experience.

---

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install opencv-python mediapipe
```

The Vosk offline speech model is already included in the repository at:

```
speech/models/vosk-model-small-en-us-0.15/
```

Both demos resolve this path automatically from the project root.

---

## Running the Demos

Run from the **project root** (the folder containing this README):

```bash
python demo/live_shopping_cv.py
python demo/live_kiosk_cv.py
```

### Keyboard controls (both demos)

| Key | Action |
|-----|--------|
| `A` | Switch to Condition A — mouse-only baseline |
| `B` | Switch to Condition B — gaze + speech |
| `M` | Push-to-talk toggle — press once to start recording, press again to send |
| `U` | Start or restart gaze calibration |
| `SPACE` | Capture the current calibration point |
| `Q` | Quit |

See `demo/README.md` for the complete guide including all supported speech
commands, calibration steps, and troubleshooting.

---

## Running the Tests

```bash
python -m pytest tests
```

All 183 unit and integration tests run offline — no webcam or microphone is
required.  See `tests/README.md` for details.

---

## Toolkit Features

| Feature | Location |
|---------|----------|
| Real-time webcam gaze tracking (MediaPipe iris) | `fusion/gazeshop/toolkit/adapters/real_gaze_adapter.py` |
| Dwell-based target locking with progress bar | `fusion/gazeshop/toolkit/dwell.py` |
| 5-point affine gaze calibration | `fusion/gazeshop/toolkit/calibration.py` |
| Push-to-talk audio capture | `fusion/gazeshop/toolkit/adapters/speech_adapter.py` |
| Offline ASR via Vosk (or Whisper) | `fusion/gazeshop/toolkit/adapters/speech_adapter.py` |
| Three-tier VAD (Silero → WebRTC → none) | `fusion/gazeshop/toolkit/adapters/speech_adapter.py` |
| Regex + fuzzy intent recognition | `fusion/gazeshop/toolkit/intent_parser.py` |
| Timing-aware late fusion | `fusion/gazeshop/toolkit/fusion_engine.py` |
| Dialogue state machine (confirm / repair) | `fusion/gazeshop/toolkit/dialogue_manager.py` |
| Injectable app vocabulary (no core changes) | `fusion/apps/shopping/`, `fusion/apps/kiosk/` |
| JSONL telemetry logging | `fusion/gazeshop/toolkit/telemetry.py` |

See `fusion/README.md` for the complete architectural reference.

---

## Known Limitations

- Gaze accuracy depends on webcam quality, ambient lighting, and calibration
  care.  A 5-point calibration is provided; poor lighting or partial face
  occlusion will reduce accuracy.
- The live demos require a working webcam and microphone.
- Vosk speech recognition accuracy is constrained by the small model variant
  used; accuracy improves with Whisper (`ASR_ENGINE = "whisper"` in `Config`).
- All `WAIT_TARGET` timeouts in the DialogueManager are configurable via
  `Config.DISAMBIGUATION_TIMEOUT_S` but are not yet exposed in the demo UI.
