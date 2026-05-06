# MMUI Toolkit — Multimodal User Interface Toolkit

> **GazeShop** projesi kapsamında geliştirilmiş, yeniden kullanılabilir,
> modüler bir **multimodal etkileşim altyapısı**.

---

## Toolkit Overview

The **MMUI Toolkit** is a reusable, domain-agnostic framework for building
applications that fuse multiple input modalities (eye-gaze, speech, and any
future modality) into a coherent user command.  It implements a **late-fusion
architecture**: each modality operates independently, producing standardised
events on a shared bus, which the fusion engine then combines into a single
high-level command.

The toolkit ships with **zero application-specific logic**.  Shopping commands,
kiosk commands, and any future vocabulary are injected by the application layer
at runtime.  This design means a developer can build a completely new
application in **under 20 lines of Python**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  apps/shopping/   apps/kiosk/   apps/your_app/                  │
│  (intents.py + actions.py — domain-specific vocabulary)          │
└───────────────────────┬──────────────────────────────────────────┘
                        │  register_action() / register_adapter()
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                   MMUIToolkit  (runtime.py)                       │
│   Developer-facing facade — wires everything together             │
└──────┬──────────────────────────────────────────────────┬────────┘
       │                                                  │
       ▼                                                  ▼
┌─────────────────────┐                        ┌──────────────────────┐
│   Modality Adapters  │                        │   UI Feedback Events │
│                      │                        │  PromptEvent         │
│  GazeAdapterStub ──► │──┐                     │  DisambiguatPrompt   │
│  SpeechAdapter    ──► │  │  EventBus (pub/sub) │  ConfirmationPrompt  │
│  DummyKeyboard    ──► │  │                     │  ActionCancelledEvent│
│  [YourAdapter]    ──► │  │                     └──────────────────────┘
└─────────────────────┘  │                                ▲
                          │                               │
                          ▼                               │
              ┌─────────────────────┐                    │
              │    FusionEngine      │                    │
              │  Late fusion rules   │                    │
              │  Lock/TTL/Ambiguous  │                    │
              └──────────┬──────────┘                    │
                         │ Internal Events                │
                         ▼                               │
              ┌─────────────────────┐                    │
              │  DialogueManager     │────────────────────┘
              │  State machine       │
              │  confirm/disambig/   │──► MultimodalCommandEvent
              │  repair/cancel       │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Action Registry    │
              │  intent → handler()  │
              └─────────────────────┘
```

**Event flow (happy path):**
1. `GazeAdapter` emits `GazeEvent(LOCK, target_id="X")`
2. `SpeechAdapter` emits `SpeechEvent(INTENT, intent="ADD_TO_CART")`
3. `FusionEngine` fuses them → `IntentReadyEvent`
4. `DialogueManager` emits `MultimodalCommandEvent`
5. `MMUIToolkit` dispatches to the registered action handler

---

## Developer Quickstart

Install dependencies (no microphone required for testing):

```bash
cd Speech_Detection-main
pip install numpy pytest
# For live speech (optional):
pip install -r requirements.txt
```

**Five lines to build your own app:**

```python
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.intents import IntentPattern

# 1. Define your vocabulary
MY_INTENTS = [
    IntentPattern("OPEN_DETAIL", target_required=True,
                  patterns=[r"\bopen\b", r"\bshow\s+detail"]),
    IntentPattern("NAVIGATE_NEXT", target_required=False,
                  patterns=[r"\bnext\b"]),
]

# 2. Create toolkit
tk = MMUIToolkit(Config())

# 3. Register adapters (share tk.bus)
parser = IntentParser(tk.config, custom_patterns=MY_INTENTS)
tk.register_adapter(GazeAdapterStub(tk.bus))
tk.register_adapter(SpeechAdapter(tk.bus, tk.config, intent_parser=parser))

# 4. Register actions
tk.register_action("OPEN_DETAIL",    lambda cmd: print("Detail:", cmd.target_id))
tk.register_action("NAVIGATE_NEXT",  lambda cmd: print("Next page"))

# 5. Run
with tk:
    input("Press Enter to quit…")
```

---

## How to Add a New Modality

Any new input device (gesture tracker, BCI headset, foot pedal, …) can be
added as a **plugin** by following three steps:

### Step 1 — Subclass `ModalityAdapter`

```python
# my_gesture_adapter.py
from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
import threading, time

class GestureAdapter(ModalityAdapter):
    """Reads a gesture sensor and emits SpeechEvents (or custom events)."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus)
        self._thread = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            gesture = self._read_sensor()   # your hardware call
            if gesture == "swipe_right":
                self.event_bus.emit(SpeechEvent(
                    type=SpeechEventType.INTENT,
                    payload={"intent": "NAVIGATE_NEXT",
                             "target_required": False, "params": {}},
                    transcript="[gesture: swipe_right]",
                    confidence=1.0,
                ))
            time.sleep(0.05)

    def _read_sensor(self) -> str:
        return ""   # replace with real sensor read
```

### Step 2 — Register with the toolkit

```python
from my_gesture_adapter import GestureAdapter

tk = MMUIToolkit(Config())
tk.register_adapter(GestureAdapter(tk.bus))   # plug in like any other adapter
```

### Step 3 — Register an action (if needed)

```python
tk.register_action("NAVIGATE_NEXT", lambda cmd: ui.go_next())
```

**That's it.**  The fusion engine, dialogue manager, and telemetry are all
unaware of the new modality — they just see events on the bus.

> See `gazeshop/toolkit/adapters/dummy_adapter.py` for a complete working
> example of a third modality (keyboard-driven).

---

## Two Demo Applications

Both demos use **identical toolkit wiring** — only the intent vocabulary and
action handlers differ.  This is the concrete proof of reusability.

### Demo #1 — Shopping Assistant (`demo/demo_shopping.py`)

An e-commerce assistant where users look at products and speak commands.

| Intent | Phrase example | Target needed |
|---|---|---|
| `ADD_TO_CART` | "add this to cart" | ✅ gaze lock |
| `SHOW_DETAILS` | "show details" | ✅ gaze lock |
| `FIND_SIMILAR` | "find similar" | ✅ gaze lock |
| `COMPARE` | "compare" | ✅ gaze lock |
| `PIN_ITEM` | "pin this" | ✅ gaze lock |
| `REMOVE_ITEM` | "remove this" | ✅ gaze lock |
| `SCROLL` | "scroll down" | ❌ global |
| `OPEN_CART` | "open cart" | ❌ global |
| `UNDO` | "undo" | ❌ global |

Run:

```bash
cd Speech_Detection-main
python demo/demo_shopping.py
```

### Demo #2 — Museum / Document Kiosk (`demo/demo_kiosk.py`)

A kiosk for a museum or document viewer — completely different domain, same core.

| Intent | Phrase example | Target needed |
|---|---|---|
| `READ_ALOUD` | "tell me about this" | ✅ exhibit lock |
| `SUMMARIZE` | "summarize" | ✅ exhibit lock |
| `ZOOM_IN` | "zoom in" | ✅ exhibit lock |
| `OPEN_DETAIL` | "open detail" | ✅ exhibit lock |
| `PIN_EXHIBIT` | "bookmark this" | ✅ exhibit lock |
| `COMPARE_ITEMS` | "compare" | ✅ exhibit lock |
| `NAVIGATE_NEXT` | "next" | ❌ global |
| `NAVIGATE_PREV` | "back" | ❌ global |

Run:

```bash
cd Speech_Detection-main
python demo/demo_kiosk.py
```

**Why this proves reusability:**  The toolkit core (`FusionEngine`,
`DialogueManager`, `EventBus`, `MMUIToolkit`) is imported unchanged by both
apps.  Not a single line of toolkit code was modified when adding the kiosk.

---

## Running Tests

All tests run **offline** (no microphone or camera required):

```bash
cd Speech_Detection-main

# Full test suite
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_event_bus.py -v           # EventBus pub/sub
python -m pytest tests/test_fusion_engine.py -v       # Fusion rules + TTL
python -m pytest tests/test_dialogue_manager.py -v    # Dialog state machine
python -m pytest tests/test_runtime.py -v             # MMUIToolkit facade
python -m pytest tests/test_dummy_adapter.py -v       # DummyKeyboardAdapter
python -m pytest tests/test_intent_parser.py -v       # Intent recognition
python -m pytest tests/test_integration.py -v         # End-to-end flow
```

Expected output (all passing):

```
tests/test_event_bus.py          ........ [PASSED]
tests/test_fusion_engine.py      ........ [PASSED]
tests/test_dialogue_manager.py   ........ [PASSED]
tests/test_runtime.py            ........ [PASSED]
tests/test_dummy_adapter.py      ........ [PASSED]
tests/test_intent_parser.py      ........ [PASSED]
tests/test_integration.py        ........ [PASSED]
```

---

## Repository Structure

```
Speech_Detection-main/
│
├── gazeshop/
│   ├── __init__.py
│   └── toolkit/                     ◄ TOOLKIT CORE (no app logic here)
│       ├── __init__.py
│       ├── runtime.py               ◄ MMUIToolkit facade (developer API)
│       ├── config.py                ◄ All tunable parameters
│       ├── event_bus.py             ◄ EventBus + GazeEvent + SpeechEvent
│       ├── events.py                ◄ Internal & UI event dataclasses
│       ├── intents.py               ◄ IntentPattern base + dialog patterns
│       ├── intent_parser.py         ◄ Regex-based intent recognition
│       ├── fusion_engine.py         ◄ Late fusion rules (Lock/TTL/Ambiguous)
│       ├── dialogue_manager.py      ◄ State machine (confirm/disambig/repair)
│       ├── telemetry.py             ◄ JSONL event logging
│       └── adapters/
│           ├── base_adapter.py      ◄ ModalityAdapter abstract base class
│           ├── gaze_adapter.py      ◄ GazeAdapterStub (simulation)
│           ├── speech_adapter.py    ◄ Full audio→ASR→intent pipeline
│           └── dummy_adapter.py     ◄ Keyboard adapter (extensibility proof)
│
├── apps/                            ◄ APPLICATION LAYER (domain logic)
│   ├── shopping/
│   │   ├── intents.py               ◄ 13 shopping commands
│   │   └── actions.py               ◄ cart / pin / scroll handlers
│   └── kiosk/
│       ├── intents.py               ◄ 10 kiosk commands
│       └── actions.py               ◄ audio guide / zoom / bookmark handlers
│
├── demo/
│   ├── demo_shopping.py             ◄ Demo #1 — Shopping (uses MMUIToolkit)
│   ├── demo_kiosk.py                ◄ Demo #2 — Kiosk   (same toolkit!)
│   └── speech_demo.py               ◄ Interactive CLI speech tester
│
├── tests/
│   ├── conftest.py
│   ├── test_event_bus.py
│   ├── test_fusion_engine.py
│   ├── test_dialogue_manager.py
│   ├── test_runtime.py              ◄ NEW — MMUIToolkit tests
│   ├── test_dummy_adapter.py        ◄ NEW — DummyKeyboardAdapter tests
│   ├── test_intent_parser.py
│   ├── test_speech_adapter.py
│   └── test_integration.py
│
├── conftest.py                      ◄ sys.path fix for pytest
├── requirements.txt
└── README.md
```

---

## Configuration Reference

All parameters live in `gazeshop/toolkit/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LOCK_TTL_S` | `4.0` | Seconds a gaze lock stays valid |
| `FUSION_TIME_WINDOW_S` | `2.0` | Max gaze-speech time gap for fusion |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this → ask for confirmation |
| `MAX_REPAIR_ATTEMPTS` | `2` | Max disambiguation retries |
| `DISAMBIGUATION_TIMEOUT_S` | `8.0` | Auto-cancel dialog after this |
| `ENABLE_TELEMETRY` | `True` | Write JSONL interaction log |
| `TELEMETRY_EXPORT_PATH` | `logs/telemetry.jsonl` | Log file path |
| `ASR_ENGINE` | `"vosk"` | `"vosk"` or `"whisper"` |
| `VAD_BACKEND` | `"silero"` | VAD backend (`"silero"`, `"webrtc"`, `"none"`) |

---

## Environment Setup

```bash
# 1. Clone / copy the project
cd Speech_Detection-main

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download Vosk model for live speech
# Place the model folder under: models/vosk-model-small-en-us-0.15/
```

For **offline testing only** (no ASR / mic needed):

```bash
pip install numpy pytest
```

---

## Migration Notes

> **Breaking changes introduced in this refactor:**

| What changed | Migration |
|---|---|
| `gazeshop/toolkit/intents.py` — `INTENT_PATTERNS` is now empty | Move your intent patterns to `apps/<yourapp>/intents.py` and pass them as `IntentParser(config, custom_patterns=YOUR_INTENTS)` |
| `IntentParser` constructor — no longer extends with `INTENT_PATTERNS` | Pass all patterns via `custom_patterns=` |
| `kiosk_app.py` `parser._intents = ...` hack | Replaced by `IntentParser(config, custom_patterns=KIOSK_INTENTS)` |
| Old `test_dialogue_manager.py` used `on_prompt`/`on_action` callbacks | Rewritten to use EventBus subscriptions (matches actual DM API) |
| `demo/fusion_demo.py`, `demo/kiosk_app.py` | Superseded by `demo/demo_shopping.py` and `demo/demo_kiosk.py` |

---

## Technology Stack

| Component | Library | Purpose |
|---|---|---|
| Audio capture | `sounddevice` | 16 kHz mono PCM recording |
| VAD (primary) | `silero-vad` | Neural-network voice activity detection |
| VAD (fallback) | `webrtcvad` | Rule-based VAD trimming |
| ASR (primary) | `vosk` | Offline, lightweight speech recognition |
| ASR (alternate)| `whisper` | Higher accuracy, hallucination-resistant |
| Key listener | `pynput` | Cross-platform PTT key detection |
| Numerical | `numpy` | Audio buffer handling |
| Testing | `pytest` | Unit and integration tests |
