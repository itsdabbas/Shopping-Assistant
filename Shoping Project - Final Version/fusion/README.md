# fusion/ — Reusable MMUI Toolkit and Application Layer

The `fusion/` folder contains everything needed to build a multimodal
gaze + speech application.  It is divided into two independent parts:

```
fusion/
├── gazeshop/toolkit/    Reusable toolkit core (no domain-specific code)
└── apps/                Application layer (shopping and kiosk vocabularies)
    ├── shopping/
    └── kiosk/
```

The toolkit core has **zero knowledge** of shopping or kiosk semantics.
Applications supply their own intent vocabulary and action handlers at
runtime; the core is never modified.

---

## 1. Toolkit Architecture

### EventBus (`toolkit/event_bus.py`)

The `EventBus` is a lightweight publish/subscribe dispatcher.  All
inter-component communication passes through it.  Components never hold
direct references to each other; they only hold a reference to the shared bus.

Two primary event types flow on the bus:

- **`GazeEvent`** — emitted by gaze adapters.  Key fields: `type`
  (`LOCK` / `UNLOCK` / `AMBIGUOUS`), `target_id`, `timestamp`.
- **`SpeechEvent`** — emitted by the speech adapter.  Key fields: `type`
  (`INTENT` / `CONFIRM` / `REPAIR` / `CANCEL` / `ERROR`),
  `payload["intent"]`, `transcript`, `confidence`.

### ModalityAdapter (`toolkit/adapters/base_adapter.py`)

Abstract base class for all input modalities.  Subclasses implement
`start()` and `stop()`; they emit typed events via `self.event_bus.emit(…)`.
The toolkit provides three concrete adapters:

| Adapter | File | Purpose |
|---------|------|---------|
| `RealGazeAdapter` | `adapters/real_gaze_adapter.py` | Live webcam iris tracking via MediaPipe |
| `SpeechAdapter` | `adapters/speech_adapter.py` | Push-to-talk audio → ASR → intent |
| `GazeAdapterStub` | `adapters/gaze_adapter.py` | Programmatic gaze simulation (testing) |
| `DummyAdapter` | `adapters/dummy_adapter.py` | Keyboard-driven test adapter |

### SpeechAdapter (`toolkit/adapters/speech_adapter.py`)

Manages the full speech pipeline in a dedicated worker thread:

```
PTT toggle ON → sounddevice capture → VAD trimming → ASR (Vosk / Whisper)
              → IntentParser.parse() → SpeechEvent emitted on EventBus
```

The three-tier VAD chain is: Silero neural VAD → WebRTC VAD → no trimming
(automatic fallback if libraries are unavailable).

Intent parsing uses a two-stage strategy:

1. **Regex matching** against the `IntentPattern` list supplied by the
   application (`custom_patterns`).  Produces a structured `SpeechEvent`
   with a heuristic confidence score.
2. **Fuzzy matching** (rapidfuzz `token_sort_ratio`) against a flat
   synonym table as a fallback for out-of-grammar transcripts.

### RealGazeAdapter (`toolkit/adapters/real_gaze_adapter.py`)

Reads frames from the webcam, runs MediaPipe FaceMesh with refined
landmarks, extracts the average iris position from landmarks 468 (left)
and 473 (right), and applies an optional 5-point affine calibration to
map the raw eye-relative feature into screen coordinates.

A `DwellTracker` monitors how long the mapped gaze point falls within
each item bounding box.  After `DWELL_S` seconds of sustained gaze, the
adapter emits `GazeEvent(LOCK, target_id=…)`.

### DwellTracker (`toolkit/dwell.py`)

Tracks elapsed dwell time per item.  Exposes `dwell_target` and
`dwell_progress` (0.0 – 1.0) for the UI progress bar.  Emits
`AMBIGUOUS` when gaze is near a bounding-box boundary.

### FusionEngine (`toolkit/fusion_engine.py`)

Subscribes to both `GazeEvent` and `SpeechEvent` on the EventBus.
When a `SpeechEvent(INTENT)` arrives, the engine checks whether a
compatible `GazeEvent(LOCK)` exists within the configured time window:

```
fused_confidence = (asr_confidence + FUSION_CONFIDENCE_BIAS) / 2

Fusion succeeds if:
  - intent.target_required is False  (global command — no gaze needed), OR
  - a LOCK event exists AND its age < FUSION_TIME_WINDOW_S AND
    the lock has not exceeded LOCK_TTL_S
```

On success, the engine emits an internal `IntentReadyEvent` for the
DialogueManager.  On failure, it emits `PromptEvent` or
`DisambiguationPromptEvent` so the dialogue layer can request repair.

### DialogueManager (`toolkit/dialogue_manager.py`)

A finite-state machine that sits between the FusionEngine and the
action registry.  It handles four situations that the FusionEngine
alone cannot resolve:

| Situation | DM Action |
|-----------|-----------|
| Intent received but no gaze target | Emits `PromptEvent("Please look at an item first")` |
| Gaze is ambiguous (near boundary) | Emits `DisambiguationPromptEvent`; waits for repair |
| Confidence below `CONFIDENCE_THRESHOLD` | Emits `ConfirmationPromptEvent`; waits for yes/no |
| User says "cancel" / timeout | Emits `ActionCancelledEvent` |

On resolution, the DialogueManager emits `MultimodalCommandEvent(intent,
target_id, params)` which the MMUIToolkit dispatches to the application
action handler.

### MMUIToolkit (`toolkit/runtime.py`)

The developer-facing façade.  Wires all components together and exposes
a minimal API:

```python
tk = MMUIToolkit(Config())
tk.register_adapter(adapter)   # add a modality
tk.register_action(intent, handler)   # bind command to handler
tk.register_feedback(event_type, callback)   # subscribe to UI events
tk.start()   # begin all adapters and the internal event loop
tk.stop()    # cleanly shut down
```

---

## 2. Late Fusion — How Gaze and Speech Are Combined

The fusion strategy is **time-window late fusion**:

1. Each modality runs independently and produces its own events.
2. The FusionEngine holds a short memory of recent gaze locks
   (governed by `LOCK_TTL_S`).
3. When a speech intent arrives, the engine looks back in time by at
   most `FUSION_TIME_WINDOW_S` seconds for a matching gaze lock.
4. If found and still fresh, the two are combined into one command.
5. If not found, the dialogue layer prompts the user to look at a target.

Key configuration parameters (all in `toolkit/config.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `FUSION_TIME_WINDOW_S` | 2.0 s | Max age of a gaze lock that can be fused |
| `LOCK_TTL_S` | 4.0 s | A lock older than this is considered stale |
| `CONFIDENCE_THRESHOLD` | 0.60 | Below this, the DM requests confirmation |
| `DISAMBIGUATION_TIMEOUT_S` | 8.0 s | Auto-cancel if no repair response |
| `MAX_REPAIR_ATTEMPTS` | 2 | Maximum disambiguation retries |

---

## 3. Dialogue Management

The DialogueManager state machine operates **below** the FusionEngine and
**above** the action registry.  It is completely application-agnostic; it
only knows about generic dialogue acts (confirm, deny, repair, cancel).

Application-specific intents enter as `IntentReadyEvent`; resolved
commands leave as `MultimodalCommandEvent`.  The dialogue state machine
handles all intermediate states:

```
IntentReadyEvent
  ↓
IDLE → WAIT_TARGET (if no gaze lock present)
        ↓ user looks at target
     CONFIRMING (if confidence < threshold)
        ↓ user says "yes"
     DISAMBIGUATING (if gaze is ambiguous)
        ↓ user says "left" / "right" / repair word
     → MultimodalCommandEvent emitted
```

Cancellation (`SpeechEventType.CANCEL`) is accepted in any state.

---

## 4. Adding a New Application Domain

To build a new application on the toolkit, create two files and wire
them into the demo:

### Step 1 — Define intents (`apps/myapp/intents.py`)

```python
from gazeshop.toolkit.intents import IntentPattern

MY_INTENTS = [
    IntentPattern(
        intent="OPEN_RECORD",
        target_required=True,
        patterns=[r"\bopen\b", r"\bshow\s+record\b"],
    ),
    IntentPattern(
        intent="PRINT",
        target_required=False,
        patterns=[r"\bprint\b"],
    ),
]

MY_VOSK_VOCAB = ["open", "show record", "print", "yes", "no", "cancel", "[unk]"]
```

### Step 2 — Define action handlers (`apps/myapp/actions.py`)

```python
MY_HANDLERS = {
    "OPEN_RECORD": lambda cmd: print(f"Opening {cmd.target_id}"),
    "PRINT":       lambda cmd: print("Printing…"),
}
```

### Step 3 — Wire into the runtime

```python
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.adapters.real_gaze_adapter import RealGazeAdapter
from apps.myapp.intents import MY_INTENTS, MY_VOSK_VOCAB
from apps.myapp.actions import MY_HANDLERS

cfg    = Config(VOSK_MODEL_PATH="speech/models/vosk-model-small-en-us-0.15")
tk     = MMUIToolkit(cfg)
parser = IntentParser(cfg, custom_patterns=MY_INTENTS)
gaze   = RealGazeAdapter(tk.bus, cam_index=0, win_w=1280, win_h=720)
speech = SpeechAdapter(tk.bus, cfg, intent_parser=parser, vosk_vocab=MY_VOSK_VOCAB)

tk.register_adapter(gaze).register_adapter(speech)
for intent, handler in MY_HANDLERS.items():
    tk.register_action(intent, handler)

with tk:
    input("Running — press Enter to quit.")
```

The toolkit core (`FusionEngine`, `DialogueManager`, `EventBus`) is
imported unchanged.  **Not a single line of core code is modified.**

---

## 5. Adding a New Input Modality

Any input device can become a modality adapter by subclassing
`ModalityAdapter` and emitting typed events on the shared `EventBus`.

```python
from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.event_bus import SpeechEvent, SpeechEventType
import threading, time

class GestureAdapter(ModalityAdapter):
    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            gesture = self._read_sensor()
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
        return ""  # replace with real hardware read
```

Register it exactly like any other adapter:

```python
tk.register_adapter(GestureAdapter(tk.bus))
```

The FusionEngine and DialogueManager are unaware of the new modality.
They only observe events on the bus.

---

## 6. Key Files at a Glance

| File | Role |
|------|------|
| `gazeshop/toolkit/runtime.py` | MMUIToolkit façade — developer entry point |
| `gazeshop/toolkit/config.py` | All tunable parameters |
| `gazeshop/toolkit/event_bus.py` | EventBus + GazeEvent + SpeechEvent definitions |
| `gazeshop/toolkit/events.py` | Internal event dataclasses |
| `gazeshop/toolkit/fusion_engine.py` | Timing-aware late fusion |
| `gazeshop/toolkit/dialogue_manager.py` | Dialogue state machine |
| `gazeshop/toolkit/intent_parser.py` | Regex + fuzzy intent matching |
| `gazeshop/toolkit/intents.py` | IntentPattern dataclass + dialog patterns |
| `gazeshop/toolkit/dwell.py` | DwellTracker |
| `gazeshop/toolkit/calibration.py` | 5-point affine calibration |
| `gazeshop/toolkit/telemetry.py` | JSONL event log |
| `gazeshop/toolkit/adapters/base_adapter.py` | ModalityAdapter abstract base |
| `gazeshop/toolkit/adapters/speech_adapter.py` | Full PTT → ASR → intent pipeline |
| `gazeshop/toolkit/adapters/real_gaze_adapter.py` | Live webcam gaze tracker |
| `apps/shopping/intents.py` | Shopping intent patterns + Vosk vocabulary |
| `apps/shopping/actions.py` | Shopping action handlers |
| `apps/kiosk/intents.py` | Kiosk intent patterns + Vosk vocabulary |
| `apps/kiosk/actions.py` | Kiosk action handlers |
