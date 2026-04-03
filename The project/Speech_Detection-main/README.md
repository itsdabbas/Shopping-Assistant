mm# GazeShop — Speech Modality Module

## Project Overview

> **Note**: This repository represents the **Speech Modality** of the larger GazeShop multimodal project. For the complete integration, visual tracking, and the Live E2E test harness, see the root repository documentation.

GazeShop is a multimodal shopping assistant that combines **eye-gaze tracking** and **speech commands** through a late-fusion architecture. This repository implements the **Speech Modality** component — responsible for capturing voice input, recognising user commands, and emitting structured events that integrate with the Gaze modality and FusionEngine.

The Speech module is designed as a **standalone, testable component** that communicates exclusively through a shared EventBus, ensuring clean integration with the Gaze module (developed separately) and the future FusionEngine.

---

## Architecture

```
gazeshop/
├── __init__.py
├── toolkit/
│   ├── __init__.py
│   ├── config.py                # Centralised configuration (thresholds, ASR settings)
│   ├── event_bus.py             # Event dataclasses + pub/sub EventBus
│   ├── intents.py               # 13 intent patterns (regex definitions)
│   ├── intent_parser.py         # Rule-based intent recognition + confidence scoring
│   ├── dialogue_manager.py      # Confirmation & disambiguation dialog flows
│   └── adapters/
│       ├── __init__.py
│       ├── base_adapter.py      # Abstract ModalityAdapter base class
│       ├── speech_adapter.py    # Audio capture + ASR + PTT + intent pipeline
│       └── gaze_adapter.py      # GazeAdapter stub (interface for compatibility)
│
├── tests/
│   ├── test_event_bus.py        # EventBus unit tests
│   ├── test_intent_parser.py    # Intent recognition tests (all 13 commands)
│   ├── test_speech_adapter.py   # SpeechAdapter pipeline tests
│   ├── test_dialogue_manager.py # Dialog flow tests
│   └── test_integration.py      # Speech + Gaze stub integration tests
│
├── demo/
│   └── speech_demo.py           # Interactive CLI demo
│
├── agent/
│   └── rules/
│       └── env.md               # Python environment guidelines
│
├── requirements.txt
└── README.md
```

---

## Module Descriptions

### `config.py` — Configuration

Centralised dataclass holding all tunable parameters:

| Parameter | Default | Description |
|---|---|---|
| `PTT_MODE` | `"hold"` | Push-to-talk mode: `"hold"` (hold key) or `"toggle"` (click to start/stop) |
| `PTT_KEY` | `"space"` | Keyboard key for push-to-talk activation |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `TAIL_SILENCE_MS` | `300` | Continue recording after key release (ms) |
| `MIN_UTTERANCE_MS` | `200` | Discard utterances shorter than this |
| `MAX_UTTERANCE_MS` | `15000` | Auto-stop utterances longer than this |
| `ASR_ENGINE` | `"vosk"` | ASR backend: `"vosk"` or `"whisper"` |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this, intent requires confirmation |
| `FUSION_TIME_WINDOW_S` | `2.0` | Max seconds between gaze lock and speech for fusion |
| `DISAMBIGUATION_TIMEOUT_S` | `8.0` | Dialog auto-cancel timeout |
| `MAX_REPAIR_ATTEMPTS` | `2` | Max disambiguation retries before cancel |

### `event_bus.py` — Event System

Defines the standardised event contracts:

**SpeechEvent** fields (emitted by SpeechAdapter):
- `timestamp` — epoch float for time-window matching with gaze
- `type` — `INTENT`, `REPAIR`, `CONFIRM`, `CANCEL`, `UNDO`, `REPEAT`, `LISTENING`, `STOPPED`, `ERROR`
- `payload` — varies by type (intent name, params, target_required, etc.)
- `transcript` — raw ASR text
- `confidence` — 0.0–1.0 heuristic score
- `requires_confirmation` — True if below threshold

**GazeEvent** fields (emitted by GazeAdapter):
- `timestamp` — epoch float
- `type` — `LOCK`, `UNLOCK`, `AMBIGUOUS`
- `payload` — `target_id` for LOCK, `candidates` list for AMBIGUOUS

**EventBus** — synchronous pub/sub dispatcher with wildcard and logging support.

### `intents.py` — Command Catalogue

13 recognisable commands split into two categories:

**Object-bound** (require gaze target, `target_required=True`):
1. `ADD_TO_CART` — "add this to cart", "buy this"
2. `SHOW_DETAILS` — "show details", "what is this" (slot: `detail_type`)
3. `FIND_SIMILAR` — "find similar", "anything like this"
4. `COMPARE` — "compare this" (slot: `compare_ref`)
5. `SHOW_ALTERNATIVES` — "show alternatives", "other options"
6. `PIN_ITEM` — "pin this", "save this"
7. `REMOVE_ITEM` — "remove this", "delete this"

**Global** (no gaze target needed, `target_required=False`):
8. `SCROLL` — "scroll down/up" (slot: `direction`)
9. `OPEN_CART` — "open cart", "show my cart"
10. `GO_BACK` — "go back", "previous page"
11. `HELP` — "help", "what can I say"
12. `CANCEL` — "cancel", "never mind"
13. `UNDO` — "undo", "take that back"

**Dialog-only** patterns (active during disambiguation/confirmation):
- CONFIRM: "yes", "yeah", "correct"
- DENY: "no", "nope", "wrong"
- REPAIR: "the left one", "the right one", "the first one"
- REPEAT: "repeat", "say again"

### `intent_parser.py` — Intent Recognition

Rule-based parser that:
1. Normalises transcript (lowercase, strip punctuation)
2. Tries dialog patterns first if a dialog is active
3. Matches against regex patterns in priority order
4. Extracts slot parameters
5. Computes heuristic confidence: `0.4×pattern + 0.4×ASR + 0.2×coverage`

### `speech_adapter.py` — Speech Adapter

Full speech pipeline:
1. **PTT Management** — Listens for push-to-talk key via `pynput`
2. **Audio Capture** — Records 16kHz mono PCM via `sounddevice`
3. **Endpointing** — Tail silence, min/max duration guards, energy gate
4. **ASR Transcription** — Vosk (offline) or Whisper
5. **Intent Parsing** — Delegates to IntentParser
6. **Event Emission** — Publishes SpeechEvent on EventBus

Also provides `process_text()` — a text-only bypass for testing without microphone.

### `gaze_adapter.py` — Gaze Adapter Stub

Interface-compatible stub for development and testing. Provides:
- `simulate_lock(target_id)` — emit a LOCK event
- `simulate_unlock()` — emit an UNLOCK event
- `simulate_ambiguous(candidates)` — emit an AMBIGUOUS event

Your teammate building the real GazeAdapter should implement the same interface.

### `dialogue_manager.py` — Dialogue Manager

Handles two dialog flows:
1. **Confirmation** — When confidence < 0.60: "Did you mean ADD TO CART?"
2. **Disambiguation** — When gaze is ambiguous: "Left or right?"

Includes timeout handling, max repair attempts, and cancel support.

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
cd "human-gaze project"
pip install -r requirements.txt
```

For testing only (no microphone/ASR needed):

```bash
pip install numpy pytest
```

---

## Testing

### Run All Tests

```bash
cd "human-gaze project"
python -m pytest tests/ -v
```

### Run Specific Test Files

```bash
# EventBus unit tests
python -m pytest tests/test_event_bus.py -v

# Intent Parser tests (all 13 commands)
python -m pytest tests/test_intent_parser.py -v

# SpeechAdapter pipeline tests
python -m pytest tests/test_speech_adapter.py -v

# DialogueManager flow tests
python -m pytest tests/test_dialogue_manager.py -v

# Integration tests (Speech + Gaze compatibility)
python -m pytest tests/test_integration.py -v
```

### What Each Test File Validates

| Test File | Count | What It Tests |
|---|---|---|
| `test_event_bus.py` | 12 | Pub/sub, wildcards, error isolation, logging |
| `test_intent_parser.py` | 40+ | All 13 intents, slots, confidence, dialog patterns, edge cases |
| `test_speech_adapter.py` | 20+ | Text pipeline, EventBus emission, dialog context, fusion field compatibility |
| `test_dialogue_manager.py` | 10+ | Confirmation flow, disambiguation, timeout, cancel, repeat |
| `test_integration.py` | 10+ | Speech + Gaze stub fusion simulation, timestamp compatibility, event logging |

---

## Interactive Demo

Run the CLI demo to manually test speech recognition:

```bash
python demo/speech_demo.py
```

### Demo Commands

Type phrases as if you spoke them:

```
You 🎤 > add this to cart
  ✓ Intent: ADD_TO_CART
    Target required: True
    Confidence: 0.82

You 🎤 > scroll down
  ✓ Intent: SCROLL
    Target required: False
    Parameters: {'direction': 'down'}
    Confidence: 0.80
```

### Gaze Simulation in Demo

```
You 🎤 > :lock item_42       # Simulate gaze lock
You 🎤 > add this to cart    # Shows fusion hint
You 🎤 > :unlock             # Simulate gaze unlock
You 🎤 > :ambiguous A B      # Simulate ambiguous gaze
You 🎤 > :dialog on          # Enable dialog-mode parsing
You 🎤 > yes                 # → CONFIRM event
You 🎤 > :status             # Show system state
You 🎤 > :log                # Show event log
```

---

## Integration with Gaze Module

### For the Gaze Developer

The Gaze module developer should:

1. **Inherit from `ModalityAdapter`** (in `base_adapter.py`)
2. **Emit `GazeEvent`** objects on the shared `EventBus`
3. **Use the same event types**: `LOCK`, `UNLOCK`, `AMBIGUOUS`
4. **Include `target_id`** in LOCK payload, `candidates` list in AMBIGUOUS payload
5. **Use `time.time()`** for timestamps (same clock as SpeechAdapter)

### Interface Contract

```python
from gazeshop.toolkit.event_bus import EventBus, GazeEvent, GazeEventType

# The real GazeAdapter must emit events like this:
bus.emit(GazeEvent(
    type=GazeEventType.LOCK,
    payload={"target_id": "item_42"},
    confidence=0.95,
))

bus.emit(GazeEvent(
    type=GazeEventType.AMBIGUOUS,
    payload={"candidates": ["item_3", "item_4"]},
    confidence=0.5,
))

bus.emit(GazeEvent(
    type=GazeEventType.UNLOCK,
    payload={},
))
```

### Testing Compatibility

Run the integration tests to verify Speech + Gaze compatibility:

```bash
python -m pytest tests/test_integration.py -v
```

These tests simulate the complete multimodal flow:
- Gaze LOCK + speech "add this to cart" → fusible
- Global command without gaze → works independently
- Ambiguous gaze + speech → disambiguation path
- Unlocked gaze + object command → prompts user
- Gaze time expiry → rejected

---

## FusionEngine Compatibility

The Speech module emits events with all fields required by the FusionEngine:

| Field | Type | Source | Used By |
|---|---|---|---|
| `timestamp` | float | `time.time()` at utterance end | Fusion time-window matching |
| `type` | SpeechEventType | IntentParser | Fusion routing logic |
| `payload.intent` | str | IntentParser | Command dispatch |
| `payload.target_required` | bool | IntentPattern definition | Gaze binding decision |
| `payload.params` | dict | Slot extractors | Action parameters |
| `payload.repair_target` | str | Dialog patterns | Disambiguation resolution |
| `payload.confirm` | bool | Dialog patterns | Confirmation resolution |
| `transcript` | str | ASR engine | Logging, fallback |
| `confidence` | float | Heuristic formula | Confirmation threshold |
| `requires_confirmation` | bool | confidence < threshold | Low-confidence routing |

---

## Technology Stack

| Component | Library | Purpose |
|---|---|---|
| Audio capture | `sounddevice` | 16kHz mono PCM recording |
| ASR (primary) | `vosk` | Offline, lightweight speech recognition |
| ASR (alternate) | `whisper` | Higher accuracy, GPU recommended |
| Key listener | `pynput` | Cross-platform PTT key detection |
| Numerical | `numpy` | Audio buffer handling |
| Testing | `pytest` | Unit and integration tests |

---

## Project Guide and Quick Start

### How Does the Project Work?

GazeShop Speech Modality is a development module built for e-commerce sites to interact with currently gazed-at products using voice commands (e.g., adding to cart or viewing details). The project operates in three fundamental steps:

1. **Listening (Push-to-Talk):** Captures your voice commands via the microphone.
2. **Recognition & Understanding (ASR & Intent Parser):** Transcribes the audio to text (using models like Vosk) and predicts the intent behind the sentence using rule-based logic. Example: "Add this to cart" -> `ADD_TO_CART`.
3. **Synchronization (Event Bus):** Emits special messages called "Events" to the system to check whether the spoken command aligns with the eye-tracking (Gaze) data.

### Folder Structure and Contents

To prevent clutter, the project is designed as an independent set of modules:

- **`gazeshop/` (Core Module):** The heart of the project where all the background logic resides.
  - `intent_parser.py`: Analyzes and interprets the intent behind the voice.
  - `speech_adapter.py`: Establishes the microphone feed and structural connection to the ASR engine.
  - `event_bus.py`: The core messaging system that enables communication between eye movements and speech.
- **`demo/`:** Contains test scripts that demonstrate how the code can be utilized externally, allowing you to speak into the microphone and watch the system's real-time reactions.
- **`tests/`:** Contains unit/integration test files that automatically verify the system's stability across various edge-case scenarios. It's executed after making any modifications to ensure no existing functionality is broken.
- **`models/`:** Houses the language models (e.g., Vosk language model) used for offline speech-to-text transcription, eliminating the need for an internet connection.
- **`agent/rules/`:** Contains the reference documentation (`env.md`) outlining Python environment limitations and general project rules.
- **`.venv/`:** The isolated virtual environment folder where all project dependencies (libraries) are installed.

### How Can I Test It? (Step-by-Step Testing Guide)

To evaluate the capabilities of the project, you can follow a two-tier testing strategy:

#### 1. Running Automated Tests (Code Verification)
You can test how all functions in the system react to various extreme scenarios embedded right into the test code (no microphone needed for this step).

You can launch the test suite by running the following command in your terminal:
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
*(This process simulates hundreds of communication breakdowns, misunderstandings, etc., proving that the code remains robust and doesn't crash.)*

#### 2. Live Audio Demo (Interactive Test)
You can interact with and test the system in real-time by speaking directly into the microphone. This is the best way to measure how fast the system can understand your commands.

Run this command in the terminal:
```bash
.\.venv\Scripts\python.exe demo/speech_demo.py
```

On the interactive screen that appears:
1. Hold down the **Space** key on your keyboard and speak commands into your microphone, such as "add this to cart" or "scroll down".
2. The system will automatically translate your speech into an `Intent`, calculate a confidence score, and log the results to the screen for you.
3. You can also type special terminal commands such as `:lock item_1` via the keyboard to trick the eye-tracking system (as if you were currently looking at an object on the screen), allowing you to manually test the system in "Multimodal Mode".
