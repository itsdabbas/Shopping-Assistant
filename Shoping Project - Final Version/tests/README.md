# tests/ — Automated Test Suite

This folder contains the automated tests for the GazeShop multimodal toolkit.
All standard tests run **offline** — no webcam or microphone is required.

---

## Running the Tests

Run from the **project root** (the folder containing `README.md`):

```bash
python -m pytest tests
```

For verbose output:

```bash
python -m pytest tests -v
```

The suite currently collects **183 tests**.  A passing run confirms that
the toolkit infrastructure is working correctly end-to-end.

---

## Test Files

| File | What it verifies |
|------|-----------------|
| `test_event_bus.py` | EventBus publish/subscribe, handler registration, event ordering |
| `test_fusion_engine.py` | Gaze–speech time-window fusion, lock TTL, confidence scoring |
| `test_dialogue_manager.py` | DialogueManager state machine — WAIT_TARGET, CONFIRMING, DISAMBIGUATING, cancellation |
| `test_intent_parser.py` | IntentParser regex matching, fuzzy fallback, confidence computation |
| `test_dummy_adapter.py` | DummyAdapter event emission and lifecycle |
| `test_speech_adapter.py` | SpeechAdapter PTT state machine (no actual audio hardware required) |
| `test_runtime.py` | MMUIToolkit façade — adapter registration, action dispatch, lifecycle |
| `test_integration.py` | End-to-end pipeline: simulated gaze lock + speech intent → action handler fired |
| `test_ptt_smoke.py` | Push-to-talk toggle smoke test |

---

## Test Scope

The automated tests cover the **toolkit core** — the reusable, domain-agnostic
components shared by all applications:

- `EventBus` — event dispatch correctness and ordering
- `FusionEngine` — timing windows, confidence thresholds, fusion logic
- `DialogueManager` — all state transitions and timeout/cancellation paths
- `IntentParser` — regex and fuzzy-matching paths, confidence weighting
- `ModalityAdapter` subclasses (stub and dummy adapters)
- `MMUIToolkit` runtime façade

The tests use **simulated events** (programmatic gaze and speech events) so
they run reliably in any environment without hardware.

---

## Manual / Interactive Tests

Three scripts in this folder require real hardware and are run manually:

| Script | Purpose |
|--------|---------|
| `run_audio_test.py` | Verify microphone capture and Vosk ASR outside the demo |
| `run_interactive_audio_test.py` | Interactive PTT session for testing speech recognition accuracy |
| `run_text_demo_test.py` | Text-input simulation of the full pipeline without a camera |

These are development aids, not part of the `pytest` suite.

---

## Optional Dependencies

Some speech-adapter tests import `vosk` and `sounddevice`.  If these
packages are not installed, the affected tests are skipped automatically.

Install all dependencies with:

```bash
python -m pip install -r requirements.txt
```

---

## Hardware Testing

Webcam and microphone functionality can only be verified by running the
live demos:

```bash
python demo/live_shopping_cv.py
python demo/live_kiosk_cv.py
```

See [`demo/README.md`](../demo/README.md) for full demo instructions.
