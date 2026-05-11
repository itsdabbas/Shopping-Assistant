# speech/ — Speech Recognition Resources

This folder contains offline speech recognition model files used by the
`SpeechAdapter` in the GazeShop toolkit.

```
speech/
└── models/
    ├── vosk-model-small-en-us-0.15/    Vosk model directory (used at runtime)
    └── vosk-model-small-en-us-0.15.zip Archive copy of the model
```

---

## Speech Model

The demos use the **Vosk** offline ASR engine with the small English model:

```
speech/models/vosk-model-small-en-us-0.15/
```

Both live demos resolve this path automatically relative to the project root.
No manual configuration is needed as long as the project is run from its root
directory.

The model is configured inside each demo via:

```python
cfg = Config(
    ASR_ENGINE      = "vosk",
    VOSK_MODEL_PATH = str(ROOT / "speech" / "models" / "vosk-model-small-en-us-0.15"),
)
```

where `ROOT` is the project root resolved at startup from `__file__`.

---

## How Speech Is Used in the Toolkit

The speech pipeline is implemented in
`fusion/gazeshop/toolkit/adapters/speech_adapter.py` and operates
entirely within a background worker thread so the UI render loop is
never blocked.

**Pipeline stages:**

```
PTT toggle ON
  → sounddevice audio capture (16 kHz, mono, int16)
  → Voice Activity Detection (Silero → WebRTC → none, automatic fallback)
  → ASR engine (Vosk or Whisper)
  → IntentParser.parse(transcript)
  → SpeechEvent emitted on EventBus
PTT toggle OFF
```

**Application vocabulary injection:**

The `SpeechAdapter` and `IntentParser` are initialised with
application-specific data supplied by the app layer — the toolkit core
contains no shopping or kiosk vocabulary.

- `SpeechAdapter(vosk_vocab=SHOPPING_VOSK_VOCAB)` restricts the Vosk
  grammar to the shopping command set, improving recognition accuracy.
- `IntentParser(custom_patterns=SHOPPING_INTENTS)` maps transcripts to
  shopping intent tags.

Vocabulary for each application lives in:

```
fusion/apps/shopping/intents.py    SHOPPING_VOSK_VOCAB, SHOPPING_INTENTS
fusion/apps/kiosk/intents.py       KIOSK_VOSK_VOCAB, KIOSK_INTENTS
```

---

## Push-to-Talk

Both live demos use a **toggle push-to-talk** interface controlled by the
`M` key in the OpenCV window:

- **First M press** — begins recording; the microphone indicator turns red.
- **Second M press** — stops recording and sends the audio for ASR processing.

The pynput key listener is disabled in the CV demos; PTT is driven by
`cv2.waitKey()` to avoid conflicts with the OpenCV window focus.

---

## Supported Commands

### Shopping demo

Commands recognised by the shopping vocabulary (all map to `ADD_TO_CART`
unless otherwise noted):

| Spoken phrase | Intent |
|---------------|--------|
| add to cart, add, cart, card | `ADD_TO_CART` |
| add cart, put in cart, add this | `ADD_TO_CART` |
| to cart, two cart, to card, two card | `ADD_TO_CART` |
| show details, details | `SHOW_DETAILS` |
| find similar | `FIND_SIMILAR` |
| compare | `COMPARE` |
| yes / no / cancel | Dialog responses |

### Kiosk demo

| Spoken phrase | Intent |
|---------------|--------|
| tell me about this, tell, me, about, this | `READ_ALOUD` |
| me about, tell me, about this, tell about, tell this | `READ_ALOUD` |
| bookmark this, bookmark, book, mark, book mark | `PIN_EXHIBIT` |
| pin, pin this, save, save this | `PIN_EXHIBIT` |
| summarize | `SUMMARIZE` |
| zoom in | `ZOOM_IN` |
| open detail | `OPEN_DETAIL` |
| next | `NAVIGATE_NEXT` |
| go back, back, previous | `NAVIGATE_PREV` |
| help | `HELP` |
| cancel, never mind | `CANCEL` |

---

## Troubleshooting

| Problem | Likely cause and fix |
|---------|----------------------|
| `FileNotFoundError` or `Model not found` at startup | The Vosk model directory is missing. Verify that `speech/models/vosk-model-small-en-us-0.15/` exists relative to the project root. |
| Microphone not detected | Ensure the microphone is connected and set as the default input device in the OS audio settings. |
| `ModuleNotFoundError: vosk` | Run `python -m pip install vosk`. |
| `ModuleNotFoundError: rapidfuzz` | Run `python -m pip install rapidfuzz`. |
| `ModuleNotFoundError: sounddevice` | Run `python -m pip install sounddevice`. |
| ASR returns empty or incorrect transcript | Speak clearly and close to the microphone; avoid background noise; the small Vosk model has limited accuracy on short or unusual phrases. |
| Raw transcript visible in demo terminal | Each processed utterance prints `[SPEECH] raw='...'  intent=...  conf=...` to the console — this is intentional debug output from `IntentParser._log_and_return()`. |
| Switching to Whisper | Edit the `Config(ASR_ENGINE="whisper", WHISPER_MODEL_SIZE="small")` block in the demo file.  Whisper provides higher accuracy but requires `pip install openai-whisper` and is slower on CPU. |
