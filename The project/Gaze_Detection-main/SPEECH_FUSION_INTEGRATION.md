# Speech/Fusion Integration (Current Runtime)

This document is **1:1 aligned** with the current project files and runtime behavior.

## Files Used at Runtime

- `gaze_server.py` - WebSocket server, gaze stream, speech transcript input, fusion output
- `fusion.py` - `FusionEngine` and `SpeechEvent`
- `intents.py` - intent patterns and parsing rules

`event_bus.py` exists in the project but is **not used** in the current runtime flow.

## WebSocket Endpoint

- Default: `ws://127.0.0.1:8765`
- Configurable with:
  - `GAZE_WS_HOST`
  - `GAZE_WS_PORT`

## Current Runtime Mapping

All of the following are implemented inside `gaze_server.py`:

1. `ws_handler()`:
   - accepts incoming commands
   - emits `CAPABILITIES` on connect
   - handles `speech_transcript`
2. `camera_loop()`:
   - emits continuous gaze frame packets
   - includes `gaze_event` in each frame
3. `FusionEngine` (`fusion.py`):
   - parses transcript using `intents.py`
   - binds target using latest gaze lock
   - emits intent/ambiguous fusion result

## Messages Sent By Server

## 1) `CAPABILITIES` (sent once after connect)

```json
{
  "type": "CAPABILITIES",
  "timestamp": 1710000000.123,
  "payload": {
    "speech_ready": true,
    "accepts_transcript_command": true,
    "command_catalog": {
      "object_bound": [],
      "global": [],
      "dialog_only": {}
    }
  }
}
```

`command_catalog` is generated from `intents.py` and reflects current patterns.

## 2) Continuous gaze frame packet

No top-level `type` field. This is the normal stream packet from `camera_loop()`.

Important keys:

- `status`, `gaze_x`, `gaze_y`
- `cal_step`, `cal_total`, `cal_target`
- `frame_jpg`
- `gaze_event`

Example `gaze_event`:

```json
{
  "timestamp": 1710000000.456,
  "type": "LOCK",
  "payload": {"target_id":"grid-0-1"}
}
```

Possible `gaze_event.type` values:

- `LOCK`
- `UNLOCK`
- `AMBIGUOUS`

## 3) `SPEECH_EVENT` (response to transcript input)

```json
{
  "type": "SPEECH_EVENT",
  "timestamp": 1710000000.789,
  "payload": {
    "timestamp": 1710000000.789,
    "type": "INTENT",
    "payload": {
      "intent_name": "ADD_TO_CART",
      "target_required": true
    },
    "transcript": "add this to cart",
    "confidence": 0.87,
    "requires_confirmation": false
  }
}
```

## 4) `FUSION_EVENT` (response to transcript input)

Success example:

```json
{
  "type": "FUSION_EVENT",
  "timestamp": 1710000000.790,
  "payload": {
    "type": "INTENT",
    "timestamp": 1710000000.789,
    "payload": {
      "intent_name": "ADD_TO_CART",
      "target_id": "grid-0-1",
      "params": {},
      "requires_confirmation": false
    },
    "speech": {}
  }
}
```

Ambiguous example:

```json
{
  "type": "FUSION_EVENT",
  "timestamp": 1710000000.790,
  "payload": {
    "type": "AMBIGUOUS",
    "timestamp": 1710000000.789,
    "payload": {
      "reason": "target_required",
      "candidates": []
    },
    "speech": {}
  }
}
```

## Commands Accepted By Server

Send these over WebSocket:

```json
{"cmd":"capture"}
```

```json
{"cmd":"reset"}
```

```json
{"cmd":"speech_transcript","transcript":"add this to cart","confidence":0.9}
```

`transcript` is required for `speech_transcript`.  
`confidence` is optional (defaults to `0.9` in `gaze_server.py`).

## Intent Source of Truth

Intent definitions and regex patterns are in `intents.py`:

- Object-bound: `ADD_TO_CART`, `SHOW_DETAILS`, `FIND_SIMILAR`, `COMPARE`, `SHOW_ALTERNATIVES`, `PIN_ITEM`, `REMOVE_ITEM`
- Global: `SCROLL`, `OPEN_CART`, `GO_BACK`, `HELP`, `CANCEL`, `UNDO`
- Dialog patterns: confirm/deny/repair/repeat

If your team changes command phrases, update `intents.py`.

## Teammate Integration Steps

1. Connect to `ws://127.0.0.1:8765`.
2. Wait for `CAPABILITIES`.
3. Forward ASR results as `speech_transcript`.
4. Handle:
   - continuous gaze packets (`gaze_event` inside packet)
   - `SPEECH_EVENT`
   - `FUSION_EVENT`

## Minimal Client Example

```js
const ws = new WebSocket("ws://127.0.0.1:8765");

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "CAPABILITIES") {
    console.log("Capabilities", msg.payload);
    return;
  }
  if (msg.type === "SPEECH_EVENT") {
    console.log("Speech event", msg.payload);
    return;
  }
  if (msg.type === "FUSION_EVENT") {
    console.log("Fusion event", msg.payload);
    return;
  }
  // Regular gaze stream packet:
  // msg.gaze_event is available here.
};

function forwardAsr(transcript, confidence = 0.9) {
  ws.send(JSON.stringify({
    cmd: "speech_transcript",
    transcript,
    confidence
  }));
}
```
