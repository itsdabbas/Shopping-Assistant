"""
WebSocket Broadcaster
=====================
Subscribe to all UI events on the EventBus and broadcast them to the browser app.
Runs an asyncio WebSocket server in a background thread.
"""

from ast import Import
import asyncio
import json
import threading

import websockets

CLIENTS: set = set()
_LOOP: asyncio.AbstractEventLoop | None = None
_HARNESS = None 

async def _handler(websocket):
    CLIENTS.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get('type') == 'PTT' and _HARNESS is not None:
                    if data['action'] == 'press':
                        def _press():
                            # Set dialog context before starting recording.
                            # When the Dialogue Manager is waiting for confirmation or
                            # disambiguation (state != "IDLE"), the Intent Parser switches
                            # to dialog mode and recognises replies like "yes", "no",
                            # "left", "right" instead of treating them as regular commands.
                            _HARNESS.speech.set_dialog_active(_HARNESS.dm.state != "IDLE")
                            _HARNESS.speech.on_ptt_press()
                        threading.Thread(target=_press, daemon=True).start()
                    elif data['action'] == 'release':
                        threading.Thread(target=_HARNESS.speech.on_ptt_release, daemon=True).start()
                elif data.get('type') == 'SIMULATE' and data.get('event') == 'AMBIGUOUS':
                    import time
                    from gazeshop.toolkit.event_bus import GazeEvent, GazeEventType                    
                    ge = GazeEvent(
                        timestamp=time.time(),
                        type=GazeEventType.AMBIGUOUS,
                        payload={"candidates": data.get("candidates", [])},
                        confidence=0.5
                    )
                    _HARNESS.bus.emit(ge)
            except Exception:
                pass
    finally:
        CLIENTS.discard(websocket)

async def _broadcast(msg: dict):
    if not CLIENTS:
        return
    data = json.dumps(msg, default=str)
    await asyncio.gather(*[c.send(data) for c in list(CLIENTS)],
                         return_exceptions=True)


def _fire(msg: dict):
    """Broadcast messages safely from any thread"""
    if _LOOP is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _LOOP)


def start_ws_server(host: str = "localhost", port: int = 8766):
    """
    Start a background WebSocket server.
    Use port 8766 to avoid conflict with gaze_server.py on 8765.
    Return the event loop for external use.
    """
    global _LOOP
    _LOOP = asyncio.new_event_loop()

    async def _run():
        async with websockets.serve(_handler, host, port):
            print(f"[WS_BROADCASTER] Listening on ws://{host}:{port}")
            await asyncio.Future()

    def _thread():
        _LOOP.run_until_complete(_run())

    t = threading.Thread(target=_thread, daemon=True)
    t.start()
    return _LOOP


def register_bus_hooks(bus, harness):
    """
    Forward all UI events from the EventBus to the WebSocket.
    Call this at the end of HarnessSystem.setup_components().
    """
    global _HARNESS
    _HARNESS = harness

    bus.subscribe("TargetLockedEvent",
        lambda e: _fire({
            "type": "LOCKED",
            "target_id": e.target_id
        }))

    bus.subscribe("TargetUnlockedEvent",
        lambda e: _fire({
            "type": "UNLOCKED"
        }))

    bus.subscribe("TargetExpiredEvent",
        lambda e: _fire({
            "type": "EXPIRED",
            "target_id": e.target_id
        }))

    bus.subscribe("MultimodalCommandEvent",
        lambda e: _fire({
            "type": "COMMAND",
            "intent": e.intent,
            "target_id": e.target_id,
            "params": e.params,
            "confidence": e.confidence
        }))

    bus.subscribe("DisambiguationPromptEvent",
        lambda e: _fire({
            "type": "DISAMBIGUATION",
            "message": e.message,
            "candidates": e.candidates
        }))

    bus.subscribe("ConfirmationPromptEvent",
        lambda e: _fire({
            "type": "CONFIRMATION",
            "message": e.message,
            "intent": e.intent,
            "target_id": e.target_id
        }))

    bus.subscribe("ActionCancelledEvent",
        lambda e: _fire({
            "type": "CANCELLED",
            "reason": e.reason,
            "message": e.message
        }))

    bus.subscribe("PromptEvent",
        lambda e: _fire({
            "type": "PROMPT",
            "message": e.message
        }))

    bus.subscribe("SpeechEvent",
        lambda e: _fire({
            "type": "SPEECH",
            "speech_type": e.type.value,
            "transcript": getattr(e, "transcript", ""),
            "confidence": e.confidence
        }))