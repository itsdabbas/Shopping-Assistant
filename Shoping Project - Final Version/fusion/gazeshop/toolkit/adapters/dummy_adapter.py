"""
Dummy Keyboard Adapter  — Extensibility Proof
==============================================

A minimal modality adapter that reads *keyboard* input and emits
toolkit events.  Its purpose is twofold:

1. **Plugin proof** — demonstrates that any developer can add a new
   input modality by subclassing :class:`~gazeshop.toolkit.adapters.base_adapter.ModalityAdapter`
   and emitting events on the shared bus.

2. **Offline testing** — lets you drive the full fusion + dialogue loop
   from the terminal without a microphone or eye-tracker.

Keyboard mappings (when running)
----------------------------------
Key     Event emitted
------  ------------------------------------------
l       GazeEvent LOCK  (target = "dummy_target")
u       GazeEvent UNLOCK
a       GazeEvent AMBIGUOUS (item_A vs item_B)
i       SpeechEvent INTENT  (configurable)
c       SpeechEvent CONFIRM (yes)
n       SpeechEvent CONFIRM (no)
r       SpeechEvent REPAIR  (left)
x       SpeechEvent CANCEL
q       Stop the adapter

Usage
-----
::

    from gazeshop.toolkit.adapters.dummy_adapter import DummyKeyboardAdapter
    from gazeshop.toolkit.runtime import MMUIToolkit

    tk = MMUIToolkit()
    dummy = DummyKeyboardAdapter(tk.bus, default_intent="OPEN_DETAIL")
    tk.register_adapter(dummy)
    tk.start()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.event_bus import (
    EventBus,
    GazeEvent,
    GazeEventType,
    SpeechEvent,
    SpeechEventType,
)

logger = logging.getLogger(__name__)

# ── Key → action map ────────────────────────────────────────────────

_HELP = """
  DummyKeyboardAdapter controls
  ─────────────────────────────
  l  → Gaze LOCK   (target='dummy_target')
  u  → Gaze UNLOCK
  a  → Gaze AMBIGUOUS (item_A / item_B)
  i  → Speech INTENT  (default_intent)
  c  → Speech CONFIRM yes
  n  → Speech CONFIRM no
  r  → Speech REPAIR  left
  x  → Speech CANCEL
  ?  → Show this help
  q  → Quit / stop adapter
"""


class DummyKeyboardAdapter(ModalityAdapter):
    """Keyboard-driven modality adapter for testing and demonstrations.

    Parameters
    ----------
    event_bus:
        Shared ``EventBus`` instance.
    default_intent:
        Intent name emitted when the user presses ``i``.
        Defaults to ``"OPEN_DETAIL"``.
    default_target:
        ``target_id`` used in LOCK events.  Defaults to ``"dummy_target"``.
    target_required:
        Whether the default intent requires a gaze target.
    blocking:
        If ``True``, :meth:`start` blocks the calling thread (useful for
        simple scripts).  If ``False``, the input loop runs in a daemon
        thread (useful when other threads are also running).
    """

    def __init__(
        self,
        event_bus: EventBus,
        default_intent: str = "OPEN_DETAIL",
        default_target: str = "dummy_target",
        target_required: bool = True,
        blocking: bool = False,
    ) -> None:
        super().__init__(event_bus)
        self.default_intent = default_intent
        self.default_target = default_target
        self.target_required = target_required
        self._blocking = blocking
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── ModalityAdapter lifecycle ────────────────────────────────────

    def start(self) -> None:
        """Start listening for keyboard input."""
        self._running = True
        self._stop_event.clear()
        print(_HELP)
        if self._blocking:
            self._input_loop()
        else:
            self._thread = threading.Thread(
                target=self._input_loop, daemon=True, name="DummyAdapter"
            )
            self._thread.start()
        logger.info("DummyKeyboardAdapter started (intent=%s).", self.default_intent)

    def stop(self) -> None:
        """Stop the keyboard input loop."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("DummyKeyboardAdapter stopped.")

    # ── Input loop ───────────────────────────────────────────────────

    def _input_loop(self) -> None:
        """Read single-character commands from stdin."""
        while self._running and not self._stop_event.is_set():
            try:
                key = input("dummy> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if key == "q":
                logger.info("DummyAdapter: quit requested.")
                self.stop()
                break
            elif key == "?":
                print(_HELP)
            else:
                self._handle_key(key)

    def _handle_key(self, key: str) -> None:
        """Dispatch the appropriate event for *key*."""
        if key == "l":
            self._emit_gaze_lock(self.default_target)
        elif key == "u":
            self._emit_gaze_unlock()
        elif key == "a":
            self._emit_gaze_ambiguous(["item_A", "item_B"])
        elif key == "i":
            self._emit_speech_intent(self.default_intent, self.target_required)
        elif key == "c":
            self._emit_speech_confirm(True)
        elif key == "n":
            self._emit_speech_confirm(False)
        elif key == "r":
            self._emit_speech_repair("left")
        elif key == "x":
            self._emit_speech_cancel()
        else:
            print(f"  Unknown key '{key}'. Press '?' for help.")

    # ── Event factories ──────────────────────────────────────────────

    def _emit_gaze_lock(self, target_id: str) -> None:
        event = GazeEvent(
            timestamp=time.time(),
            type=GazeEventType.LOCK,
            payload={"target_id": target_id},
            confidence=1.0,
        )
        self.event_bus.emit(event)
        logger.debug("DummyAdapter: emitted GAZE LOCK → %s", target_id)
        print(f"  [GAZE] LOCK → {target_id}")

    def _emit_gaze_unlock(self) -> None:
        event = GazeEvent(
            timestamp=time.time(),
            type=GazeEventType.UNLOCK,
            payload={},
            confidence=0.0,
        )
        self.event_bus.emit(event)
        logger.debug("DummyAdapter: emitted GAZE UNLOCK")
        print("  [GAZE] UNLOCK")

    def _emit_gaze_ambiguous(self, candidates: list[str]) -> None:
        formatted = [
            {"id": cid, "pos": ("left" if i == 0 else "right")}
            for i, cid in enumerate(candidates)
        ]
        event = GazeEvent(
            timestamp=time.time(),
            type=GazeEventType.AMBIGUOUS,
            payload={"candidates": formatted},
            confidence=0.5,
        )
        self.event_bus.emit(event)
        logger.debug("DummyAdapter: emitted GAZE AMBIGUOUS: %s", candidates)
        print(f"  [GAZE] AMBIGUOUS → {candidates}")

    def _emit_speech_intent(self, intent: str, target_required: bool) -> None:
        event = SpeechEvent(
            timestamp=time.time(),
            type=SpeechEventType.INTENT,
            payload={
                "intent": intent,
                "params": {},
                "target_required": target_required,
            },
            transcript=intent.lower().replace("_", " "),
            confidence=0.90,
            requires_confirmation=False,
        )
        self.event_bus.emit(event)
        logger.debug("DummyAdapter: emitted SPEECH INTENT: %s", intent)
        print(f"  [SPEECH] INTENT → {intent} (target_required={target_required})")

    def _emit_speech_confirm(self, value: bool) -> None:
        event = SpeechEvent(
            timestamp=time.time(),
            type=SpeechEventType.CONFIRM,
            payload={"confirm": value},
            transcript="yes" if value else "no",
            confidence=1.0,
        )
        self.event_bus.emit(event)
        print(f"  [SPEECH] CONFIRM → {'yes' if value else 'no'}")

    def _emit_speech_repair(self, pos: str) -> None:
        event = SpeechEvent(
            timestamp=time.time(),
            type=SpeechEventType.REPAIR,
            payload={"repair_target": pos},
            transcript=pos,
            confidence=1.0,
        )
        self.event_bus.emit(event)
        print(f"  [SPEECH] REPAIR → {pos}")

    def _emit_speech_cancel(self) -> None:
        event = SpeechEvent(
            timestamp=time.time(),
            type=SpeechEventType.CANCEL,
            payload={},
            transcript="cancel",
            confidence=1.0,
        )
        self.event_bus.emit(event)
        print("  [SPEECH] CANCEL")
