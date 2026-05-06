"""
Event Bus & Event Dataclasses
=============================

This module defines:
1. **Event dataclasses** — the standardised messages that flow between
   modality adapters, the FusionEngine, and the DialogueManager.
2. **EventBus** — a lightweight synchronous publish / subscribe
   dispatcher that decouples producers from consumers.

Design notes
------------
* ``SpeechEvent`` and ``GazeEvent`` share the ``Modality`` enum so the
  FusionEngine can distinguish sources without type-checking.
* Timestamps use ``time.time()`` (POSIX epoch float) so gaze and speech
  events are directly comparable on the same clock.
* The ``EventBus`` supports both type-specific and wildcard (``"*"``)
  subscriptions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    """Identifies which sensory channel produced an event."""
    GAZE = "gaze"
    SPEECH = "speech"


class SpeechEventType(str, Enum):
    """Types of events emitted by the SpeechAdapter."""
    INTENT = "INTENT"           # Parsed command intent
    REPAIR = "REPAIR"           # Disambiguation repair (left/right/etc.)
    CONFIRM = "CONFIRM"         # Yes / No confirmation
    CANCEL = "CANCEL"           # Cancel current action
    UNDO = "UNDO"               # Undo last action
    REPEAT = "REPEAT"           # Re-prompt request
    LISTENING = "LISTENING"     # Mic activated (UI state change)
    STOPPED = "STOPPED"         # Mic deactivated
    ERROR = "ERROR"             # ASR / parse failure


class GazeEventType(str, Enum):
    """Types of events emitted by the GazeAdapter."""
    LOCK = "LOCK"               # User gaze locked on a target
    UNLOCK = "UNLOCK"           # User gaze left the target
    AMBIGUOUS = "AMBIGUOUS"     # Gaze is between two candidates


# ────────────────────────────────────────────────────────────────────
# Event dataclasses
# ────────────────────────────────────────────────────────────────────

@dataclass
class SpeechEvent:
    """Standardised event emitted by SpeechAdapter.

    Payload keys vary by ``type``:

    =========  ========================================================
    Type       Payload
    =========  ========================================================
    INTENT     ``{ intent: str, params: dict, target_required: bool }``
    REPAIR     ``{ repair_target: str }``   e.g. ``"left"``
    CONFIRM    ``{ confirm: bool }``        True = yes, False = no
    CANCEL     ``{}``
    UNDO       ``{}``
    REPEAT     ``{}``
    ERROR      ``{ error: str }``
    LISTENING  ``{}``
    STOPPED    ``{}``
    =========  ========================================================
    """

    timestamp: float = field(default_factory=time)
    modality: Modality = Modality.SPEECH
    type: SpeechEventType = SpeechEventType.INTENT
    payload: dict[str, Any] = field(default_factory=dict)
    transcript: str = ""
    confidence: float = 0.0
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (e.g. for JSON logging)."""
        return {
            "timestamp": self.timestamp,
            "modality": self.modality.value,
            "type": self.type.value,
            "payload": self.payload,
            "transcript": self.transcript,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
        }

    def __repr__(self) -> str:
        return (
            f"SpeechEvent(type={self.type.value}, "
            f"transcript={self.transcript!r}, "
            f"confidence={self.confidence:.2f}, "
            f"payload={self.payload})"
        )


@dataclass
class GazeEvent:
    """Standardised event emitted by GazeAdapter.

    Payload keys vary by ``type``:

    =========  =====================================================
    Type       Payload
    =========  =====================================================
    LOCK       ``{ target_id: str }``
    UNLOCK     ``{}``
    AMBIGUOUS  ``{ candidates: list[str] }``  e.g. ``["item_3", "item_4"]``
    =========  =====================================================
    """

    timestamp: float = field(default_factory=time)
    modality: Modality = Modality.GAZE
    type: GazeEventType = GazeEventType.LOCK
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "modality": self.modality.value,
            "type": self.type.value,
            "payload": self.payload,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        return (
            f"GazeEvent(type={self.type.value}, "
            f"payload={self.payload}, "
            f"confidence={self.confidence:.2f})"
        )


# ────────────────────────────────────────────────────────────────────
# EventBus — lightweight pub/sub
# ────────────────────────────────────────────────────────────────────

class EventBus:
    """Simple synchronous publish / subscribe event dispatcher.

    Subscribers register for a specific event *class name* or ``"*"``
    (wildcard) to receive every event.

    Example
    -------
    >>> bus = EventBus()
    >>> bus.subscribe("SpeechEvent", lambda e: print(e))
    >>> bus.emit(SpeechEvent(type=SpeechEventType.LISTENING))
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._event_log: list[dict] = []
        self._logging_enabled: bool = False

    # ── Public API ──────────────────────────────────────────────────

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register *callback* for events whose class name is *event_type*.

        Use ``"*"`` to subscribe to all events.

        Parameters
        ----------
        event_type:
            Class name (e.g. ``"SpeechEvent"``, ``"GazeEvent"``)
            or ``"*"`` for wildcard.
        callback:
            Callable that accepts a single event argument.
        """
        self._subscribers[event_type].append(callback)
        logger.debug("Subscriber registered for %s", event_type)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Remove a previously registered callback."""
        try:
            self._subscribers[event_type].remove(callback)
        except ValueError:
            logger.warning(
                "Attempted to unsubscribe a callback that was not registered "
                "for event type '%s'.",
                event_type,
            )

    def emit(self, event: Any) -> None:
        """Dispatch *event* to all matching subscribers.

        Matching rules:
        1. Subscribers registered for ``type(event).__name__``.
        2. Subscribers registered for ``"*"`` (wildcard).

        Parameters
        ----------
        event:
            An event instance.
        """
        event_class = type(event).__name__

        if self._logging_enabled:
            if hasattr(event, "to_dict"):
                self._event_log.append(event.to_dict())
            else:
                from dataclasses import asdict, is_dataclass
                if is_dataclass(event):
                    try:
                        # Some nested fields like GazeEvent aren't clean, fail gracefully
                        self._event_log.append(asdict(event))
                    except Exception:
                        self._event_log.append({"type": event_class, "repr": repr(event)})
                else:
                    self._event_log.append({"type": event_class, "repr": repr(event)})

        # Type-specific subscribers
        for cb in self._subscribers.get(event_class, []):
            try:
                cb(event)
            except Exception:
                logger.exception(
                    "Error in subscriber for %s", event_class,
                )

        # Wildcard subscribers
        for cb in self._subscribers.get("*", []):
            try:
                cb(event)
            except Exception:
                logger.exception("Error in wildcard subscriber")

    # ── Logging helpers ─────────────────────────────────────────────

    def enable_logging(self) -> None:
        """Start recording every emitted event for later inspection."""
        self._logging_enabled = True

    def disable_logging(self) -> None:
        """Stop recording events."""
        self._logging_enabled = False

    def get_log(self) -> list[dict]:
        """Return a copy of the event log."""
        return list(self._event_log)

    def clear_log(self) -> None:
        """Discard all recorded events."""
        self._event_log.clear()
