"""
Event Definitions
=================

Extends event_bus.py definitions with Fission (UI Feedback) events
and Internal Fusion events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from gazeshop.toolkit.event_bus import SpeechEvent


# ── Internal Fusion Events (FusionEngine -> DialogueManager) ────────

@dataclass
class IntentReadyEvent:
    """An intent and target were successfully fused (no dialog needed)."""
    intent: str
    target_id: str | None
    params: dict[str, Any]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    source_speech: SpeechEvent | None = None
    source_gaze: dict | None = None


@dataclass
class IntentNeedsTargetEvent:
    """An object-bound intent was parsed, but no valid gaze lock exists."""
    reason: str  # "no_lock" | "lock_expired"
    intent: str
    timestamp: float = field(default_factory=time.time)
    source_speech: SpeechEvent | None = None


@dataclass
class IntentNeedsDisambiguationEvent:
    """An object-bound intent was parsed, but gaze is AMBIGUOUS."""
    intent: str
    candidates: list[dict[str, str]]  # e.g., [{"id": "item1", "pos": "left"}, ...]
    timestamp: float = field(default_factory=time.time)
    source_speech: SpeechEvent | None = None


@dataclass
class IntentNeedsConfirmationEvent:
    """An intent was parsed with low confidence."""
    intent: str
    target_id: str | None
    timestamp: float = field(default_factory=time.time)
    source_speech: SpeechEvent | None = None


# ── Fission / UI Feedback Events (DialogueManager -> UI) ────────────

@dataclass
class TargetLockedEvent:
    """Indicates that a gaze target is now locked."""
    target_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TargetUnlockedEvent:
    """Indicates that the user looked away and the lock cleared."""
    timestamp: float = field(default_factory=time.time)


@dataclass
class TargetExpiredEvent:
    """Indicates that an existing lock expired (TTL passed)."""
    target_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PromptEvent:
    """Generic prompt (e.g., 'Please look at an item.')."""
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class DisambiguationPromptEvent:
    """UI overlay for disambiguation (Left vs Right)."""
    message: str
    candidates: list[dict[str, str]]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConfirmationPromptEvent:
    """UI overlay for low-confidence confirmation (Yes/No)."""
    message: str
    intent: str
    target_id: str | None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionCancelledEvent:
    """Notifies UI that the active interaction was cancelled or timed out."""
    reason: str  # "timeout" | "user_cancelled" | "max_retries"
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MultimodalCommandEvent:
    """The final resolved command that the UI must execute."""
    intent: str
    target_id: str | None
    params: dict[str, Any]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MultimodalCommandEvent",
            "intent": self.intent,
            "target_id": self.target_id,
            "params": self.params,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
