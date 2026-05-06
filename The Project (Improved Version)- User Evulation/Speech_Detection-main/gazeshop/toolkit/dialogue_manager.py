"""
Dialogue Manager — State Machine for Multimodal Interaction
===========================================================

Listens to Internal Fusion Events (IntentNeedsDisambiguation, etc.) 
and raw SpeechEvents (REPAIR, CONFIRM, CANCEL) to manage dialog.
Emits Fission (UI) events like MultimodalCommandEvent and PromptEvent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType, GazeEvent, GazeEventType
from gazeshop.toolkit.events import (
    IntentReadyEvent,
    IntentNeedsTargetEvent,
    IntentNeedsDisambiguationEvent,
    IntentNeedsConfirmationEvent,
    PromptEvent,
    DisambiguationPromptEvent,
    ConfirmationPromptEvent,
    MultimodalCommandEvent,
    ActionCancelledEvent,
)

logger = logging.getLogger(__name__)


class DialogueManager:
    """State machine mapping Fusion outcomes and user repairs to commands."""
    
    def __init__(self, event_bus: EventBus, config: Config | None = None) -> None:
        self.event_bus = event_bus
        self.config = config or Config()
        
        self.state = "IDLE"
        self._pending_intent: str | None = None
        self._pending_params: dict[str, Any] | None = None
        self._pending_candidates: list[dict[str, str]] = []
        self._pending_target: str | None = None
        self._repair_attempts = 0
        self._state_ts = 0.0
        
        # Subscribe to Internal Fusion Decisions
        self.event_bus.subscribe("IntentReadyEvent", self._on_ready)
        self.event_bus.subscribe("IntentNeedsTargetEvent", self._on_needs_target)
        self.event_bus.subscribe("IntentNeedsDisambiguationEvent", self._on_needs_disambig)
        self.event_bus.subscribe("IntentNeedsConfirmationEvent", self._on_needs_confirm)
        
        # Subscribe to Raw Events for resolution
        self.event_bus.subscribe("SpeechEvent", self._on_speech)
        self.event_bus.subscribe("GazeEvent", self._on_gaze)
        
        logger.info("DialogueManager State Machine initialised.")

    def _reset(self) -> None:
        self.state = "IDLE"
        self._pending_intent = None
        self._pending_params = None
        self._pending_candidates = []
        self._pending_target = None
        self._repair_attempts = 0
        self._state_ts = time.time()

    def _emit_command(self, intent: str, target_id: str | None, params: dict, conf: float) -> None:
        logger.info("Decision Reached! Executing command %s on %s", intent, target_id)
        self.event_bus.emit(MultimodalCommandEvent(
            intent=intent, target_id=target_id, params=params, confidence=conf
        )) # type: ignore
        self._reset()
        
    def _cancel(self, reason: str, msg: str) -> None:
        logger.info("Interaction Cancelled (%s): %s", reason, msg)
        self.event_bus.emit(ActionCancelledEvent(reason=reason, message=msg)) # type: ignore
        self._reset()

    # ── Fusion Subscribers ──────────────────────────────────────────

    def _on_ready(self, event: IntentReadyEvent) -> None:
        self._emit_command(event.intent, event.target_id, event.params, event.confidence)

    def _on_needs_target(self, event: IntentNeedsTargetEvent) -> None:
        self._reset()
        self.state = "WAIT_TARGET"
        self._pending_intent = event.intent
        self._pending_params = event.source_speech.payload.get("params", {}) if event.source_speech else {}
        self._state_ts = time.time()
        
        msg = "Please look at an item to apply the command."
        if event.reason == "lock_expired":
            msg = "Lock expired. Please look at the item again."
            
        self.event_bus.emit(PromptEvent(message=msg)) # type: ignore

    def _on_needs_disambig(self, event: IntentNeedsDisambiguationEvent) -> None:
        self._reset()
        self.state = "WAIT_DISAMBIGUATION"
        self._pending_intent = event.intent
        self._pending_params = event.source_speech.payload.get("params", {}) if event.source_speech else {}
        self._pending_candidates = event.candidates
        self._state_ts = time.time()
        
        msg = f"Which one? Options: {', '.join(c.get('pos', '?') for c in event.candidates)}"
        self.event_bus.emit(DisambiguationPromptEvent(message=msg, candidates=event.candidates)) # type: ignore

    def _on_needs_confirm(self, event: IntentNeedsConfirmationEvent) -> None:
        self._reset()
        self.state = "WAIT_CONFIRMATION"
        self._pending_intent = event.intent
        self._pending_target = event.target_id
        if event.source_speech:
             self._pending_params = event.source_speech.payload.get("params", {}) 
        self._state_ts = time.time()
        
        target_str = f" on {event.target_id}" if event.target_id else ""
        msg = f"Did you mean '{event.intent}'{target_str}? Say yes or no."
        self.event_bus.emit(ConfirmationPromptEvent(message=msg, intent=event.intent, target_id=event.target_id)) # type: ignore

    # ── Input Subscribers ─────────────────────────────────────────

    def _on_gaze(self, event: GazeEvent) -> None:
        if self.state == "WAIT_TARGET" and event.type == GazeEventType.LOCK:
            target_id = event.payload.get("target_id")
            logger.info("Gaze lock acquired while WAITING -> resolving intent.")
            self._emit_command(self._pending_intent or "", target_id, self._pending_params or {}, 0.9)

    def _on_speech(self, event: SpeechEvent) -> None:
        # Timeout handling
        if self.state in ["WAIT_DISAMBIGUATION", "WAIT_CONFIRMATION"]:
            if time.time() - self._state_ts > self.config.DISAMBIGUATION_TIMEOUT_S:
                self._cancel("timeout", "Dialog timed out.")
                return

        # Cancel logic
        if event.type == SpeechEventType.CANCEL:
            if self.state != "IDLE":
                self._cancel("user_cancelled", "Action cancelled by user.")
            return

        # Resolution Logic
        if self.state == "WAIT_DISAMBIGUATION" and event.type == SpeechEventType.REPAIR:
            self._repair_attempts += 1
            pos = event.payload.get("repair_target", "")
            
            resolved_id = next((c["id"] for c in self._pending_candidates if c["pos"] == pos), None)
            if not resolved_id: # fallback mapping 
                if pos == "first": pos = "left"
                elif pos == "second": pos = "right"
                resolved_id = next((c["id"] for c in self._pending_candidates if c["pos"] == pos), None)

            if resolved_id:
                logger.info("Disambiguation resolved to %s.", resolved_id)
                self._emit_command(self._pending_intent or "", resolved_id, self._pending_params or {}, 0.9)
            else:
                if self._repair_attempts >= self.config.MAX_REPAIR_ATTEMPTS:
                    self._cancel("max_retries", "Exceeded max repair attempts.")
                else:
                    msg = f"Could not find '{pos}'. Please say 'left' or 'right'."
                    self.event_bus.emit(DisambiguationPromptEvent(message=msg, candidates=self._pending_candidates)) # type: ignore

        elif self.state == "WAIT_CONFIRMATION":
            if event.type == SpeechEventType.CONFIRM:
                is_confirmed = event.payload.get("confirm", True)
                if is_confirmed:
                    logger.info("Confirmation received.")
                    self._emit_command(self._pending_intent or "", self._pending_target, self._pending_params or {}, 0.9)
                else:
                    self._cancel("user_deny", "Action denied by user.")
