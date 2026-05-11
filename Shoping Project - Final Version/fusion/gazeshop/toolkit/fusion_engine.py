"""
Fusion Engine — Late Fusion for GazeShop (Refactored)
=====================================================

Responsible ONLY for matching multimodal inputs (Gaze + Speech).
Evaluates the rules (intent vs gaze lock state, TTL) and emits 
Internal Events to the Dialogue Manager.

Does NOT handle dialog state.
"""

from __future__ import annotations

import logging
import time

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import (
    EventBus,
    GazeEvent,
    GazeEventType,
    SpeechEvent,
    SpeechEventType,
)
from gazeshop.toolkit.events import (
    IntentReadyEvent,
    IntentNeedsTargetEvent,
    IntentNeedsDisambiguationEvent,
    IntentNeedsConfirmationEvent,
    TargetLockedEvent,
    TargetUnlockedEvent,
    TargetExpiredEvent,
)

logger = logging.getLogger(__name__)


class FusionEngine:
    def __init__(self, event_bus: EventBus, config: Config | None = None) -> None:
        self.event_bus = event_bus
        self.config = config or Config()

        self._gaze_state: str = "idle"
        self._locked_target: str | None = None
        self._lock_timestamp: float | None = None
        self._ambiguous_candidates: list[dict[str, str]] = []

        self.event_bus.subscribe("GazeEvent", self._on_gaze_event)
        self.event_bus.subscribe("SpeechEvent", self._on_speech_event)
        
        logger.info(
            "FusionEngine initialised (window=%.1fs, lock_ttl=%.1fs).",
            self.config.FUSION_TIME_WINDOW_S,
            self.config.LOCK_TTL_S,
        )

    def _on_gaze_event(self, event: GazeEvent) -> None:
        if event.type == GazeEventType.LOCK:
            self._locked_target = event.payload.get("target_id")
            self._lock_timestamp = event.timestamp
            self._gaze_state = "locked"
            self._ambiguous_candidates = []
            
            logger.debug("Gaze LOCK → %s", self._locked_target)
            self.event_bus.emit(TargetLockedEvent(
                target_id=self._locked_target, 
                timestamp=event.timestamp
            )) # type: ignore

        elif event.type == GazeEventType.UNLOCK:
            self._locked_target = None
            self._lock_timestamp = None
            self._gaze_state = "idle"
            
            logger.debug("Gaze UNLOCK")
            self.event_bus.emit(TargetUnlockedEvent(timestamp=event.timestamp)) # type: ignore

        elif event.type == GazeEventType.AMBIGUOUS:
            self._locked_target = None
            self._lock_timestamp = None
            self._gaze_state = "ambiguous"
            self._ambiguous_candidates = event.payload.get("candidates", [])
            logger.debug("Gaze AMBIGUOUS: %s", self._ambiguous_candidates)

    def _on_speech_event(self, event: SpeechEvent) -> None:
        if event.type != SpeechEventType.INTENT:
            return

        intent = event.payload.get("intent", "UNKNOWN")
        target_required = event.payload.get("target_required", False)
        params = event.payload.get("params", {})

        # Rule 6: Global Command
        if not target_required:
            self.event_bus.emit(IntentReadyEvent(
                intent=intent,
                target_id=None,
                params=params,
                confidence=event.confidence,
                source_speech=event,
            )) # type: ignore
            return

        # Rule 4: Low Confidence -> Confirmation 
        if event.requires_confirmation:
            target = self._locked_target if self._gaze_state == "locked" else None
            self.event_bus.emit(IntentNeedsConfirmationEvent(
                intent=intent,
                target_id=target,
                source_speech=event,
            )) # type: ignore
            return

        # Rule 3: Ambiguous Gaze -> Disambiguation
        if self._gaze_state == "ambiguous" and self._ambiguous_candidates:
            self.event_bus.emit(IntentNeedsDisambiguationEvent(
                intent=intent,
                candidates=self._ambiguous_candidates,
                source_speech=event,
            )) # type: ignore
            return

        # Check Active Lock & TTL
        #
        # Two independent timing gates protect fusion quality:
        #
        #   LOCK_TTL_S (default 4.0 s) — hard ceiling on lock validity.
        #     When exceeded the lock is considered physically stale; the
        #     gaze state is reset and the user must look at the item again.
        #     Emits TargetExpiredEvent so the UI can update accordingly.
        #
        #   FUSION_TIME_WINDOW_S (default 2.0 s) — multimodal co-occurrence
        #     window.  Even if the lock is still within TTL, speech that
        #     arrives more than this many seconds after the lock timestamp
        #     is not treated as a deliberate gaze+speech gesture.  The lock
        #     itself is NOT invalidated; only this particular speech event is
        #     considered too late to fuse, and the user is asked to re-look.
        #
        if self._gaze_state == "locked" and self._lock_timestamp:
            age = time.time() - self._lock_timestamp

            # ── Gate 1: hard TTL — lock is stale, reset everything ───
            if age > self.config.LOCK_TTL_S:
                logger.warning("Lock passed TTL (%.1fs). Requesting re-lock.", age)
                self.event_bus.emit(TargetExpiredEvent(target_id=self._locked_target)) # type: ignore
                self.event_bus.emit(IntentNeedsTargetEvent(
                    reason="lock_expired",
                    intent=intent,
                    source_speech=event,
                )) # type: ignore

                # Reset stale state
                self._gaze_state = "idle"
                self._locked_target = None
                return

            # ── Gate 2: fusion window — lock alive but speech too late ─
            if age > self.config.FUSION_TIME_WINDOW_S:
                logger.info(
                    "Speech arrived %.2fs after lock (fusion window=%.1fs); "
                    "too late to fuse — requesting re-look.",
                    age, self.config.FUSION_TIME_WINDOW_S,
                )
                self.event_bus.emit(IntentNeedsTargetEvent(
                    reason="fusion_window_expired",
                    intent=intent,
                    source_speech=event,
                )) # type: ignore
                # Do NOT reset gaze state — the lock is still physically valid
                # and will accept the next deliberate gaze+speech pair.
                return

            # ── Both gates passed → fused successfully! ───────────────
            self.event_bus.emit(IntentReadyEvent(
                intent=intent,
                target_id=self._locked_target,
                params=params,
                confidence=round(
                    (event.confidence + self.config.FUSION_CONFIDENCE_BIAS) / 2, 4
                ),
                source_speech=event,
                source_gaze={"state": "locked", "target_id": self._locked_target}
            )) # type: ignore
            return

        # Rule 2: No valid lock
        self.event_bus.emit(IntentNeedsTargetEvent(
            reason="no_lock",
            intent=intent,
            source_speech=event,
        )) # type: ignore
