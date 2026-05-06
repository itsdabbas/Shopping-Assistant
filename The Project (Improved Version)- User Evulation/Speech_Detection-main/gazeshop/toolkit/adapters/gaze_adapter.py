"""
Gaze Adapter — Stub
===================

This module provides interface stubs for the Gaze modality adapter.
The full implementation resides in a separate gaze module; this stub
exists so that:

1. The **event contracts** (``GazeEvent``, ``GazeEventType``) are
   importable from the same toolkit.
2. The **FusionEngine** can be developed and tested against a mock
   gaze source without requiring real eye-tracking hardware.
3. Developers can see the expected interface when building the real
   ``GazeAdapter``.

Usage (simulation)
------------------
    bus = EventBus()
    gaze = GazeAdapterStub(bus)
    gaze.simulate_lock("item_42")
    gaze.simulate_ambiguous(["item_42", "item_43"])
    gaze.simulate_unlock()
"""

from __future__ import annotations

import logging
import time as _time

from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.event_bus import EventBus, GazeEvent, GazeEventType

logger = logging.getLogger(__name__)


class GazeAdapterStub(ModalityAdapter):
    """Stub gaze adapter for development and testing.

    This adapter does **not** connect to real eye-tracking hardware.
    Instead, it exposes programmatic methods to simulate gaze events
    that the FusionEngine can consume.

    Parameters
    ----------
    event_bus:
        Shared event dispatcher.
    dwell_to_lock_s:
        Seconds the user must dwell before a LOCK is emitted.
        (Not enforced by the stub — included for interface parity.)
    """

    def __init__(
        self,
        event_bus: EventBus,
        dwell_to_lock_s: float = 1.0,
    ) -> None:
        super().__init__(event_bus)
        self.dwell_to_lock_s = dwell_to_lock_s
        self._current_target: str | None = None
        self._state: str = "unlocked"  # "locked" | "unlocked" | "ambiguous"

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the stub adapter (no-op for hardware; logs readiness)."""
        self._running = True
        logger.info(
            "GazeAdapterStub started (dwell_to_lock=%.1f s).",
            self.dwell_to_lock_s,
        )

    def stop(self) -> None:
        """Stop the stub adapter."""
        self._running = False
        self._current_target = None
        self._state = "unlocked"
        logger.info("GazeAdapterStub stopped.")

    # ── Simulation API ──────────────────────────────────────────────

    def simulate_lock(self, target_id: str, confidence: float = 1.0) -> GazeEvent:
        """Simulate the user locking their gaze on *target_id*.

        Parameters
        ----------
        target_id:
            Identifier of the item the user is looking at.
        confidence:
            Simulated gaze confidence (0–1).

        Returns
        -------
        GazeEvent
            The emitted LOCK event.
        """
        self._current_target = target_id
        self._state = "locked"

        event = GazeEvent(
            timestamp=_time.time(),
            type=GazeEventType.LOCK,
            payload={"target_id": target_id},
            confidence=confidence,
        )
        self.event_bus.emit(event)
        logger.debug("Simulated LOCK on %s", target_id)
        return event

    def simulate_unlock(self) -> GazeEvent:
        """Simulate the user looking away from all items."""
        self._current_target = None
        self._state = "unlocked"

        event = GazeEvent(
            timestamp=_time.time(),
            type=GazeEventType.UNLOCK,
            payload={},
            confidence=0.0,
        )
        self.event_bus.emit(event)
        logger.debug("Simulated UNLOCK")
        return event

    def simulate_ambiguous(
        self,
        candidates: list[str],
        confidence: float = 0.5,
    ) -> GazeEvent:
        """Simulate an ambiguous gaze between multiple candidates.

        Parameters
        ----------
        candidates:
            List of item IDs the gaze could be targeting.
        confidence:
            Simulated gaze confidence (typically lower for ambiguous).
        """
        self._current_target = None
        self._state = "ambiguous"
        
        # Convert simple string IDs to dict format with generated positions
        formatted_cands = []
        for i, cid in enumerate(candidates):
            pos = "left" if i == 0 else ("right" if i == 1 else "other")
            formatted_cands.append({"id": cid, "pos": pos})

        event = GazeEvent(
            timestamp=_time.time(),
            type=GazeEventType.AMBIGUOUS,
            payload={"candidates": formatted_cands},
            confidence=confidence,
        )
        self.event_bus.emit(event)
        logger.debug("Simulated AMBIGUOUS gaze: %s", candidates)
        return event

    # ── State queries ───────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current gaze state: ``'locked'``, ``'unlocked'``, or ``'ambiguous'``."""
        return self._state

    @property
    def current_target(self) -> str | None:
        """ID of the currently locked target, or ``None``."""
        return self._current_target
