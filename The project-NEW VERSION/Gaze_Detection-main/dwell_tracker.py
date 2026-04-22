"""
Dwell Tracker — Active Gaze Lock via Sustained Fixation
========================================================

Implements "dwell-to-lock": the user must fixate on the same target
for at least ``dwell_to_lock_s`` seconds before a LOCK event fires.

Emitted event types
-------------------
LOCK      – user held gaze on target_id for the full dwell window
UNLOCK    – gaze left a previously locked target (face lost or moved away)
AMBIGUOUS – gaze is oscillating between ≥2 candidates within one dwell window

Usage
-----
    tracker = DwellTracker(dwell_to_lock_s=1.0, ambiguous_switch_count=2)

    # Call once per frame with the current target_id (or None)
    event = tracker.update(target_id="grid-0-1", ts=time.time())
    if event:
        print(event)   # {"type": "LOCK"|"UNLOCK"|"AMBIGUOUS", ...}
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


# ── Public data structures ────────────────────────────────────────────

@dataclass
class DwellEvent:
    """Lightweight event produced by DwellTracker.

    Compatible with Gaze_Detection-main event schema so it can be
    serialised directly into the WebSocket stream.
    """
    type: str                  # "LOCK" | "UNLOCK" | "AMBIGUOUS"
    timestamp: float
    payload: dict = field(default_factory=dict)
    confidence: float = 0.0


# ── DwellTracker ─────────────────────────────────────────────────────

class DwellTracker:
    """Converts a raw per-frame gaze target into LOCK / UNLOCK / AMBIGUOUS events.

    State machine
    -------------
    IDLE  →  [same target seen for dwell_to_lock_s]  →  LOCKED
    LOCKED → [target changes]                         →  IDLE (+ UNLOCK)
    IDLE  →  [target switches ≥ ambiguous_switch_count times in window] → AMBIGUOUS
    AMBIGUOUS → [one target held for dwell_to_lock_s] → LOCKED

    Parameters
    ----------
    dwell_to_lock_s:
        Seconds a target must be held before LOCK fires.
    ambiguous_switch_count:
        Number of distinct target switches within the dwell window that
        triggers AMBIGUOUS instead of continuing to wait for a single lock.
    """

    _STATE_IDLE      = "idle"
    _STATE_LOCKED    = "locked"
    _STATE_AMBIGUOUS = "ambiguous"

    def __init__(
        self,
        dwell_to_lock_s: float = 1.0,
        ambiguous_switch_count: int = 3,
    ) -> None:
        self.dwell_to_lock_s = dwell_to_lock_s
        self.ambiguous_switch_count = ambiguous_switch_count

        self._state: str = self._STATE_IDLE
        self._current_target: str | None = None
        self._dwell_start: float | None = None

        # Ring-buffer of (timestamp, target_id) for ambiguity detection
        self._recent_targets: Deque[tuple[float, str]] = deque(maxlen=30)
        self._last_event_type: str | None = None

    # ── Public API ────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def locked_target(self) -> str | None:
        """Currently locked target_id, or None."""
        return self._current_target if self._state == self._STATE_LOCKED else None

    def update(
        self,
        target_id: str | None,
        ts: float | None = None,
        gaze_confidence: float = 1.0,
    ) -> DwellEvent | None:
        """Process one frame's gaze target.

        Parameters
        ----------
        target_id:
            The target grid/item the gaze is currently pointing at,
            or ``None`` when no face / no target is detected.
        ts:
            Frame timestamp (``time.time()``). Defaults to now.
        gaze_confidence:
            Confidence from the upstream tracker (0–1).

        Returns
        -------
        DwellEvent | None
            An event if a state change occurred, otherwise ``None``.
        """
        ts = ts if ts is not None else time.time()

        # ── No face / no target ──────────────────────────────────────
        if target_id is None:
            return self._handle_no_target(ts, gaze_confidence)

        # ── Track recent switches ────────────────────────────────────
        if not self._recent_targets or self._recent_targets[-1][1] != target_id:
            self._recent_targets.append((ts, target_id))

        # ── State dispatch ───────────────────────────────────────────
        if self._state == self._STATE_IDLE:
            return self._handle_idle(target_id, ts, gaze_confidence)
        elif self._state == self._STATE_LOCKED:
            return self._handle_locked(target_id, ts, gaze_confidence)
        elif self._state == self._STATE_AMBIGUOUS:
            return self._handle_ambiguous(target_id, ts, gaze_confidence)
        return None

    def reset(self) -> None:
        """Reset all internal state (e.g. on calibration reset)."""
        self._state = self._STATE_IDLE
        self._current_target = None
        self._dwell_start = None
        self._recent_targets.clear()
        self._last_event_type = None

    # ── State handlers ────────────────────────────────────────────────

    def _handle_no_target(self, ts: float, confidence: float) -> DwellEvent | None:
        if self._state == self._STATE_LOCKED:
            prev = self._current_target
            self._reset_dwell()
            self._state = self._STATE_IDLE
            return DwellEvent(
                type="UNLOCK",
                timestamp=ts,
                payload={"previous_target": prev},
                confidence=confidence,
            )
        self._reset_dwell()
        return None

    def _handle_idle(self, target_id: str, ts: float, confidence: float) -> DwellEvent | None:
        # New target — start or reset dwell timer
        if target_id != self._current_target:
            self._current_target = target_id
            self._dwell_start = ts

        # Check ambiguity: too many switches in recent window?
        candidates = self._candidates_in_window(ts)
        if len(candidates) >= self.ambiguous_switch_count:
            self._state = self._STATE_AMBIGUOUS
            self._dwell_start = ts  # restart timer for disambiguation
            self._current_target = None
            
            # Determine deterministic layout positions
            # Let's sort candidates alphabetically so position assignment is stable
            cand_list = sorted(list(candidates))
            formatted_cands = []
            for i, cid in enumerate(cand_list):
                pos = "left" if i == 0 else ("right" if i == 1 else "other")
                formatted_cands.append({"id": cid, "pos": pos})

            return DwellEvent(
                type="AMBIGUOUS",
                timestamp=ts,
                payload={"candidates": formatted_cands},
                confidence=confidence * 0.5,
            )

        # Check if dwell time exceeded → LOCK
        if self._dwell_start is not None and (ts - self._dwell_start) >= self.dwell_to_lock_s:
            self._state = self._STATE_LOCKED
            return DwellEvent(
                type="LOCK",
                timestamp=ts,
                payload={"target_id": target_id, "dwell_s": round(ts - self._dwell_start, 3)},
                confidence=min(1.0, confidence),
            )
        return None

    def _handle_locked(self, target_id: str, ts: float, confidence: float) -> DwellEvent | None:
        if target_id == self._current_target:
            return None  # still on same target, no new event

        # Target changed → unlock then restart idle
        prev = self._current_target
        self._reset_dwell()
        self._state = self._STATE_IDLE
        self._current_target = target_id
        self._dwell_start = ts
        return DwellEvent(
            type="UNLOCK",
            timestamp=ts,
            payload={"previous_target": prev, "new_target": target_id},
            confidence=confidence,
        )

    def _handle_ambiguous(self, target_id: str, ts: float, confidence: float) -> DwellEvent | None:
        # In ambiguous state, wait for user to hold one target for dwell_to_lock_s
        if target_id != self._current_target:
            self._current_target = target_id
            self._dwell_start = ts
            return None

        if self._dwell_start is not None and (ts - self._dwell_start) >= self.dwell_to_lock_s:
            self._state = self._STATE_LOCKED
            return DwellEvent(
                type="LOCK",
                timestamp=ts,
                payload={"target_id": target_id, "dwell_s": round(ts - self._dwell_start, 3), "resolved_from_ambiguous": True},
                confidence=min(1.0, confidence),
            )
        return None

    # ── Helpers ───────────────────────────────────────────────────────

    def _reset_dwell(self) -> None:
        self._current_target = None
        self._dwell_start = None

    def _candidates_in_window(self, now: float) -> set[str]:
        """Return distinct target IDs seen within the dwell window."""
        cutoff = now - self.dwell_to_lock_s
        return {tid for ts, tid in self._recent_targets if ts >= cutoff}
