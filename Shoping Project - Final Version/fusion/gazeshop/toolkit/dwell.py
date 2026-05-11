"""
DwellTracker
============

Tracks how long a user's gaze has been on a single target.
Once the configured dwell duration elapses the tracker signals a lock.

Import
------
    from gazeshop.toolkit.dwell import DwellTracker

Design notes
------------
* ``DwellTracker`` is intentionally dependency-free (only stdlib ``time``).
* It does **not** emit events — callers inspect the return value of
  :meth:`update` and decide what action to take.
* The tracker auto-resets after each lock so it is ready for the next dwell
  immediately without requiring an explicit ``reset()`` call.
"""
from __future__ import annotations

import time


class DwellTracker:
    """Time-based gaze dwell tracker.

    A lock is declared when the user's gaze rests on the same *target_id*
    continuously for at least *dwell_s* seconds.  Any change in target_id
    restarts the dwell timer.

    Parameters
    ----------
    dwell_s:
        Seconds the user must gaze at a target continuously before a lock
        is declared.  Defaults to 1.2 s (matching the demo configuration).
    """

    def __init__(self, dwell_s: float = 1.2) -> None:
        self.dwell_s = dwell_s
        self._target: str | None = None
        self._start: float = 0.0

    def update(self, target_id: str | None) -> tuple[str | None, float]:
        """Process one gaze frame.

        Parameters
        ----------
        target_id:
            ID of the item currently under the gaze point, or ``None`` if no
            item is hit.

        Returns
        -------
        tuple[str | None, float]
            ``(locked_id, progress)`` where:

            * *locked_id* — the item ID that just completed its dwell and
              locked, or ``None`` if no lock occurred this frame.
            * *progress* — dwell progress in ``[0.0, 1.0]``.  Useful for
              drawing a progress arc or bar on the UI.

        Notes
        -----
        After a lock fires the tracker resets automatically so it can
        immediately begin tracking the next dwell.
        """
        if target_id != self._target:
            self._target = target_id
            self._start = time.time()
        if self._target is None:
            return None, 0.0
        progress = min((time.time() - self._start) / self.dwell_s, 1.0)
        if progress >= 1.0:
            locked = self._target
            self._target = None
            self._start = 0.0
            return locked, 1.0
        return None, progress
