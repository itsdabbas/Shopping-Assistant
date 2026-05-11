"""
Base Modality Adapter
=====================

Abstract base class that all modality adapters (Speech, Gaze, etc.)
must inherit from.  This ensures a uniform interface so the
FusionEngine and application code can treat every adapter identically.

A toolkit developer extending GazeShop with a new input modality
(e.g. gesture, brain-computer interface) would subclass this and
implement ``start()`` / ``stop()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gazeshop.toolkit.event_bus import EventBus


class ModalityAdapter(ABC):
    """Base class for all modality adapters.

    Parameters
    ----------
    event_bus:
        Shared ``EventBus`` instance used to publish events
        produced by this adapter.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._running: bool = False

    # ── Lifecycle ───────────────────────────────────────────────────

    @abstractmethod
    def start(self) -> None:
        """Begin listening for input from the modality hardware.

        Implementations should set ``self._running = True`` and start
        any background threads / streams required.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and release resources.

        Implementations should set ``self._running = False`` and join
        any background threads.
        """

    @property
    def is_running(self) -> bool:
        """Whether this adapter is actively capturing input."""
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
