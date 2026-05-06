"""
GazeShop MMUI Toolkit — Runtime Facade
=======================================

``MMUIToolkit`` is the single entry-point that a developer uses to wire
together the toolkit components.  It hides all internal plumbing and
exposes a clean, five-step API:

    1.  Create a toolkit instance with configuration.
    2.  Register one or more modality adapters.
    3.  Register *intent → action* handlers.
    4.  Call ``start()`` to begin capturing input.
    5.  Call ``stop()`` when done.

Example (minimal)
-----------------
::

    from gazeshop.toolkit.runtime import MMUIToolkit
    from gazeshop.toolkit.config import Config
    from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
    from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter

    tk = MMUIToolkit(Config())

    gaze   = GazeAdapterStub(tk.bus)
    speech = SpeechAdapter(tk.bus, tk.config)

    tk.register_adapter(gaze)
    tk.register_adapter(speech)

    tk.register_action("OPEN_DETAIL",  lambda cmd: print("Detail:", cmd.target_id))
    tk.register_action("NAVIGATE_NEXT", lambda cmd: print("Next page"))

    with tk:          # calls start() / stop() automatically
        input("Press Enter to quit…")

Design notes
------------
* ``MMUIToolkit`` never imports application-specific code.
* Adapters are started/stopped in registration order.
* Action handlers receive a :class:`~gazeshop.toolkit.events.MultimodalCommandEvent`.
* Unhandled intents are logged as warnings (not exceptions).
"""

from __future__ import annotations

import logging
from typing import Callable

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager
from gazeshop.toolkit.telemetry import TelemetryLogger
from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.events import MultimodalCommandEvent

logger = logging.getLogger(__name__)


class MMUIToolkit:
    """Developer-facing facade for the MMUI (Multimodal User Interface) Toolkit.

    Parameters
    ----------
    config:
        Configuration instance.  Defaults to :class:`~gazeshop.toolkit.config.Config`
        with factory defaults if not supplied.
    enable_telemetry:
        Override ``config.ENABLE_TELEMETRY`` for this session.
    """

    def __init__(
        self,
        config: Config | None = None,
        enable_telemetry: bool | None = None,
    ) -> None:
        self.config = config or Config()
        if enable_telemetry is not None:
            self.config.ENABLE_TELEMETRY = enable_telemetry

        # ── Core infrastructure ──────────────────────────────────────
        self.bus = EventBus()
        self._fusion = FusionEngine(self.bus, self.config)
        self._dialogue = DialogueManager(self.bus, self.config)
        self._telemetry = TelemetryLogger(self.bus, self.config)

        # ── Registry ─────────────────────────────────────────────────
        self._adapters: list[ModalityAdapter] = []
        self._action_handlers: dict[str, Callable[[MultimodalCommandEvent], None]] = {}

        # Wire the action dispatcher
        self.bus.subscribe("MultimodalCommandEvent", self._dispatch_action)

        logger.info("MMUIToolkit initialised (telemetry=%s).", self.config.ENABLE_TELEMETRY)

    # ── Adapter registration ─────────────────────────────────────────

    def register_adapter(self, adapter: ModalityAdapter) -> "MMUIToolkit":
        """Register a modality adapter with the toolkit.

        The adapter will be started when :meth:`start` is called and stopped
        when :meth:`stop` is called.

        Parameters
        ----------
        adapter:
            Any object that implements :class:`~gazeshop.toolkit.adapters.base_adapter.ModalityAdapter`.

        Returns
        -------
        MMUIToolkit
            Returns *self* to allow method chaining.
        """
        if not isinstance(adapter, ModalityAdapter):
            raise TypeError(
                f"adapter must be a ModalityAdapter subclass, got {type(adapter).__name__}"
            )
        self._adapters.append(adapter)
        logger.debug("Registered adapter: %s", type(adapter).__name__)
        return self

    # ── Action registration ──────────────────────────────────────────

    def register_action(
        self,
        intent: str,
        handler: Callable[[MultimodalCommandEvent], None],
    ) -> "MMUIToolkit":
        """Bind an intent name to an action handler function.

        The handler is called every time the toolkit resolves a
        :class:`~gazeshop.toolkit.events.MultimodalCommandEvent` with
        ``event.intent == intent``.

        Parameters
        ----------
        intent:
            Canonical intent name (e.g. ``"OPEN_DETAIL"``).  Must match
            what your :class:`~gazeshop.toolkit.intents.IntentPattern` emits.
        handler:
            Callable that receives the ``MultimodalCommandEvent``.

        Returns
        -------
        MMUIToolkit
            Returns *self* to allow method chaining.

        Example
        -------
        ::

            tk.register_action("ADD_TO_CART", lambda cmd: cart.add(cmd.target_id))
            tk.register_action("SCROLL",      lambda cmd: ui.scroll(cmd.params.get("direction")))
        """
        self._action_handlers[intent] = handler
        logger.debug("Registered action handler for intent: %s", intent)
        return self

    def register_feedback(
        self,
        event_type: str,
        handler: Callable,
    ) -> "MMUIToolkit":
        """Subscribe to any toolkit event by class name.

        Use this for UI feedback events such as ``PromptEvent``,
        ``DisambiguationPromptEvent``, ``TargetLockedEvent``, etc.

        Parameters
        ----------
        event_type:
            Class name string (e.g. ``"PromptEvent"``) or ``"*"`` wildcard.
        handler:
            Callable that receives the event instance.

        Returns
        -------
        MMUIToolkit
            Returns *self* to allow method chaining.
        """
        self.bus.subscribe(event_type, handler)
        return self

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all registered adapters.

        Adapters are started in the order they were registered.
        """
        for adapter in self._adapters:
            if not adapter.is_running:
                adapter.start()
        logger.info(
            "MMUIToolkit started (%d adapters).", len(self._adapters)
        )

    def stop(self) -> None:
        """Stop all registered adapters in reverse registration order."""
        for adapter in reversed(self._adapters):
            if adapter.is_running:
                adapter.stop()
        logger.info("MMUIToolkit stopped.")

    def __enter__(self) -> "MMUIToolkit":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # ── Internal dispatch ────────────────────────────────────────────

    def _dispatch_action(self, event: MultimodalCommandEvent) -> None:
        """Route a resolved command to the registered handler."""
        handler = self._action_handlers.get(event.intent)
        if handler is None:
            logger.warning(
                "No action handler registered for intent '%s' (target=%s). "
                "Register one with tk.register_action('%s', handler).",
                event.intent, event.target_id, event.intent,
            )
            return
        try:
            handler(event)
        except Exception:
            logger.exception(
                "Action handler for '%s' raised an exception.", event.intent
            )

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def event_log(self) -> list[dict]:
        """Return the in-memory event log (requires ``bus.enable_logging()``)."""
        return self.bus.get_log()

    def enable_logging(self) -> "MMUIToolkit":
        """Enable in-memory event logging on the shared bus."""
        self.bus.enable_logging()
        return self
