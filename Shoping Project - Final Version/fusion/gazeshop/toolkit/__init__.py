"""
GazeShop MMUI Toolkit
======================

Reusable multimodal interaction toolkit.  Provides:

* Event bus (pub/sub) for inter-module communication.
* Modality adapters (speech, gaze, dummy keyboard, …).
* Intent parsing (rule-based, regex, extensible).
* Late fusion engine (FusionEngine).
* Dialogue management for confirmation & disambiguation.
* ``MMUIToolkit`` facade — 5-step developer API.

Quick start
-----------
::

    from gazeshop.toolkit.runtime import MMUIToolkit
    from gazeshop.toolkit.config import Config

    tk = MMUIToolkit(Config())
    tk.register_adapter(gaze_adapter)
    tk.register_adapter(speech_adapter)
    tk.register_action("MY_INTENT", lambda cmd: print(cmd))
    tk.start()
"""

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.event_bus import (
    EventBus,
    Modality,
    SpeechEvent,
    SpeechEventType,
    GazeEvent,
    GazeEventType,
)
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager
from gazeshop.toolkit.telemetry import TelemetryLogger
from gazeshop.toolkit.intents import IntentPattern
from gazeshop.toolkit.events import (
    IntentReadyEvent,
    IntentNeedsTargetEvent,
    IntentNeedsDisambiguationEvent,
    IntentNeedsConfirmationEvent,
    TargetLockedEvent,
    TargetUnlockedEvent,
    TargetExpiredEvent,
    PromptEvent,
    DisambiguationPromptEvent,
    ConfirmationPromptEvent,
    ActionCancelledEvent,
    MultimodalCommandEvent,
)

__all__ = [
    # Facade
    "MMUIToolkit",
    # Config
    "Config",
    # Event bus
    "EventBus",
    "Modality",
    "SpeechEvent",
    "SpeechEventType",
    "GazeEvent",
    "GazeEventType",
    # Core engines
    "FusionEngine",
    "DialogueManager",
    "TelemetryLogger",
    # Intent
    "IntentPattern",
    # Internal fusion events
    "IntentReadyEvent",
    "IntentNeedsTargetEvent",
    "IntentNeedsDisambiguationEvent",
    "IntentNeedsConfirmationEvent",
    # UI / fission events
    "TargetLockedEvent",
    "TargetUnlockedEvent",
    "TargetExpiredEvent",
    "PromptEvent",
    "DisambiguationPromptEvent",
    "ConfirmationPromptEvent",
    "ActionCancelledEvent",
    "MultimodalCommandEvent",
]
