"""
GazeShop Toolkit
================

Reusable multimodal interaction toolkit providing:
- Event bus (pub/sub) for inter-module communication
- Modality adapters (speech, gaze)
- Intent parsing (rule-based, regex)
- Late fusion engine (FusionEngine)
- Dialogue management for confirmation & disambiguation
"""

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import (
    EventBus,
    Modality,
    SpeechEvent,
    SpeechEventType,
    GazeEvent,
    GazeEventType,
)
from gazeshop.toolkit.fusion_engine import (
    FusionEngine
)
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
    "Config",
    "EventBus",
    "Modality",
    "SpeechEvent",
    "SpeechEventType",
    "GazeEvent",
    "GazeEventType",
    "FusionEngine",
    # Internal Fusion Events
    "IntentReadyEvent",
    "IntentNeedsTargetEvent",
    "IntentNeedsDisambiguationEvent",
    "IntentNeedsConfirmationEvent",
    # Fission / UI Events
    "TargetLockedEvent",
    "TargetUnlockedEvent",
    "TargetExpiredEvent",
    "PromptEvent",
    "DisambiguationPromptEvent",
    "ConfirmationPromptEvent",
    "ActionCancelledEvent",
    "MultimodalCommandEvent",
]

