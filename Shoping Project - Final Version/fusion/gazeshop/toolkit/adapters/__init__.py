"""
Modality Adapters
=================

Adapters bridge raw hardware input (microphone, eye tracker)
to standardized toolkit events via the EventBus.
"""

from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter

__all__ = ["ModalityAdapter", "SpeechAdapter"]
