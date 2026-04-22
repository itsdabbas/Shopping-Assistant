"""
Telemetry Logger
================

Subscribes to EventBus and records system interactions into a JSONL file 
for offline evaluation (e.g. success rate, repair counts, cancellation reasons).
"""

import json
import logging
import os
import time
from typing import Any

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus
from gazeshop.toolkit.events import (
    MultimodalCommandEvent,
    ActionCancelledEvent,
    PromptEvent,
    DisambiguationPromptEvent,
    ConfirmationPromptEvent,
)

logger = logging.getLogger(__name__)


class TelemetryLogger:
    def __init__(self, event_bus: EventBus, config: Config | None = None) -> None:
        self.config = config or Config()
        self.event_bus = event_bus

        if not self.config.ENABLE_TELEMETRY:
            logger.info("Telemetry is DISABLED.")
            return

        export_dir = os.path.dirname(self.config.TELEMETRY_EXPORT_PATH)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)

        # Basic interaction tracking
        self.event_bus.subscribe("MultimodalCommandEvent", self._on_command)
        self.event_bus.subscribe("ActionCancelledEvent", self._on_cancel)
        self.event_bus.subscribe("PromptEvent", lambda e: self._log_interaction({"event": "prompt", "message": e.message}))
        self.event_bus.subscribe("DisambiguationPromptEvent", lambda e: self._log_interaction({"event": "disambiguation_prompt"}))
        self.event_bus.subscribe("ConfirmationPromptEvent", lambda e: self._log_interaction({"event": "confirmation_prompt"}))
        
        logger.info("Telemetry tracking enabled: %s", self.config.TELEMETRY_EXPORT_PATH)

    def _log_interaction(self, data: dict[str, Any]) -> None:
        if not self.config.ENABLE_TELEMETRY:
            return
        
        data["timestamp"] = time.time()
        try:
            with open(self.config.TELEMETRY_EXPORT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error("Failed to write telemetry trace: %s", e)

    def _on_command(self, e: MultimodalCommandEvent) -> None:
        self._log_interaction({
            "event": "interaction_success",
            "intent": e.intent,
            "target_id": e.target_id,
            "confidence": e.confidence,
        })

    def _on_cancel(self, e: ActionCancelledEvent) -> None:
        self._log_interaction({
            "event": "interaction_cancelled",
            "reason": e.reason,
            "message": e.message,
        })
