from dataclasses import dataclass
from time import time
from typing import Literal, Any

from intents import parse_intent


@dataclass
class SpeechEvent:
    timestamp: float
    type: Literal["INTENT", "REPAIR", "CONFIRM", "CANCEL", "UNDO", "REPEAT", "LISTENING", "STOPPED", "ERROR"]
    payload: dict[str, Any]
    transcript: str
    confidence: float
    requires_confirmation: bool


class FusionEngine:
    def __init__(self, confirm_threshold: float = 0.65) -> None:
        self.confirm_threshold = confirm_threshold
        self.last_locked_target: str | None = None

    def on_gaze(self, event: Any) -> None:
        if event.type == "LOCK":
            self.last_locked_target = event.payload.get("target_id")
        elif event.type == "UNLOCK":
            self.last_locked_target = None

    def on_speech_transcript(self, transcript: str, confidence: float = 0.9, ts: float | None = None) -> tuple[SpeechEvent | None, dict[str, Any] | None]:
        ts = ts or time()
        parsed = parse_intent(transcript)
        if not parsed:
            return None, None

        requires_confirmation = confidence < self.confirm_threshold
        speech_event = SpeechEvent(
            timestamp=ts,
            type=parsed["type"],
            payload=parsed["payload"],
            transcript=transcript,
            confidence=confidence,
            requires_confirmation=requires_confirmation,
        )

        intent_name = speech_event.payload.get("intent_name")
        target_required = bool(speech_event.payload.get("target_required", False))
        target_id = self.last_locked_target if target_required else None

        if target_required and not target_id:
            fused = {
                "type": "AMBIGUOUS",
                "timestamp": ts,
                "payload": {"reason": "target_required", "candidates": []},
                "speech": speech_event.__dict__,
            }
            return speech_event, fused

        fused = {
            "type": "INTENT",
            "timestamp": ts,
            "payload": {
                "intent_name": intent_name,
                "target_id": target_id,
                "params": speech_event.payload.get("params", {}),
                "requires_confirmation": requires_confirmation,
            },
            "speech": speech_event.__dict__,
        }
        return speech_event, fused
