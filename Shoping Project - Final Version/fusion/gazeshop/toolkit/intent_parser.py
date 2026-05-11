"""
Intent Parser
=============

Rule-based intent parser that converts ASR transcript text into
structured ``SpeechEvent`` objects.

Processing pipeline
-------------------
1. Normalise transcript (lowercase, strip punctuation).
2. If a dialog is active → try REPAIR / CONFIRM / DENY / REPEAT
   patterns first.
3. Try each ``IntentPattern`` regex in priority order.
4. On match → extract slots, compute heuristic confidence.
5. Emit a ``SpeechEvent`` with the parsed intent.

Confidence scoring (§D.3 of the spec)
--------------------------------------
Since we use regex (no ML), confidence is a weighted heuristic:

    confidence = 0.4 × pattern_match_score
               + 0.4 × asr_confidence
               + 0.2 × word_coverage

where:
    pattern_match_score = 1.0 (full-line match) or 0.7 (substring)
    asr_confidence      = mean word-level confidence from ASR engine
    word_coverage       = matched_words / total_words
"""

from __future__ import annotations

import json
import logging
import re
from time import time
from typing import Any

from rapidfuzz import fuzz, process

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import SpeechEvent, SpeechEventType
from gazeshop.toolkit.intents import (
    CONFIRM_PATTERNS,
    DENY_PATTERNS,
    INTENT_PATTERNS,
    REPAIR_PATTERNS,
    REPEAT_PATTERNS,
    IntentPattern,
)

logger = logging.getLogger(__name__)

# Pre-compiled regex for stripping non-alphanumeric characters
_CLEAN_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Canonical phrases → (intent_tag, synonym_list)
INTENT_SYNONYMS: dict[str, tuple[str, list[str]]] = {
    "add to cart":    ("ADD_TO_CART",  ["add cart", "put in cart", "add this", "cart add", "add", "cart", "card", "two cart", "two card", "to cart", "to card"]),
    "show details":   ("SHOW_DETAILS", ["details", "show detail", "more info", "tell me more"]),
    "find similar":   ("FIND_SIMILAR", ["similar items", "find like", "show similar"]),
    "compare items":  ("COMPARE",      ["compare", "comparison", "compare these"]),
    "go back":        ("GO_BACK",      ["back", "previous", "go previous"]),
    "next item":      ("NAV_NEXT",     ["next", "next one", "move next"]),
    "zoom in please": ("ZOOM_IN",      ["zoom in", "zoom", "enlarge"]),
    "read summary":   ("SUMMARIZE",    ["summarize", "summary", "give summary"]),
    "get help":       ("HELP",         ["help", "assist", "what can i do"]),
    "confirm yes":    ("CONFIRM_YES",  ["yes", "confirm", "ok", "correct"]),
    "cancel no":      ("CONFIRM_NO",   ["no", "cancel", "wrong", "stop"]),
}

# Flat lookup built once at import time: every canonical phrase and synonym → intent tag
_FLAT: dict[str, str] = {}
for _phrase, (_tag, _synonyms) in INTENT_SYNONYMS.items():
    _FLAT[_phrase] = _tag
    for _syn in _synonyms:
        _FLAT[_syn] = _tag


class IntentParser:
    """Parse natural-language transcripts into structured intents.

    Parameters
    ----------
    config:
        ``Config`` instance providing confidence weights and threshold.
    custom_patterns:
        Optional extra ``IntentPattern`` list appended after the
        built-in patterns.
    """

    def __init__(
        self,
        config: Config | None = None,
        custom_patterns: list[IntentPattern] | None = None,
    ) -> None:
        self.config = config or Config()
        # INTENT_PATTERNS in core is empty by design.
        # Applications supply their vocabulary via custom_patterns.
        self._patterns: list[IntentPattern] = list(custom_patterns or INTENT_PATTERNS)

        # Pre-compile all regex patterns for performance
        self._compiled: list[tuple[IntentPattern, list[re.Pattern]]] = [
            (ip, [re.compile(p, re.IGNORECASE) for p in ip.patterns])
            for ip in self._patterns
        ]
        self._compiled_confirm = [
            re.compile(p, re.IGNORECASE) for p in CONFIRM_PATTERNS
        ]
        self._compiled_deny = [
            re.compile(p, re.IGNORECASE) for p in DENY_PATTERNS
        ]
        self._compiled_repair: dict[str, list[re.Pattern]] = {
            target: [re.compile(p, re.IGNORECASE) for p in pats]
            for target, pats in REPAIR_PATTERNS.items()
        }
        self._compiled_repeat = [
            re.compile(p, re.IGNORECASE) for p in REPEAT_PATTERNS
        ]

    # ── Public API ──────────────────────────────────────────────────

    def parse(
        self,
        transcript: str,
        asr_confidence: float = 1.0,
        dialog_active: bool = False,
    ) -> SpeechEvent:
        """Parse *transcript* into a ``SpeechEvent``.

        Parameters
        ----------
        transcript:
            Raw text string from the ASR engine.
        asr_confidence:
            Mean word-level confidence reported by the ASR engine
            (0.0–1.0).  Default ``1.0`` if the engine does not
            provide one.
        dialog_active:
            ``True`` when the DialogueManager has an open
            confirmation or disambiguation dialog.  In this mode
            REPAIR / CONFIRM patterns are tried first.

        Returns
        -------
        SpeechEvent
            Populated event ready to be emitted on the EventBus.
        """
        cleaned = self._normalise(transcript)

        if not cleaned or not cleaned.strip():
            return self._log_and_return(self._error_event(transcript, "Empty transcript"), transcript)

        # ── Dialog-mode patterns first ──────────────────────────────
        if dialog_active:
            event = self._try_dialog_patterns(cleaned, transcript, asr_confidence)
            if event is not None:
                return self._log_and_return(event, transcript)

        # ── Standard intent patterns ────────────────────────────────
        for intent_pattern, compiled_regexes in self._compiled:
            for regex in compiled_regexes:
                match = regex.search(cleaned)
                if match:
                    return self._log_and_return(
                        self._build_intent_event(
                            intent_pattern,
                            match,
                            cleaned,
                            transcript,
                            asr_confidence,
                        ),
                        transcript,
                    )

        # ── No regex match — fall back to fuzzy matching ─────────────
        logger.info("No regex match for %r — trying fuzzy match.", transcript)
        return self._log_and_return(self._fuzzy_parse(transcript, asr_confidence), transcript)

    def register_pattern(self, pattern: IntentPattern) -> None:
        """Dynamically add a new intent pattern at runtime.

        Parameters
        ----------
        pattern:
            ``IntentPattern`` to append to the pattern list.
        """
        self._patterns.append(pattern)
        self._compiled.append(
            (pattern, [re.compile(p, re.IGNORECASE) for p in pattern.patterns])
        )
        logger.info("Registered new intent pattern: %s", pattern.intent)

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _normalise(transcript: str) -> str:
        """Lowercase and strip non-essential punctuation."""
        return _CLEAN_RE.sub("", transcript.lower()).strip()

    def _try_dialog_patterns(
        self,
        cleaned: str,
        raw_transcript: str,
        asr_confidence: float,
    ) -> SpeechEvent | None:
        """Try CONFIRM → DENY → REPAIR → REPEAT (dialog-mode only)."""

        # CONFIRM
        for regex in self._compiled_confirm:
            if regex.search(cleaned):
                return SpeechEvent(
                    timestamp=time(),
                    type=SpeechEventType.CONFIRM,
                    payload={"confirm": True},
                    transcript=raw_transcript,
                    confidence=self._compute_confidence(1.0, asr_confidence, cleaned, cleaned),
                )

        # DENY
        for regex in self._compiled_deny:
            if regex.search(cleaned):
                return SpeechEvent(
                    timestamp=time(),
                    type=SpeechEventType.CONFIRM,
                    payload={"confirm": False},
                    transcript=raw_transcript,
                    confidence=self._compute_confidence(1.0, asr_confidence, cleaned, cleaned),
                )

        # REPAIR
        for target, regexes in self._compiled_repair.items():
            for regex in regexes:
                if regex.search(cleaned):
                    return SpeechEvent(
                        timestamp=time(),
                        type=SpeechEventType.REPAIR,
                        payload={"repair_target": target},
                        transcript=raw_transcript,
                        confidence=self._compute_confidence(
                            1.0, asr_confidence, cleaned, cleaned,
                        ),
                    )

        # REPEAT
        for regex in self._compiled_repeat:
            if regex.search(cleaned):
                return SpeechEvent(
                    timestamp=time(),
                    type=SpeechEventType.REPEAT,
                    payload={},
                    transcript=raw_transcript,
                    confidence=1.0,
                )

        # Cancel (reuse the standard CANCEL intent in dialog mode too)
        cancel_re = re.compile(
            r"\b(cancel|never\s*mind|stop|forget\s+it)\b", re.IGNORECASE,
        )
        if cancel_re.search(cleaned):
            return SpeechEvent(
                timestamp=time(),
                type=SpeechEventType.CANCEL,
                payload={},
                transcript=raw_transcript,
                confidence=1.0,
            )

        return None

    def _build_intent_event(
        self,
        intent_pattern: IntentPattern,
        match: re.Match,
        cleaned: str,
        raw_transcript: str,
        asr_confidence: float,
    ) -> SpeechEvent:
        """Construct a SpeechEvent from a successful regex match."""

        # Extract slots
        params: dict[str, Any] = {}
        for slot_name, slot_regex in intent_pattern.slot_extractors.items():
            slot_match = re.search(slot_regex, cleaned, re.IGNORECASE)
            if slot_match:
                params[slot_name] = slot_match.group(1) if slot_match.lastindex else slot_match.group(0)

        # Compute confidence
        matched_text = match.group(0)
        pattern_match_score = 1.0 if len(matched_text) == len(cleaned) else 0.7
        confidence = self._compute_confidence(
            pattern_match_score, asr_confidence, matched_text, cleaned,
        )

        requires_confirmation = confidence < self.config.CONFIDENCE_THRESHOLD

        return SpeechEvent(
            timestamp=time(),
            type=SpeechEventType.INTENT,
            payload={
                "intent": intent_pattern.intent,
                "params": params,
                "target_required": intent_pattern.target_required,
            },
            transcript=raw_transcript,
            confidence=confidence,
            requires_confirmation=requires_confirmation,
        )

    def _compute_confidence(
        self,
        pattern_match_score: float,
        asr_confidence: float,
        matched_text: str,
        full_text: str,
    ) -> float:
        """Heuristic confidence from three weighted signals (§D.3).

        confidence = w1 * pattern_match_score
                   + w2 * asr_confidence
                   + w3 * word_coverage
        """
        w = self.config.CONFIDENCE_WEIGHTS
        matched_words = len(matched_text.split())
        total_words = max(len(full_text.split()), 1)
        word_coverage = min(matched_words / total_words, 1.0)

        conf = (
            w["pattern_match"] * pattern_match_score
            + w["asr_confidence"] * asr_confidence
            + w["word_coverage"] * word_coverage
        )
        return round(min(max(conf, 0.0), 1.0), 4)

    def _fuzzy_parse(self, transcript: str, asr_confidence: float) -> SpeechEvent:
        """Match *transcript* against _FLAT using rapidfuzz token_sort_ratio.

        score >= 82  → confident match  → INTENT with matched tag
        score >= 60  → low-confidence   → AMBIGUOUS (DialogueManager resolves)
        score <  60  → no usable match  → UNKNOWN
        raw_transcript is always stored in params, never as a SpeechEvent field.
        """
        normalized = transcript.lower().strip()
        result = process.extractOne(
            normalized, _FLAT.keys(), scorer=fuzz.token_sort_ratio
        )

        if result is None:
            return SpeechEvent(
                timestamp=time(),
                type=SpeechEventType.INTENT,
                payload={"intent": "UNKNOWN", "params": {"raw_transcript": transcript}},
                transcript=transcript,
                confidence=0.0,
            )

        matched_phrase, score, _ = result
        tag = _FLAT[matched_phrase]
        conf = round(score / 100, 4)

        if score >= 82:
            logger.info("Fuzzy match: %r → %s (score=%s)", transcript, tag, score)
            return SpeechEvent(
                timestamp=time(),
                type=SpeechEventType.INTENT,
                payload={"intent": tag, "params": {"raw_transcript": transcript}},
                transcript=transcript,
                confidence=conf,
            )
        if score >= 60:
            logger.info(
                "Fuzzy ambiguous: %r → candidate=%s (score=%s)", transcript, tag, score
            )
            return SpeechEvent(
                timestamp=time(),
                type=SpeechEventType.INTENT,
                payload={
                    "intent": "AMBIGUOUS",
                    "params": {"candidate": tag, "raw_transcript": transcript},
                },
                transcript=transcript,
                confidence=conf,
            )
        logger.info("Fuzzy no match: %r (best score=%s)", transcript, score)
        return SpeechEvent(
            timestamp=time(),
            type=SpeechEventType.INTENT,
            payload={"intent": "UNKNOWN", "params": {"raw_transcript": transcript}},
            transcript=transcript,
            confidence=conf,
        )

    @staticmethod
    def _log_and_return(event: SpeechEvent, transcript: str) -> SpeechEvent:
        """Log a single-line JSON record for every produced SpeechEvent, then return it."""
        raw = event.payload.get("params", {}).get("raw_transcript") or transcript
        intent = event.payload.get("intent", event.type.value)
        record = {
            "raw": raw,
            "intent": intent,
            "confidence": event.confidence,
            "timestamp": event.timestamp,
        }
        logger.info(json.dumps(record, separators=(",", ":")))
        print(f"[SPEECH] raw={raw!r}  intent={intent}  conf={event.confidence:.0%}", flush=True)
        return event

    @staticmethod
    def _error_event(transcript: str, error_msg: str) -> SpeechEvent:
        """Create an ERROR-type SpeechEvent."""
        return SpeechEvent(
            timestamp=time(),
            type=SpeechEventType.ERROR,
            payload={"error": error_msg},
            transcript=transcript,
            confidence=0.0,
        )
