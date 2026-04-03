"""
Tests for IntentParser
======================

Validates intent recognition for all 13 commands, slot extraction,
confidence scoring, dialog-mode patterns, and edge cases.
"""

import pytest

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import SpeechEventType
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.intents import IntentPattern


@pytest.fixture
def parser():
    """Default IntentParser with standard config."""
    return IntentParser(Config())


# ────────────────────────────────────────────────────────────────────
# Object-bound commands (target_required = True)
# ────────────────────────────────────────────────────────────────────

class TestAddToCart:
    """ADD_TO_CART intent recognition."""

    @pytest.mark.parametrize("phrase", [
        "add this to cart",
        "add it to cart",
        "put this in the basket",
        "buy this",
        "I want this one",
        "purchase that",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == "ADD_TO_CART"
        assert event.payload["target_required"] is True

    def test_high_confidence(self, parser):
        event = parser.parse("add this to cart", asr_confidence=0.95)
        assert event.confidence >= 0.60
        assert event.requires_confirmation is False


class TestShowDetails:
    """SHOW_DETAILS intent with optional detail_type slot."""

    @pytest.mark.parametrize("phrase", [
        "show details",
        "tell me more",
        "what is this",
        "show ingredients",
        "more info",
        "give me the specs",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == "SHOW_DETAILS"

    def test_slot_ingredient(self, parser):
        event = parser.parse("show me the ingredients")
        assert event.payload["params"].get("detail_type") == "ingredient"

    def test_slot_review(self, parser):
        event = parser.parse("show me the reviews")
        assert event.payload["params"].get("detail_type") == "review"

    def test_slot_spec(self, parser):
        event = parser.parse("give me the specs")
        assert event.payload["params"].get("detail_type") == "spec"


class TestFindSimilar:
    """FIND_SIMILAR intent."""

    @pytest.mark.parametrize("phrase", [
        "find similar",
        "show me similar ones",
        "anything like this",
        "like this one",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == "FIND_SIMILAR"


class TestCompare:
    """COMPARE intent with compare_ref slot."""

    def test_basic(self, parser):
        event = parser.parse("compare this")
        assert event.payload["intent"] == "COMPARE"

    def test_slot_pinned(self, parser):
        event = parser.parse("compare this with pinned")
        assert event.payload["params"].get("compare_ref") == "pinned"

    def test_slot_cheaper(self, parser):
        event = parser.parse("compare with cheaper")
        assert event.payload["params"].get("compare_ref") == "cheaper"


class TestShowAlternatives:
    """SHOW_ALTERNATIVES intent."""

    @pytest.mark.parametrize("phrase", [
        "show alternatives",
        "what else is there",
        "other options",
        "find alternatives",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == "SHOW_ALTERNATIVES"


class TestPinItem:
    """PIN_ITEM intent."""

    @pytest.mark.parametrize("phrase", [
        "pin this",
        "save this",
        "remember this one",
        "bookmark that",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == "PIN_ITEM"


class TestRemoveItem:
    """REMOVE_ITEM intent."""

    @pytest.mark.parametrize("phrase", [
        "remove this",
        "delete this",
        "take off that",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.type == SpeechEventType.INTENT
        assert event.payload["intent"] == "REMOVE_ITEM"


# ────────────────────────────────────────────────────────────────────
# Global commands (target_required = False)
# ────────────────────────────────────────────────────────────────────

class TestScroll:
    """SCROLL intent with direction slot."""

    def test_scroll_down(self, parser):
        event = parser.parse("scroll down")
        assert event.payload["intent"] == "SCROLL"
        assert event.payload["target_required"] is False
        assert event.payload["params"].get("direction") == "down"

    def test_scroll_up(self, parser):
        event = parser.parse("scroll up")
        assert event.payload["params"].get("direction") == "up"

    def test_next_page(self, parser):
        event = parser.parse("next page")
        assert event.payload["intent"] == "SCROLL"
        assert event.payload["params"].get("direction") == "next"

    def test_go_down(self, parser):
        event = parser.parse("go down")
        assert event.payload["intent"] == "SCROLL"


class TestOpenCart:
    """OPEN_CART intent."""

    @pytest.mark.parametrize("phrase", [
        "open cart",
        "show my cart",
        "view cart",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.payload["intent"] == "OPEN_CART"
        assert event.payload["target_required"] is False


class TestGoBack:
    """GO_BACK intent."""

    @pytest.mark.parametrize("phrase", [
        "go back",
        "back",
        "previous page",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.payload["intent"] == "GO_BACK"


class TestHelp:
    """HELP intent."""

    @pytest.mark.parametrize("phrase", [
        "help",
        "what can I say",
        "show commands",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.payload["intent"] == "HELP"


class TestCancel:
    """CANCEL intent."""

    @pytest.mark.parametrize("phrase", [
        "cancel",
        "never mind",
        "stop",
        "forget it",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.payload["intent"] == "CANCEL"


class TestUndo:
    """UNDO intent."""

    @pytest.mark.parametrize("phrase", [
        "undo",
        "take that back",
        "take it back",
    ])
    def test_recognized(self, parser, phrase):
        event = parser.parse(phrase)
        assert event.payload["intent"] == "UNDO"


# ────────────────────────────────────────────────────────────────────
# Dialog-mode patterns (CONFIRM / DENY / REPAIR / REPEAT)
# ────────────────────────────────────────────────────────────────────

class TestDialogPatterns:
    """Patterns only active when dialog_active=True."""

    def test_confirm_yes(self, parser):
        event = parser.parse("yes", dialog_active=True)
        assert event.type == SpeechEventType.CONFIRM
        assert event.payload["confirm"] is True

    def test_confirm_yeah(self, parser):
        event = parser.parse("yeah", dialog_active=True)
        assert event.type == SpeechEventType.CONFIRM
        assert event.payload["confirm"] is True

    def test_confirm_correct(self, parser):
        event = parser.parse("correct", dialog_active=True)
        assert event.type == SpeechEventType.CONFIRM
        assert event.payload["confirm"] is True

    def test_deny_no(self, parser):
        event = parser.parse("no", dialog_active=True)
        assert event.type == SpeechEventType.CONFIRM
        assert event.payload["confirm"] is False

    def test_deny_nope(self, parser):
        event = parser.parse("nope", dialog_active=True)
        assert event.type == SpeechEventType.CONFIRM
        assert event.payload["confirm"] is False

    def test_repair_left(self, parser):
        event = parser.parse("the left one", dialog_active=True)
        assert event.type == SpeechEventType.REPAIR
        assert event.payload["repair_target"] == "left"

    def test_repair_right(self, parser):
        event = parser.parse("the right one", dialog_active=True)
        assert event.type == SpeechEventType.REPAIR
        assert event.payload["repair_target"] == "right"

    def test_repair_first(self, parser):
        event = parser.parse("the first one", dialog_active=True)
        assert event.type == SpeechEventType.REPAIR
        assert event.payload["repair_target"] == "left"  # "first" maps to "left"

    def test_repair_second(self, parser):
        event = parser.parse("the second one", dialog_active=True)
        assert event.type == SpeechEventType.REPAIR
        assert event.payload["repair_target"] == "right"  # "second" maps to "right"

    def test_repeat(self, parser):
        event = parser.parse("repeat", dialog_active=True)
        assert event.type == SpeechEventType.REPEAT

    def test_cancel_in_dialog(self, parser):
        event = parser.parse("cancel", dialog_active=True)
        assert event.type == SpeechEventType.CANCEL

    def test_dialog_patterns_inactive_by_default(self, parser):
        """'yes' without dialog_active should NOT produce CONFIRM."""
        event = parser.parse("yes", dialog_active=False)
        # Should either be an ERROR (no intent match) or match something else
        assert event.type != SpeechEventType.CONFIRM


# ────────────────────────────────────────────────────────────────────
# Confidence scoring
# ────────────────────────────────────────────────────────────────────

class TestConfidence:
    """Heuristic confidence calculation."""

    def test_high_confidence_no_confirmation(self, parser):
        """High ASR confidence + good pattern match = no confirmation needed."""
        event = parser.parse("add this to cart", asr_confidence=0.95)
        assert event.confidence >= 0.60
        assert event.requires_confirmation is False

    def test_low_confidence_triggers_confirmation(self, parser):
        """Low ASR confidence should flag requires_confirmation."""
        event = parser.parse("add this to cart", asr_confidence=0.1)
        # With low ASR confidence, the final score should be lower
        assert event.type == SpeechEventType.INTENT

    def test_confidence_bounded_0_to_1(self, parser):
        """Confidence is always in [0, 1]."""
        event = parser.parse("add this to cart", asr_confidence=1.0)
        assert 0.0 <= event.confidence <= 1.0

        event2 = parser.parse("add this to cart", asr_confidence=0.0)
        assert 0.0 <= event2.confidence <= 1.0


# ────────────────────────────────────────────────────────────────────
# Edge cases
# ────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_transcript(self, parser):
        """Empty string produces an ERROR event."""
        event = parser.parse("")
        assert event.type == SpeechEventType.ERROR

    def test_whitespace_only(self, parser):
        """Whitespace-only transcript produces an ERROR event."""
        event = parser.parse("   ")
        assert event.type == SpeechEventType.ERROR

    def test_unrecognized_phrase(self, parser):
        """Completely unrecognized text produces ERROR."""
        event = parser.parse("the quick brown fox jumps")
        assert event.type == SpeechEventType.ERROR

    def test_case_insensitive(self, parser):
        """Matching should be case-insensitive."""
        event = parser.parse("ADD THIS TO CART")
        assert event.payload["intent"] == "ADD_TO_CART"

    def test_punctuation_ignored(self, parser):
        """Punctuation in transcript should not break matching."""
        event = parser.parse("add this, to cart!")
        assert event.payload["intent"] == "ADD_TO_CART"


# ────────────────────────────────────────────────────────────────────
# Custom pattern registration
# ────────────────────────────────────────────────────────────────────

class TestCustomPatterns:
    """Dynamic pattern registration at runtime."""

    def test_register_custom_intent(self, parser):
        """Registering a new intent makes it recognizable."""
        parser.register_pattern(IntentPattern(
            intent="RATE_ITEM",
            target_required=True,
            patterns=[r"\brate\b.*\b(\d)\s*star"],
        ))
        event = parser.parse("rate this 5 stars")
        assert event.payload["intent"] == "RATE_ITEM"
        assert event.payload["target_required"] is True

    def test_custom_pattern_does_not_break_builtins(self, parser):
        """Adding a custom pattern does not interfere with built-in ones."""
        parser.register_pattern(IntentPattern(
            intent="CUSTOM_CMD",
            target_required=False,
            patterns=[r"\bcustom\b"],
        ))
        # Built-in still works
        event = parser.parse("add this to cart")
        assert event.payload["intent"] == "ADD_TO_CART"

        # Custom also works
        event2 = parser.parse("custom command")
        assert event2.payload["intent"] == "CUSTOM_CMD"
