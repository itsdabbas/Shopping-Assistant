"""
Tests for DialogueManager
==========================

Validates confirmation flow, disambiguation flow, timeout,
repair attempts, and integration with EventBus.
"""

import time
import pytest

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.dialogue_manager import DialogueManager


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def prompts():
    """Collects DM prompt messages."""
    return []


@pytest.fixture
def actions():
    """Collects executed actions."""
    return []


@pytest.fixture
def dm(bus, prompts, actions):
    """DialogueManager with captured prompts and actions."""
    cfg = Config()
    cfg.DISAMBIGUATION_TIMEOUT_S = 2.0  # shorter for tests
    cfg.MAX_REPAIR_ATTEMPTS = 2

    return DialogueManager(
        event_bus=bus,
        config=cfg,
        on_prompt=lambda msg: prompts.append(msg),
        on_action=lambda intent, target, params: actions.append(
            {"intent": intent, "target": target, "params": params}
        ),
    )


def _make_intent_event(intent="ADD_TO_CART", target_required=True, confidence=0.8):
    """Helper to create an INTENT SpeechEvent."""
    return SpeechEvent(
        type=SpeechEventType.INTENT,
        payload={
            "intent": intent,
            "params": {},
            "target_required": target_required,
        },
        transcript="test",
        confidence=confidence,
        requires_confirmation=confidence < 0.60,
    )


# ────────────────────────────────────────────────────────────────────
# Confirmation flow
# ────────────────────────────────────────────────────────────────────

class TestConfirmation:
    """Low-confidence intent confirmation dialog."""

    def test_confirm_intent_opens_dialog(self, dm, prompts):
        """confirm_intent() opens a dialog and prompts the user."""
        event = _make_intent_event(confidence=0.4)
        dm.confirm_intent(event)

        assert dm.is_active
        assert len(prompts) == 1
        assert "ADD TO CART" in prompts[0]

    def test_user_confirms_yes(self, dm, bus, prompts, actions):
        """User saying 'yes' executes the pending action."""
        event = _make_intent_event(confidence=0.4)
        dm.confirm_intent(event)

        # Simulate user saying "yes" via EventBus
        bus.emit(SpeechEvent(
            type=SpeechEventType.CONFIRM,
            payload={"confirm": True},
        ))

        assert not dm.is_active
        assert len(actions) == 1
        assert actions[0]["intent"] == "ADD_TO_CART"

    def test_user_confirms_no(self, dm, bus, prompts, actions):
        """User saying 'no' cancels the action."""
        event = _make_intent_event(confidence=0.4)
        dm.confirm_intent(event)

        bus.emit(SpeechEvent(
            type=SpeechEventType.CONFIRM,
            payload={"confirm": False},
        ))

        assert not dm.is_active
        assert len(actions) == 0
        assert "try again" in prompts[-1].lower()


class TestConfirmAction:
    """confirm_action() with target binding."""

    def test_high_confidence_executes_directly(self, dm, actions):
        """High confidence skips confirmation dialog."""
        event = _make_intent_event(confidence=0.9)
        dm.confirm_action(event, target_id="item_42")

        assert not dm.is_active
        assert len(actions) == 1
        assert actions[0]["target"] == "item_42"

    def test_low_confidence_asks_confirmation(self, dm, prompts, actions):
        """Low confidence opens confirmation dialog."""
        event = _make_intent_event(confidence=0.4)
        dm.confirm_action(event, target_id="item_42")

        assert dm.is_active
        assert len(actions) == 0
        assert "item_42" in prompts[0]


# ────────────────────────────────────────────────────────────────────
# Disambiguation flow
# ────────────────────────────────────────────────────────────────────

class TestDisambiguation:
    """Ambiguous gaze target disambiguation dialog."""

    def test_disambiguate_opens_dialog(self, dm, prompts):
        """disambiguate() opens a dialog and prompts with options."""
        event = _make_intent_event()
        dm.disambiguate(["item_3", "item_4"], event)

        assert dm.is_active
        assert len(prompts) == 1
        assert "left" in prompts[0].lower() or "right" in prompts[0].lower()

    def test_repair_left_resolves(self, dm, bus, prompts, actions):
        """User saying 'left' resolves to first candidate."""
        event = _make_intent_event()
        dm.disambiguate(["item_3", "item_4"], event)

        bus.emit(SpeechEvent(
            type=SpeechEventType.REPAIR,
            payload={"repair_target": "left"},
        ))

        assert not dm.is_active
        assert len(actions) == 1
        assert actions[0]["target"] == "item_3"

    def test_repair_right_resolves(self, dm, bus, prompts, actions):
        """User saying 'right' resolves to second candidate."""
        event = _make_intent_event()
        dm.disambiguate(["item_3", "item_4"], event)

        bus.emit(SpeechEvent(
            type=SpeechEventType.REPAIR,
            payload={"repair_target": "right"},
        ))

        assert actions[0]["target"] == "item_4"

    def test_max_repair_attempts_cancels(self, dm, bus, prompts, actions):
        """After MAX_REPAIR_ATTEMPTS failed repairs, action is cancelled."""
        event = _make_intent_event()
        dm.disambiguate(["item_3", "item_4"], event)

        # Send two invalid repairs
        for _ in range(2):
            bus.emit(SpeechEvent(
                type=SpeechEventType.REPAIR,
                payload={"repair_target": "other"},  # "other" index may not match
            ))

        # After 2 failed attempts, should be cancelled
        assert not dm.is_active
        assert len(actions) == 0
        assert "cancelled" in prompts[-1].lower()


# ────────────────────────────────────────────────────────────────────
# Cancel and Repeat
# ────────────────────────────────────────────────────────────────────

class TestCancelAndRepeat:
    """Cancel and repeat during dialog."""

    def test_cancel_during_dialog(self, dm, bus, prompts, actions):
        """CANCEL event closes the dialog without executing."""
        event = _make_intent_event()
        dm.confirm_intent(event)

        bus.emit(SpeechEvent(type=SpeechEventType.CANCEL, payload={}))

        assert not dm.is_active
        assert len(actions) == 0
        assert "cancelled" in prompts[-1].lower()

    def test_repeat_reprompts(self, dm, bus, prompts):
        """REPEAT event re-displays the prompt."""
        event = _make_intent_event()
        dm.confirm_intent(event)
        initial_prompts = len(prompts)

        bus.emit(SpeechEvent(type=SpeechEventType.REPEAT, payload={}))

        assert len(prompts) > initial_prompts

    def test_events_ignored_when_no_dialog(self, dm, bus, actions):
        """Speech events are ignored when no dialog is active."""
        bus.emit(SpeechEvent(
            type=SpeechEventType.CONFIRM,
            payload={"confirm": True},
        ))

        assert len(actions) == 0


# ────────────────────────────────────────────────────────────────────
# Timeout
# ────────────────────────────────────────────────────────────────────

class TestTimeout:
    """Dialog timeout behaviour."""

    def test_timeout_cancels_dialog(self, dm, prompts, actions):
        """After timeout, the dialog is auto-cancelled."""
        event = _make_intent_event()
        dm.confirm_intent(event)

        # Wait for timeout (config set to 2.0s for tests)
        time.sleep(2.5)

        assert not dm.is_active
        assert len(actions) == 0
        assert any("cancelled" in p.lower() for p in prompts)
