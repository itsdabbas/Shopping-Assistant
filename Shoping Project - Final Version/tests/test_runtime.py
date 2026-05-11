"""
Tests for MMUIToolkit Facade (runtime.py)
==========================================

Verifies the developer-facing API:
- Adapter registration + lifecycle
- Action dispatch
- Feedback subscription
- Unhandled intent warning (no crash)
- Context manager usage
"""

from __future__ import annotations

import pytest

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.event_bus import EventBus, GazeEvent, GazeEventType, SpeechEvent, SpeechEventType
from gazeshop.toolkit.events import MultimodalCommandEvent, PromptEvent


# ── Minimal stub adapter for testing ────────────────────────────────

class _StubAdapter(ModalityAdapter):
    def __init__(self, bus: EventBus):
        super().__init__(bus)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self._running = True
        self.started = True

    def stop(self) -> None:
        self._running = False
        self.stopped = True


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def tk() -> MMUIToolkit:
    cfg = Config(ENABLE_TELEMETRY=False)
    return MMUIToolkit(cfg)


@pytest.fixture
def stub(tk: MMUIToolkit) -> _StubAdapter:
    adapter = _StubAdapter(tk.bus)
    tk.register_adapter(adapter)
    return adapter


# ── Tests ─────────────────────────────────────────────────────────────

class TestAdapterLifecycle:
    def test_start_starts_adapters(self, tk, stub):
        tk.start()
        assert stub.started
        tk.stop()

    def test_stop_stops_adapters(self, tk, stub):
        tk.start()
        tk.stop()
        assert stub.stopped

    def test_context_manager(self, tk, stub):
        with tk:
            assert stub.started
        assert stub.stopped

    def test_invalid_adapter_raises(self, tk):
        with pytest.raises(TypeError):
            tk.register_adapter("not_an_adapter")  # type: ignore

    def test_chaining(self, tk):
        a1 = _StubAdapter(tk.bus)
        a2 = _StubAdapter(tk.bus)
        result = tk.register_adapter(a1).register_adapter(a2)
        assert result is tk


class TestActionDispatch:
    def test_registered_action_called(self, tk):
        received = []
        tk.register_action("TEST_INTENT", lambda cmd: received.append(cmd))

        cmd = MultimodalCommandEvent(
            intent="TEST_INTENT",
            target_id="item_1",
            params={},
            confidence=0.9,
        )
        tk.bus.emit(cmd)
        assert len(received) == 1
        assert received[0].intent == "TEST_INTENT"

    def test_unregistered_intent_no_crash(self, tk):
        """Emitting an unhandled intent should warn but not raise."""
        cmd = MultimodalCommandEvent(
            intent="UNKNOWN_INTENT",
            target_id=None,
            params={},
            confidence=0.8,
        )
        # Should not raise
        tk.bus.emit(cmd)

    def test_chained_action_registration(self, tk):
        result = tk.register_action("A", lambda cmd: None).register_action("B", lambda cmd: None)
        assert result is tk


class TestFeedbackSubscription:
    def test_feedback_subscriber_receives_event(self, tk):
        prompts = []
        tk.register_feedback("PromptEvent", lambda e: prompts.append(e))

        tk.bus.emit(PromptEvent(message="Look at an item"))
        assert len(prompts) == 1
        assert prompts[0].message == "Look at an item"

    def test_wildcard_feedback(self, tk):
        all_events = []
        tk.register_feedback("*", lambda e: all_events.append(e))

        tk.bus.emit(PromptEvent(message="test"))
        tk.bus.emit(MultimodalCommandEvent(intent="X", target_id=None, params={}, confidence=1.0))
        assert len(all_events) == 2


class TestEventLogging:
    def test_enable_logging(self, tk):
        tk.enable_logging()
        tk.bus.emit(PromptEvent(message="hello"))
        assert len(tk.event_log) == 1
