from collections import defaultdict
from typing import Any, Callable


class EventBus:
    def __init__(self, enable_logging: bool = True) -> None:
        self._subs: dict[str, list[Callable[[str, Any], None]]] = defaultdict(list)
        self.enable_logging = enable_logging

    def subscribe(self, topic: str, handler: Callable[[str, Any], None]) -> None:
        self._subs[topic].append(handler)

    def publish(self, topic: str, event: Any) -> None:
        if self.enable_logging:
            print(f"[EventBus] {topic}: {event}")
        for handler in self._subs.get(topic, []):
            handler(topic, event)
        for handler in self._subs.get("*", []):
            handler(topic, event)
