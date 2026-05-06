from dataclasses import dataclass
import re
from typing import Any


@dataclass
class IntentSpec:
    name: str
    target_required: bool
    patterns: list[str]


INTENTS: list[IntentSpec] = [
    IntentSpec("ADD_TO_CART", True, [r"\badd (this|it) to cart\b", r"\bbuy this\b"]),
    IntentSpec("SHOW_DETAILS", True, [r"\bshow details\b", r"\bwhat is this\b"]),
    IntentSpec("FIND_SIMILAR", True, [r"\bfind similar\b", r"\banything like this\b"]),
    IntentSpec("COMPARE", True, [r"\bcompare this\b"]),
    IntentSpec("SHOW_ALTERNATIVES", True, [r"\bshow alternatives\b", r"\bother options\b"]),
    IntentSpec("PIN_ITEM", True, [r"\bpin this\b", r"\bsave this\b"]),
    IntentSpec("REMOVE_ITEM", True, [r"\bremove this\b", r"\bdelete this\b"]),
    IntentSpec("SCROLL", False, [r"\bscroll (down|up)\b"]),
    IntentSpec("OPEN_CART", False, [r"\bopen cart\b", r"\bshow my cart\b"]),
    IntentSpec("GO_BACK", False, [r"\bgo back\b", r"\bprevious page\b"]),
    IntentSpec("HELP", False, [r"\bhelp\b", r"\bwhat can i say\b"]),
    IntentSpec("CANCEL", False, [r"\bcancel\b", r"\bnever mind\b"]),
    IntentSpec("UNDO", False, [r"\bundo\b", r"\btake that back\b"]),
]

CONFIRM_PATTERNS = [r"\byes\b", r"\byeah\b", r"\bcorrect\b"]
DENY_PATTERNS = [r"\bno\b", r"\bnope\b", r"\bwrong\b"]
REPAIR_PATTERNS = [r"\bleft one\b", r"\bright one\b", r"\bfirst one\b"]
REPEAT_PATTERNS = [r"\brepeat\b", r"\bsay again\b"]


def match_dialog_type(text: str) -> str | None:
    t = text.lower().strip()
    if any(re.search(p, t) for p in CONFIRM_PATTERNS):
        return "CONFIRM"
    if any(re.search(p, t) for p in DENY_PATTERNS):
        return "CANCEL"
    if any(re.search(p, t) for p in REPAIR_PATTERNS):
        return "REPAIR"
    if any(re.search(p, t) for p in REPEAT_PATTERNS):
        return "REPEAT"
    return None


def parse_intent(text: str) -> dict[str, Any] | None:
    t = text.lower().strip()
    dialog_type = match_dialog_type(t)
    if dialog_type:
        return {
            "type": dialog_type,
            "payload": {"intent_name": dialog_type, "target_required": False},
        }

    for spec in INTENTS:
        for p in spec.patterns:
            m = re.search(p, t)
            if not m:
                continue
            payload: dict[str, Any] = {"intent_name": spec.name, "target_required": spec.target_required}
            if spec.name == "SCROLL" and m.groups():
                payload["params"] = {"direction": m.group(1)}
            return {"type": "INTENT", "payload": payload}
    return None
