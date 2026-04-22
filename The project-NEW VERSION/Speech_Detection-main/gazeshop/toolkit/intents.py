"""
Intent Pattern Definitions
==========================

This module defines the full catalogue of recognisable speech
commands as ``IntentPattern`` dataclasses.  Each pattern specifies:

* **intent** — canonical name (e.g. ``"ADD_TO_CART"``).
* **target_required** — whether the command needs a gaze-locked
  target to be actionable.
* **patterns** — list of case-insensitive regex strings to match
  against ASR transcripts.
* **slot_extractors** — optional dict mapping slot names to regex
  patterns that extract additional parameters from the transcript.

The patterns are applied *in list order*.  The ``IntentParser``
tries dialog-context patterns first (REPAIR / CONFIRM) when a
dialog is active, then falls through to these.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IntentPattern:
    """A single intent definition with its recognition patterns."""

    intent: str
    target_required: bool
    patterns: list[str]
    slot_extractors: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"IntentPattern(intent={self.intent!r}, "
            f"target_required={self.target_required}, "
            f"patterns={len(self.patterns)})"
        )


# ────────────────────────────────────────────────────────────────────
# Object-bound commands  (target_required = True)
# ────────────────────────────────────────────────────────────────────

_OBJECT_BOUND: list[IntentPattern] = [
    IntentPattern(
        intent="ADD_TO_CART",
        target_required=True,
        patterns=[
            r"\b(add|put)\b.*(cart|basket)",
            r"\b(buy|purchase|want)\b.*\b(this|it|that)\b",
            r"\badd\s*(this|it)\b",
        ],
    ),
    IntentPattern(
        intent="SHOW_DETAILS",
        target_required=True,
        patterns=[
            r"\b(show|tell|give)\b.*(detail|info|more|ingredient|spec|review)",
            r"\bwhat\s+is\s+(this|that|it)\b",
            r"\bmore\s+info\b",
        ],
        slot_extractors={
            "detail_type": r"\b(ingredient|spec|review)s?\b",
        },
    ),
    IntentPattern(
        intent="FIND_SIMILAR",
        target_required=True,
        patterns=[
            r"\b(find|show|get)\b.*\bsimilar\b",
            r"\banything\s+like\b",
            r"\blike\s+this\s+one\b",
        ],
    ),
    IntentPattern(
        intent="COMPARE",
        target_required=True,
        patterns=[
            r"\bcompare\b",
        ],
        slot_extractors={
            "compare_ref": r"\b(pinned|cheaper|left|right|previous)\b",
        },
    ),
    IntentPattern(
        intent="SHOW_ALTERNATIVES",
        target_required=True,
        patterns=[
            r"\b(show|get|find)\b.*\b(alternative|option)s?\b",
            r"\bwhat\s+else\b",
            r"\bother\s+(option|choice)s?\b",
        ],
    ),
    IntentPattern(
        intent="PIN_ITEM",
        target_required=True,
        patterns=[
            r"\b(pin|save|remember|bookmark)\b.*\b(this|it|that)\b",
            r"\b(pin|save|remember)\b",
        ],
    ),
    IntentPattern(
        intent="REMOVE_ITEM",
        target_required=True,
        patterns=[
            r"\b(remove|delete|take\s+off)\b.*\b(this|it|that)\b",
            r"\b(remove|delete)\b",
        ],
    ),
]

# ────────────────────────────────────────────────────────────────────
# Global commands  (target_required = False)
# ────────────────────────────────────────────────────────────────────

_GLOBAL: list[IntentPattern] = [
    IntentPattern(
        intent="SCROLL",
        target_required=False,
        patterns=[
            r"\bscroll\s+(up|down)\b",
            r"\bnext\s+page\b",
            r"\bgo\s+(up|down)\b",
        ],
        slot_extractors={
            "direction": r"\b(up|down|next|previous)\b",
        },
    ),
    IntentPattern(
        intent="OPEN_CART",
        target_required=False,
        patterns=[
            r"\b(open|show|view)\b.*\bcart\b",
        ],
    ),
    IntentPattern(
        intent="HELP",
        target_required=False,
        patterns=[
            r"\bhelp\b",
            r"\bwhat\s+can\s+I\s+say\b",
            r"\bshow\s+commands\b",
        ],
    ),
    IntentPattern(
        intent="CANCEL",
        target_required=False,
        patterns=[
            r"\b(cancel|never\s*mind|stop|forget\s+it)\b",
        ],
    ),
    IntentPattern(
        intent="UNDO",
        target_required=False,
        patterns=[
            r"\bundo\b",
            r"\btake\s+(that|it)\s+back\b",
        ],
    ),
    IntentPattern(
        intent="GO_BACK",
        target_required=False,
        patterns=[
            r"\bgo\s*back\b",
            r"\bprevious\s+page\b",
            r"^back$",
        ],
    ),
]

# ────────────────────────────────────────────────────────────────────
# Repair / Confirmation patterns  (dialog-context only)
# ────────────────────────────────────────────────────────────────────

CONFIRM_PATTERNS: list[str] = [
    r"\b(yes|yeah|yep|correct|confirm|sure|absolutely)\b",
]

DENY_PATTERNS: list[str] = [
    r"\b(no|nope|nah|wrong|not\s+that)\b",
]

REPAIR_PATTERNS: dict[str, list[str]] = {
    "left":   [r"\bleft\b", r"\bfirst\b"],
    "right":  [r"\bright\b", r"\bsecond\b"],
    "other":  [r"\bother\b"],
    "top":    [r"\btop\b"],
    "bottom": [r"\bbottom\b"],
}

REPEAT_PATTERNS: list[str] = [
    r"\b(repeat|again|what|pardon)\b",
]

# ────────────────────────────────────────────────────────────────────
# Combined list used by IntentParser
# ────────────────────────────────────────────────────────────────────

INTENT_PATTERNS: list[IntentPattern] = _OBJECT_BOUND + _GLOBAL
"""All intent patterns in priority order (object-bound first)."""


def get_pattern_by_intent(intent_name: str) -> IntentPattern | None:
    """Look up an IntentPattern by its canonical intent name."""
    for pat in INTENT_PATTERNS:
        if pat.intent == intent_name:
            return pat
    return None
