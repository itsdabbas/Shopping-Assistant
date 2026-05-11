"""
Intent Pattern Definitions  (Toolkit Core)
==========================================

This module defines the **dialog-resolution patterns** (REPAIR / CONFIRM /
DENY / REPEAT / CANCEL) that are shared across *all* applications built on
top of this toolkit.

Application-specific intent vocabularies (e.g. shopping commands, kiosk
commands) live **outside** the core in ``apps/<app_name>/intents.py`` and
are registered at runtime via ``IntentParser.register_pattern()`` or passed
as ``custom_patterns`` to the constructor.

The toolkit ships with *no* default object-bound or global commands —
those are the developer's responsibility.  This keeps the core free of
any domain knowledge.

Usage
-----
    from gazeshop.toolkit.intents import IntentPattern, DIALOG_PATTERNS

    parser = IntentParser(config, custom_patterns=MY_APP_INTENTS)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IntentPattern:
    """A single intent definition with its recognition patterns.

    Parameters
    ----------
    intent:
        Canonical intent name (e.g. ``"OPEN_DETAIL"``).
    target_required:
        Whether the command needs a gaze-locked target to execute.
    patterns:
        List of case-insensitive regex strings matched against ASR transcripts.
    slot_extractors:
        Optional dict mapping slot names to regex patterns that extract
        additional parameters from the transcript.
    """

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
# Dialog-resolution patterns  (shared across ALL applications)
# These are NOT intents — they are meta-commands used during an
# active disambiguation / confirmation dialog.
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

#: The combined structure returned by :func:`get_dialog_patterns`.
DIALOG_PATTERNS = {
    "confirm": CONFIRM_PATTERNS,
    "deny": DENY_PATTERNS,
    "repair": REPAIR_PATTERNS,
    "repeat": REPEAT_PATTERNS,
}

# ────────────────────────────────────────────────────────────────────
# Default empty intent list — applications must supply their own
# ────────────────────────────────────────────────────────────────────

#: Toolkit core ships with NO default intent patterns.
#: Applications register their own via IntentParser(custom_patterns=...).
INTENT_PATTERNS: list[IntentPattern] = []


def get_pattern_by_intent(
    intent_name: str,
    patterns: list[IntentPattern] | None = None,
) -> IntentPattern | None:
    """Look up an IntentPattern by its canonical intent name.

    Parameters
    ----------
    intent_name:
        Canonical name to search for (e.g. ``"OPEN_DETAIL"``).
    patterns:
        Pattern list to search.  Defaults to :data:`INTENT_PATTERNS`
        (which is empty in the core — pass your app-specific list).
    """
    search_in = patterns if patterns is not None else INTENT_PATTERNS
    for pat in search_in:
        if pat.intent == intent_name:
            return pat
    return None
