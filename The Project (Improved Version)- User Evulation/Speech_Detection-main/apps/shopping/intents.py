"""
Shopping App — Intent Vocabulary
=================================

All shopping-specific intent patterns live here.
The toolkit core knows nothing about these commands.
"""

from __future__ import annotations

from gazeshop.toolkit.intents import IntentPattern

# ── Object-bound commands  (require gaze lock) ───────────────────────

SHOPPING_INTENTS: list[IntentPattern] = [
    IntentPattern(
        intent="ADD_TO_CART",
        target_required=True,
        patterns=[
            r"\b(add|put)\b.*(cart|basket)",
            r"\b(buy|purchase|want)\b.*\b(this|it|that)\b",
            r"\badd\s*(this|it)\b",
            r"^(add|cart|card)$",
            r"^(two|to)\s+(cart|card)$",
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
        patterns=[r"\bcompare\b"],
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
    # ── Global commands  (no gaze needed) ───────────────────────────
    IntentPattern(
        intent="SCROLL",
        target_required=False,
        patterns=[
            r"\bscroll\s+(up|down)\b",
            r"\bnext\s+page\b",
            r"\bgo\s+(up|down)\b",
        ],
        slot_extractors={"direction": r"\b(up|down|next|previous)\b"},
    ),
    IntentPattern(
        intent="OPEN_CART",
        target_required=False,
        patterns=[r"\b(open|show|view)\b.*\bcart\b"],
    ),
    IntentPattern(
        intent="GO_BACK",
        target_required=False,
        patterns=[r"\bgo\s*back\b", r"\bprevious\s+page\b", r"^back$"],
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
        patterns=[r"\b(cancel|never\s*mind|stop|forget\s+it)\b"],
    ),
    IntentPattern(
        intent="UNDO",
        target_required=False,
        patterns=[r"\bundo\b", r"\btake\s+(that|it)\s+back\b"],
    ),
]

#: Whisper vocabulary-priming prompt for shopping commands.
WHISPER_PROMPT = (
    "Add to cart. Show details. Find similar. Compare. Show alternatives. "
    "Pin this. Remove this. Scroll up. Scroll down. Open cart. "
    "Help. Cancel. Never mind. Undo. Go back. "
    "Yes. No. Left. Right. First. Second."
)
