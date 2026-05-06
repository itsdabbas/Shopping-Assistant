"""
Kiosk App — Intent Vocabulary
==============================

Museum / Document Kiosk commands.
Completely different vocabulary from the Shopping app;
proves that the toolkit is domain-agnostic.

Supported tasks demonstrated in the demo:
  1. READ_ALOUD    — "tell me about this exhibit"
  2. SUMMARIZE     — "give me the short version"
  3. ZOOM_IN       — "zoom in" / "enlarge"
  4. OPEN_DETAIL   — "open detail panel" / "show full info"
  5. NAVIGATE_NEXT — "next" / "go forward"
  6. NAVIGATE_PREV — "back" / "previous"
  7. PIN_EXHIBIT   — "bookmark this" / "save for later"
  8. COMPARE_ITEMS — "compare" (two gaze-pinned exhibits)
  9. HELP          — "help" / "what can I say"
 10. CANCEL        — "cancel" / "never mind"
"""

from __future__ import annotations

from gazeshop.toolkit.intents import IntentPattern

KIOSK_INTENTS: list[IntentPattern] = [
    # ── Object-bound (target_required=True) ─────────────────────────
    IntentPattern(
        intent="READ_ALOUD",
        target_required=True,
        patterns=[
            r"\bread\s+(this|aloud)\b",
            r"\baudio\s+guide\b",
            r"\btell\s+me\s+about\s+(this|that)\b",
            r"\bdescribe\s+(this|that|it)\b",
        ],
    ),
    IntentPattern(
        intent="SUMMARIZE",
        target_required=True,
        patterns=[
            r"\bsummar(ize|y)\b",
            r"\bgive\s+me\s+the\s+short\s+version\b",
            r"\btl\s*[;:]?\s*dr\b",
            r"\bin\s+brief\b",
        ],
    ),
    IntentPattern(
        intent="ZOOM_IN",
        target_required=True,
        patterns=[
            r"\bzoom\s+in\b",
            r"\benlarge\b",
            r"\bbigger\b",
            r"\bcloser\b",
        ],
    ),
    IntentPattern(
        intent="OPEN_DETAIL",
        target_required=True,
        patterns=[
            r"\b(open|show)\s+(detail|full\s+info|more\s+info)\b",
            r"\bmore\s+about\s+(this|that|it)\b",
            r"\blearn\s+more\b",
        ],
    ),
    IntentPattern(
        intent="PIN_EXHIBIT",
        target_required=True,
        patterns=[
            r"\b(bookmark|save\s+for\s+later|remember|pin)\s+(this|it|that)?\b",
            r"\badd\s+to\s+(favourites|favorites|my\s+list)\b",
        ],
    ),
    IntentPattern(
        intent="COMPARE_ITEMS",
        target_required=True,
        patterns=[
            r"\bcompare\s+(this|it|that)?\b",
            r"\bside\s+by\s+side\b",
        ],
    ),
    # ── Global navigation (target_required=False) ────────────────────
    IntentPattern(
        intent="NAVIGATE_NEXT",
        target_required=False,
        patterns=[
            r"\bnext\b",
            r"\bgo\s+forward\b",
            r"\bnext\s+(page|exhibit|item)\b",
        ],
    ),
    IntentPattern(
        intent="NAVIGATE_PREV",
        target_required=False,
        patterns=[
            r"\b(go\s+)?back\b",
            r"\bprevious\b",
            r"\bprev\b",
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
        patterns=[r"\b(cancel|never\s*mind|stop|forget\s+it)\b"],
    ),
]

#: Whisper vocabulary hint for kiosk commands.
WHISPER_PROMPT = (
    "Read aloud. Audio guide. Tell me about this. Summarize. Zoom in. "
    "Open detail. Learn more. Navigate next. Go back. Bookmark this. "
    "Compare. Help. Cancel. Yes. No. Left. Right."
)
