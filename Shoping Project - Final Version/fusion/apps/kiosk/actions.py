"""
Kiosk App -- Action Handlers
==============================

Handlers for the Museum / Document Kiosk demo.
All kiosk-specific domain logic lives here; the toolkit core is untouched.
"""

from __future__ import annotations

from typing import Any

from gazeshop.toolkit.events import MultimodalCommandEvent

# -- Simulated in-memory kiosk state --
_bookmarks: list[str] = []
_history: list[str] = []
_current_page: int = 1


def _c(color: str, text: str) -> str:
    return f"{color}{text}\033[0m"


def handle_read_aloud(cmd: MultimodalCommandEvent) -> None:
    exhibit = cmd.target_id or "current exhibit"
    print(_c("\033[95m", f"  [AUDIO GUIDE] Reading aloud: '{exhibit}'"))
    print(_c("\033[90m", f"       'This remarkable piece dates back to the 15th century...'"))


def handle_summarize(cmd: MultimodalCommandEvent) -> None:
    exhibit = cmd.target_id or "current exhibit"
    print(_c("\033[95m", f"  [SUMMARY] '{exhibit}': A brief overview displayed."))


def handle_zoom_in(cmd: MultimodalCommandEvent) -> None:
    exhibit = cmd.target_id or "current exhibit"
    print(_c("\033[96m", f"  [ZOOM] Enlarging view on '{exhibit}'"))


def handle_open_detail(cmd: MultimodalCommandEvent) -> None:
    exhibit = cmd.target_id or "current exhibit"
    print(_c("\033[94m", f"  [DETAIL] Opening full information for '{exhibit}'"))


def handle_pin_exhibit(cmd: MultimodalCommandEvent) -> None:
    exhibit = cmd.target_id or "unknown"
    _bookmarks.append(exhibit)
    print(_c("\033[93m", f"  [BOOKMARK] Saved '{exhibit}'  (bookmarks: {_bookmarks})"))


def handle_compare_items(cmd: MultimodalCommandEvent) -> None:
    exhibit = cmd.target_id or "current"
    prev = _bookmarks[-1] if _bookmarks else "none"
    print(_c("\033[96m", f"  [COMPARE] Side-by-side: '{exhibit}' vs '{prev}'"))


def handle_navigate_next(cmd: MultimodalCommandEvent) -> None:
    global _current_page
    _current_page += 1
    print(_c("\033[90m", f"  [NEXT] Navigating -> page {_current_page}"))


def handle_navigate_prev(cmd: MultimodalCommandEvent) -> None:
    global _current_page
    _current_page = max(1, _current_page - 1)
    print(_c("\033[90m", f"  [BACK] Navigating <- page {_current_page}"))


def handle_help(cmd: MultimodalCommandEvent) -> None:
    print(_c("\033[96m", "  [HELP] Kiosk commands: read aloud, summarize, zoom in, "
              "open detail, next, back, bookmark, compare, cancel"))


def handle_cancel(cmd: MultimodalCommandEvent) -> None:
    print(_c("\033[91m", "  [CANCEL] Action cancelled"))


# -- Handler registry --

KIOSK_ACTION_HANDLERS: dict[str, Any] = {
    "READ_ALOUD":    handle_read_aloud,
    "SUMMARIZE":     handle_summarize,
    "ZOOM_IN":       handle_zoom_in,
    "OPEN_DETAIL":   handle_open_detail,
    "PIN_EXHIBIT":   handle_pin_exhibit,
    "COMPARE_ITEMS": handle_compare_items,
    "NAVIGATE_NEXT": handle_navigate_next,
    "NAVIGATE_PREV": handle_navigate_prev,
    "HELP":          handle_help,
    "CANCEL":        handle_cancel,
}
