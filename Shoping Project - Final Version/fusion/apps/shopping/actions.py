"""
Shopping App -- Action Handlers
================================

Application-specific handlers that react to MultimodalCommandEvent.
Each handler corresponds to one intent from apps.shopping.intents.
"""

from __future__ import annotations

from typing import Any

from gazeshop.toolkit.events import MultimodalCommandEvent

# -- Simulated in-memory state --
_cart: list[dict] = []
_pinned: list[str] = []
_history: list[str] = []


def _c(color: str, text: str) -> str:
    return f"{color}{text}\033[0m"


def handle_add_to_cart(cmd: MultimodalCommandEvent) -> None:
    item = cmd.target_id or "unknown_item"
    _cart.append({"item": item, "params": cmd.params})
    _history.append(item)
    print(_c("\033[92m", f"  [CART+] Added '{item}' to cart  (cart size: {len(_cart)})"))


def handle_show_details(cmd: MultimodalCommandEvent) -> None:
    item = cmd.target_id or "unknown_item"
    detail_type = cmd.params.get("detail_type", "general")
    print(_c("\033[94m", f"  [DETAIL] Showing {detail_type} details for '{item}'"))


def handle_find_similar(cmd: MultimodalCommandEvent) -> None:
    item = cmd.target_id or "unknown_item"
    print(_c("\033[94m", f"  [SEARCH] Finding products similar to '{item}'"))


def handle_compare(cmd: MultimodalCommandEvent) -> None:
    item = cmd.target_id or "unknown_item"
    ref = cmd.params.get("compare_ref", "pinned")
    print(_c("\033[94m", f"  [COMPARE] Comparing '{item}' vs {ref}"))


def handle_show_alternatives(cmd: MultimodalCommandEvent) -> None:
    item = cmd.target_id or "unknown_item"
    print(_c("\033[94m", f"  [ALT] Showing alternatives to '{item}'"))


def handle_pin_item(cmd: MultimodalCommandEvent) -> None:
    item = cmd.target_id or "unknown_item"
    _pinned.append(item)
    print(_c("\033[93m", f"  [PIN] Pinned '{item}'  (pinned: {_pinned})"))


def handle_remove_item(cmd: MultimodalCommandEvent) -> None:
    item = cmd.target_id or "unknown_item"
    before = len(_cart)
    _cart[:] = [c for c in _cart if c["item"] != item]
    removed = before - len(_cart)
    print(_c("\033[91m", f"  [REMOVE] Removed {removed}x '{item}' from cart"))


def handle_scroll(cmd: MultimodalCommandEvent) -> None:
    direction = cmd.params.get("direction", "down")
    print(_c("\033[90m", f"  [SCROLL] Scrolling {direction}"))


def handle_open_cart(cmd: MultimodalCommandEvent) -> None:
    print(_c("\033[92m", f"  [CART] Opening cart -- {len(_cart)} item(s): "
              + str([c['item'] for c in _cart])))


def handle_go_back(cmd: MultimodalCommandEvent) -> None:
    print(_c("\033[90m", "  [BACK] Going back"))


def handle_help(cmd: MultimodalCommandEvent) -> None:
    print(_c("\033[96m", "  [HELP] Available commands: add to cart, show details, "
              "find similar, compare, scroll up/down, open cart, go back, undo, cancel"))


def handle_cancel(cmd: MultimodalCommandEvent) -> None:
    print(_c("\033[91m", "  [CANCEL] Action cancelled by user"))


def handle_undo(cmd: MultimodalCommandEvent) -> None:
    if _history:
        last = _history.pop()
        _cart[:] = [c for c in _cart if c["item"] != last]
        print(_c("\033[93m", f"  [UNDO] Undid last action (removed '{last}' from cart)"))
    else:
        print(_c("\033[93m", "  [UNDO] Nothing to undo"))


# -- Handler registry --

SHOPPING_ACTION_HANDLERS: dict[str, Any] = {
    "ADD_TO_CART":       handle_add_to_cart,
    "SHOW_DETAILS":      handle_show_details,
    "FIND_SIMILAR":      handle_find_similar,
    "COMPARE":           handle_compare,
    "SHOW_ALTERNATIVES": handle_show_alternatives,
    "PIN_ITEM":          handle_pin_item,
    "REMOVE_ITEM":       handle_remove_item,
    "SCROLL":            handle_scroll,
    "OPEN_CART":         handle_open_cart,
    "GO_BACK":           handle_go_back,
    "HELP":              handle_help,
    "CANCEL":            handle_cancel,
    "UNDO":              handle_undo,
}
