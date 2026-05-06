# -*- coding: utf-8 -*-
"""
MMUI Toolkit - Live Shopping Demo
===================================
streamlit run demo/live_shopping_app.py
"""

from __future__ import annotations
import sys, time, queue, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.event_bus import SpeechEventType
from apps.shopping.intents import SHOPPING_INTENTS, WHISPER_PROMPT
from apps.shopping.actions import SHOPPING_ACTION_HANDLERS

# ── Products ────────────────────────────────────────────────────────
PRODUCTS = [
    {"id": "product_1",  "name": "Organic Coffee",    "price": "$12.99", "icon": "☕"},
    {"id": "product_2",  "name": "Green Tea",         "price": "$8.49",  "icon": "🍵"},
    {"id": "product_3",  "name": "Dark Chocolate",    "price": "$5.99",  "icon": "🍫"},
    {"id": "product_4",  "name": "Almond Milk",       "price": "$3.99",  "icon": "🥛"},
    {"id": "product_5",  "name": "Granola Bar",       "price": "$2.49",  "icon": "🌾"},
    {"id": "product_6",  "name": "Sparkling Water",   "price": "$1.99",  "icon": "💧"},
    {"id": "product_7",  "name": "Blueberry Muffin",  "price": "$3.49",  "icon": "🫐"},
    {"id": "product_8",  "name": "Avocado Toast",     "price": "$7.99",  "icon": "🥑"},
    {"id": "product_9",  "name": "Greek Yogurt",      "price": "$4.29",  "icon": "🥣"},
    {"id": "product_10", "name": "Orange Juice",      "price": "$4.99",  "icon": "🍊"},
    {"id": "product_11", "name": "Protein Bar",       "price": "$3.99",  "icon": "💪"},
    {"id": "product_12", "name": "Chia Pudding",      "price": "$5.49",  "icon": "🌱"},
]

DWELL_S = 1.2  # seconds to auto-lock after hovering

CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
.stApp { background: #0f1117; }
.card { background:#1e2130; border:2px solid #2d3148; border-radius:12px;
        padding:14px; text-align:center; margin:4px; cursor:pointer;
        transition:all .2s; }
.card.locked { border-color:#7c3aed; background:#2a1f4a;
               box-shadow:0 0 18px #7c3aed88; }
.card.hovering { border-color:#f59e0b; background:#2a2010; }
.card h4 { color:#e2e8f0; font-size:13px; margin:6px 0 2px; }
.card p  { color:#94a3b8; font-size:11px; margin:0; }
.card .icon { font-size:28px; }
.tag-locked  { background:#7c3aed; color:white; padding:2px 8px;
               border-radius:8px; font-size:11px; font-weight:700; }
.tag-idle    { background:#334155; color:#94a3b8; padding:2px 8px;
               border-radius:8px; font-size:11px; }
.tag-prompt  { background:#f59e0b; color:#1e2130; padding:4px 10px;
               border-radius:8px; font-size:12px; font-weight:700; }
.tag-cmd     { background:#059669; color:white; padding:4px 10px;
               border-radius:8px; font-size:12px; font-weight:700; }
.tag-cancel  { background:#dc2626; color:white; padding:4px 10px;
               border-radius:8px; font-size:12px; }
.ev-row { font-size:11px; color:#94a3b8; padding:2px 0; border-bottom:1px solid #1e2130; }
</style>
"""

# ── Shared singleton (survive reruns) ───────────────────────────────
_toolkit_ref: dict = {}

def _get_toolkit() -> dict:
    return _toolkit_ref

# ── Init ─────────────────────────────────────────────────────────────

def init():
    ss = st.session_state
    if ss.get("initialized"):
        return

    eq: queue.Queue = queue.Queue()
    ss.event_queue = eq

    cfg = Config(FUSION_TIME_WINDOW_S=2.0, LOCK_TTL_S=5.0,
                 MAX_REPAIR_ATTEMPTS=2, ENABLE_TELEMETRY=False,
                 WHISPER_INITIAL_PROMPT=WHISPER_PROMPT)

    tk = MMUIToolkit(cfg)
    parser = IntentParser(cfg, custom_patterns=SHOPPING_INTENTS)
    gaze = GazeAdapterStub(event_bus=tk.bus)
    speech = SpeechAdapter(event_bus=tk.bus, config=cfg, intent_parser=parser)

    tk.register_adapter(gaze).register_adapter(speech)
    for intent, handler in SHOPPING_ACTION_HANDLERS.items():
        tk.register_action(intent, handler)

    # ── EventBus → Queue bridge ──────────────────────────────────────
    def _put(kind, data=None):
        eq.put((kind, data or {}))

    tk.register_feedback("TargetLockedEvent",
        lambda e: _put("locked", {"id": e.target_id}))
    tk.register_feedback("TargetUnlockedEvent",
        lambda e: _put("unlocked", {}))
    tk.register_feedback("TargetExpiredEvent",
        lambda e: _put("expired", {"id": e.target_id}))
    tk.register_feedback("PromptEvent",
        lambda e: _put("prompt", {"msg": e.message}))
    tk.register_feedback("DisambiguationPromptEvent",
        lambda e: _put("disambig", {"msg": e.message}))
    tk.register_feedback("ConfirmationPromptEvent",
        lambda e: _put("confirm_prompt", {"msg": e.message}))
    tk.register_feedback("ActionCancelledEvent",
        lambda e: _put("cancelled", {"msg": e.message}))
    tk.register_feedback("MultimodalCommandEvent",
        lambda e: _put("command", {"intent": e.intent,
                                   "target": e.target_id,
                                   "conf": e.confidence}))
    tk.bus.subscribe("SpeechEvent",
        lambda e: _put("speech", {"type": e.type.name,
                                   "transcript": e.transcript,
                                   "payload": e.payload,
                                   "conf": e.confidence}) \
                  if e.type == SpeechEventType.INTENT else None)

    tk.start()
    _toolkit_ref.update({"tk": tk, "gaze": gaze, "speech": speech})

    ss.locked_target  = None
    ss.fusion_state   = "IDLE"
    ss.prompt_msg     = ""
    ss.last_intent    = ""
    ss.last_transcript= ""
    ss.last_conf      = 0.0
    ss.disambig_active= False
    ss.confirm_active = False
    ss.event_log      = []
    ss.cart           = []
    ss.dwell_target   = None
    ss.dwell_start    = None
    ss.dwell_progress = 0.0
    ss.last_cmd       = ""
    ss.initialized    = True


def log_event(tag: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.event_log.insert(0, f"{ts}  {tag}  {msg}")
    if len(st.session_state.event_log) > 12:
        st.session_state.event_log.pop()


def drain():
    ss = st.session_state
    while not ss.event_queue.empty():
        kind, data = ss.event_queue.get_nowait()

        if kind == "locked":
            ss.locked_target  = data["id"]
            ss.fusion_state   = "LOCKED"
            ss.disambig_active= False
            ss.confirm_active = False
            ss.prompt_msg     = ""
            log_event("[GAZE]", f"Locked -> {data['id']}")

        elif kind in ("unlocked", "expired"):
            ss.locked_target = None
            ss.fusion_state  = "IDLE"
            label = "Unlocked" if kind == "unlocked" else f"Expired ({data.get('id','')})"
            log_event("[GAZE]", label)

        elif kind == "prompt":
            ss.fusion_state = "NEEDS_TARGET"
            ss.prompt_msg   = data["msg"]
            log_event("[PROMPT]", data["msg"])

        elif kind == "disambig":
            ss.fusion_state   = "DISAMBIG"
            ss.prompt_msg     = data["msg"]
            ss.disambig_active= True
            ss.confirm_active = False
            log_event("[DISAMBIG]", data["msg"])

        elif kind == "confirm_prompt":
            ss.fusion_state  = "CONFIRM"
            ss.prompt_msg    = data["msg"]
            ss.confirm_active= True
            ss.disambig_active= False
            log_event("[CONFIRM?]", data["msg"])

        elif kind == "command":
            ss.fusion_state   = "COMMAND"
            ss.last_cmd       = f"{data['intent']} on {data['target']} ({data['conf']:.0%})"
            ss.disambig_active= False
            ss.confirm_active = False
            ss.prompt_msg     = ""
            log_event("[CMD]", ss.last_cmd)
            if data["intent"] == "ADD_TO_CART" and data["target"]:
                ss.cart.append(data["target"])

        elif kind == "cancelled":
            ss.fusion_state   = "CANCELLED"
            ss.prompt_msg     = data["msg"]
            ss.disambig_active= False
            ss.confirm_active = False
            log_event("[CANCEL]", data["msg"])

        elif kind == "speech":
            ss.last_intent     = data["payload"].get("intent", "")
            ss.last_transcript = data["transcript"]
            ss.last_conf       = data["conf"]
            log_event("[SPEECH]", f"'{data['transcript']}' -> {ss.last_intent} ({data['conf']:.0%})")


def update_dwell():
    ss = st.session_state
    if ss.dwell_target and ss.dwell_start:
        elapsed = time.time() - ss.dwell_start
        ss.dwell_progress = min(elapsed / DWELL_S, 1.0)
        if ss.dwell_progress >= 1.0:
            ref = _get_toolkit()
            if ref.get("gaze"):
                ref["gaze"].simulate_lock(ss.dwell_target)
            ss.dwell_target  = None
            ss.dwell_start   = None
            ss.dwell_progress = 0.0


# ── UI ────────────────────────────────────────────────────────────────

def render_header():
    ss = st.session_state
    color_map = {"IDLE": "#64748b", "LOCKED": "#7c3aed", "NEEDS_TARGET": "#f59e0b",
                 "DISAMBIG": "#f97316", "CONFIRM": "#3b82f6",
                 "COMMAND": "#059669", "CANCELLED": "#dc2626"}
    col1, col2, col3 = st.columns([5, 2, 1])
    with col1:
        st.markdown("## 🛒 MMUI Toolkit — Shopping Live Demo")
    with col2:
        c = color_map.get(ss.fusion_state, "#64748b")
        st.markdown(f"<div style='margin-top:10px'>"
                    f"<span style='background:{c};color:white;padding:4px 10px;"
                    f"border-radius:8px;font-weight:700;font-size:13px'>"
                    f"● {ss.fusion_state}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='margin-top:10px;font-size:14px;color:#94a3b8'>"
                    f"🛒 <b>{len(ss.cart)}</b></div>", unsafe_allow_html=True)
    st.divider()


def render_grid():
    ss = st.session_state
    ref = _get_toolkit()
    gaze = ref.get("gaze")
    cols = st.columns(4)
    for i, p in enumerate(PRODUCTS):
        locked = ss.locked_target == p["id"]
        hovering = ss.dwell_target == p["id"]
        css_class = "locked" if locked else ("hovering" if hovering else "")
        border = "#7c3aed" if locked else ("#f59e0b" if hovering else "#2d3148")
        bg = "#2a1f4a" if locked else ("#2a2010" if hovering else "#1e2130")
        label = "LOCKED" if locked else ("HOVERING..." if hovering else "")
        label_html = (f"<br><span class='tag-locked'>{label}</span>" if locked else
                      f"<br><span style='color:#f59e0b;font-size:10px'>● dwelling...</span>" if hovering else "")

        with cols[i % 4]:
            st.markdown(
                f"<div class='card {css_class}' style='border-color:{border};background:{bg}'>"
                f"<div class='icon'>{p['icon']}</div>"
                f"<h4>{p['name']}</h4>"
                f"<p>{p['price']}{label_html}</p>"
                f"</div>", unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                if st.button("👁 Look", key=f"lock_{p['id']}", use_container_width=True):
                    ss.dwell_target  = None; ss.dwell_start = None
                    if gaze: gaze.simulate_lock(p["id"])
            with b2:
                if st.button("⏱ Hover", key=f"dwell_{p['id']}", use_container_width=True):
                    ss.dwell_target = p["id"]
                    ss.dwell_start  = time.time()
                    ss.dwell_progress = 0.0


def render_controls(speech_adapter):
    ss = st.session_state
    ref = _get_toolkit()
    gaze = ref.get("gaze")

    st.markdown("### Gaze Control")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔓 Release Gaze", use_container_width=True):
            if gaze: gaze.simulate_unlock()
            ss.dwell_target = None; ss.dwell_start = None
    with c2:
        if st.button("⚡ Ambiguous", use_container_width=True):
            items = [p["id"] for p in PRODUCTS[:2]]
            if gaze: gaze.simulate_ambiguous(items)

    # Dwell progress
    if ss.dwell_target:
        st.markdown(f"<small style='color:#f59e0b'>Dwelling on: {ss.dwell_target}</small>",
                    unsafe_allow_html=True)
        st.progress(ss.dwell_progress)

    st.divider()
    st.markdown("### 🎙 Speech Input")
    txt = st.text_input("Type command:", key="speech_input",
                        placeholder="e.g. add this to cart")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Process", use_container_width=True):
            if txt and speech_adapter:
                speech_adapter.set_dialog_active(
                    ss.disambig_active or ss.confirm_active)
                speech_adapter.process_text(txt)
    with col_b:
        if st.button("🔄 Dialog ON", use_container_width=True):
            if speech_adapter: speech_adapter.set_dialog_active(True)

    # Quick command buttons
    st.markdown("<small style='color:#64748b'>Quick commands:</small>",
                unsafe_allow_html=True)
    qcols = st.columns(3)
    quick = [("Add Cart","add this to cart"),("Details","show details"),
             ("Similar","find similar"),("Yes","yes"),("No","no"),("Left","left")]
    for i,(label,cmd) in enumerate(quick):
        with qcols[i % 3]:
            if st.button(label, key=f"qcmd_{i}", use_container_width=True):
                if speech_adapter:
                    speech_adapter.set_dialog_active(
                        ss.disambig_active or ss.confirm_active)
                    speech_adapter.process_text(cmd)

    # Disambiguation buttons
    if ss.disambig_active:
        st.divider()
        st.markdown("<div class='tag-prompt'>DISAMBIGUATION</div>", unsafe_allow_html=True)
        st.markdown(f"<small>{ss.prompt_msg}</small>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            if st.button("← LEFT", use_container_width=True, type="primary"):
                if speech_adapter:
                    speech_adapter.set_dialog_active(True)
                    speech_adapter.process_text("left")
        with d2:
            if st.button("RIGHT →", use_container_width=True):
                if speech_adapter:
                    speech_adapter.set_dialog_active(True)
                    speech_adapter.process_text("right")

    # Confirmation buttons
    if ss.confirm_active:
        st.divider()
        st.markdown("<div class='tag-prompt'>CONFIRMATION NEEDED</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<small>{ss.prompt_msg}</small>", unsafe_allow_html=True)
        y1, y2 = st.columns(2)
        with y1:
            if st.button("✅ YES", use_container_width=True, type="primary"):
                if speech_adapter:
                    speech_adapter.set_dialog_active(True)
                    speech_adapter.process_text("yes")
        with y2:
            if st.button("❌ NO", use_container_width=True):
                if speech_adapter:
                    speech_adapter.set_dialog_active(True)
                    speech_adapter.process_text("no")

    # Prompt / Status
    if ss.prompt_msg and not ss.disambig_active and not ss.confirm_active:
        st.divider()
        st.warning(ss.prompt_msg)

    st.divider()
    st.markdown("### Status")
    st.markdown(f"**Target:** `{ss.locked_target or 'none'}`")
    st.markdown(f"**Intent:** `{ss.last_intent or '-'}`")
    st.markdown(f"**Transcript:** _{ss.last_transcript or '-'}_")
    if ss.last_conf:
        st.progress(ss.last_conf, text=f"Confidence: {ss.last_conf:.0%}")
    if ss.last_cmd:
        st.markdown(f"<div class='tag-cmd'>CMD: {ss.last_cmd}</div>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### Cart")
    if ss.cart:
        for item in ss.cart[-6:]:
            st.markdown(f"• `{item}`")
    else:
        st.markdown("<small style='color:#64748b'>Empty</small>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### Event Log")
    for ev in ss.event_log[:10]:
        st.markdown(f"<div class='ev-row'>{ev}</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="MMUI Shopping Live Demo",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    init()
    drain()
    update_dwell()

    ref = _get_toolkit()
    speech = ref.get("speech")

    render_header()

    grid_col, ctrl_col = st.columns([3, 1])
    with grid_col:
        render_grid()
    with ctrl_col:
        render_controls(speech)

    time.sleep(0.18)
    st.rerun()


main()
