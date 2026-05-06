# -*- coding: utf-8 -*-
"""
MMUI Toolkit - Live Kiosk Demo
================================
streamlit run demo/live_kiosk_app.py
"""

from __future__ import annotations
import sys, time, queue
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.event_bus import SpeechEventType
from apps.kiosk.intents import KIOSK_INTENTS, WHISPER_PROMPT
from apps.kiosk.actions import KIOSK_ACTION_HANDLERS

# ── Exhibits ─────────────────────────────────────────────────────────
EXHIBITS = [
    {"id": "Mona_Lisa",       "name": "Mona Lisa",          "era": "1503–1519", "icon": "🖼"},
    {"id": "Starry_Night",    "name": "The Starry Night",   "era": "1889",      "icon": "🌌"},
    {"id": "The_Scream",      "name": "The Scream",         "era": "1893",      "icon": "😱"},
    {"id": "Girl_Pearl",      "name": "Girl with a Pearl",  "era": "1665",      "icon": "💎"},
    {"id": "Last_Supper",     "name": "The Last Supper",    "era": "1495–1498", "icon": "🍞"},
    {"id": "Guernica",        "name": "Guernica",           "era": "1937",      "icon": "✊"},
    {"id": "Birth_Venus",     "name": "Birth of Venus",     "era": "1484–1486", "icon": "🌊"},
    {"id": "David",           "name": "David (Michelangelo)","era": "1501–1504","icon": "🗿"},
    {"id": "Sunflowers",      "name": "Sunflowers",         "era": "1888",      "icon": "🌻"},
    {"id": "Persistence",     "name": "Persistence of Mem.","era": "1931",      "icon": "🕰"},
    {"id": "Night_Watch",     "name": "The Night Watch",    "era": "1642",      "icon": "🏛"},
    {"id": "Liberty",         "name": "Liberty Leading",    "era": "1830",      "icon": "🗽"},
]

DWELL_S = 1.2

CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #0d0f14; }
.stApp { background: #0d0f14; }
.card { background:#161a26; border:2px solid #252a3e; border-radius:14px;
        padding:14px; text-align:center; margin:4px; transition:all .2s; }
.card.locked { border-color:#d97706; background:#231a06;
               box-shadow:0 0 20px #d9770666; }
.card.hovering { border-color:#2563eb; background:#0d1a2e; }
.card h4 { color:#e2e8f0; font-size:12px; margin:6px 0 2px; }
.card p  { color:#64748b; font-size:10px; margin:0; }
.card .icon { font-size:30px; }
.tag-locked  { background:#d97706; color:white; padding:2px 8px;
               border-radius:8px; font-size:11px; font-weight:700; }
.tag-prompt  { background:#f59e0b; color:#1e2130; padding:4px 10px;
               border-radius:8px; font-size:12px; font-weight:700; }
.tag-cmd     { background:#059669; color:white; padding:4px 10px;
               border-radius:8px; font-size:12px; font-weight:700; }
.ev-row { font-size:11px; color:#94a3b8; padding:2px 0;
          border-bottom:1px solid #1e2130; }
.detail-box { background:#161a26; border:1px solid #252a3e; border-radius:10px;
              padding:14px; color:#e2e8f0; font-size:13px; min-height:80px; }
</style>
"""

_toolkit_ref: dict = {}

def _get_toolkit() -> dict:
    return _toolkit_ref


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
    parser = IntentParser(cfg, custom_patterns=KIOSK_INTENTS)
    gaze = GazeAdapterStub(event_bus=tk.bus)
    speech = SpeechAdapter(event_bus=tk.bus, config=cfg, intent_parser=parser)

    tk.register_adapter(gaze).register_adapter(speech)
    for intent, handler in KIOSK_ACTION_HANDLERS.items():
        tk.register_action(intent, handler)

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
        lambda e: _put("speech", {"transcript": e.transcript,
                                   "payload": e.payload,
                                   "conf": e.confidence})
                  if e.type == SpeechEventType.INTENT else None)

    tk.start()
    _toolkit_ref.update({"tk": tk, "gaze": gaze, "speech": speech})

    ss.locked_target   = None
    ss.fusion_state    = "IDLE"
    ss.prompt_msg      = ""
    ss.last_intent     = ""
    ss.last_transcript = ""
    ss.last_conf       = 0.0
    ss.disambig_active = False
    ss.confirm_active  = False
    ss.event_log       = []
    ss.bookmarks       = []
    ss.detail_text     = ""
    ss.detail_exhibit  = ""
    ss.current_page    = 1
    ss.dwell_target    = None
    ss.dwell_start     = None
    ss.dwell_progress  = 0.0
    ss.last_cmd        = ""
    ss.initialized     = True


def log_event(tag: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.event_log.insert(0, f"{ts}  {tag}  {msg}")
    if len(st.session_state.event_log) > 12:
        st.session_state.event_log.pop()


DETAIL_TEXTS = {
    "Mona_Lisa":    "Painted by Leonardo da Vinci, the Mona Lisa is famed for her enigmatic smile and pioneering use of sfumato technique.",
    "Starry_Night": "Vincent van Gogh's Starry Night depicts a swirling night sky over a village, painted from his room at Saint-Paul-de-Mausole asylum.",
    "The_Scream":   "Edvard Munch's The Scream represents the anxiety of modern life — a figure with an agonized expression against a turbulent orange sky.",
    "Sunflowers":   "Van Gogh's Sunflowers series explores complementary colors and the beauty of ordinary objects in brilliant yellow tones.",
    "Guernica":     "Picasso's Guernica is a powerful anti-war statement, depicting the bombing of the Basque town during the Spanish Civil War.",
}


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
            intent = data["intent"]
            target = data.get("target") or ""
            if intent == "PIN_EXHIBIT" and target:
                if target not in ss.bookmarks:
                    ss.bookmarks.append(target)
            elif intent == "OPEN_DETAIL":
                ss.detail_exhibit = target
                ss.detail_text = DETAIL_TEXTS.get(target,
                    f"Detailed information for '{target}' would appear here.")
            elif intent == "READ_ALOUD":
                ss.detail_text = f"[Audio Guide] {DETAIL_TEXTS.get(target, target)}"
                ss.detail_exhibit = target
            elif intent == "NAVIGATE_NEXT":
                ss.current_page += 1
                ss.detail_text = f"Page {ss.current_page}"
            elif intent == "NAVIGATE_PREV":
                ss.current_page = max(1, ss.current_page - 1)
                ss.detail_text = f"Page {ss.current_page}"

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


def render_header():
    ss = st.session_state
    color_map = {"IDLE": "#64748b", "LOCKED": "#d97706", "NEEDS_TARGET": "#f59e0b",
                 "DISAMBIG": "#f97316", "CONFIRM": "#3b82f6",
                 "COMMAND": "#059669", "CANCELLED": "#dc2626"}
    col1, col2, col3 = st.columns([5, 2, 1])
    with col1:
        st.markdown("## 🏛 MMUI Toolkit — Museum Kiosk Live Demo")
    with col2:
        c = color_map.get(ss.fusion_state, "#64748b")
        st.markdown(f"<div style='margin-top:10px'>"
                    f"<span style='background:{c};color:white;padding:4px 10px;"
                    f"border-radius:8px;font-weight:700;font-size:13px'>"
                    f"● {ss.fusion_state}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='margin-top:10px;font-size:14px;color:#94a3b8'>"
                    f"📌 <b>{len(ss.bookmarks)}</b></div>", unsafe_allow_html=True)
    st.divider()


def render_grid():
    ss = st.session_state
    ref = _get_toolkit()
    gaze = ref.get("gaze")
    cols = st.columns(4)
    for i, ex in enumerate(EXHIBITS):
        locked   = ss.locked_target == ex["id"]
        hovering = ss.dwell_target == ex["id"]
        border = "#d97706" if locked else ("#2563eb" if hovering else "#252a3e")
        bg     = "#231a06" if locked else ("#0d1a2e" if hovering else "#161a26")
        bm_badge = " 📌" if ex["id"] in ss.bookmarks else ""
        label_html = (f"<br><span class='tag-locked'>FOCUSED</span>" if locked else
                      f"<br><span style='color:#2563eb;font-size:10px'>● dwelling...</span>" if hovering else "")
        with cols[i % 4]:
            st.markdown(
                f"<div class='card' style='border-color:{border};background:{bg}'>"
                f"<div class='icon'>{ex['icon']}</div>"
                f"<h4>{ex['name']}{bm_badge}</h4>"
                f"<p>{ex['era']}{label_html}</p>"
                f"</div>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("👁 Focus", key=f"lock_{ex['id']}", use_container_width=True):
                    ss.dwell_target = None; ss.dwell_start = None
                    if gaze: gaze.simulate_lock(ex["id"])
            with b2:
                if st.button("⏱ Dwell", key=f"dwell_{ex['id']}", use_container_width=True):
                    ss.dwell_target = ex["id"]
                    ss.dwell_start  = time.time()
                    ss.dwell_progress = 0.0


def render_controls(speech_adapter):
    ss = st.session_state
    ref = _get_toolkit()
    gaze = ref.get("gaze")

    st.markdown("### Gaze Control")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔓 Release", use_container_width=True):
            if gaze: gaze.simulate_unlock()
            ss.dwell_target = None; ss.dwell_start = None
    with c2:
        if st.button("⚡ Ambiguous", use_container_width=True):
            items = [ex["id"] for ex in EXHIBITS[:2]]
            if gaze: gaze.simulate_ambiguous(items)

    if ss.dwell_target:
        st.markdown(f"<small style='color:#2563eb'>Dwelling: {ss.dwell_target}</small>",
                    unsafe_allow_html=True)
        st.progress(ss.dwell_progress)

    st.divider()
    st.markdown("### 🎙 Speech Input")
    txt = st.text_input("Type command:", key="speech_input",
                        placeholder="e.g. tell me about this")
    ca, cb = st.columns(2)
    with ca:
        if st.button("▶ Process", use_container_width=True):
            if txt and speech_adapter:
                speech_adapter.set_dialog_active(
                    ss.disambig_active or ss.confirm_active)
                speech_adapter.process_text(txt)
    with cb:
        if st.button("🔄 Dialog ON", use_container_width=True):
            if speech_adapter: speech_adapter.set_dialog_active(True)

    st.markdown("<small style='color:#64748b'>Quick commands:</small>",
                unsafe_allow_html=True)
    qcols = st.columns(3)
    quick = [("Read Aloud","tell me about this"),("Summarize","summarize"),
             ("Zoom In","zoom in"),("Next","next"),("Back","back"),
             ("Bookmark","bookmark this"),("Yes","yes"),("No","no"),("Left","left")]
    for i,(label,cmd) in enumerate(quick):
        with qcols[i % 3]:
            if st.button(label, key=f"qcmd_{i}", use_container_width=True):
                if speech_adapter:
                    speech_adapter.set_dialog_active(
                        ss.disambig_active or ss.confirm_active)
                    speech_adapter.process_text(cmd)

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

    if ss.prompt_msg and not ss.disambig_active and not ss.confirm_active:
        st.divider()
        st.warning(ss.prompt_msg)

    st.divider()
    st.markdown("### Status")
    st.markdown(f"**Focused:** `{ss.locked_target or 'none'}`")
    st.markdown(f"**Page:** `{ss.current_page}`")
    st.markdown(f"**Intent:** `{ss.last_intent or '-'}`")
    if ss.last_conf:
        st.progress(ss.last_conf, text=f"Confidence: {ss.last_conf:.0%}")
    if ss.last_cmd:
        st.markdown(f"<div class='tag-cmd'>CMD: {ss.last_cmd}</div>",
                    unsafe_allow_html=True)

    # Detail panel
    if ss.detail_text:
        st.divider()
        st.markdown("### Detail Panel")
        st.markdown(f"<div class='detail-box'><b>{ss.detail_exhibit}</b><br>"
                    f"{ss.detail_text}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Bookmarks")
    if ss.bookmarks:
        for bm in ss.bookmarks[-5:]:
            name = next((e["name"] for e in EXHIBITS if e["id"] == bm), bm)
            st.markdown(f"📌 `{name}`")
    else:
        st.markdown("<small style='color:#64748b'>None yet</small>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### Event Log")
    for ev in ss.event_log[:10]:
        st.markdown(f"<div class='ev-row'>{ev}</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="MMUI Kiosk Live Demo",
        page_icon="🏛",
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
