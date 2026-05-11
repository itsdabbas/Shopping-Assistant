# -*- coding: utf-8 -*-
"""
MMUI Toolkit – Museum Kiosk Demo  (REAL Camera + Real Mic)
===========================================================
Run from repo root:
    python demo/live_kiosk_cv.py

Keys (OpenCV window):
    M  – toggle PTT recording
    U  – start / restart gaze calibration
    SPACE – capture calibration point (during calibration mode)
    Q  – quit

Gaze pipeline:
    MediaPipe FaceMesh (refine_landmarks=True) → iris landmarks 468 (left)
    + 473 (right) averaged → optional 5-point affine calibration →
    hit-test against exhibit bounding boxes → DwellTracker (1.2 s) →
    LOCK / AMBIGUOUS / UNLOCK events on EventBus.
"""
from __future__ import annotations

import sys
import time
import threading
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── repo path ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = ROOT / "fusion"
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.adapters.real_gaze_adapter import RealGazeAdapter
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.event_bus import SpeechEvent, SpeechEventType
from apps.kiosk.intents import KIOSK_INTENTS, KIOSK_VOSK_VOCAB, WHISPER_PROMPT
from apps.kiosk.actions import KIOSK_ACTION_HANDLERS
from gazeshop.toolkit.events import MultimodalCommandEvent

logging.basicConfig(
    level=logging.INFO,          # INFO shows PTT/ASR log lines for debugging
    format="%(levelname)s %(name)s: %(message)s"
)

# ══════════════════════════════════════════════════════════════════════
# EXHIBITS  (3×2 large-card layout)
# ══════════════════════════════════════════════════════════════════════
EXHIBITS = [
    {"id": "exhibit_1", "name": "Egyptian Mummy",      "era": "1350 BC",
     "cat": "EGYPT",  "desc": "Wrapped in linen, gilded mask."},
    {"id": "exhibit_2", "name": "Roman Gladius",        "era": "100 AD",
     "cat": "ROME",   "desc": "Iron short sword, legionnaire issue."},
    {"id": "exhibit_3", "name": "Greek Amphora",        "era": "500 BC",
     "cat": "GREECE", "desc": "Red-figure pottery, wine vessel."},
    {"id": "exhibit_4", "name": "Viking Helmet",        "era": "900 AD",
     "cat": "VIKING", "desc": "Iron spectacle helmet, Gjermundbu."},
    {"id": "exhibit_5", "name": "Aztec Calendar Stone", "era": "1427 AD",
     "cat": "AZTEC",  "desc": "Solar disk, 365-day cycle carved."},
    {"id": "exhibit_6", "name": "Ming Dynasty Vase",    "era": "1403 AD",
     "cat": "CHINA",  "desc": "Blue-and-white porcelain ewer."},
]

COLS, ROWS = 3, 2
DWELL_S    = 1.2
AMBIG_PX   = 35
LOCK_TTL_S = 5.0

# ── Layout constants ──────────────────────────────────────────────────
# FOOTER_H reserves screen space at the bottom for gaze-debug labels,
# the status bar, event log, and key hints.  Cards must end above this
# region so click buttons never overlap status text.
FOOTER_H = 165   # pixels reserved at the bottom of WIN_H


# ══════════════════════════════════════════════════════════════════════
# App State
# ══════════════════════════════════════════════════════════════════════
@dataclass
class AppState:
    locked_target:   str | None = None
    fusion_state:    str        = "IDLE"
    prompt_msg:      str        = ""
    last_intent:     str        = ""
    last_transcript: str        = ""
    last_conf:       float      = 0.0
    last_cmd:        str        = ""
    mic_active:      bool       = False
    bookmarks:       list       = field(default_factory=list)
    current_page:    int        = 1
    event_log:       list       = field(default_factory=list)

    def log(self, tag: str, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.event_log.insert(0, f"{ts} {tag} {msg}")
        if len(self.event_log) > 12:
            self.event_log.pop()

# ══════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════
def build_bboxes(w: int, h: int,
                 grid_top: int = 80,
                 grid_bot: int = 90) -> list[dict]:
    usable_h = h - grid_top - grid_bot
    cell_w   = w // COLS
    cell_h   = usable_h // ROWS
    bboxes   = []
    for idx, ex in enumerate(EXHIBITS):
        col = idx % COLS
        row = idx // COLS
        x1  = col * cell_w + 6
        y1  = grid_top + row * cell_h + 6
        x2  = x1 + cell_w - 12
        y2  = y1 + cell_h - 12
        bboxes.append({**ex, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return bboxes

# ══════════════════════════════════════════════════════════════════════
# Palette & renderer
# ══════════════════════════════════════════════════════════════════════
CLR = {
    "bg":      (12, 14, 20),
    "card":    (22, 26, 40),
    "locked":  (99, 102, 241),
    "hover":   (16, 185, 129),
    "ambig":   (245, 158,  11),
    "text":    (226, 232, 240),
    "sub":     (100, 116, 139),
    "gold":    (251, 191,  36),
    "green":   (16,  185, 129),
    "red":     (239,  68,  68),
    "blue":    (59,  130, 246),
    "hdr_bg":  (18,  22,  36),
}

def _put(img, text, x, y, scale=0.45, color=(226, 232, 240),
         thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

def _rect(img, x1, y1, x2, y2, color, thick=-1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

def render(canvas: np.ndarray, state: AppState,
           bboxes: list[dict], gaze: RealGazeAdapter,
           condition_mode: str = "B") -> None:
    H, W = canvas.shape[:2]
    canvas[:] = CLR["bg"]

    # ── header ──────────────────────────────────────────────────────
    _rect(canvas, 0, 0, W, 52, CLR["hdr_bg"])
    _put(canvas, "MMUI Toolkit  Museum Kiosk", 12, 34,
         scale=0.75, color=CLR["text"], thick=2)

    # page indicator
    _put(canvas, f"PAGE {state.current_page}",
         W // 2 - 35, 34, scale=0.55, color=CLR["gold"], thick=2)

    # fusion badge
    fs = state.fusion_state
    fc = {"IDLE": CLR["sub"], "LOCKED": CLR["locked"],
          "NEEDS_TARGET": CLR["ambig"], "DISAMBIG": CLR["ambig"],
          "CONFIRM": CLR["blue"], "COMMAND": CLR["green"],
          "CANCELLED": CLR["red"]}.get(fs, CLR["sub"])
    _rect(canvas, W - 195, 8, W - 5, 44, fc)
    _put(canvas, fs, W - 190, 34, scale=0.5,
         color=(255, 255, 255), thick=2)

    # bookmark count
    _put(canvas, f"BMK: {len(state.bookmarks)}",
         W - 330, 34, scale=0.5, color=CLR["gold"])

    # mic indicator
    mic_c = CLR["red"] if state.mic_active else CLR["sub"]
    cv2.circle(canvas, (W - 390, 26), 12, mic_c, -1)
    _put(canvas, "MIC" if state.mic_active else "[M]",
         W - 378, 31, scale=0.38,
         color=CLR["red"] if state.mic_active else CLR["sub"])

    # ── exhibit cards ────────────────────────────────────────────────
    # Gaze highlights (hover / lock) only shown in Condition B.
    cond_b    = (condition_mode == "B")
    dwell_id  = gaze.dwell_target   if cond_b else None
    dwell_prg = gaze.dwell_progress if cond_b else 0.0
    stable_id = gaze.stable_target  if cond_b else None

    for b in bboxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        is_locked   = cond_b and (state.locked_target == b["id"])
        is_hovering = cond_b and (dwell_id == b["id"])

        if is_locked:
            bg_c  = (25, 26, 65)
            bdr_c = CLR["locked"]
            thick = 3
        elif is_hovering:
            bg_c  = (15, 42, 35)
            bdr_c = CLR["hover"]
            thick = 2
        else:
            bg_c  = CLR["card"]
            bdr_c = (40, 48, 72)
            thick = 1

        _rect(canvas, x1, y1, x2, y2, bg_c)
        _rect(canvas, x1, y1, x2, y2, bdr_c, thick)

        # category label band
        band_h = 22
        _rect(canvas, x1, y1, x2, y1 + band_h, bdr_c)
        _put(canvas, b["cat"], x1 + 6, y1 + 15,
             scale=0.45, color=(255, 255, 255), thick=1)

        # exhibit name
        _put(canvas, b["name"], x1 + 6, y1 + band_h + 22,
             scale=0.5, color=CLR["text"], thick=1)
        # era
        _put(canvas, b["era"], x1 + 6, y1 + band_h + 42,
             scale=0.38, color=CLR["gold"])
        # description (word-wrap simple)
        desc = b["desc"]
        _put(canvas, desc[:38], x1 + 6, y1 + band_h + 62,
             scale=0.36, color=CLR["sub"])
        if len(desc) > 38:
            _put(canvas, desc[38:70], x1 + 6, y1 + band_h + 78,
                 scale=0.36, color=CLR["sub"])

        # locked badge
        if is_locked:
            _rect(canvas, x1 + 2, y2 - 20, x1 + 58, y2 - 4, CLR["locked"])
            _put(canvas, "LOCKED", x1 + 4, y2 - 8,
                 scale=0.32, color=(255, 255, 255))

        # dwell progress bar
        if is_hovering and not is_locked:
            bw = x2 - x1 - 4
            pw = int(bw * dwell_prg)
            _rect(canvas, x1 + 2, y2 - 10, x2 - 2, y2 - 2, (30, 30, 30))
            _rect(canvas, x1 + 2, y2 - 10, x1 + 2 + pw, y2 - 2, CLR["hover"])

        # Snap indicator: small crosshair at exhibit centre when this is the
        # stable snap target (even before dwell fills).
        if cond_b and stable_id == b["id"] and not is_locked:
            scx, scy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(canvas, (scx, scy), (180, 240, 200),
                           cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)

    # ── gaze crosshair + debug labels (Condition B only) ────────────
    if cond_b:
        gpt = gaze.gaze_pt
        if gpt:
            cv2.drawMarker(canvas, gpt, (80, 220, 120),
                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            cv2.circle(canvas, gpt, 8, (80, 220, 120), 1, cv2.LINE_AA)

        face_str = (f"Face: OK [{gaze._gaze_mode}]" if gaze.face_detected
                    else f"Face: NOT DETECTED [{gaze._gaze_mode}]")
        face_c   = CLR["green"] if gaze.face_detected else CLR["red"]

        cal_ready = gaze.is_calibrated()
        track_label = ("Tracking: eye-relative iris  |  Feature smoothing: on  |  Smoothed gaze active"
                       if cal_ready else "Tracking: eye-relative iris  |  Feature smoothing: on")
        _put(canvas, track_label, 12, H - 136, scale=0.36, color=CLR["blue"])
        cal_label = "Gaze calibrated" if cal_ready else "Gaze not calibrated  [U] to calibrate"
        _put(canvas, cal_label, 12, H - 120, scale=0.38,
             color=CLR["green"] if cal_ready else CLR["sub"])
        feat = gaze.raw_gaze_feature
        if feat is not None:
            _put(canvas, f"EyeFeat: {feat[0]:.3f}, {feat[1]:.3f}",
                 12, H - 104, scale=0.35, color=CLR["sub"])
        elif gaze.face_detected:
            _put(canvas, "No stable eyes detected - open eyes wider",
                 12, H - 104, scale=0.36, color=CLR["red"])
        _put(canvas, face_str, 12, H - 88, scale=0.40, color=face_c)

    # ── status bar ──────────────────────────────────────────────────
    sb = H - 80
    _rect(canvas, 0, sb, W, H, (16, 20, 32))

    col2 = W // 2

    _put(canvas, f"Target : {state.locked_target or 'none'}",
         10, sb + 16, color=CLR["text"])
    _put(canvas, f"Intent : {state.last_intent or '-'}",
         10, sb + 34, color=CLR["text"])
    _put(canvas, f"Speech : {state.last_transcript[:55] or '-'}",
         10, sb + 52, color=CLR["sub"])
    conf_t = f"Conf: {state.last_conf:.0%}" if state.last_conf else ""
    _put(canvas, conf_t, 10, sb + 70, color=CLR["green"])

    if state.prompt_msg:
        _put(canvas, f">> {state.prompt_msg[:80]}",
             col2, sb + 16, scale=0.42, color=CLR["ambig"])

    if state.last_cmd:
        _put(canvas, f"CMD: {state.last_cmd[:65]}",
             col2, sb + 34, scale=0.45, color=CLR["green"], thick=2)

    # event log (right panel)
    ex_x = W - 420
    _put(canvas, "Log:", ex_x, sb + 8, scale=0.38, color=CLR["sub"])
    for i, ev in enumerate(state.event_log[:5]):
        _put(canvas, ev[:58], ex_x, sb + 22 + i * 14,
             scale=0.32, color=CLR["sub"])

    _put(canvas, "[A] Mouse  [B] Gaze+Speech  [M] PTT  [U] Calib  [D] Debug  [Q] Quit",
         col2 - 210, H - 6, scale=0.38, color=CLR["sub"])

# ══════════════════════════════════════════════════════════════════════
# Condition A — Mouse / Click baseline
# ══════════════════════════════════════════════════════════════════════

# (label, intent_name, button_bg_color_bgr)
_KIOSK_BTNS: list[tuple[str, str, tuple]] = [
    ("Info",  "OPEN_DETAIL",   (20,  55, 140)),
    ("Sum",   "SUMMARIZE",     (85,  40, 110)),
    ("Zoom",  "ZOOM_IN",       (20, 110,  70)),
    ("Pin",   "PIN_EXHIBIT",   (30,  80, 140)),
]
_BTN_H = 16   # button height in pixels
_PAD_B = 12   # gap between button row and card bottom edge


def build_click_regions(bboxes: list[dict]) -> list[dict]:
    """Build per-exhibit clickable button rects for Condition A (mouse baseline).

    Each entry holds: rect, card_id, action, label, color.
    """
    regions: list[dict] = []
    n_btns = len(_KIOSK_BTNS)
    for b in bboxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        btn_w = (x2 - x1) // n_btns
        by1   = y2 - _PAD_B - _BTN_H
        by2   = y2 - _PAD_B
        for i, (label, action, color) in enumerate(_KIOSK_BTNS):
            bx1 = x1 + i * btn_w
            bx2 = bx1 + btn_w - 1
            regions.append({
                "rect":    (bx1, by1, bx2, by2),
                "card_id": b["id"],
                "action":  action,
                "label":   label,
                "color":   color,
            })
    return regions


def draw_click_buttons(canvas: np.ndarray, click_regions: list[dict]) -> None:
    """Overlay Condition A action buttons on already-drawn exhibit cards."""
    for r in click_regions:
        bx1, by1, bx2, by2 = r["rect"]
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), r["color"], -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (170, 170, 170), 1)
        (tw, th), _ = cv2.getTextSize(
            r["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1,
        )
        tx = bx1 + max(((bx2 - bx1) - tw) // 2, 0)
        ty = by1 + ((by2 - by1) + th) // 2
        cv2.putText(
            canvas, r["label"], (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (230, 230, 230), 1, cv2.LINE_AA,
        )


def draw_condition_bar(canvas: np.ndarray, condition_mode: str = "B") -> None:
    """Draw active-mode badge + full key-hint in the legend strip below the header."""
    if condition_mode == "A":
        badge_c = (30, 150, 55)
        badge_t = "MODE: A  Mouse-only"
    else:
        badge_c = (170, 80, 20)
        badge_t = "MODE: B  Gaze+Speech"
    cv2.rectangle(canvas, (10, 54), (248, 73), badge_c, -1)
    cv2.putText(canvas, badge_t, (14, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "[A] Mouse-only  [B] Gaze+Speech  |  Cond A: click buttons  "
        "Cond B: look + speak",
        (255, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 80), 1, cv2.LINE_AA,
    )


# ══════════════════════════════════════════════════════════════════════
# Gaze debug overlay  (toggled with D key, Condition B only)
# ══════════════════════════════════════════════════════════════════════

def draw_gaze_debug_overlay(canvas: np.ndarray, gaze: RealGazeAdapter) -> None:
    """Draw a compact gaze diagnostics panel in the top-right corner.

    Shows raw/smoothed eye features, calibration quality, snap/hysteresis state,
    dwell progress, and frame rejection reason.  Toggle with D key (Condition B).
    """
    H, W = canvas.shape[:2]
    PW, PH = 318, 224   # 12 rows × 17 px + 14 header + 6 padding
    px1, py1 = W - PW - 8, 76
    px2, py2 = px1 + PW, py1 + PH

    # Semi-transparent dark panel
    panel = canvas[py1:py2, px1:px2].copy()
    panel[:] = (10, 14, 24)
    cv2.addWeighted(panel, 0.85, canvas[py1:py2, px1:px2], 0.15, 0,
                    canvas[py1:py2, px1:px2])
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (55, 75, 115), 1)

    def _t(text: str, row: int, color=(160, 190, 220)) -> None:
        cv2.putText(canvas, text, (px1 + 6, py1 + 14 + row * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)

    # ── row 0: header ────────────────────────────────────────────────
    _t("GAZE DEBUG  snap=ON  [D] to hide", 0, (80, 200, 100))

    # ── row 1: face / iris ───────────────────────────────────────────
    face_c = (80, 200, 80) if gaze.face_detected else (80, 80, 220)
    iris_ok = gaze.raw_gaze_feature is not None
    _t(f"Face: {'YES' if gaze.face_detected else 'NO '}"
       f"   Iris: {'OK  ' if iris_ok else 'NONE'}", 1, face_c)

    # ── rows 2-3: eye features ───────────────────────────────────────
    rf = gaze.raw_gaze_feature
    _t(f"Raw  feat : {f'{rf[0]:.3f},{rf[1]:.3f}' if rf else 'NONE':>14}", 2)
    sf = gaze.smooth_gaze_feature
    _t(f"Smooth    : {f'{sf[0]:.3f},{sf[1]:.3f}' if sf else 'NONE':>14}", 3)

    # ── row 4: screen gaze point ──────────────────────────────────────
    gp = gaze.gaze_pt
    held_tag = " [HELD]" if gaze.gaze_using_held else ""
    _t(f"Screen: {f'({gp[0]:4d},{gp[1]:4d})' if gp else 'NONE'}{held_tag}", 4,
       (220, 180, 60) if gaze.gaze_using_held else (160, 190, 220))

    # ── row 5: calibration ───────────────────────────────────────────
    cal_ready = gaze.is_calibrated()
    rmse = gaze.get_calibration_rmse()
    if cal_ready:
        rmse_s = f"  RMSE {rmse:.0f} px" if rmse is not None else ""
        _t(f"Calib: OK{rmse_s}", 5, (80, 200, 80))
    else:
        _t("Calib: NOT CALIBRATED  [U]", 5, (80, 80, 220))

    # ── row 6: frame rejection ───────────────────────────────────────
    rej = gaze.frame_rejection_reason
    rej_c = (80, 80, 220) if rej else (100, 120, 150)
    _t(f"Rejected: {rej if rej else 'none'}", 6, rej_c)

    # ── divider ──────────────────────────────────────────────────────
    div_y = py1 + 14 + 7 * 17 - 4
    cv2.line(canvas, (px1 + 4, div_y), (px2 - 4, div_y), (50, 65, 100), 1)

    # ── row 7: nearest snap candidate (single-frame) ──────────────────
    sc = gaze.snap_candidate
    dist = gaze.snap_dist_px
    dist_s = f"{dist:.0f} px" if dist != float('inf') else "inf"
    sc_c = (80, 200, 80) if sc else (100, 120, 150)
    _t(f"Snap cand : {(sc or 'none')[:14]}  {dist_s}", 7, sc_c)

    # ── row 8: candidate confirmation streak ──────────────────────────
    cf = gaze.snap_candidate_frames
    _t(f"Cand streak: {cf} / 4 frames"
       f"{'  CONFIRMED' if cf >= 4 else ''}",
       8, (80, 200, 80) if cf >= 4 else (160, 190, 220))

    # ── row 9: stable (hysteresis-confirmed) target ───────────────────
    st = gaze.stable_target
    sw = gaze.target_switch_count
    st_c = (100, 220, 100) if st else (100, 120, 150)
    _t(f"Stable    : {(st or 'none')[:14]}  sw:{sw}", 9, st_c)

    # ── row 10: dwell progress ────────────────────────────────────────
    dt = gaze.dwell_target
    dp = gaze.dwell_progress
    dp_bar = "█" * int(dp * 10) + "░" * (10 - int(dp * 10))
    _t(f"Dwell: {(dt or 'none')[:12]}  {dp_bar} {dp:.0%}", 10, (150, 155, 210))

    # ── row 11: backend ───────────────────────────────────────────────
    _t(f"Backend: {gaze._gaze_mode}", 11, (100, 120, 150))


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    WIN_W, WIN_H = 1280, 720
    WIN_NAME     = "MMUI Kiosk – Real Input"

    cfg = Config(
        ASR_ENGINE            = "vosk",
        VOSK_MODEL_PATH       = str(ROOT / "speech" / "models" / "vosk-model-small-en-us-0.15"),
        PTT_KEY               = "",       # CV demo drives PTT via cv2.waitKey
        PTT_MODE              = "toggle",
        FUSION_TIME_WINDOW_S  = 2.0,
        LOCK_TTL_S            = LOCK_TTL_S,
        MAX_REPAIR_ATTEMPTS   = 2,
        ENABLE_TELEMETRY      = False,
        WHISPER_INITIAL_PROMPT= WHISPER_PROMPT,
    )

    tk     = MMUIToolkit(cfg)
    parser = IntentParser(cfg, custom_patterns=KIOSK_INTENTS)
    gaze   = RealGazeAdapter(event_bus=tk.bus, cam_index=0,
                             win_w=WIN_W, win_h=WIN_H,
                             dwell_s=DWELL_S, ambig_px=AMBIG_PX)
    speech = SpeechAdapter(event_bus=tk.bus, config=cfg, intent_parser=parser,
                           vosk_vocab=KIOSK_VOSK_VOCAB)

    tk.register_adapter(gaze).register_adapter(speech)
    for intent, handler in KIOSK_ACTION_HANDLERS.items():
        tk.register_action(intent, handler)

    state = AppState()

    def _on_locked(e):
        state.locked_target = e.target_id
        state.fusion_state  = "LOCKED"
        state.prompt_msg    = ""
        state.log("[GAZE]", f"Locked → {e.target_id}")

    def _on_unlocked(e):
        state.locked_target = None
        state.fusion_state  = "IDLE"
        state.log("[GAZE]", "Unlocked")

    def _on_expired(e):
        state.locked_target = None
        state.fusion_state  = "IDLE"
        state.log("[GAZE]", f"Expired {e.target_id}")

    def _on_prompt(e):
        state.fusion_state = "NEEDS_TARGET"
        state.prompt_msg   = e.message
        state.log("[PROMPT]", e.message)

    def _on_disambig(e):
        state.fusion_state = "DISAMBIG"
        state.prompt_msg   = e.message
        state.log("[DISAMBIG]", e.message)

    def _on_confirm(e):
        state.fusion_state = "CONFIRM"
        state.prompt_msg   = e.message
        state.log("[CONFIRM?]", e.message)

    def _on_cancelled(e):
        state.fusion_state = "CANCELLED"
        state.prompt_msg   = e.message
        state.log("[CANCEL]", e.message)

    def _on_command(e):
        state.fusion_state = "COMMAND"
        state.last_cmd     = f"{e.intent} on {e.target_id} ({e.confidence:.0%})"
        state.prompt_msg   = ""
        state.log("[CMD]", state.last_cmd)
        if e.intent == "PIN_EXHIBIT" and e.target_id:
            state.bookmarks.append(e.target_id)
        elif e.intent == "NAVIGATE_NEXT":
            state.current_page += 1
        elif e.intent == "NAVIGATE_PREV":
            state.current_page = max(1, state.current_page - 1)

    def _on_speech(e):
        state.mic_active = (e.type == SpeechEventType.LISTENING)
        if e.type == SpeechEventType.INTENT:
            state.last_intent     = e.payload.get("intent", "")
            state.last_transcript = e.transcript
            state.last_conf       = e.confidence
            state.log("[SPEECH]",
                      f"'{e.transcript[:28]}' → {state.last_intent} "
                      f"({e.confidence:.0%})")
        elif e.type == SpeechEventType.STOPPED:
            state.mic_active = False

    tk.register_feedback("TargetLockedEvent",           _on_locked)
    tk.register_feedback("TargetUnlockedEvent",         _on_unlocked)
    tk.register_feedback("TargetExpiredEvent",          _on_expired)
    tk.register_feedback("PromptEvent",                 _on_prompt)
    tk.register_feedback("DisambiguationPromptEvent",   _on_disambig)
    tk.register_feedback("ConfirmationPromptEvent",     _on_confirm)
    tk.register_feedback("ActionCancelledEvent",        _on_cancelled)
    tk.register_feedback("MultimodalCommandEvent",      _on_command)
    tk.bus.subscribe("SpeechEvent",                     _on_speech)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    bboxes = build_bboxes(WIN_W, WIN_H, grid_bot=FOOTER_H)
    gaze.set_bboxes(bboxes)
    click_regions = build_click_regions(bboxes)   # Condition A button rects

    tk.start()
    print("[Kiosk CV] Started.  Hold M to speak, Q to quit.")

    # ── Condition A: mouse callback ──────────────────────────────────
    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        """Left-click on an exhibit card button → emit MultimodalCommandEvent
        on the shared bus.  _dispatch_action and _on_command handle action
        execution and AppState update automatically.
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for r in click_regions:
            bx1, by1, bx2, by2 = r["rect"]
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                tk.bus.emit(MultimodalCommandEvent(
                    intent=r["action"],
                    target_id=r["card_id"],
                    params={},
                    confidence=1.0,
                ))
                break

    cv2.setMouseCallback(WIN_NAME, on_mouse)

    canvas            = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    ptt_active        = False
    calib_complete_ts: float = 0.0
    show_gaze_debug:  bool  = False
    # ── Condition mode ───────────────────────────────────────────────
    # "B" = gaze+speech (default); "A" = mouse-only baseline
    condition_mode: str = "B"

    try:
        while True:
            render(canvas, state, bboxes, gaze, condition_mode)

            # Flash calibration result banner for 2.5 s
            if calib_complete_ts and time.time() - calib_complete_ts < 2.5:
                banner  = gaze.get_calibration_status() or "Calibration complete!"
                is_poor = "poor" in banner.lower() or "failed" in banner.lower()
                b_color = (60, 80, 220) if is_poor else (80, 220, 120)
                _put(canvas, banner, WIN_W // 2 - 200, WIN_H // 2,
                     scale=0.75, color=b_color, thick=2)

            # Per-frame calibration tick (handles stable-sample collection window)
            all_done = gaze.tick_calibration()
            if all_done:
                calib_complete_ts = time.time()
                print(f"[Calib] {gaze.get_calibration_status()}")

            # Draw calibration overlay on top of the normal canvas
            gaze.draw_calibration_overlay(canvas)

            # Condition A — draw clickable buttons (hidden during calibration)
            if not gaze.is_calibrating():
                draw_click_buttons(canvas, click_regions)

            # Always-visible condition legend (shows active mode)
            draw_condition_bar(canvas, condition_mode)

            # Gaze debug overlay (D key, Condition B only)
            if show_gaze_debug and condition_mode == "B" and not gaze.is_calibrating():
                draw_gaze_debug_overlay(canvas, gaze)

            cv2.imshow(WIN_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            # D/d – toggle gaze debug overlay (Condition B only)
            if key in (ord('d'), ord('D')):
                show_gaze_debug = not show_gaze_debug
                print(f"[Debug] Gaze debug overlay {'ON' if show_gaze_debug else 'OFF'}.")

            # A/a – switch to Condition A (mouse-only baseline)
            if key in (ord('a'), ord('A')):
                condition_mode = "A"
                gaze.item_bboxes = []
                gaze.reset_dwell()
                state.locked_target  = None
                state.fusion_state   = "IDLE"
                state.prompt_msg     = ""
                print("[Mode] Condition A – mouse-only.  Gaze UI hidden.")

            # B/b – switch to Condition B (gaze + speech)
            if key in (ord('b'), ord('B')):
                condition_mode = "B"
                gaze.item_bboxes = bboxes
                print("[Mode] Condition B – gaze + speech active.")

            # U/u – gaze calibration (Condition B only)
            if key in (ord('u'), ord('U')):
                if condition_mode == "A":
                    print("[Calib] Press B first to switch to Condition B "
                          "before calibrating gaze.")
                else:
                    gaze.start_calibration()
                    print("[Calib] Calibration started – look at each target and press SPACE.")

            # SPACE – begin stable-sample collection for current calibration target
            if key == ord(' ') and gaze.is_calibrating():
                gaze.begin_calibration_capture()  # no-op if already collecting

            # M – PTT (disabled while calibrating or in Condition A)
            if key == ord('m') and not gaze.is_calibrating():
                if condition_mode == "A":
                    print("[PTT] Speech disabled in Condition A (mouse-only mode).")
                elif not ptt_active:
                    ptt_active = True
                    print(f"[PTT] TOGGLE ON   t={time.time():.3f}")
                    speech.begin_listening()
                else:
                    ptt_active = False
                    print(f"[PTT] TOGGLE OFF  t={time.time():.3f}")
                    speech.end_listening()

    finally:
        tk.stop()
        cv2.destroyAllWindows()
        print("[Kiosk CV] Stopped.")


if __name__ == "__main__":
    main()
