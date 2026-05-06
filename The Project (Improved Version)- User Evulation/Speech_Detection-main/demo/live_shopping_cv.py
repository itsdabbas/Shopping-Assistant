# -*- coding: utf-8 -*-
"""
MMUI Toolkit – Shopping Demo  (REAL Camera + Real Mic)
=======================================================
Run from repo root:
    python demo/live_shopping_cv.py

Keys (OpenCV window):
    M  – toggle PTT recording
    U  – start / restart gaze calibration
    SPACE – capture calibration point (during calibration mode)
    Q  – quit

Dependencies:
    pip install opencv-python mediapipe sounddevice vosk numpy
    (or openai-whisper instead of vosk)

Gaze pipeline:
    MediaPipe FaceMesh (refine_landmarks=True) → iris landmarks 468 (left)
    + 473 (right) averaged → optional 5-point affine calibration →
    hit-test against item bounding boxes → DwellTracker (1.2 s) →
    LOCK / AMBIGUOUS / UNLOCK events emitted on EventBus.
"""
from __future__ import annotations

import sys
import time
import threading
import math
import logging
import queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# ── repo path ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.event_bus import (
    EventBus, GazeEvent, GazeEventType, SpeechEvent, SpeechEventType
)
from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.calibration import (
    GazeCalibrator5, CalibSession5, eye_relative_gaze_from_landmarks,
    GazeSmoother, GazeFeatureSmoother,
)
from apps.shopping.intents import SHOPPING_INTENTS, WHISPER_PROMPT
from apps.shopping.actions import SHOPPING_ACTION_HANDLERS
from gazeshop.toolkit.events import MultimodalCommandEvent

logging.basicConfig(
    level=logging.INFO,          # INFO shows PTT/ASR log lines for debugging
    format="%(levelname)s %(name)s: %(message)s"
)

# ══════════════════════════════════════════════════════════════════════
# ITEMS  (laid out in a 4×3 grid; bbox set at runtime from canvas size)
# ══════════════════════════════════════════════════════════════════════
ITEMS = [
    {"id": "product_1",  "name": "Organic Coffee",   "price": "$12.99", "icon": "COFFEE"},
    {"id": "product_2",  "name": "Green Tea",         "price": "$8.49",  "icon": "TEA"},
    {"id": "product_3",  "name": "Dark Chocolate",    "price": "$5.99",  "icon": "CHOC"},
    {"id": "product_4",  "name": "Almond Milk",       "price": "$3.99",  "icon": "MILK"},
    {"id": "product_5",  "name": "Granola Bar",        "price": "$2.49",  "icon": "GRA"},
    {"id": "product_6",  "name": "Sparkling Water",   "price": "$1.99",  "icon": "H2O"},
    {"id": "product_7",  "name": "Blueberry Muffin",  "price": "$3.49",  "icon": "MUF"},
    {"id": "product_8",  "name": "Avocado Toast",     "price": "$7.99",  "icon": "AVO"},
    {"id": "product_9",  "name": "Greek Yogurt",      "price": "$4.29",  "icon": "YOG"},
    {"id": "product_10", "name": "Orange Juice",      "price": "$4.99",  "icon": "OJ"},
    {"id": "product_11", "name": "Protein Bar",       "price": "$3.99",  "icon": "PRO"},
    {"id": "product_12", "name": "Chia Pudding",      "price": "$5.49",  "icon": "CHIA"},
]

COLS, ROWS = 4, 3
DWELL_S    = 1.2    # seconds to lock
AMBIG_PX   = 30     # pixels from boundary → AMBIGUOUS
LOCK_TTL_S = 5.0

# ── Layout constants ──────────────────────────────────────────────────
# FOOTER_H reserves screen space at the bottom for gaze-debug labels,
# the status panel, event log, and key hints.  Cards must end above this
# region so click buttons never overlap status text.
FOOTER_H = 165   # pixels reserved at the bottom of WIN_H

# ══════════════════════════════════════════════════════════════════════
# DwellTracker
# ══════════════════════════════════════════════════════════════════════
class DwellTracker:
    def __init__(self, dwell_s: float = DWELL_S) -> None:
        self.dwell_s = dwell_s
        self._target: str | None = None
        self._start: float = 0.0

    def update(self, target_id: str | None) -> tuple[str | None, float]:
        """Returns (locked_id_or_None, progress 0-1)."""
        if target_id != self._target:
            self._target = target_id
            self._start  = time.time()
        if self._target is None:
            return None, 0.0
        progress = min((time.time() - self._start) / self.dwell_s, 1.0)
        if progress >= 1.0:
            locked = self._target
            self._target = None
            self._start  = 0.0
            return locked, 1.0
        return None, progress

# ══════════════════════════════════════════════════════════════════════
# RealGazeAdapter  – MediaPipe FaceMesh → GazeEvent
# ══════════════════════════════════════════════════════════════════════
class RealGazeAdapter(ModalityAdapter):
    """Reads webcam, detects face, maps nose/face-center to items.

    Gaze backend priority:
      1. MediaPipe FaceLandmarker Tasks API  (mediapipe >= 0.10)
      2. MediaPipe FaceMesh solutions API    (mediapipe <  0.10)
      3. OpenCV Haar cascade                 (always available)
    """

    def __init__(self, event_bus: EventBus, cam_index: int = 0) -> None:
        super().__init__(event_bus)
        self.cam_index   = cam_index
        self._thread: threading.Thread | None = None
        # shared state (written by bg thread, read by main)
        self.gaze_pt: tuple[int, int] | None = None   # screen coords
        self.frame: np.ndarray | None = None
        self.face_detected = False
        # bboxes set by main before start()
        self.item_bboxes: list[dict] = []   # [{"id":…,"x1","y1","x2","y2"}]
        self._dwell = DwellTracker(DWELL_S)
        self._current_state = "idle"
        self._locked_id: str | None = None
        self._cam: Any = None
        # gaze backend slots
        self._face_mesh: Any = None          # solutions FaceMesh
        self._mp_landmarker: Any = None      # Tasks FaceLandmarker
        self._haar: Any = None               # OpenCV cascade
        self._gaze_mode: str = "none"        # "tasks"|"solutions"|"haar"|"none"
        # gaze / calibration state (written by bg thread, read by main thread)
        # raw_gaze_feature: eye-relative iris position (NOT absolute image coords)
        self.raw_gaze_feature: tuple[float, float] | None = None
        self.calibrating: bool = False
        self._calibrator: GazeCalibrator5 = GazeCalibrator5(1280, 720)
        self._feat_smoother: GazeFeatureSmoother = GazeFeatureSmoother()
        self._smoother: GazeSmoother = GazeSmoother()
        self._last_valid_gaze_pt: tuple[int, int] | None = None
        self._last_gaze_valid_ts: float = 0.0

    def _init_gaze_backend(self) -> None:
        """Try each backend in order; set self._gaze_mode accordingly."""
        # ── 1. MediaPipe Tasks API (mediapipe >= 0.10) ───────────────
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as _mp_python
            from mediapipe.tasks.python import vision as _mp_vision
            # Tasks API needs a .task model file; skip if unavailable
            import urllib.request, os, tempfile
            model_url = ("https://storage.googleapis.com/mediapipe-models/"
                         "face_landmarker/face_landmarker/float16/1/"
                         "face_landmarker.task")
            model_path = Path(tempfile.gettempdir()) / "face_landmarker.task"
            if not model_path.exists():
                print("[GazeAdapter] Downloading MediaPipe FaceLandmarker model …")
                urllib.request.urlretrieve(model_url, model_path)
            base_opts = _mp_python.BaseOptions(
                model_asset_path=str(model_path))
            opts = _mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self._mp_landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)
            self._gaze_mode = "tasks"
            print("[GazeAdapter] Using MediaPipe Tasks API (FaceLandmarker).")
            return
        except Exception as e:
            logging.debug("Tasks API unavailable: %s", e)

        # ── 2. MediaPipe solutions API (mediapipe < 0.10) ────────────
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._gaze_mode = "solutions"
            print("[GazeAdapter] Using MediaPipe solutions API (FaceMesh).")
            return
        except Exception as e:
            logging.debug("solutions API unavailable: %s", e)

        # ── 3. OpenCV Haar cascade (always available) ────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        self._gaze_mode = "haar"
        print("[GazeAdapter] Falling back to OpenCV Haar cascade face detection.")

    def start(self) -> None:
        self._running = True
        self._cam = cv2.VideoCapture(self.cam_index)
        if not self._cam.isOpened():
            logging.error("Cannot open camera %d", self.cam_index)
        self._init_gaze_backend()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cam:
            self._cam.release()

    def set_bboxes(self, bboxes: list[dict]) -> None:
        self.item_bboxes = bboxes

    # ── background capture loop ──────────────────────────────────────
    def _loop(self) -> None:
        WIN_W, WIN_H = 1280, 720   # canvas size (must match overlay)
        while self._running:
            ok, raw = self._cam.read()
            if not ok:
                time.sleep(0.03)
                continue

            raw = cv2.flip(raw, 1)
            cam_h, cam_w = raw.shape[:2]
            gpt: tuple[int, int] | None = None

            # ── Tasks API ────────────────────────────────────────────
            if self._gaze_mode == "tasks" and self._mp_landmarker is not None:
                try:
                    import mediapipe as mp
                    mp_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                    )
                    result = self._mp_landmarker.detect(mp_img)
                    if result.face_landmarks:
                        self.face_detected = True
                        lms = result.face_landmarks[0]
                        feat = eye_relative_gaze_from_landmarks(lms)
                        self.raw_gaze_feature = feat
                        smooth_feat = self._feat_smoother.update(feat)
                        if smooth_feat is not None:
                            if self._calibrator.is_calibrated and not self.calibrating:
                                raw_gpt = self._calibrator.predict(smooth_feat)
                                gpt = self._smoother.update(raw_gpt)
                            else:
                                # Rough uncalibrated visualisation (eye-centred)
                                gpt = (
                                    max(0, min(WIN_W, int(WIN_W // 2 + (smooth_feat[0] - 0.5) * WIN_W))),
                                    max(0, min(WIN_H, int(WIN_H // 2 + (smooth_feat[1] - 0.5) * WIN_H))),
                                )
                        # feat=None → iris not detected (eyes closed / occluded)
                    else:
                        self.face_detected = False
                        self.raw_gaze_feature = None
                        self._feat_smoother.update(None)
                except Exception as exc:
                    logging.debug("Tasks detect error: %s", exc)
                    self.face_detected = False
                    self.raw_gaze_feature = None

            # ── solutions API ─────────────────────────────────────────
            elif self._gaze_mode == "solutions" and self._face_mesh is not None:
                rgb     = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                results = self._face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    self.face_detected = True
                    lms = results.multi_face_landmarks[0].landmark
                    feat = eye_relative_gaze_from_landmarks(lms)
                    self.raw_gaze_feature = feat
                    smooth_feat = self._feat_smoother.update(feat)
                    if smooth_feat is not None:
                        if self._calibrator.is_calibrated and not self.calibrating:
                            raw_gpt = self._calibrator.predict(smooth_feat)
                            gpt = self._smoother.update(raw_gpt)
                        else:
                            gpt = (
                                max(0, min(WIN_W, int(WIN_W // 2 + (smooth_feat[0] - 0.5) * WIN_W))),
                                max(0, min(WIN_H, int(WIN_H // 2 + (smooth_feat[1] - 0.5) * WIN_H))),
                            )
                else:
                    self.face_detected = False
                    self.raw_gaze_feature = None
                    self._feat_smoother.update(None)

            # ── Haar cascade (no iris data — gaze calibration unavailable) ──
            elif self._gaze_mode == "haar" and self._haar is not None:
                gray  = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                faces = self._haar.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    self.face_detected = True
                    # raw_gaze_feature stays None: no iris landmarks from Haar
                    self.raw_gaze_feature = None
                    # gpt stays None: cannot do eye-relative gaze without iris
                else:
                    self.face_detected = False
                    self.raw_gaze_feature = None
            else:
                self.face_detected = False
                self.raw_gaze_feature = None

            # Keep last valid gaze point briefly when eyes temporarily lost
            if gpt is None and self._last_valid_gaze_pt is not None:
                if time.time() - self._last_gaze_valid_ts < 0.5:
                    gpt = self._last_valid_gaze_pt
            if gpt is not None:
                self._last_valid_gaze_pt = gpt
                self._last_gaze_valid_ts = time.time()

            self.gaze_pt = gpt
            self.frame   = raw

            # ── hit-test (skipped during calibration) ────────────────
            if not self.calibrating:
                self._update_gaze_events(gpt)
            time.sleep(0.03)   # ~30 fps

    def _update_gaze_events(self, gpt: tuple[int, int] | None) -> None:
        if not self.item_bboxes or gpt is None:
            if self._current_state != "idle":
                self._current_state = "idle"
                self._emit_unlock()
            return

        x, y = gpt
        hit_items = []
        near_boundary = False

        for bbox in self.item_bboxes:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit_items.append(bbox["id"])
                # check proximity to edges
                if (x - x1 < AMBIG_PX or x2 - x < AMBIG_PX or
                        y - y1 < AMBIG_PX or y2 - y < AMBIG_PX):
                    near_boundary = True

        if len(hit_items) == 0:
            locked, _ = self._dwell.update(None)
            if self._current_state != "idle":
                self._current_state = "idle"
                self._emit_unlock()
            return

        if len(hit_items) >= 2 or near_boundary:
            # AMBIGUOUS
            if self._current_state != "ambiguous":
                self._current_state = "ambiguous"
                candidates = [{"id": h, "pos": ("left" if i == 0 else "right")}
                              for i, h in enumerate(hit_items[:2])]
                ev = GazeEvent(
                    timestamp=time.time(),
                    type=GazeEventType.AMBIGUOUS,
                    payload={"candidates": candidates},
                    confidence=0.4,
                )
                self.event_bus.emit(ev)
            return

        # single hit → dwell
        locked, progress = self._dwell.update(hit_items[0])
        self.dwell_progress = progress
        self.dwell_target   = hit_items[0]
        if locked:
            self._current_state = "locked"
            self._locked_id = locked
            ev = GazeEvent(
                timestamp=time.time(),
                type=GazeEventType.LOCK,
                payload={"target_id": locked},
                confidence=0.95,
            )
            self.event_bus.emit(ev)

    def _emit_unlock(self) -> None:
        self._locked_id = None
        ev = GazeEvent(
            timestamp=time.time(),
            type=GazeEventType.UNLOCK,
            payload={},
            confidence=0.0,
        )
        self.event_bus.emit(ev)

    @property
    def dwell_progress(self) -> float:
        return getattr(self, "_dwell_progress", 0.0)

    @dwell_progress.setter
    def dwell_progress(self, v: float) -> None:
        self._dwell_progress = v

    @property
    def dwell_target(self) -> str | None:
        return getattr(self, "_dwell_target", None)

    @dwell_target.setter
    def dwell_target(self, v: str | None) -> None:
        self._dwell_target = v

# ══════════════════════════════════════════════════════════════════════
# State  (shared between event callbacks and render loop)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class AppState:
    locked_target: str | None = None
    fusion_state:  str = "IDLE"
    prompt_msg:    str = ""
    last_intent:   str = ""
    last_transcript: str = ""
    last_conf:     float = 0.0
    last_cmd:      str = ""
    mic_active:    bool = False
    cart:          list = field(default_factory=list)
    event_log:     list = field(default_factory=list)
    lock_ts:       float = 0.0

    def log(self, tag: str, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.event_log.insert(0, f"{ts} {tag} {msg}")
        if len(self.event_log) > 12:
            self.event_log.pop()

# ══════════════════════════════════════════════════════════════════════
# Build item bboxes for a given canvas size
# ══════════════════════════════════════════════════════════════════════
def build_bboxes(w: int, h: int,
                 grid_top: int = 80, grid_bot: int = 60) -> list[dict]:
    usable_h = h - grid_top - grid_bot
    cell_w   = w  // COLS
    cell_h   = usable_h // ROWS
    bboxes   = []
    for idx, item in enumerate(ITEMS):
        col = idx % COLS
        row = idx // COLS
        x1  = col * cell_w + 4
        y1  = grid_top + row * cell_h + 4
        x2  = x1 + cell_w - 8
        y2  = y1 + cell_h - 8
        bboxes.append({**item, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return bboxes

# ══════════════════════════════════════════════════════════════════════
# Overlay renderer
# ══════════════════════════════════════════════════════════════════════
CLR = {
    "bg":       (15,  17,  23),
    "card":     (30,  33,  48),
    "locked":   (124, 58, 237),
    "hover":    (245, 158, 11),
    "ambig":    (249, 115, 22),
    "text":     (226, 232, 240),
    "sub":      (148, 163, 184),
    "green":    (5,  150, 105),
    "red":      (220, 38,  38),
    "blue":     (59, 130, 246),
    "mic_on":   (220, 38,  38),
    "mic_off":  (51,  65,  85),
}

def _put(img, text, x, y, scale=0.45, color=(226,232,240),
         thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

def _rect(img, x1, y1, x2, y2, color, thick=-1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

def render(canvas: np.ndarray, state: AppState,
           bboxes: list[dict], gaze_adapter: RealGazeAdapter,
           condition_mode: str = "B") -> None:
    H, W = canvas.shape[:2]
    canvas[:] = CLR["bg"]

    # ── header bar ──────────────────────────────────────────────────
    _rect(canvas, 0, 0, W, 50, (20, 23, 36))
    _put(canvas, "MMUI Toolkit  Shopping Demo", 10, 30,
         scale=0.7, color=CLR["text"], thick=2)

    # fusion state badge
    fs = state.fusion_state
    fc = {"IDLE": CLR["sub"], "LOCKED": CLR["locked"],
          "NEEDS_TARGET": CLR["hover"], "DISAMBIG": CLR["ambig"],
          "CONFIRM": CLR["blue"], "COMMAND": CLR["green"],
          "CANCELLED": CLR["red"]}.get(fs, CLR["sub"])
    _rect(canvas, W - 200, 8, W - 10, 42, fc)
    _put(canvas, fs, W - 195, 32, scale=0.5, color=(255,255,255), thick=2)

    # cart count
    _put(canvas, f"CART: {len(state.cart)}", W - 340, 32,
         scale=0.55, color=CLR["green"], thick=2)

    # mic indicator
    mic_c = CLR["mic_on"] if state.mic_active else CLR["mic_off"]
    cv2.circle(canvas, (W - 420, 25), 12, mic_c, -1)
    _put(canvas, "MIC" if state.mic_active else "[M]",
         W - 408, 30, scale=0.4,
         color=CLR["mic_on"] if state.mic_active else CLR["sub"])

    # ── item grid ────────────────────────────────────────────────────
    # In Condition A gaze highlights are suppressed so the mouse baseline
    # looks clean.  Hover / lock visuals only appear in Condition B.
    cond_b    = (condition_mode == "B")
    dwell_id  = gaze_adapter.dwell_target  if cond_b else None
    dwell_prg = gaze_adapter.dwell_progress if cond_b else 0.0

    for b in bboxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        is_locked   = cond_b and (state.locked_target == b["id"])
        is_hovering = cond_b and (dwell_id == b["id"])

        if is_locked:
            card_c = (42, 31, 74)
            bdr_c  = CLR["locked"]
            thick  = 3
        elif is_hovering:
            card_c = (42, 32, 16)
            bdr_c  = CLR["hover"]
            thick  = 2
        else:
            card_c = CLR["card"]
            bdr_c  = (45, 49, 72)
            thick  = 1

        _rect(canvas, x1, y1, x2, y2, card_c)
        _rect(canvas, x1, y1, x2, y2, bdr_c, thick)

        cx = (x1 + x2) // 2
        # icon
        _put(canvas, b["icon"], cx - 18, y1 + 28,
             scale=0.55, color=(200, 200, 200), thick=1)
        # name
        name = b["name"][:14]
        _put(canvas, name, x1 + 6, y1 + 52,
             scale=0.38, color=CLR["text"])
        # price
        _put(canvas, b["price"], x1 + 6, y1 + 70,
             scale=0.35, color=CLR["sub"])

        # locked badge
        if is_locked:
            _rect(canvas, x1 + 2, y1 + 2, x1 + 52, y1 + 16, CLR["locked"])
            _put(canvas, "LOCKED", x1 + 3, y1 + 13,
                 scale=0.3, color=(255,255,255))

        # dwell progress bar
        if is_hovering and not is_locked:
            bw = x2 - x1 - 4
            pw = int(bw * dwell_prg)
            _rect(canvas, x1 + 2, y2 - 8, x2 - 2, y2 - 2, (40,40,40))
            _rect(canvas, x1 + 2, y2 - 8, x1 + 2 + pw, y2 - 2, CLR["hover"])

    # ── gaze dot + debug labels (Condition B only) ───────────────────
    if cond_b:
        gpt = gaze_adapter.gaze_pt
        if gpt:
            cv2.circle(canvas, gpt, 10, (100, 220, 100), 2)
            cv2.circle(canvas, gpt, 3,  (100, 220, 100), -1)
        mode_str = getattr(gaze_adapter, "_gaze_mode", "?")
        face_str = (f"Face: YES [{mode_str}]" if gaze_adapter.face_detected
                    else f"Face: NO  [{mode_str}] (look at camera)")
        face_c = CLR["green"] if gaze_adapter.face_detected else CLR["red"]

        cal_ready = gaze_adapter._calibrator.is_calibrated
        track_label = ("Tracking: eye-relative iris  |  Feature smoothing: on  |  Smoothed gaze active"
                       if cal_ready else "Tracking: eye-relative iris  |  Feature smoothing: on")
        _put(canvas, track_label, 10, H - 160, scale=0.36, color=CLR["blue"])
        cal_label = "Gaze calibrated" if cal_ready else "Gaze not calibrated  [U] to calibrate"
        _put(canvas, cal_label, 10, H - 144, scale=0.38,
             color=CLR["green"] if cal_ready else CLR["sub"])
        feat = gaze_adapter.raw_gaze_feature
        if feat is not None:
            _put(canvas, f"EyeFeat: {feat[0]:.3f}, {feat[1]:.3f}",
                 10, H - 128, scale=0.35, color=CLR["sub"])
        elif gaze_adapter.face_detected:
            _put(canvas, "No stable eyes detected - open eyes wider",
                 10, H - 128, scale=0.36, color=CLR["red"])
        _put(canvas, face_str, 10, H - 112, scale=0.40, color=face_c)

    # ── status panel (bottom) ────────────────────────────────────────
    py = H - 96
    _rect(canvas, 0, py - 4, W, H, (20, 23, 36))

    _put(canvas, f"Target : {state.locked_target or 'none'}",
         10, py + 14, color=CLR["text"])
    _put(canvas, f"Intent : {state.last_intent or '-'}",
         10, py + 32, color=CLR["text"])
    _put(canvas, f"Transcript: {state.last_transcript[:60] or '-'}",
         10, py + 50, color=CLR["sub"])
    conf_txt = f"Conf: {state.last_conf:.0%}" if state.last_conf else ""
    _put(canvas, conf_txt, 10, py + 68, color=CLR["green"])

    if state.prompt_msg:
        _put(canvas, f">> {state.prompt_msg[:80]}",
             10, py + 86, color=CLR["hover"], scale=0.42)

    # last cmd
    if state.last_cmd:
        _put(canvas, f"CMD: {state.last_cmd[:70]}",
             W // 2, py + 14, color=CLR["green"], scale=0.45, thick=2)

    # event log (right side)
    ex = W - 400
    _put(canvas, "Event Log:", ex, py + 4, scale=0.38, color=CLR["sub"])
    for i, ev in enumerate(state.event_log[:5]):
        _put(canvas, ev[:55], ex, py + 18 + i * 16,
             scale=0.33, color=CLR["sub"])

    # keys hint
    _put(canvas, "[A] Mouse-only  [B] Gaze+Speech  [M] PTT  [U] Calib  [Q] Quit",
         W // 2 - 200, H - 8, scale=0.38, color=CLR["sub"])

# ══════════════════════════════════════════════════════════════════════
# Condition A — Mouse / Click baseline
# ══════════════════════════════════════════════════════════════════════

# (label, intent_name, button_bg_color_bgr)
_SHOP_BTNS: list[tuple[str, str, tuple]] = [
    ("Add",  "ADD_TO_CART",  (20, 100,  45)),
    ("Info", "SHOW_DETAILS", (20,  55, 140)),
    ("Sim",  "FIND_SIMILAR", (85,  40, 110)),
    ("Cmp",  "COMPARE",      (30,  75, 140)),
]
_BTN_H = 16   # button height in pixels
_PAD_B = 10   # gap between button row and card bottom edge


def build_click_regions(bboxes: list[dict]) -> list[dict]:
    """Build per-card clickable button rects for Condition A (mouse baseline).

    Returns a flat list; each entry holds:
        rect    : (bx1, by1, bx2, by2) in canvas pixels
        card_id : product id string
        action  : canonical intent name
        label   : short display text
        color   : BGR button background colour
    """
    regions: list[dict] = []
    n_btns = len(_SHOP_BTNS)
    for b in bboxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        btn_w = (x2 - x1) // n_btns
        by1   = y2 - _PAD_B - _BTN_H
        by2   = y2 - _PAD_B
        for i, (label, action, color) in enumerate(_SHOP_BTNS):
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
    """Overlay Condition A action buttons on already-drawn cards."""
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
        badge_c = (30, 150, 55)          # green  – mouse-only mode
        badge_t = "MODE: A  Mouse-only"
    else:
        badge_c = (170, 80, 20)          # amber  – gaze+speech mode
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
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    WIN_W, WIN_H = 1280, 720
    WIN_NAME     = "MMUI Shopping – Real Input"

    # ── Config ───────────────────────────────────────────────────────
    cfg = Config(
        ASR_ENGINE            = "vosk",
        VOSK_MODEL_PATH       = str(ROOT / "models" / "vosk-model-small-en-us-0.15"),
        PTT_KEY               = "m",
        PTT_MODE              = "hold",
        FUSION_TIME_WINDOW_S  = 2.0,
        LOCK_TTL_S            = LOCK_TTL_S,
        MAX_REPAIR_ATTEMPTS   = 2,
        ENABLE_TELEMETRY      = False,
        WHISPER_INITIAL_PROMPT= WHISPER_PROMPT,
    )
    # Disable pynput listener: the OpenCV window owns keyboard focus on Windows,
    # so we drive PTT entirely from cv2.waitKey() below.
    cfg.PTT_KEY = ""   # empty string → _start_key_listener does nothing

    # ── Toolkit ──────────────────────────────────────────────────────
    tk     = MMUIToolkit(cfg)
    parser = IntentParser(cfg, custom_patterns=SHOPPING_INTENTS)
    gaze   = RealGazeAdapter(event_bus=tk.bus, cam_index=0)
    speech = SpeechAdapter(event_bus=tk.bus, config=cfg, intent_parser=parser)

    tk.register_adapter(gaze).register_adapter(speech)
    for intent, handler in SHOPPING_ACTION_HANDLERS.items():
        tk.register_action(intent, handler)

    # ── App state ────────────────────────────────────────────────────
    state = AppState()

    def _on_locked(e):
        state.locked_target = e.target_id
        state.fusion_state  = "LOCKED"
        state.lock_ts       = e.timestamp
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
        if e.intent == "ADD_TO_CART" and e.target_id:
            state.cart.append(e.target_id)

    def _on_speech(e):
        state.mic_active = e.type == SpeechEventType.LISTENING
        if e.type == SpeechEventType.INTENT:
            state.last_intent     = e.payload.get("intent", "")
            state.last_transcript = e.transcript
            state.last_conf       = e.confidence
            state.log("[SPEECH]",
                      f"'{e.transcript[:30]}' → {state.last_intent} "
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

    # ── Window setup ─────────────────────────────────────────────────
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    bboxes = build_bboxes(WIN_W, WIN_H, grid_bot=FOOTER_H)
    gaze.set_bboxes(bboxes)
    click_regions = build_click_regions(bboxes)   # Condition A button rects

    # ── Start adapters ───────────────────────────────────────────────
    tk.start()
    print("[Shopping CV] Started.  Hold M to speak, Q to quit.")

    # ── Condition A: mouse callback ──────────────────────────────────
    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        """Left-click on a card button → emit MultimodalCommandEvent on the
        shared bus.  _dispatch_action and _on_command handle the rest
        (action execution + AppState update) automatically.
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
    calib_session: CalibSession5 | None = None
    calib_complete_ts: float = 0.0
    # ── Condition mode ───────────────────────────────────────────────
    # "B" = gaze+speech (default, preserves original behaviour)
    # "A" = mouse-only baseline (hides gaze UI, pauses hit-testing)
    condition_mode: str = "B"

    try:
        while True:
            render(canvas, state, bboxes, gaze, condition_mode)

            # Flash calibration result banner for 2.5 s
            if calib_complete_ts and time.time() - calib_complete_ts < 2.5:
                banner  = calib_session.status_msg if calib_session else "Calibration complete!"
                is_poor = "poor" in banner.lower() or "failed" in banner.lower()
                b_color = (60, 80, 220) if is_poor else (80, 220, 120)
                _put(canvas, banner, WIN_W // 2 - 200, WIN_H // 2,
                     scale=0.75, color=b_color, thick=2)

            # Per-frame calibration tick (handles stable-sample collection window)
            if calib_session and calib_session.active:
                all_done = calib_session.tick(gaze.raw_gaze_feature)
                if all_done:
                    gaze.calibrating = False
                    gaze._feat_smoother.reset()
                    gaze._smoother.reset()
                    calib_complete_ts = time.time()
                    status = calib_session.status_msg
                    print(f"[Calib] {status}")

            # Draw calibration overlay on top of the normal canvas
            if calib_session and calib_session.active:
                calib_session.draw_overlay(canvas, gaze.raw_gaze_feature)
            elif calib_session and not calib_session.active and calib_complete_ts:
                # Show final calibration status message briefly after session ends
                pass   # calib_complete_ts banner drawn above handles this

            # Condition A — draw clickable buttons (hidden during calibration
            # so they don't appear on the dimmed calibration overlay)
            if not (calib_session and calib_session.active):
                draw_click_buttons(canvas, click_regions)

            # Always-visible condition legend (shows active mode)
            draw_condition_bar(canvas, condition_mode)

            cv2.imshow(WIN_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            # A/a – switch to Condition A (mouse-only baseline)
            if key in (ord('a'), ord('A')):
                condition_mode = "A"
                # pause gaze hit-testing by clearing bboxes
                gaze.item_bboxes    = []
                gaze._dwell_target   = None
                gaze._dwell_progress = 0.0
                state.locked_target  = None
                state.fusion_state   = "IDLE"
                state.prompt_msg     = ""
                print("[Mode] Condition A – mouse-only.  Gaze UI hidden.")

            # B/b – switch to Condition B (gaze + speech)
            if key in (ord('b'), ord('B')):
                condition_mode = "B"
                gaze.item_bboxes = bboxes   # restore hit-testing
                print("[Mode] Condition B – gaze + speech active.")

            # U/u – gaze calibration (Condition B only)
            if key in (ord('u'), ord('U')):
                if condition_mode == "A":
                    print("[Calib] Press B first to switch to Condition B "
                          "before calibrating gaze.")
                else:
                    gaze.calibrating = True
                    gaze._feat_smoother.reset()
                    gaze._smoother.reset()
                    calib_session = CalibSession5(gaze._calibrator)
                    calib_session.start()
                    print("[Calib] Calibration started – look at each target and press SPACE.")

            # SPACE – begin stable-sample collection for current calibration target
            if key == ord(' ') and calib_session and calib_session.active:
                started = calib_session.begin_capture()
                if not started and calib_session.is_collecting:
                    pass   # already collecting; silently ignore extra SPACE presses

            # M – PTT (disabled while calibrating or in Condition A)
            if key == ord('m') and (not calib_session or not calib_session.active):
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
        print("[Shopping CV] Stopped.")


if __name__ == "__main__":
    main()
