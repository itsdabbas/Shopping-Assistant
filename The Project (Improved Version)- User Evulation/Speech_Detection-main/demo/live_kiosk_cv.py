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
sys.path.insert(0, str(ROOT))

from gazeshop.toolkit.config import Config
from gazeshop.toolkit.runtime import MMUIToolkit
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.calibration import (
    GazeCalibrator5, CalibSession5, eye_relative_gaze_from_landmarks,
    GazeSmoother, GazeFeatureSmoother,
)
from gazeshop.toolkit.intent_parser import IntentParser
from gazeshop.toolkit.event_bus import (
    EventBus, GazeEvent, GazeEventType, SpeechEvent, SpeechEventType,
)
from apps.kiosk.intents import KIOSK_INTENTS, WHISPER_PROMPT
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
# DwellTracker
# ══════════════════════════════════════════════════════════════════════
class DwellTracker:
    def __init__(self, dwell_s: float = DWELL_S) -> None:
        self.dwell_s = dwell_s
        self._target: str | None = None
        self._start:  float = 0.0

    def update(self, target_id: str | None) -> tuple[str | None, float]:
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
# RealGazeAdapter – MediaPipe FaceMesh → GazeEvent
# ══════════════════════════════════════════════════════════════════════
class RealGazeAdapter(ModalityAdapter):
    """Reads webcam, detects face, maps nose/face-center to exhibits.

    Gaze backend priority:
      1. MediaPipe FaceLandmarker Tasks API  (mediapipe >= 0.10)
      2. MediaPipe FaceMesh solutions API    (mediapipe <  0.10)
      3. OpenCV Haar cascade                 (always available)
    """

    def __init__(self, event_bus: EventBus, cam_index: int = 0) -> None:
        super().__init__(event_bus)
        self.cam_index     = cam_index
        self._thread: threading.Thread | None = None
        self.gaze_pt: tuple[int, int] | None = None
        self.frame: np.ndarray | None        = None
        self.face_detected                   = False
        self.item_bboxes: list[dict]         = []
        self._dwell                          = DwellTracker(DWELL_S)
        self._current_state                  = "idle"
        self._locked_id: str | None          = None
        self._cam: Any                       = None
        self._face_mesh: Any                 = None
        self._mp_landmarker: Any             = None
        self._haar: Any                      = None
        self._gaze_mode: str                 = "none"
        self._dwell_progress: float          = 0.0
        self._dwell_target: str | None       = None
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
        # ── 1. MediaPipe Tasks API (mediapipe >= 0.10) ──────────────
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as _mp_python
            from mediapipe.tasks.python import vision as _mp_vision
            import urllib.request, tempfile
            model_url = ("https://storage.googleapis.com/mediapipe-models/"
                         "face_landmarker/face_landmarker/float16/1/"
                         "face_landmarker.task")
            model_path = Path(tempfile.gettempdir()) / "face_landmarker.task"
            if not model_path.exists():
                print("[GazeAdapter] Downloading MediaPipe FaceLandmarker model …")
                urllib.request.urlretrieve(model_url, model_path)
            base_opts = _mp_python.BaseOptions(model_asset_path=str(model_path))
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

        # ── 2. MediaPipe solutions API (mediapipe < 0.10) ───────────
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

        # ── 3. OpenCV Haar cascade (always available) ──────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        self._gaze_mode = "haar"
        print("[GazeAdapter] Falling back to OpenCV Haar cascade face detection.")

    def start(self) -> None:
        self._running = True
        self._cam     = cv2.VideoCapture(self.cam_index)
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

    @property
    def dwell_progress(self) -> float:
        return self._dwell_progress

    @property
    def dwell_target(self) -> str | None:
        return self._dwell_target

    def _loop(self) -> None:
        WIN_W, WIN_H = 1280, 720
        while self._running:
            ok, raw = self._cam.read()
            if not ok:
                time.sleep(0.03)
                continue
            raw = cv2.flip(raw, 1)
            cam_h, cam_w = raw.shape[:2]
            gpt: tuple[int, int] | None = None

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
                    else:
                        self.face_detected = False
                        self.raw_gaze_feature = None
                        self._feat_smoother.update(None)
                except Exception as exc:
                    logging.debug("Tasks detect error: %s", exc)
                    self.face_detected = False
                    self.raw_gaze_feature = None

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
            time.sleep(0.03)

    def _update_gaze_events(self, gpt: tuple[int, int] | None) -> None:
        if not self.item_bboxes or gpt is None:
            if self._current_state != "idle":
                self._current_state  = "idle"
                self._dwell_target   = None
                self._dwell_progress = 0.0
                self._emit_unlock()
            return

        x, y      = gpt
        hit_items = []
        near_edge = False

        for bbox in self.item_bboxes:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit_items.append(bbox["id"])
                if (x - x1 < AMBIG_PX or x2 - x < AMBIG_PX or
                        y - y1 < AMBIG_PX or y2 - y < AMBIG_PX):
                    near_edge = True

        if not hit_items:
            self._dwell_target, self._dwell_progress = None, 0.0
            self._dwell.update(None)
            if self._current_state != "idle":
                self._current_state = "idle"
                self._emit_unlock()
            return

        if len(hit_items) >= 2 or near_edge:
            if self._current_state != "ambiguous":
                self._current_state = "ambiguous"
                cands = [{"id": h, "pos": "left" if i == 0 else "right"}
                         for i, h in enumerate(hit_items[:2])]
                self.event_bus.emit(GazeEvent(
                    timestamp=time.time(),
                    type=GazeEventType.AMBIGUOUS,
                    payload={"candidates": cands},
                    confidence=0.4,
                ))
            return

        locked, progress = self._dwell.update(hit_items[0])
        self._dwell_target   = hit_items[0]
        self._dwell_progress = progress
        if locked:
            self._current_state = "locked"
            self._locked_id     = locked
            self.event_bus.emit(GazeEvent(
                timestamp=time.time(),
                type=GazeEventType.LOCK,
                payload={"target_id": locked},
                confidence=0.95,
            ))

    def _emit_unlock(self) -> None:
        self._locked_id = None
        self.event_bus.emit(GazeEvent(
            timestamp=time.time(),
            type=GazeEventType.UNLOCK,
            payload={},
            confidence=0.0,
        ))

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
    dwell_id  = gaze.dwell_target  if cond_b else None
    dwell_prg = gaze.dwell_progress if cond_b else 0.0

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

        cal_ready = gaze._calibrator.is_calibrated
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

    _put(canvas, "[A] Mouse-only  [B] Gaze+Speech  [M] PTT  [U] Calib  [Q] Quit",
         col2 - 200, H - 6, scale=0.38, color=CLR["sub"])

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
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    WIN_W, WIN_H = 1280, 720
    WIN_NAME     = "MMUI Kiosk – Real Input"

    cfg = Config(
        ASR_ENGINE            = "vosk",
        VOSK_MODEL_PATH       = str(ROOT / "models" / "vosk-model-small-en-us-0.15"),
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
    gaze   = RealGazeAdapter(event_bus=tk.bus, cam_index=0)
    speech = SpeechAdapter(event_bus=tk.bus, config=cfg, intent_parser=parser)

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
    calib_session: CalibSession5 | None = None
    calib_complete_ts: float = 0.0
    # ── Condition mode ───────────────────────────────────────────────
    # "B" = gaze+speech (default); "A" = mouse-only baseline
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
                    print(f"[Calib] {calib_session.status_msg}")

            # Draw calibration overlay on top of the normal canvas
            if calib_session and calib_session.active:
                calib_session.draw_overlay(canvas, gaze.raw_gaze_feature)

            # Condition A — draw clickable buttons (hidden during calibration)
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
                gaze.item_bboxes     = []
                gaze._dwell_target   = None
                gaze._dwell_progress = 0.0
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
        print("[Kiosk CV] Stopped.")


if __name__ == "__main__":
    main()
