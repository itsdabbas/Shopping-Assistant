"""
RealGazeAdapter
===============

Production gaze adapter that drives a webcam through MediaPipe FaceMesh
(or the newer Tasks API) to produce eye-relative gaze coordinates,
applies feature-space smoothing, runs the affine calibrator, and emits
:class:`~gazeshop.toolkit.event_bus.GazeEvent` objects on the shared bus.

Backend priority (auto-selected at start-time)
-----------------------------------------------
1. MediaPipe FaceLandmarker Tasks API  (mediapipe ≥ 0.10)
2. MediaPipe FaceMesh solutions API    (mediapipe <  0.10)
3. OpenCV Haar cascade                 (always available; no iris data)

Usage
-----
::

    from gazeshop.toolkit.adapters.real_gaze_adapter import RealGazeAdapter

    gaze = RealGazeAdapter(
        event_bus=tk.bus,
        cam_index=0,
        win_w=WIN_W,
        win_h=WIN_H,
        dwell_s=DWELL_S,
        ambig_px=AMBIG_PX,
    )
    tk.register_adapter(gaze)

Design notes
------------
* The background thread writes ``gaze_pt``, ``frame``, ``face_detected``,
  ``raw_gaze_feature``.  The main (render) thread reads them.  No lock is
  used — reads of a Python reference are atomic enough for display purposes.
* ``item_bboxes`` is written by the main thread before ``start()`` and
  updated whenever the Condition A/B mode changes.  The background thread
  reads it on every frame; the list swap is GIL-safe.
* ``calibrating`` is set True by the main thread during the calibration
  flow.  While True the hit-test loop is bypassed and no GazeEvents fire.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.calibration import (
    CalibSession5,
    GazeCalibrator5,
    GazeFeatureSmoother,
    GazeSmoother,
    eye_relative_gaze_from_landmarks,
)
from gazeshop.toolkit.dwell import DwellTracker
from gazeshop.toolkit.event_bus import EventBus, GazeEvent, GazeEventType

logger = logging.getLogger(__name__)

# ── Card-level snap / magnetic targeting ─────────────────────────────────────
# Gaze does not need to land exactly inside a card.  Instead, the nearest card
# within _CARD_SNAP_MARGIN_PX (measured as point-to-rect Euclidean distance)
# is selected.  The two-stage hysteresis (candidate → stable) prevents rapid
# flickering between cards when the raw gaze dot jitters near a card boundary.
_CARD_SNAP_MARGIN_PX: int = 50   # outward snap margin around each card (px)
_CAND_CONFIRM_FRAMES: int = 4    # new candidate must persist this many frames
_STABLE_HOLD_FRAMES:  int = 8    # frames to hold stable target after snap clears


class RealGazeAdapter(ModalityAdapter):
    """Webcam-driven gaze adapter for live MMUI demos.

    Parameters
    ----------
    event_bus:
        Shared event dispatcher.
    cam_index:
        OpenCV camera index (default 0).
    win_w:
        Canvas width in pixels.  Must match the OpenCV window width so that
        uncalibrated gaze visualisation and calibrated predictions land on
        the correct screen coordinates.  Default 1280.
    win_h:
        Canvas height in pixels.  Default 720.
    dwell_s:
        Seconds the user must look at an item before a LOCK event fires.
        Default 1.2.
    ambig_px:
        Pixel margin inside an item bounding box within which the gaze is
        considered ambiguous (near the boundary between adjacent items).
        Smaller values are more conservative (fewer false ambiguities).
        Typical values: 30 (dense grids) – 35 (sparse grids).  Default 30.
    """

    def __init__(
        self,
        event_bus: EventBus,
        cam_index: int = 0,
        win_w: int = 1280,
        win_h: int = 720,
        dwell_s: float = 1.2,
        ambig_px: int = 30,
    ) -> None:
        super().__init__(event_bus)

        self.cam_index = cam_index
        self._win_w = win_w
        self._win_h = win_h
        self._ambig_px = ambig_px

        # ── Threading ───────────────────────────────────────────────
        self._thread: threading.Thread | None = None

        # ── Shared state (written by bg thread, read by main) ───────
        self.gaze_pt: tuple[int, int] | None = None      # screen coords
        self.frame: np.ndarray | None = None
        self.face_detected: bool = False
        self.raw_gaze_feature: tuple[float, float] | None = None

        # ── Debug diagnostics (written by bg thread, read by render) ─
        self.smooth_gaze_feature: tuple[float, float] | None = None
        self.frame_rejection_reason: str = ""   # "" = ok, else why rejected
        self.gaze_using_held: bool = False      # True when gaze_pt is held old value

        # ── Bboxes (written by main thread) ─────────────────────────
        # [{"id": …, "x1": …, "y1": …, "x2": …, "y2": …}, …]
        self.item_bboxes: list[dict] = []

        # ── Dwell tracking ───────────────────────────────────────────
        self._dwell = DwellTracker(dwell_s)
        self._dwell_progress: float = 0.0
        self._dwell_target: str | None = None
        self._current_state: str = "idle"
        self._locked_id: str | None = None

        # ── Snap / target hysteresis ──────────────────────────────────
        # Public so the debug overlay can read them.
        self.snap_candidate: str | None = None    # best snap result this frame
        self.snap_candidate_frames: int = 0       # consecutive frames same candidate
        self.stable_target: str | None = None     # hysteresis-confirmed target
        self.snap_dist_px: float = float('inf')   # distance to nearest card (px)
        self.target_switch_count: int = 0         # debug: total target switches
        self._stable_exit_frames: int = 0         # frames since stable last confirmed

        # ── Camera & gaze backend ────────────────────────────────────
        self._cam: Any = None
        self._face_mesh: Any = None
        self._mp_landmarker: Any = None
        self._haar: Any = None
        self._gaze_mode: str = "none"   # "tasks"|"solutions"|"haar"|"none"

        # ── Calibration & smoothing ──────────────────────────────────
        self.calibrating: bool = False
        self._calibrator: GazeCalibrator5 = GazeCalibrator5(win_w, win_h)
        self._calib_session: CalibSession5 | None = None
        self._feat_smoother: GazeFeatureSmoother = GazeFeatureSmoother()
        self._smoother: GazeSmoother = GazeSmoother()
        self._last_valid_gaze_pt: tuple[int, int] | None = None
        self._last_gaze_valid_ts: float = 0.0

    # ── Public read-only dwell properties ────────────────────────────

    @property
    def dwell_progress(self) -> float:
        """Current dwell progress for the hovered item, in ``[0.0, 1.0]``."""
        return self._dwell_progress

    @property
    def dwell_target(self) -> str | None:
        """Item ID currently being dwelled on, or ``None``."""
        return self._dwell_target

    def reset_dwell(self) -> None:
        """Reset the dwell tracker and snap/hysteresis state to idle.

        Call whenever the application switches modes (e.g. Condition A → B)
        to prevent a stale dwell from immediately triggering a LOCK event.
        """
        self._dwell_progress = 0.0
        self._dwell_target = None
        self._dwell = DwellTracker(self._dwell.dwell_s)
        self._reset_snap_state()

    # ── Calibration API ──────────────────────────────────────────────

    def start_calibration(self) -> None:
        """Begin a fresh 5-point gaze calibration session.

        Resets both smoothers so calibration samples are not polluted by
        stale EMA state, sets :attr:`calibrating` to True (which pauses
        the hit-test loop in the background thread), and starts a new
        :class:`~gazeshop.toolkit.calibration.CalibSession5`.

        Call this when the user presses **U**.
        """
        self.calibrating = True
        self._feat_smoother.reset()
        self._smoother.reset()
        self._calib_session = CalibSession5(self._calibrator)
        self._calib_session.start()
        logger.info("Calibration session started.")

    def begin_calibration_capture(self) -> bool:
        """Open the stable-sample collection window for the current target.

        Call this when the user presses **SPACE** during calibration.

        Returns
        -------
        bool
            ``True`` if collection started, ``False`` if already collecting,
            no active session, or all points already collected.
        """
        if self._calib_session is None or not self._calib_session.active:
            return False
        return self._calib_session.begin_capture()

    def tick_calibration(self) -> bool:
        """Advance the calibration session by one frame.

        Must be called every render frame while a session is active.
        When the last calibration point is collected and validated the
        method returns ``True`` and performs adapter-side cleanup
        (clears :attr:`calibrating`, resets both smoothers so the first
        post-calibration gaze output is not polluted by calibration noise).

        Returns
        -------
        bool
            ``True`` exactly once — the frame the session completes.
            ``False`` every other frame (including when no session is active).
        """
        if self._calib_session is None or not self._calib_session.active:
            return False
        all_done = self._calib_session.tick(self.raw_gaze_feature)
        if all_done:
            self.calibrating = False
            self._feat_smoother.reset()
            self._smoother.reset()
            logger.info("Calibration session complete — %s", self._calib_session.status_msg)
        return all_done

    def is_calibrating(self) -> bool:
        """Return ``True`` while a calibration session is active."""
        return bool(self._calib_session and self._calib_session.active)

    def is_calibrated(self) -> bool:
        """Return ``True`` if the affine calibrator has a valid mapping."""
        return self._calibrator.is_calibrated

    def get_calibration_status(self) -> str:
        """Return the current (or most recent) calibration status message.

        The message is set by :class:`~gazeshop.toolkit.calibration.CalibSession5`
        and reflects the last instruction shown to the user (e.g.
        ``"Calibration complete!"`` or ``"Calibration FAILED – press U to retry."``).
        Returns an empty string if no session has been started yet.
        """
        return self._calib_session.status_msg if self._calib_session else ""

    def get_calibration_rmse(self) -> float | None:
        """Return RMSE (px) from the last completed calibration, or None."""
        return getattr(self._calibrator, 'last_rmse', None)

    def draw_calibration_overlay(self, canvas: "np.ndarray") -> None:
        """Draw the calibration UI overlay onto *canvas* in-place.

        Delegates to :meth:`~gazeshop.toolkit.calibration.CalibSession5.draw_overlay`
        with the current raw gaze feature for live numeric debug.
        A no-op when no session is active.

        Parameters
        ----------
        canvas:
            The OpenCV BGR frame to annotate.  Modified in-place.
        """
        if self._calib_session and self._calib_session.active:
            self._calib_session.draw_overlay(canvas, self.raw_gaze_feature)

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Open camera, initialise gaze backend, start background thread."""
        self._running = True
        self._cam = cv2.VideoCapture(self.cam_index)
        if not self._cam.isOpened():
            logger.error("Cannot open camera %d", self.cam_index)
        self._init_gaze_backend()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            "RealGazeAdapter started (cam=%d, backend=%s, win=%dx%d, "
            "dwell=%.1fs, ambig_px=%d).",
            self.cam_index, self._gaze_mode,
            self._win_w, self._win_h,
            self._dwell.dwell_s, self._ambig_px,
        )

    def stop(self) -> None:
        """Stop background thread and release camera."""
        self._running = False
        if self._cam:
            self._cam.release()
        logger.info("RealGazeAdapter stopped.")

    def set_bboxes(self, bboxes: list[dict]) -> None:
        """Replace the hit-test bounding boxes (GIL-safe list swap)."""
        self.item_bboxes = bboxes

    # ── Backend initialisation ────────────────────────────────────────

    def _init_gaze_backend(self) -> None:
        """Try each backend in priority order; set ``_gaze_mode`` accordingly."""

        # ── 1. MediaPipe Tasks API (mediapipe >= 0.10) ───────────────
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as _mp_python
            from mediapipe.tasks.python import vision as _mp_vision
            import urllib.request
            import tempfile

            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task"
            )
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
            logger.debug("Tasks API unavailable: %s", e)

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
            logger.debug("solutions API unavailable: %s", e)

        # ── 3. OpenCV Haar cascade (always available) ────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        self._gaze_mode = "haar"
        print("[GazeAdapter] Falling back to OpenCV Haar cascade face detection.")

    # ── Background capture loop ───────────────────────────────────────

    def _loop(self) -> None:
        win_w = self._win_w
        win_h = self._win_h

        while self._running:
            ok, raw = self._cam.read()
            if not ok:
                time.sleep(0.03)
                continue

            raw = cv2.flip(raw, 1)
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
                        self.frame_rejection_reason = "" if feat is not None else "blink/bad eye"
                        smooth_feat = self._feat_smoother.update(feat)
                        self.smooth_gaze_feature = smooth_feat
                        if smooth_feat is not None:
                            if self._calibrator.is_calibrated and not self.calibrating:
                                raw_gpt = self._calibrator.predict(smooth_feat)
                                gpt = self._smoother.update(raw_gpt)
                            else:
                                # Rough uncalibrated visualisation (eye-centred)
                                gpt = (
                                    max(0, min(win_w, int(win_w // 2 + (smooth_feat[0] - 0.5) * win_w))),
                                    max(0, min(win_h, int(win_h // 2 + (smooth_feat[1] - 0.5) * win_h))),
                                )
                    else:
                        self.face_detected = False
                        self.raw_gaze_feature = None
                        self.smooth_gaze_feature = None
                        self.frame_rejection_reason = "no face"
                        self._feat_smoother.update(None)
                except Exception as exc:
                    logger.debug("Tasks detect error: %s", exc)
                    self.face_detected = False
                    self.raw_gaze_feature = None
                    self.smooth_gaze_feature = None
                    self.frame_rejection_reason = "detection error"

            # ── solutions API ─────────────────────────────────────────
            elif self._gaze_mode == "solutions" and self._face_mesh is not None:
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                results = self._face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    self.face_detected = True
                    lms = results.multi_face_landmarks[0].landmark
                    feat = eye_relative_gaze_from_landmarks(lms)
                    self.raw_gaze_feature = feat
                    self.frame_rejection_reason = "" if feat is not None else "blink/bad eye"
                    smooth_feat = self._feat_smoother.update(feat)
                    self.smooth_gaze_feature = smooth_feat
                    if smooth_feat is not None:
                        if self._calibrator.is_calibrated and not self.calibrating:
                            raw_gpt = self._calibrator.predict(smooth_feat)
                            gpt = self._smoother.update(raw_gpt)
                        else:
                            gpt = (
                                max(0, min(win_w, int(win_w // 2 + (smooth_feat[0] - 0.5) * win_w))),
                                max(0, min(win_h, int(win_h // 2 + (smooth_feat[1] - 0.5) * win_h))),
                            )
                else:
                    self.face_detected = False
                    self.raw_gaze_feature = None
                    self.smooth_gaze_feature = None
                    self.frame_rejection_reason = "no face"
                    self._feat_smoother.update(None)

            # ── Haar cascade (no iris data — gaze calibration unavailable) ──
            elif self._gaze_mode == "haar" and self._haar is not None:
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                faces = self._haar.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                if len(faces) > 0:
                    self.face_detected = True
                    self.raw_gaze_feature = None
                    self.smooth_gaze_feature = None
                    self.frame_rejection_reason = "haar: no iris"
                else:
                    self.face_detected = False
                    self.raw_gaze_feature = None
                    self.smooth_gaze_feature = None
                    self.frame_rejection_reason = "no face"
            else:
                self.face_detected = False
                self.raw_gaze_feature = None
                self.smooth_gaze_feature = None
                self.frame_rejection_reason = "no backend"

            # ── Keep last valid gaze briefly when eyes temporarily lost ──
            # Only refresh timestamp on fresh detections, so the 0.5 s hold
            # actually expires instead of refreshing itself indefinitely.
            _gpt_was_fresh = (gpt is not None)
            if gpt is None and self._last_valid_gaze_pt is not None:
                if time.time() - self._last_gaze_valid_ts < 0.5:
                    gpt = self._last_valid_gaze_pt
            if _gpt_was_fresh and gpt is not None:
                self._last_valid_gaze_pt = gpt
                self._last_gaze_valid_ts = time.time()
            self.gaze_using_held = (not _gpt_was_fresh) and (gpt is not None)

            self.gaze_pt = gpt
            self.frame = raw

            # ── Hit-test (skipped during calibration) ────────────────
            if not self.calibrating:
                self._update_gaze_events(gpt)

            time.sleep(0.03)   # ~30 fps

    # ── Snap / hysteresis helpers ─────────────────────────────────────

    def _reset_snap_state(self) -> None:
        """Clear all snap/hysteresis state (call on mode switch or calibration)."""
        self.snap_candidate = None
        self.snap_candidate_frames = 0
        self.stable_target = None
        self.snap_dist_px = float('inf')
        self._stable_exit_frames = 0
        self.target_switch_count = 0

    def _find_snap_target(
        self, gpt: tuple[int, int]
    ) -> tuple[str | None, float]:
        """Return (nearest_card_id, distance_px) for the closest card.

        Distance is measured as the Euclidean distance from *gpt* to the
        nearest point *on* the card rectangle (0 when strictly inside).
        Returns (None, inf) when ``item_bboxes`` is empty.
        """
        x, y = gpt
        best_id: str | None = None
        best_dist = float('inf')

        for bbox in self.item_bboxes:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            # Clamp to rectangle → nearest point on card
            cx = max(x1, min(x, x2))
            cy = max(y1, min(y, y2))
            dist = math.hypot(x - cx, y - cy)
            if dist < best_dist:
                best_dist = dist
                best_id = bbox["id"]

        return best_id, best_dist

    def _resolve_stable_target(
        self, gpt: tuple[int, int] | None
    ) -> str | None:
        """Return the hysteresis-filtered stable target card ID.

        Two-stage logic:
        1. **Snap**: pick the nearest card within *_CARD_SNAP_MARGIN_PX*.
        2. **Candidate accumulation**: a new snap result must persist for
           *_CAND_CONFIRM_FRAMES* consecutive frames before it replaces
           the current stable target.
        3. **Stable hold**: when snap disappears the stable target is kept for
           *_STABLE_HOLD_FRAMES* frames before being cleared.

        This means brief gaze jitter (landing in a gap or near another card
        for < *_CAND_CONFIRM_FRAMES* frames) does not disrupt the stable
        target, and the dwell timer keeps advancing smoothly.
        """
        if gpt is not None:
            snap_id, snap_dist = self._find_snap_target(gpt)
            if snap_dist > _CARD_SNAP_MARGIN_PX:
                snap_id = None   # too far from any card
        else:
            snap_id, snap_dist = None, float('inf')

        self.snap_dist_px = snap_dist

        # ── Step 1: update candidate streak ──────────────────────────
        if snap_id == self.snap_candidate:
            self.snap_candidate_frames = min(
                self.snap_candidate_frames + 1, _CAND_CONFIRM_FRAMES + 2
            )
        else:
            self.snap_candidate = snap_id
            self.snap_candidate_frames = 1

        # ── Step 2: promote candidate → stable, or hold/clear stable ─
        if snap_id == self.stable_target:
            # Snap agrees with current stable → reset exit counter
            self._stable_exit_frames = 0

        elif (
            self.snap_candidate is not None
            and self.snap_candidate_frames >= _CAND_CONFIRM_FRAMES
        ):
            # New candidate confirmed — switch stable target
            if self.snap_candidate != self.stable_target:
                self.stable_target = self.snap_candidate
                self._stable_exit_frames = 0
                self.target_switch_count += 1

        elif snap_id is None:
            # No snap at all → count toward stable hold expiry
            self._stable_exit_frames += 1
            if self._stable_exit_frames > _STABLE_HOLD_FRAMES:
                if self.stable_target is not None:
                    self.stable_target = None
                    self.target_switch_count += 1
        # else: snap is some card other than stable, but not confirmed yet
        #   → hold stable as-is (no change to exit counter)

        return self.stable_target

    # ── Hit-test & event emission ─────────────────────────────────────

    def _update_gaze_events(self, gpt: tuple[int, int] | None) -> None:
        """Resolve the stable snap target and advance dwell / emit GazeEvents.

        Replaces the old strict point-in-rectangle test.  The magnetic snap
        (_find_snap_target) and two-stage hysteresis (_resolve_stable_target)
        ensure that brief gaze jitter near card boundaries does not interrupt
        an in-progress dwell.
        """
        # Condition A or bboxes not set — clear everything cleanly
        if not self.item_bboxes:
            self._reset_snap_state()
            if self._current_state != "idle":
                self._current_state = "idle"
                self._dwell_target = None
                self._dwell_progress = 0.0
                self._emit_unlock()
            return

        stable = self._resolve_stable_target(gpt)

        if stable is not None:
            locked, progress = self._dwell.update(stable)
            self._dwell_target = stable
            self._dwell_progress = progress
            if locked:
                self._current_state = "locked"
                self._locked_id = locked
                self.event_bus.emit(
                    GazeEvent(
                        timestamp=time.time(),
                        type=GazeEventType.LOCK,
                        payload={"target_id": locked},
                        confidence=0.95,
                    )
                )
        else:
            self._dwell.update(None)
            self._dwell_target = None
            self._dwell_progress = 0.0
            if self._current_state != "idle":
                self._current_state = "idle"
                self._emit_unlock()

    def _emit_unlock(self) -> None:
        self._locked_id = None
        self.event_bus.emit(
            GazeEvent(
                timestamp=time.time(),
                type=GazeEventType.UNLOCK,
                payload={},
                confidence=0.0,
            )
        )
