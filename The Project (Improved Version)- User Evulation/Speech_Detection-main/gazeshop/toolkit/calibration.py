"""
Gaze Calibration Module
=======================

Maps an **eye-relative** gaze feature to screen (canvas) coordinates using a
5-point affine regression (numpy lstsq — no sklearn needed).

The key insight: instead of using the absolute image position of the iris
center (which tracks head/face movement), we normalize the iris position
*inside each eye bounding box* (corners + eyelids).  This value changes when
the user moves their eyes and is nearly constant when the user moves only
their head.

Usage
-----
    cal = GazeCalibrator5(win_w=1280, win_h=720)
    session = CalibSession5(cal)
    session.start()

    # Per frame — during calibration:
    feat = eye_relative_gaze_from_landmarks(lms)   # (rel_x, rel_y) or None
    all_done = session.tick(feat)                   # call every frame
    session.draw_overlay(canvas, feat)
    if all_done:
        calibrating = False

    # On SPACE key press:
    session.begin_capture()     # start stable-sample collection window

    # During normal operation:
    smoother = GazeSmoother()
    raw_gpt  = cal.predict(feat)            # → (screen_x, screen_y) or None
    gaze_gpt = smoother.update(raw_gpt)    # smoothed, dead-zoned output
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ── Tuning constants ──────────────────────────────────────────────────────────
_COLLECT_S        = 0.70    # seconds to collect stable samples per calibration point
_MIN_COLLECT      = 8       # minimum valid frames required; ~30 fps × 0.7 s ≈ 21 frames
_QUALITY_THRESH   = 180.0   # max acceptable RMSE in px before rejecting calibration


# ── MediaPipe FaceMesh landmark indices ───────────────────────────────────────
# These require refine_landmarks=True (solutions API) or the Tasks API model.
# Left eye  (anatomical left = appears on LEFT side of mirrored webcam image)
_LEFT_IRIS      = 468   # iris centre
_LEFT_CORNER_A  = 33    # temporal/outer corner
_LEFT_CORNER_B  = 133   # nasal/inner corner
_LEFT_TOP       = 159   # top eyelid centre
_LEFT_BOTTOM    = 145   # bottom eyelid centre

# Right eye (anatomical right = appears on RIGHT side of mirrored webcam image)
_RIGHT_IRIS     = 473   # iris centre
_RIGHT_CORNER_A = 362   # nasal/inner corner
_RIGHT_CORNER_B = 263   # temporal/outer corner
_RIGHT_TOP      = 386   # top eyelid centre
_RIGHT_BOTTOM   = 374   # bottom eyelid centre


# ── Eye-relative gaze feature ─────────────────────────────────────────────────

def eye_relative_gaze_from_landmarks(lms) -> tuple[float, float] | None:
    """Compute an eye-relative gaze feature from MediaPipe FaceMesh landmarks.

    For each eye the iris centre is normalised inside the eye bounding box
    defined by the two corners (horizontal) and two eyelid points (vertical):

        rel_x = (iris_x - left_corner_x) / eye_width
        rel_y = (iris_y - top_lid_y)     / eye_height

    Both eyes are averaged into a single (rel_x, rel_y) pair.

    This feature changes when the user moves their EYES and stays nearly
    constant when the user moves only their HEAD — unlike the absolute iris
    position which is dominated by face/head translation.

    Requires iris landmarks 468 and 473 (refine_landmarks=True or Tasks API).
    Returns None if those landmarks are not present, if eyes are too narrow
    (blink/squint), or if the computed feature is implausibly out of range.
    """
    if len(lms) < 474:
        return None

    contributions: list[tuple[float, float]] = []

    for iris_idx, ca_idx, cb_idx, top_idx, bot_idx in (
        (_LEFT_IRIS,  _LEFT_CORNER_A,  _LEFT_CORNER_B,  _LEFT_TOP,    _LEFT_BOTTOM),
        (_RIGHT_IRIS, _RIGHT_CORNER_A, _RIGHT_CORNER_B, _RIGHT_TOP,   _RIGHT_BOTTOM),
    ):
        try:
            iris = lms[iris_idx]
            ca   = lms[ca_idx]
            cb   = lms[cb_idx]
            top  = lms[top_idx]
            bot  = lms[bot_idx]

            left_x  = min(ca.x, cb.x)
            right_x = max(ca.x, cb.x)
            top_y   = min(top.y, bot.y)
            bot_y   = max(top.y, bot.y)

            eye_w = right_x - left_x
            eye_h = bot_y   - top_y

            if eye_w < 1e-5 or eye_h < 1e-5:
                continue

            # Reject blink / heavily squinted eye: opening ratio too small
            if eye_h / eye_w < 0.08:
                continue

            rel_x = (iris.x - left_x) / eye_w
            rel_y = (iris.y - top_y)  / eye_h

            # Reject wildly implausible values (landmark noise / partial occlusion)
            if not (-0.2 <= rel_x <= 1.2 and -0.2 <= rel_y <= 1.2):
                continue

            contributions.append((rel_x, rel_y))
        except (IndexError, AttributeError):
            continue

    if not contributions:
        return None

    avg_x = sum(r[0] for r in contributions) / len(contributions)
    avg_y = sum(r[1] for r in contributions) / len(contributions)
    return avg_x, avg_y


# ── Legacy absolute-iris helper (kept for back-compat; NOT used for calib) ───

def iris_xy_from_landmarks(lms) -> tuple[float, float] | None:
    """Return averaged iris centre in normalised [0,1] image coordinates.

    WARNING: This is the ABSOLUTE iris position and is dominated by head/face
    movement.  Do NOT use it for gaze calibration.  Use
    ``eye_relative_gaze_from_landmarks`` instead.

    Returns None if iris landmarks (>=474) are unavailable.
    Nose-tip fallback has been intentionally removed.
    """
    n = len(lms)
    if n > 473:
        l, r = lms[468], lms[473]
        return (l.x + r.x) / 2.0, (l.y + r.y) / 2.0
    if n > 468:
        lm = lms[468]
        return lm.x, lm.y
    return None


def extract_iris_xy(
    raw_bgr: "np.ndarray",
    face_mesh: object | None = None,
    mp_landmarker: object | None = None,
) -> tuple[float, float] | None:
    """Extract averaged absolute iris centre from a BGR camera frame.

    Returns (x, y) in [0,1] normalised camera coordinates, or None.

    NOTE: For gaze calibration prefer ``eye_relative_gaze_from_landmarks``
    which is invariant to head translation.
    """
    try:
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        if face_mesh is not None:
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                return iris_xy_from_landmarks(lms)

        if mp_landmarker is not None:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = mp_landmarker.detect(mp_img)
            if result.face_landmarks:
                lms = result.face_landmarks[0]
                return iris_xy_from_landmarks(lms)

    except Exception as exc:
        logger.debug("[CAL] extract_iris_xy error: %s", exc)

    return None


# ── Gaze smoother ─────────────────────────────────────────────────────────────

class GazeSmoother:
    """Exponential moving average with dead-zone and per-frame jump limiter.

    Prevents the calibrated gaze dot from flying across the screen due to
    momentary iris detection noise.

    Parameters
    ----------
    alpha     : EMA weight for the *previous* position (0 = no smoothing,
                1 = frozen).  0.75–0.80 is a good starting point.
    dead_zone : movements smaller than this many pixels are ignored entirely.
    max_jump  : per-frame movement is capped at this many pixels before EMA,
                preventing sudden large jumps from being applied instantly.
    """

    def __init__(self, alpha: float = 0.78,
                 dead_zone: int = 10,
                 max_jump: int  = 150) -> None:
        self.alpha     = alpha
        self.dead_zone = dead_zone
        self.max_jump  = max_jump
        self._prev: tuple[int, int] | None = None

    def reset(self) -> None:
        """Clear history (call when calibration changes)."""
        self._prev = None

    def update(self, new_pt: tuple[int, int] | None) -> tuple[int, int] | None:
        """Return the smoothed gaze point, or None if no history yet."""
        if new_pt is None:
            return self._prev         # hold last known position briefly
        if self._prev is None:
            self._prev = new_pt
            return new_pt

        px, py = self._prev
        nx, ny = new_pt
        dx, dy = nx - px, ny - py
        dist = math.sqrt(dx * dx + dy * dy)

        # Dead-zone: ignore tiny jitter
        if dist < self.dead_zone:
            return self._prev

        # Jump limiter: clamp single-frame displacement
        if dist > self.max_jump:
            scale = self.max_jump / dist
            nx = int(px + dx * scale)
            ny = int(py + dy * scale)

        # EMA
        sx = int(self.alpha * px + (1.0 - self.alpha) * nx)
        sy = int(self.alpha * py + (1.0 - self.alpha) * ny)
        self._prev = (sx, sy)
        return self._prev


# ── Gaze feature smoother ────────────────────────────────────────────────────

class GazeFeatureSmoother:
    """EMA smoother for the raw eye-relative feature (rel_x, rel_y).

    Applied BEFORE calibrator.predict() so that per-frame landmark jitter
    is dampened in feature space rather than in screen space.  This is
    important because the affine matrix can amplify small feature changes
    into large screen-space jumps.

    Parameters
    ----------
    alpha       : EMA weight for the *previous* feature (0 = no smoothing,
                  1 = frozen).  0.80 works well for ~30 fps input.
    dead_zone   : feature-space dead-zone; movements smaller than this are
                  ignored.  Expressed in normalised units (0–1 scale).
    max_jump    : maximum allowed feature displacement per frame before EMA.
                  Prevents a sudden bad detection from being applied at all.
    hold_frames : number of frames to hold the last good value when the
                  current frame returns None (e.g. brief blinks).
    """

    def __init__(
        self,
        alpha: float = 0.80,
        dead_zone: float = 0.005,
        max_jump: float = 0.15,
        hold_frames: int = 6,
    ) -> None:
        self.alpha       = alpha
        self.dead_zone   = dead_zone
        self.max_jump    = max_jump
        self.hold_frames = hold_frames
        self._prev: tuple[float, float] | None = None
        self._none_count: int = 0

    def reset(self) -> None:
        """Clear history (call when calibration starts/ends)."""
        self._prev = None
        self._none_count = 0

    def update(
        self, feat: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Return the smoothed feature, or None if no stable history yet."""
        if feat is None:
            self._none_count += 1
            if self._none_count <= self.hold_frames and self._prev is not None:
                return self._prev   # hold last good value during brief blink
            return None

        self._none_count = 0

        if self._prev is None:
            self._prev = feat
            return feat

        px, py = self._prev
        nx, ny = feat
        dx, dy = nx - px, ny - py
        dist = math.sqrt(dx * dx + dy * dy)

        # Dead-zone: ignore tiny sub-threshold jitter
        if dist < self.dead_zone:
            return self._prev

        # Jump limiter: clamp single-frame displacement
        if dist > self.max_jump:
            scale = self.max_jump / dist
            nx = px + dx * scale
            ny = py + dy * scale

        # EMA
        sx = self.alpha * px + (1.0 - self.alpha) * nx
        sy = self.alpha * py + (1.0 - self.alpha) * ny
        self._prev = (sx, sy)
        return self._prev


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CalibPoint:
    screen_x: int
    screen_y: int
    samples: list[tuple[float, float]] = field(default_factory=list)

    def mean_feature(self) -> tuple[float, float] | None:
        """Return the median of collected samples (robust to outliers)."""
        if not self.samples:
            return None
        xs = [s[0] for s in self.samples]
        ys = [s[1] for s in self.samples]
        return float(np.median(xs)), float(np.median(ys))

    # Back-compat alias
    def mean_iris(self) -> tuple[float, float] | None:
        return self.mean_feature()


# ── Calibrator ────────────────────────────────────────────────────────────────

class GazeCalibrator:
    """N-point gaze calibration using affine regression.

    Input feature: (rel_x, rel_y) — eye-relative normalised iris position.
    Output: (screen_x, screen_y) in canvas pixel coordinates.

    The affine model:
        [sx]   [a b c]   [rx]
        [sy] = [d e f] * [ry]
                          [ 1]
    Solved with numpy.linalg.lstsq over all calibration samples.
    """

    _GRID: list[tuple[float, float]] = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    ]

    SAMPLES_PER_POINT: int = 10
    DWELL_S:           float = 2.0

    def __init__(self, win_w: int = 1280, win_h: int = 720) -> None:
        self.win_w = win_w
        self.win_h = win_h
        self._points: list[CalibPoint] = []
        self._matrix: np.ndarray | None = None   # (3, 2)
        self.is_calibrated: bool = False
        # Feature range seen during calibration (for input clamping in predict)
        self._feat_min: tuple[float, float] | None = None
        self._feat_max: tuple[float, float] | None = None
        self._build_points()

    def _build_points(self) -> None:
        self._points = [
            CalibPoint(
                screen_x=int(rx * self.win_w),
                screen_y=int(ry * self.win_h),
            )
            for rx, ry in self._GRID
        ]

    @property
    def calibration_points(self) -> list[CalibPoint]:
        return self._points

    def reset(self) -> None:
        self._build_points()
        self._matrix = None
        self.is_calibrated = False
        self._feat_min = None
        self._feat_max = None
        logger.info("[CAL] Reset.")

    def add_sample(self, point_idx: int, gaze_feat: tuple[float, float]) -> None:
        """Add one eye-relative gaze feature sample for calibration point *point_idx*."""
        if 0 <= point_idx < len(self._points):
            self._points[point_idx].samples.append(gaze_feat)

    def fit(self) -> bool:
        """Fit affine regression from eye-relative feature → screen coords.

        Returns True on success.  Does NOT perform a quality check — use
        ``fit_and_validate`` if you want automatic rejection of bad results.
        """
        feat_rows, screen_rows = [], []
        for pt in self._points:
            mean = pt.mean_feature()
            if mean is None:
                continue
            rx, ry = mean
            feat_rows.append([rx, ry, 1.0])
            screen_rows.append([pt.screen_x, pt.screen_y])

        if len(feat_rows) < 4:
            logger.warning("[CAL] Not enough calibration points (%d < 4).", len(feat_rows))
            return False

        X = np.array(feat_rows,   dtype=np.float64)   # (N, 3)
        Y = np.array(screen_rows, dtype=np.float64)   # (N, 2)

        M, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
        self._matrix = M   # (3, 2)
        self.is_calibrated = True

        # Store feature range seen during calibration for clamping in predict()
        rx_vals = [r[0] for r in feat_rows]
        ry_vals = [r[1] for r in feat_rows]
        self._feat_min = (min(rx_vals), min(ry_vals))
        self._feat_max = (max(rx_vals), max(ry_vals))

        logger.info(
            "[CAL] Fitted on %d points  rank=%d  residuals=%s  "
            "feat_range x=[%.3f,%.3f] y=[%.3f,%.3f]",
            len(feat_rows), rank,
            residuals.tolist() if len(residuals) else "n/a",
            self._feat_min[0], self._feat_max[0],
            self._feat_min[1], self._feat_max[1],
        )
        return True

    def fit_and_validate(
        self,
        max_error_px: float = _QUALITY_THRESH,
        prev_matrix: "np.ndarray | None" = None,
    ) -> str:
        """Fit and validate calibration quality.

        Computes RMSE between predicted and target screen positions using the
        calibration samples collected.  If RMSE exceeds *max_error_px*, the
        new calibration is rejected and *prev_matrix* (if provided) is restored.

        Returns
        -------
        'ok'     – calibration accepted and stored.
        'poor'   – RMSE too high; calibration rejected, previous restored.
        'failed' – not enough calibration points to fit at all.
        """
        if not self.fit():
            return 'failed'

        # Evaluate RMSE on the training points
        errors: list[float] = []
        for pt in self._points:
            feat = pt.mean_feature()
            if feat is None:
                continue
            pred = self.predict(feat)
            if pred is None:
                continue
            dx = pred[0] - pt.screen_x
            dy = pred[1] - pt.screen_y
            errors.append(math.sqrt(dx * dx + dy * dy))

        if not errors:
            return 'failed'

        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        logger.info("[CAL] Calibration RMSE: %.1f px  (threshold %.0f px)", rmse, max_error_px)

        if rmse > max_error_px:
            # Reject new calibration — restore previous if available
            if prev_matrix is not None:
                self._matrix = prev_matrix
                self.is_calibrated = True
                logger.warning(
                    "[CAL] Rejected (RMSE %.1f > %.0f).  Previous calibration restored.",
                    rmse, max_error_px,
                )
            else:
                self._matrix = None
                self.is_calibrated = False
                logger.warning(
                    "[CAL] Rejected (RMSE %.1f > %.0f).  No previous calibration.",
                    rmse, max_error_px,
                )
            return 'poor'

        return 'ok'

    def predict(self, gaze_feat: tuple[float, float]) -> tuple[int, int] | None:
        """Map eye-relative feature (rx, ry) → screen (sx, sy).

        Clamps the input feature to the range seen during calibration (with a
        15 % margin) before applying the affine transform.  This prevents
        extrapolation from amplifying out-of-range landmark noise into large
        screen-position errors.

        Returns None if not calibrated.
        """
        if not self.is_calibrated or self._matrix is None:
            return None

        rx, ry = gaze_feat

        # Clamp to calibration feature range + 15 % margin
        if self._feat_min is not None and self._feat_max is not None:
            margin_x = (self._feat_max[0] - self._feat_min[0]) * 0.15
            margin_y = (self._feat_max[1] - self._feat_min[1]) * 0.15
            rx = float(np.clip(rx, self._feat_min[0] - margin_x,
                                   self._feat_max[0] + margin_x))
            ry = float(np.clip(ry, self._feat_min[1] - margin_y,
                                   self._feat_max[1] + margin_y))

        feat = np.array([rx, ry, 1.0], dtype=np.float64)
        pred = feat @ self._matrix   # (2,)
        sx = int(np.clip(pred[0], 0, self.win_w))
        sy = int(np.clip(pred[1], 0, self.win_h))
        return sx, sy


# ── 5-point calibrator (inset grid — avoids extreme corner eye movement) ─────

class GazeCalibrator5(GazeCalibrator):
    """5-point variant: center + 4 inset corner targets.

    Targets are at 25 %/75 % of the window instead of 10 %/90 %, which keeps
    calibration points within a comfortable eye-movement range and avoids
    triggering head movement.
    """

    _GRID: list[tuple[float, float]] = [
        (0.50, 0.50),   # center       (shown first — easiest anchor)
        (0.25, 0.25),   # top-left
        (0.75, 0.25),   # top-right
        (0.25, 0.75),   # bottom-left
        (0.75, 0.75),   # bottom-right
    ]


# ── Interactive 9-point calibration session (dwell-based) ────────────────────

class CalibrationSession:
    """Drives a 9-point dwell-based calibration sequence frame-by-frame.

    Usage (inside OpenCV main loop)::

        session = CalibrationSession(calibrator, on_complete=lambda: None)
        session.start()
        while session.active:
            feat = eye_relative_gaze_from_landmarks(lms)
            session.update(feat)
            session.draw_overlay(canvas)
    """

    def __init__(
        self,
        calibrator: GazeCalibrator,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self._cal = calibrator
        self._on_complete = on_complete
        self._point_idx: int = 0
        self._point_start: float = 0.0
        self._collected: int = 0
        self.active: bool = False
        self.status_msg: str = ""

    def start(self) -> None:
        self._cal.reset()
        self._point_idx = 0
        self._point_start = time.time()
        self._collected = 0
        self.active = True
        self.status_msg = "Calibration started – look at the dot"
        logger.info("[CAL SESSION] Started.")

    def update(self, gaze_feat: tuple[float, float] | None) -> None:
        if not self.active:
            return
        pts = self._cal.calibration_points
        if self._point_idx >= len(pts):
            self._finish()
            return

        elapsed = time.time() - self._point_start
        progress = min(elapsed / self._cal.DWELL_S, 1.0)

        if gaze_feat is not None:
            self._cal.add_sample(self._point_idx, gaze_feat)
            self._collected += 1

        n_pts = len(pts)
        self.status_msg = (
            f"Calibration  {self._point_idx + 1}/{n_pts} – "
            f"look at dot ({progress:.0%})"
        )

        if elapsed >= self._cal.DWELL_S:
            logger.info(
                "[CAL SESSION] Point %d collected %d samples.",
                self._point_idx, self._collected,
            )
            self._point_idx += 1
            self._point_start = time.time()
            self._collected = 0

    def _finish(self) -> None:
        ok = self._cal.fit()
        if ok:
            self.status_msg = "Calibration complete!  Gaze mapping active."
            logger.info("[CAL SESSION] Complete.")
        else:
            self.status_msg = "Calibration FAILED – not enough data. Press U to retry."
            logger.warning("[CAL SESSION] Failed.")
        self.active = False
        if self._on_complete:
            self._on_complete()

    def draw_overlay(self, canvas: "np.ndarray") -> None:
        try:
            import cv2
        except ImportError:
            return

        pts = self._cal.calibration_points
        if self._point_idx >= len(pts):
            return

        pt = pts[self._point_idx]
        cx, cy = pt.screen_x, pt.screen_y
        elapsed = time.time() - self._point_start
        progress = min(elapsed / self._cal.DWELL_S, 1.0)

        overlay = canvas.copy()
        overlay[:] = (10, 10, 20)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        angle = int(360 * progress)
        cv2.ellipse(canvas, (cx, cy), (28, 28), -90, 0, angle,
                    (80, 220, 120), 4, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 12, (80, 220, 120), 2, cv2.LINE_AA)

        W = canvas.shape[1]
        cv2.putText(canvas, "GAZE CALIBRATION MODE",
                    (W // 2 - 162, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Keep your head still – move only your eyes",
                    (W // 2 - 232, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 220, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, self.status_msg,
                    (W // 2 - 260, canvas.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"Point {self._point_idx + 1} / {len(pts)}",
                    (cx - 35, cy - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ── SPACE-key-driven 5-point calibration session ──────────────────────────────

class CalibSession5:
    """Interactive 5-point gaze calibration driven by SPACE-key presses.

    Calibrates from the eye-relative gaze feature to window/screen coordinates.

    Key design decisions
    --------------------
    * Targets are placed at 25 %/75 % insets — comfortable eye movement,
      no head movement required.
    * Pressing SPACE opens a short collection window (_COLLECT_S seconds).
      All valid frames during that window are collected; their median is used
      as the representative sample — one bad frame cannot ruin a point.
    * After all 5 points, ``fit_and_validate`` checks RMSE.  If the error
      exceeds _QUALITY_THRESH pixels, the new calibration is rejected and the
      previous good calibration (if any) is restored.

    Usage (inside OpenCV main loop)::

        session = CalibSession5(GazeCalibrator5(win_w=1280, win_h=720))
        session.start()

        # every frame:
        all_done = session.tick(raw_gaze_feature)      # ← add this
        session.draw_overlay(canvas, raw_gaze_feature)
        if all_done:
            calibrating = False
            smoother.reset()

        # on SPACE key:
        session.begin_capture()
    """

    def __init__(self, calibrator: "GazeCalibrator5") -> None:
        self._cal = calibrator
        self._idx = 0
        self.active = False
        self.status_msg = ""
        # Stable-sample collection state
        self._collecting: bool = False
        self._collect_buf: list[tuple[float, float]] = []
        self._collect_start: float = 0.0
        # Saved calibration matrix to restore on rejection
        self._prev_matrix: "np.ndarray | None" = None

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        # Snapshot the current good calibration before resetting
        if self._cal.is_calibrated and self._cal._matrix is not None:
            self._prev_matrix = self._cal._matrix.copy()
        else:
            self._prev_matrix = None
        self._cal.reset()
        self._idx = 0
        self.active = True
        self._collecting = False
        self._collect_buf = []
        self.status_msg = "Look at the target and press SPACE"
        logger.info("[CAL5] Started (%d points).  Previous cal saved: %s",
                    len(self._cal.calibration_points),
                    self._prev_matrix is not None)

    def begin_capture(self) -> bool:
        """Start the stable-sample collection window for the current target.

        Call this when the user presses SPACE.  Returns True if collection
        started, False if already collecting or session is inactive.
        """
        if not self.active or self._collecting or self._idx >= self.n_points:
            return False
        self._collecting = True
        self._collect_buf = []
        self._collect_start = time.time()
        self.status_msg = "Collecting stable eye samples..."
        logger.info("[CAL5] Collection started for point %d", self._idx)
        return True

    def tick(self, gaze_feat: tuple[float, float] | None) -> bool:
        """Process one frame.  Must be called every frame while session is active.

        If a collection window is open, valid features are accumulated.  When
        the window closes (_COLLECT_S seconds), the median feature is computed
        and stored.  Returns True when the entire calibration sequence is done
        (session becomes inactive).
        """
        if not self.active or not self._collecting:
            return False

        if gaze_feat is not None:
            self._collect_buf.append(gaze_feat)

        elapsed = time.time() - self._collect_start
        if elapsed < _COLLECT_S:
            return False   # still collecting

        # Collection window closed — finalise this point
        n_valid = len(self._collect_buf)
        if n_valid < _MIN_COLLECT:
            self.status_msg = (
                f"Not enough stable eye samples ({n_valid}/{_MIN_COLLECT}). "
                "Try again."
            )
            logger.warning("[CAL5] Point %d rejected: only %d valid samples.",
                           self._idx, n_valid)
            self._collecting = False
            self._collect_buf = []
            return False

        # Compute robust median across the collected window
        xs = [f[0] for f in self._collect_buf]
        ys = [f[1] for f in self._collect_buf]
        med_feat = (float(np.median(xs)), float(np.median(ys)))

        # Add a repeated copy so lstsq sees this point with reasonable weight
        for _ in range(12):
            self._cal.add_sample(self._idx, med_feat)

        logger.info("[CAL5] Point %d stored: (%.4f, %.4f)  from %d frames",
                    self._idx, *med_feat, n_valid)

        self._idx += 1
        self._collecting = False
        self._collect_buf = []

        if self._idx >= self.n_points:
            return self._finish()

        self.status_msg = "Look at the target and press SPACE"
        return False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_points(self) -> int:
        return len(self._cal.calibration_points)

    @property
    def current_target(self) -> tuple[int, int]:
        pts = self._cal.calibration_points
        if self._idx < len(pts):
            return pts[self._idx].screen_x, pts[self._idx].screen_y
        return (self._cal.win_w // 2, self._cal.win_h // 2)

    @property
    def is_collecting(self) -> bool:
        return self._collecting

    # ── Internal ──────────────────────────────────────────────────────────────

    def _finish(self) -> bool:
        result = self._cal.fit_and_validate(
            max_error_px=_QUALITY_THRESH,
            prev_matrix=self._prev_matrix,
        )
        if result == 'ok':
            self.status_msg = "Calibration complete!"
        elif result == 'poor':
            self.status_msg = "Calibration poor. Please try again."
        else:
            self.status_msg = "Calibration FAILED – press U to retry."
        self.active = False
        logger.info("[CAL5] Finished — result: %s", result)
        return True   # signals to caller that the session is over

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw_overlay(self, canvas: "np.ndarray",
                     gaze_feat: tuple[float, float] | None = None) -> None:
        """Draw calibration UI on top of *canvas* in-place.

        Pass the current *gaze_feat* to display the live numeric debug value
        and to show "no iris" feedback when it is None.
        """
        try:
            import cv2 as _cv2
        except ImportError:
            return

        if not self.active or self._idx >= self.n_points:
            return

        H, W = canvas.shape[:2]

        # Dim the scene
        overlay = canvas.copy()
        overlay[:] = (8, 10, 18)
        _cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        cx, cy = self.current_target

        # Target marker
        _cv2.drawMarker(canvas, (cx, cy), (255, 255, 255),
                        _cv2.MARKER_CROSS, 40, 2, _cv2.LINE_AA)
        _cv2.circle(canvas, (cx, cy), 18, (80, 220, 120), 2, _cv2.LINE_AA)
        _cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1, _cv2.LINE_AA)

        # If collecting: show progress ring
        if self._collecting:
            elapsed = time.time() - self._collect_start
            progress = min(elapsed / _COLLECT_S, 1.0)
            angle = int(360 * progress)
            _cv2.ellipse(canvas, (cx, cy), (26, 26), -90, 0, angle,
                         (255, 200, 60), 3, _cv2.LINE_AA)

        # Header text
        _cv2.putText(canvas, "GAZE CALIBRATION MODE",
                     (W // 2 - 162, 58),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 120), 2, _cv2.LINE_AA)
        _cv2.putText(canvas, "Keep your head still",
                     (W // 2 - 110, 86),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 220, 160), 1, _cv2.LINE_AA)

        # Dynamic status (changes with collection state)
        status_color = (255, 200, 60) if self._collecting else (200, 200, 200)
        _cv2.putText(canvas, self.status_msg,
                     (W // 2 - 220, 112),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1, _cv2.LINE_AA)

        # Point counter near target
        _cv2.putText(canvas, f"Point {self._idx + 1} / {self.n_points}",
                     (cx - 42, cy - 28),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, _cv2.LINE_AA)

        # Bottom: live eye-relative feature or error message
        if gaze_feat is not None:
            _cv2.putText(
                canvas,
                f"EyeFeat: {gaze_feat[0]:.3f}, {gaze_feat[1]:.3f}",
                (10, H - 18),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 180, 120), 1, _cv2.LINE_AA,
            )
        else:
            _cv2.putText(
                canvas, "No iris detected – open your eyes wider",
                (10, H - 18),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 220), 1, _cv2.LINE_AA,
            )
