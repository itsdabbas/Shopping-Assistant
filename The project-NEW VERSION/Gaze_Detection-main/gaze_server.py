"""
Gaze tracking server — MediaPipe Face Landmarker + OpenCV + WebSocket.

Uses the Tasks API (Face Landmarker). Legacy mp.solutions.face_mesh is not
available in mediapipe 0.10+ wheels (e.g. Python 3.13).

Install: pip install opencv-python mediapipe websockets numpy
Run:     python gaze_server.py

Optional env: GAZE_WS_HOST (default 127.0.0.1), GAZE_WS_PORT (default 8765).
If port is busy (WinError 10048), close the other server or set GAZE_WS_PORT.

On first run, face_landmarker.task is downloaded next to this script.

Controls:
  Ctrl+C in terminal - stop server
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import websockets
from dwell_tracker import DwellTracker
from fusion import FusionEngine
from intents import INTENTS, CONFIRM_PATTERNS, DENY_PATTERNS, REPAIR_PATTERNS, REPEAT_PATTERNS

# --- Calibration and tracking ---

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_face_landmarker_model() -> Path:
    """Download Google's face_landmarker.task if missing (once per machine)."""
    path = Path(__file__).resolve().parent / "face_landmarker.task"
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    print(f"[Model] Downloading Face Landmarker to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("[Model] Ready.")
    return path


LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


class GazeCalibrator:
    """9-point calibration: iris feature -> normalized screen (0-1).

    Collects multiple samples per point and fits polynomial regression.
    """

    SCREEN_POINTS = [
        (0.1, 0.1),
        (0.5, 0.1),
        (0.9, 0.1),
        (0.1, 0.5),
        (0.5, 0.5),
        (0.9, 0.5),
        (0.1, 0.9),
        (0.5, 0.9),
        (0.9, 0.9),
    ]
    SAMPLES_PER_POINT = 3

    def __init__(self) -> None:
        self.iris_samples: list[tuple[float, float]] = []
        self.screen_targets: list[tuple[float, float]] = []
        self.current_idx = 0
        self._pending_samples: list[tuple[float, float]] = []
        self.calibrated = False
        self.coef_x: np.ndarray | None = None
        self.coef_y: np.ndarray | None = None

    def is_done(self) -> bool:
        return self.current_idx >= len(self.SCREEN_POINTS)

    def current_target(self) -> tuple[float, float] | None:
        if self.is_done():
            return None
        return self.SCREEN_POINTS[self.current_idx]

    def samples_at_current(self) -> int:
        return len(self._pending_samples)

    def add_sample(self, iris_x: float, iris_y: float) -> None:
        target = self.current_target()
        if target is None:
            return
        self._pending_samples.append((iris_x, iris_y))
        print(
            f"[Calibration] Point {self.current_idx + 1} sample "
            f"{len(self._pending_samples)}/{self.SAMPLES_PER_POINT}"
        )

        if len(self._pending_samples) >= self.SAMPLES_PER_POINT:
            avg_x = sum(s[0] for s in self._pending_samples) / len(self._pending_samples)
            avg_y = sum(s[1] for s in self._pending_samples) / len(self._pending_samples)
            self.iris_samples.append((avg_x, avg_y))
            self.screen_targets.append(target)
            self._pending_samples = []
            self.current_idx += 1
            print(f"[Calibration] Point {self.current_idx}/{len(self.SCREEN_POINTS)} committed.")
            if self.is_done():
                self._fit()

    def _poly_features(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Degree-2 polynomial features: [1, x, y, x^2, xy, y^2]."""
        return np.column_stack(
            [
                np.ones_like(ix),
                ix,
                iy,
                ix**2,
                ix * iy,
                iy**2,
            ]
        )

    def _fit(self) -> None:
        ix = np.array([s[0] for s in self.iris_samples], dtype=np.float64)
        iy = np.array([s[1] for s in self.iris_samples], dtype=np.float64)
        sx = np.array([t[0] for t in self.screen_targets], dtype=np.float64)
        sy = np.array([t[1] for t in self.screen_targets], dtype=np.float64)

        a = self._poly_features(ix, iy)
        lam = 0.01
        ata = a.T @ a + lam * np.eye(a.shape[1])
        self.coef_x = np.linalg.solve(ata, a.T @ sx)
        self.coef_y = np.linalg.solve(ata, a.T @ sy)
        self.calibrated = True
        print("[Calibration] Done. Polynomial (degree=2) mapping fitted.")

    def predict(self, iris_x: float, iris_y: float) -> tuple[float, float] | None:
        if not self.calibrated or self.coef_x is None or self.coef_y is None:
            return None
        ix = np.array([iris_x], dtype=np.float64)
        iy = np.array([iris_y], dtype=np.float64)
        feat = self._poly_features(ix, iy)
        sx = float(feat @ self.coef_x)
        sy = float(feat @ self.coef_y)
        sx = max(0.0, min(1.0, sx))
        sy = max(0.0, min(1.0, sy))
        return sx, sy


class GazeTracker:
    def __init__(self) -> None:
        model_path = str(ensure_face_landmarker_model())
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._video_ts_ms = 0
        self.calibrator = GazeCalibrator()
        self.ema_x: float | None = None
        self.ema_y: float | None = None
        self.alpha = 0.40

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _iris_center(
        self,
        landmarks: Any,
        indices: list[int],
        img_w: int,
        img_h: int,
    ) -> tuple[float, float]:
        pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return cx, cy

    def _eye_ratio(
        self,
        landmarks: Any,
        inner: int,
        outer: int,
        top: int,
        bottom: int,
        img_w: int,
        img_h: int,
        iris_x: float,
        iris_y: float,
    ) -> tuple[float, float]:
        ex1 = landmarks[outer].x * img_w
        ex2 = landmarks[inner].x * img_w
        ey1 = landmarks[top].y * img_h
        ey2 = landmarks[bottom].y * img_h
        eye_w = abs(ex2 - ex1)
        eye_h = abs(ey2 - ey1)
        if eye_w < 1 or eye_h < 1:
            return 0.5, 0.5
        rx = (iris_x - min(ex1, ex2)) / eye_w
        ry = (iris_y - min(ey1, ey2)) / eye_h
        return max(0.0, min(1.0, rx)), max(0.0, min(1.0, ry))

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._video_ts_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect_for_video(mp_image, self._video_ts_ms)

        if not results.face_landmarks:
            return {
                "status": "no_face",
                "cal_target": self.calibrator.current_target(),
                "cal_step": self.calibrator.current_idx,
                "cal_total": len(GazeCalibrator.SCREEN_POINTS),
                "gaze_x": None,
                "gaze_y": None,
                "iris_raw": None,
            }

        lm = results.face_landmarks[0]
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        face_box = [
            max(0.0, min(xs)),
            max(0.0, min(ys)),
            min(1.0, max(xs)),
            min(1.0, max(ys)),
        ]

        lx, ly = self._iris_center(lm, LEFT_IRIS, w, h)
        rx, ry = self._iris_center(lm, RIGHT_IRIS, w, h)
        iris_x = (lx + rx) / 2
        iris_y = (ly + ry) / 2

        l_rx, l_ry = self._eye_ratio(
            lm,
            LEFT_EYE_INNER,
            LEFT_EYE_OUTER,
            LEFT_EYE_TOP,
            LEFT_EYE_BOTTOM,
            w,
            h,
            lx,
            ly,
        )
        r_rx, r_ry = self._eye_ratio(
            lm,
            RIGHT_EYE_INNER,
            RIGHT_EYE_OUTER,
            RIGHT_EYE_TOP,
            RIGHT_EYE_BOTTOM,
            w,
            h,
            rx,
            ry,
        )
        rel_x = (l_rx + r_rx) / 2
        rel_y = (l_ry + r_ry) / 2

        status = "calibrating" if not self.calibrator.is_done() else "tracking"

        gaze_x: float | None = None
        gaze_y: float | None = None
        if self.calibrator.calibrated:
            pred = self.calibrator.predict(rel_x, rel_y)
            if pred is not None:
                raw_x, raw_y = pred
                if self.ema_x is None:
                    self.ema_x, self.ema_y = raw_x, raw_y
                else:
                    self.ema_x = self.alpha * raw_x + (1 - self.alpha) * self.ema_x
                    self.ema_y = self.alpha * raw_y + (1 - self.alpha) * self.ema_y
                gaze_x = round(self.ema_x, 4)
                gaze_y = round(self.ema_y, 4)

        cal_target = self.calibrator.current_target()
        return {
            "status": status,
            "cal_target": [cal_target[0], cal_target[1]] if cal_target else None,
            "cal_step": self.calibrator.current_idx,
            "cal_total": len(GazeCalibrator.SCREEN_POINTS),
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "iris_raw": [round(rel_x, 4), round(rel_y, 4)],
            "debug": {
                "face_box": [round(v, 4) for v in face_box],
                "left_iris": [round(lx / w, 4), round(ly / h, 4)],
                "right_iris": [round(rx / w, 4), round(ry / h, 4)],
            },
        }


def draw_face_debug(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw lightweight face/eye debug guides on preview frame."""
    out = frame.copy()
    iris = result.get("iris_raw")
    debug = result.get("debug") or {}
    status = result.get("status")

    h, w = out.shape[:2]
    # Draw target during calibration.
    target = result.get("cal_target")
    if target is not None:
        tx = int(float(target[0]) * w)
        ty = int(float(target[1]) * h)
        cv2.circle(out, (tx, ty), 10, (0, 0, 255), 2)

    if status == "no_face":
        cv2.putText(
            out,
            "NO FACE",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return out

    if iris and isinstance(iris, list) and len(iris) == 2:
        # iris_raw is normalized-in-eye-space [0..1], visualize as HUD only.
        cv2.putText(
            out,
            f"Iris rel: {iris[0]:.2f}, {iris[1]:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Draw face box and iris markers to confirm detection on the face.
    face_box = debug.get("face_box")
    if isinstance(face_box, list) and len(face_box) == 4:
        x1 = int(float(face_box[0]) * w)
        y1 = int(float(face_box[1]) * h)
        x2 = int(float(face_box[2]) * w)
        y2 = int(float(face_box[3]) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        cv2.rectangle(out, (4, 4), (w - 4, h - 4), (0, 255, 0), 2)

    for key, color in (("left_iris", (255, 255, 0)), ("right_iris", (0, 255, 255))):
        pt = debug.get(key)
        if isinstance(pt, list) and len(pt) == 2:
            px = int(float(pt[0]) * w)
            py = int(float(pt[1]) * h)
            cv2.circle(out, (px, py), 5, color, 2)

    cv2.putText(
        out,
        "FACE DETECTED",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return out


connected_clients: set[Any] = set()
pending_capture = False
pending_reset = False
fusion_engine = FusionEngine()

# Active gaze: dwell-to-lock tracker (1 s dwell, 3 fast switches → AMBIGUOUS)
dwell_tracker = DwellTracker(dwell_to_lock_s=1.0, ambiguous_switch_count=3)


def build_command_catalog() -> dict[str, Any]:
    object_bound = []
    global_cmds = []
    for spec in INTENTS:
        item = {
            "name": spec.name,
            "target_required": bool(spec.target_required),
            "patterns": list(spec.patterns),
        }
        if spec.target_required:
            object_bound.append(item)
        else:
            global_cmds.append(item)
    return {
        "object_bound": object_bound,
        "global": global_cmds,
        "dialog_only": {
            "CONFIRM": list(CONFIRM_PATTERNS),
            "DENY": list(DENY_PATTERNS),
            "REPAIR": list(REPAIR_PATTERNS),
            "REPEAT": list(REPEAT_PATTERNS),
        },
    }


async def send_capabilities(websocket: Any) -> None:
    payload = {
        "type": "CAPABILITIES",
        "timestamp": time.time(),
        "payload": {
            "speech_ready": True,
            "accepts_transcript_command": True,
            "command_catalog": build_command_catalog(),
        },
    }
    await websocket.send(json.dumps(payload))


def infer_gaze_event(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a tracker frame result into a DwellTracker-driven gaze event.

    Active gaze: a LOCK is only emitted after the user fixates the same
    grid cell for >= dwell_to_lock_s seconds.  Fast switches produce
    AMBIGUOUS; face loss produces UNLOCK.
    """
    status = result.get("status")
    ts = time.time()

    if status == "no_face":
        dwell_event = dwell_tracker.update(target_id=None, ts=ts)
    else:
        gx = result.get("gaze_x")
        gy = result.get("gaze_y")
        if gx is None or gy is None:
            dwell_event = dwell_tracker.update(target_id=None, ts=ts)
        else:
            # Map normalised (gx, gy) → stable grid cell ID
            col = min(2, max(0, int(float(gx) * 3.0)))
            row = min(1, max(0, int(float(gy) * 2.0)))
            target_id = f"grid-{row}-{col}"
            dwell_event = dwell_tracker.update(target_id=target_id, ts=ts)

    if dwell_event is None:
        # No state change this frame — report current state without triggering
        state = dwell_tracker.state
        if state == "locked":
            return {
                "timestamp": ts,
                "type": "LOCK",
                "payload": {"target_id": dwell_tracker.locked_target},
            }
        return {"timestamp": ts, "type": "IDLE", "payload": {}}

    return {
        "timestamp": dwell_event.timestamp,
        "type": dwell_event.type,
        "payload": dwell_event.payload,
        "confidence": dwell_event.confidence,
    }


async def ws_handler(websocket: Any) -> None:
    global pending_capture, pending_reset
    connected_clients.add(websocket)
    print(f"[WS] Connected: {getattr(websocket, 'remote_address', '?')}")
    try:
        await send_capabilities(websocket)
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            cmd = msg.get("cmd")
            if cmd == "capture":
                pending_capture = True
            elif cmd == "reset":
                pending_reset = True
            elif cmd == "speech_transcript":
                transcript = str(msg.get("transcript", "")).strip()
                confidence = float(msg.get("confidence", 0.9))
                if not transcript:
                    continue
                speech_event, fused = fusion_engine.on_speech_transcript(
                    transcript=transcript,
                    confidence=confidence,
                    ts=time.time(),
                )
                if speech_event is not None:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "SPEECH_EVENT",
                                "timestamp": speech_event.timestamp,
                                "payload": speech_event.__dict__,
                            }
                        )
                    )
                if fused is not None:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "FUSION_EVENT",
                                "timestamp": fused.get("timestamp", time.time()),
                                "payload": fused,
                            }
                        )
                    )
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print("[WS] Disconnected")


def result_to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


async def broadcast(data: dict[str, Any]) -> None:
    if not connected_clients:
        return
    payload = json.dumps(result_to_jsonable(data))
    await asyncio.gather(
        *[ws.send(payload) for ws in list(connected_clients)],
        return_exceptions=True,
    )


async def camera_loop(tracker: GazeTracker) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    global pending_capture, pending_reset
    print("[Camera] Running in headless mode (no OpenCV window).")
    print("[Calibration] Use web page controls to capture each point.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.03)
                continue

            # Keep native camera orientation (no mirror flip).
            result = tracker.process(frame)

            if pending_reset:
                tracker.calibrator = GazeCalibrator()
                tracker.ema_x = None
                tracker.ema_y = None
                dwell_tracker.reset()
                pending_reset = False
                print("[Calibration] Reset.")

            if pending_capture:
                pending_capture = False
                if tracker.calibrator.is_done():
                    print("[Calibration] Already complete.")
                else:
                    iris_raw = result.get("iris_raw")
                    if iris_raw and result["status"] != "no_face":
                        tracker.calibrator.add_sample(float(iris_raw[0]), float(iris_raw[1]))
                        n_samples = tracker.calibrator.samples_at_current()
                        n_points = tracker.calibrator.current_idx
                        total = len(GazeCalibrator.SCREEN_POINTS)
                        if n_samples > 0:
                            print(
                                f"[Calibration] Sample {n_samples}/"
                                f"{GazeCalibrator.SAMPLES_PER_POINT} for point {n_points + 1}"
                            )
                        if tracker.calibrator.is_done():
                            print("[Calibration] Complete. Tracking active.")
                    else:
                        print("[Warning] No face; capture ignored.")

            ts = time.time()
            debug_frame = draw_face_debug(frame, result)
            preview = cv2.resize(debug_frame, (320, 180))
            ok, jpg = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            frame_jpg = base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None
            await broadcast(
                {
                    **result_to_jsonable(result),
                    "t": ts,
                    "server": "mediapipe",
                    "frame_jpg": frame_jpg,
                    "gaze_event": infer_gaze_event(result),
                }
            )
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("[Server] Camera loop cancelled.")
        raise
    finally:
        cap.release()


def _ws_host() -> str:
    return os.environ.get("GAZE_WS_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _ws_port() -> int:
    return int(os.environ.get("GAZE_WS_PORT", "8765").strip())


async def main() -> None:
    host = _ws_host()
    port = _ws_port()
    tracker = GazeTracker()
    try:
        async with websockets.serve(ws_handler, host, port):
            print(f"[WS] WebSocket listening on ws://{host}:{port}")
            await asyncio.gather(
                camera_loop(tracker),
                return_exceptions=True,
            )
    except asyncio.CancelledError:
        # Normal shutdown path; keep exit clean.
        pass
    except OSError as e:
        winerr = getattr(e, "winerror", None)
        if e.errno == 10048 or winerr == 10048:
            print(
                f"[ERROR] Port {port} is already in use. "
                "Stop the other gaze_server (or whatever holds that port), then retry.\n"
                "Or use another port, e.g. PowerShell:\n"
                '  $env:GAZE_WS_PORT="8766"; python gaze_server.py\n'
                "and point index.html at ws://localhost:8766"
            )
            raise SystemExit(1) from e
        raise
    finally:
        tracker.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Server] Stopped.")
