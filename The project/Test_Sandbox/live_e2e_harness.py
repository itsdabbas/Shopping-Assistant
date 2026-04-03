"""
End-to-End Test Harness for GazeShop MMUI
=========================================
Bu araç GazeShop Toolkit test edilebilmesi için kurgulanmıştır.
İki modu bulunur:
1. LIVE MODE (--mode live): OpenCV kamerasını açar, mikrofon PTT okur.
2. SCRIPTED MODE (--mode scripted): CLI üzerinden hazırlanan senaryoları deterministik 
   şekilde yürütür ve istatistikleri çıkarır.
"""

import argparse
import sys
import os
import time
import threading
import json
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np

# --- YOL AYARLAMALARI ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLKIT_DIR = PROJECT_ROOT / "Speech_Detection-main"
GAZE_SERVER_DIR = PROJECT_ROOT / "Gaze_Detection-main"

sys.path.insert(0, str(TOOLKIT_DIR))
sys.path.insert(0, str(GAZE_SERVER_DIR))

# Toolkit Sınıfları
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, GazeEvent, GazeEventType, SpeechEventType
from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
from gazeshop.toolkit.fusion_engine import FusionEngine
from gazeshop.toolkit.dialogue_manager import DialogueManager
from gazeshop.toolkit.telemetry import TelemetryLogger

# Gaze Sınıfları (Eski sistemdeki algoritmadan sadece tespit katmanı alınır)
try:
    from dwell_tracker import DwellTracker as OldDwellTracker
    from gaze_server import GazeTracker, GazeCalibrator, draw_face_debug
except ImportError:
    print("UYARI: GazeTracker modülleri Gaze_Detection-main altında bulunamadı.")
    GazeTracker = None

# ─── BBOX TANIMLAMALARI (LIVE MODE İÇİN) ───
# Eklenecek sanal kutular (Normalized coordinates: 0.0 - 1.0)
VIRTUAL_BBOXES = {
    "item_1": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4, "color": (200,0,0)},
    "item_2": {"x1": 0.6, "y1": 0.1, "x2": 0.9, "y2": 0.4, "color": (0,200,0)},
    "item_3": {"x1": 0.1, "y1": 0.6, "x2": 0.4, "y2": 0.9, "color": (0,0,200)},
    "item_4": {"x1": 0.6, "y1": 0.6, "x2": 0.9, "y2": 0.9, "color": (200,200,0)},
}

def get_target_for_gaze(gx: float, gy: float) -> str | None:
    """Normalize x,y (0-1) koordinatından virtual bbox eşleştirmesi."""
    for item_id, box in VIRTUAL_BBOXES.items():
        if box["x1"] <= gx <= box["x2"] and box["y1"] <= gy <= box["y2"]:
            return item_id
    return None


class HarnessSystem:
    def __init__(self, mode: str):
        self.mode = mode
        self.bus = EventBus()
        self.config = Config(
            FUSION_TIME_WINDOW_S=3.0, 
            GAZE_DWELL_TO_LOCK_S=1.0, 
            LOCK_TTL_S=4.0,
            ENABLE_TELEMETRY=True,
            TELEMETRY_EXPORT_PATH=f"logs/harness_{int(time.time())}.jsonl",
            ASR_ENGINE="whisper",
            WHISPER_MODEL_SIZE="base",
            MIN_UTTERANCE_MS=100,       # 100 milisaniye bile olsa es geçme!
            ENERGY_THRESHOLD=0.0        # Mikrofon ne kadar fısıltılı olsa da her sesi yakala
        )
        
        # SCRIPTED simülasyonları ve canlı takip için sayaçlar
        self.stats = {
            "commands": 0,
            "prompts": 0,
            "disambiguations": 0,
            "confirmations": 0,
            "cancels": 0
        }
        self.ui_state = {
            "gaze_lock": None,
            "prompt": "",
            "flash": "",
            "flash_time": 0
        }
        
        self.setup_components()
        
    def setup_components(self):
        # UI/Harness Logger (Log to screen and update state)
        self.bus.subscribe("MultimodalCommandEvent", self._h_command)
        self.bus.subscribe("PromptEvent", self._h_prompt)
        self.bus.subscribe("DisambiguationPromptEvent", self._h_disambiguation)
        self.bus.subscribe("ConfirmationPromptEvent", self._h_confirmation)
        self.bus.subscribe("ActionCancelledEvent", self._h_cancel)
        self.bus.subscribe("TargetLockedEvent", self._h_locked)
        self.bus.subscribe("TargetUnlockedEvent", self._h_unlocked)
        self.bus.subscribe("TargetExpiredEvent", self._h_expired)
        
        # Audio/Speech Logları (Sessiz hataları görmek için)
        self.bus.subscribe("SpeechEvent", lambda e: print(
            f"\033[{'91' if e.type == SpeechEventType.ERROR else '36'}m"
            f">>> [SPEECH_LOG] {e.type.value} | {e.payload} | {getattr(e, 'transcript', '')}\033[0m"
        ))

        # Gaze Dwell Tracker'ı (Arayüz adaptörü simüle ederiz)
        self.dwell = OldDwellTracker(dwell_to_lock_s=self.config.GAZE_DWELL_TO_LOCK_S, ambiguous_switch_count=3)
        self.speech = SpeechAdapter(self.bus, self.config)
        self.engine = FusionEngine(self.bus, self.config)
        self.dm = DialogueManager(self.bus, self.config)
        self.telemetry = TelemetryLogger(self.bus, self.config)

    def _ui_flash(self, msg: str):
        self.ui_state["flash"] = msg
        self.ui_state["flash_time"] = time.time() + 2.0
        print(f"\033[93m >>> HARNESS_EVENT: {msg}\033[0m")

    def _h_command(self, e): 
        self.stats["commands"] += 1
        self._ui_flash(f"CMD_EXEC: {e.intent} on {e.target_id}")
        self.ui_state["prompt"] = ""
    def _h_prompt(self, e): 
        self.stats["prompts"] += 1
        self.ui_state["prompt"] = f"PROMPT: {e.message}"
        self._ui_flash(self.ui_state["prompt"])
    def _h_disambiguation(self, e): 
        self.stats["disambiguations"] += 1
        self.ui_state["prompt"] = f"DISAMBIG: {e.message}"
        self._ui_flash(self.ui_state["prompt"])
    def _h_confirmation(self, e): 
        self.stats["confirmations"] += 1
        self.ui_state["prompt"] = f"CONFIRM: {e.message}"
        self._ui_flash(self.ui_state["prompt"])
    def _h_cancel(self, e): 
        self.stats["cancels"] += 1
        self.ui_state["prompt"] = f"CANCELLED: {e.message}"
        self._ui_flash(self.ui_state["prompt"])
    def _h_locked(self, e):
        self.ui_state["gaze_lock"] = e.target_id
    def _h_unlocked(self, e):
        self.ui_state["gaze_lock"] = None
    def _h_expired(self, e):
        self._ui_flash("LOCK EXPIRED")
        self.ui_state["gaze_lock"] = None

    def emit_dwell(self, target_id: str | None, ts: float):
        """Update old DwellTracker model and generate proper framework GazeEvents"""
        ev = self.dwell.update(target_id=target_id, ts=ts)
        if ev:
            # Map Old DwellEvent to Framework GazeEvent
            if ev.type == "LOCK":
                ge = GazeEvent(timestamp=ts, type=GazeEventType.LOCK, payload={"target_id": self.dwell.locked_target}, confidence=1.0)
            elif ev.type == "UNLOCK":
                ge = GazeEvent(timestamp=ts, type=GazeEventType.UNLOCK, payload={}, confidence=1.0)
            elif ev.type == "AMBIGUOUS":
                cands = ev.payload.get("candidates", [])
                
                # Dinamik olarak Ambiguity yarat, sağ/sol pozisyonu ver
                fmt = [{"id": cid, "pos": "left" if i==0 else "right"} for i, cid in enumerate(cands)]
                ge = GazeEvent(timestamp=ts, type=GazeEventType.AMBIGUOUS, payload={"candidates": fmt}, confidence=0.5)
            else:
                return
            
            self.bus.emit(ge)


# ==========================================
# 1. LIVE MODE (OpenCV CAMERA)
# ==========================================
def run_live(harness: HarnessSystem):
    if GazeTracker is None:
        print("MediaPipe Gaze_Detection not found. Exiting.")
        return
        
    print("\n[KAMERA BAŞLATILIYOR]")
    print(" - 'm' tuşuna BİR KERE basarak konuşun.")
    print(" - Sonuçları beklemek için terminali veya OpenCV penceresini takip edin.")
    print(" - Çıkış için OpenCV üzerinde 'q' tuşuna basın.")
    
    cap = cv2.VideoCapture(0)
    tracker = GazeTracker()
    
    print("[INFO] Speech Module initialized. Starting ASR Engine (Whisper) ...")
    harness.speech.start()
    
    mic_active = False
    last_m_press = 0.0

    def audio_task(action="press"):
        """Speech adapter calls are moved to a local thread to avoid OpenCV freezing"""
        try:
            if action == "press":
                # Ensure speech adapter knows if it should expect dialog interactions (yes, no, etc.)
                harness.speech.set_dialog_active(harness.dm.state != "IDLE")
                harness.speech.on_ptt_press()
            else:
                harness.speech.on_ptt_release()
        except Exception as e:
            print(f"Ses hatası: {e}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            frame = cv2.flip(frame, 1) # Mirror mode
            h, w = frame.shape[:2]
            
            # Gaze Prediction
            result = tracker.process(frame)
            gx, gy = result.get("gaze_x"), result.get("gaze_y")
            
            # Kalibrasyon Atlandığı için doğrudan Kafanın (Yüzün) merkezini ekran yüzdesine çeviriyoruz (Çok daha stabil)
            if gx is None and gy is None:
                debug_info = result.get("debug", {})
                fb = debug_info.get("face_box")
                if isinstance(fb, list) and len(fb) == 4:
                    gx = (fb[0] + fb[2]) / 2.0
                    gy = (fb[1] + fb[3]) / 2.0
            
            # Overlay Drawings
            debug_frame = draw_face_debug(frame, result)
            
            ts = time.time()
            if gx is not None and gy is not None:
                # Ekranda bakış hedefini (Imleci) belirten Mavi Nokta Çizelim
                px, py = int(gx * w), int(gy * h)
                cv2.circle(debug_frame, (px, py), 12, (255, 0, 0), -1)
                cv2.circle(debug_frame, (px, py), 6, (0, 255, 255), -1)
                
                target_id = get_target_for_gaze(gx, gy)
                harness.emit_dwell(target_id, ts)
            else:
                harness.emit_dwell(None, ts)
            
            # Draw Virtual Shelves
            for key, box in VIRTUAL_BBOXES.items():
                x1, y1 = int(box["x1"]*w), int(box["y1"]*h)
                x2, y2 = int(box["x2"]*w), int(box["y2"]*h)
                
                # Active Target Color
                color = (255,255,255) if harness.ui_state["gaze_lock"] == key else box["color"]
                thickness = 4 if harness.ui_state["gaze_lock"] == key else 2
                
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(debug_frame, key, (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw UI Prompts
            if time.time() < harness.ui_state["flash_time"]:
                cv2.putText(debug_frame, harness.ui_state["flash"], (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                
            if harness.ui_state["prompt"]:
                cv2.putText(debug_frame, harness.ui_state["prompt"], (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
            if mic_active:
                cv2.circle(debug_frame, (w - 30, 30), 10, (0,0,255), -1)
                cv2.putText(debug_frame, "MIC LISTENING", (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("GazeShop E2E Live Harness", debug_frame)
            
            # Keyboard Inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                current_time = time.time()
                if current_time - last_m_press > 0.5:
                    last_m_press = current_time
                    if not mic_active:
                        threading.Thread(target=audio_task, args=("press",)).start()
                        mic_active = True
                    else:
                        threading.Thread(target=audio_task, args=("release",)).start()
                        mic_active = False

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==========================================
# 2. SCRIPTED MODE (DETERMINISTIC SIMULATION)
# ==========================================
def cmd_say(harness: HarnessSystem, payload: str):
    harness.speech.set_dialog_active(harness.dm.state != "IDLE")
    harness.speech.process_text(payload)

def cmd_lock(harness: HarnessSystem, payload: str):
    ge = GazeEvent(timestamp=time.time(), type=GazeEventType.LOCK, payload={"target_id": payload}, confidence=1.0)
    harness.bus.emit(ge)

def cmd_unlock(harness: HarnessSystem, payload: str):
    ge = GazeEvent(timestamp=time.time(), type=GazeEventType.UNLOCK, payload={}, confidence=1.0)
    harness.bus.emit(ge)

def cmd_ambig(harness: HarnessSystem, payload: str):
    cands = payload.split()
    fmt = [{"id": cid, "pos": "left" if i==0 else "right"} for i, cid in enumerate(cands)]
    ge = GazeEvent(timestamp=time.time(), type=GazeEventType.AMBIGUOUS, payload={"candidates": fmt}, confidence=0.5)
    harness.bus.emit(ge)

def cmd_sleep(harness: HarnessSystem, payload: str):
    time.sleep(float(payload))


SCRIPT_SCENARIOS = [
    # a) LOCK + object INTENT → command
    [":lock item_1", ":say add to cart"],
    
    # b) no LOCK + object INTENT → WAIT_TARGET prompt
    [":unlock", ":say show details"],
    
    # c) AMBIGUOUS + intent → disambiguation prompt → REPAIR → command
    [":ambig item_3 item_4", ":say buy this", ":sleep 0.1", ":say right"],
    
    # d) low confidence (simulated via manual event for strict determination, or just text intent if supported)
    # the harness config supports text intent easily, we just emit a raw confirmable speech event.
    [":lock test_c", ":say delete this"], # actually "delete this" usually has high conf in text simulation
    
    # e) stale lock TTL → “expired” prompt → tekrar lock iste
    [":lock old_item", ":sleep 4.5", ":say save this"],
]

def run_scripted(harness: HarnessSystem):
    print("\n=== STARTING SCRIPTED SCENARIOS ===")
    funcs = {
        ":lock": cmd_lock,
        ":unlock": cmd_unlock,
        ":say": cmd_say,
        ":ambig": cmd_ambig,
        ":sleep": cmd_sleep,
    }
    
    for i, scenario in enumerate(SCRIPT_SCENARIOS):
        print(f"\n[Test Case {i+1}] Running: {scenario}")
        for step in scenario:
            parts = step.split(" ", 1)
            cmd = parts[0]
            payload = parts[1] if len(parts) > 1 else ""
            
            if cmd in funcs:
                funcs[cmd](harness, payload)
            time.sleep(0.05) # EventBus processing time
                
    print("\n=== SCRIPTED SIMULATION RESULTS ===")
    for k, v in harness.stats.items():
        print(f"  {k:15}: {v}")
    print("\n[INFO] End-to-End telemetry is written to /logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GazeShop E2E Test Harness")
    parser.add_argument("--mode", choices=["live", "scripted"], default="scripted", help="live (Kamera+Mikrofon) or scripted (Deterministik Test)")
    
    args = parser.parse_args()
    
    harness = HarnessSystem(args.mode)
    if args.mode == "live":
        run_live(harness)
    else:
        run_scripted(harness)
