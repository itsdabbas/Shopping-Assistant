"""
Gaze Modülü Test Harness — DwellTracker + infer_gaze_event
===========================================================
Gereksinimleri olmayan bağımsız testler.
Çalıştır: python test_gaze_server.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Gaze_Detection-main klasöründen import çalışsın
sys.path.insert(0, str(Path(__file__).parent))

from dwell_tracker import DwellTracker, DwellEvent


# ── Renk çıktısı ──────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

_passed = 0
_failed = 0


def _check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {GREEN}✓{RESET} {name}")
    else:
        _failed += 1
        print(f"  {RED}✗{RESET} {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n{BOLD}{YELLOW}── {title}{RESET}")


# ─────────────────────────────────────────────────────────────────────
# 1. Dwell-to-lock: LOCK 1 s sonra üretilmeli
# ─────────────────────────────────────────────────────────────────────
section("1. Dwell-to-lock: LOCK after 1 s fixation")

tracker = DwellTracker(dwell_to_lock_s=1.0)
now = 1000.0

# 0.0 s — ilk frame, henüz kilitleme yok
e = tracker.update("grid-0-1", ts=now)
_check("Frame @0.0s → no event yet", e is None, str(e))

# 0.5 s — hâlâ dwell süresi dolmadı
e = tracker.update("grid-0-1", ts=now + 0.5)
_check("Frame @0.5s → no event yet", e is None, str(e))

# 1.0 s — dwell süresi doldu → LOCK bekleniyor
e = tracker.update("grid-0-1", ts=now + 1.0)
_check("Frame @1.0s → LOCK emitted", e is not None and e.type == "LOCK", str(e))
_check("LOCK payload has target_id", e is not None and e.payload.get("target_id") == "grid-0-1")

# ─────────────────────────────────────────────────────────────────────
# 2. Hedef değişince UNLOCK üretilmeli
# ─────────────────────────────────────────────────────────────────────
section("2. Target change after LOCK → UNLOCK")

tracker = DwellTracker(dwell_to_lock_s=1.0)
now = 2000.0

tracker.update("grid-0-0", ts=now)
tracker.update("grid-0-0", ts=now + 1.1)   # LOCK tetikle

e = tracker.update("grid-0-1", ts=now + 1.5)  # farklı hedef
_check("Target change → UNLOCK", e is not None and e.type == "UNLOCK", str(e))
_check("UNLOCK has previous_target", e is not None and e.payload.get("previous_target") == "grid-0-0")
_check("State is now idle", tracker.state == "idle")

# ─────────────────────────────────────────────────────────────────────
# 3. Yüz yoksa UNLOCK (sadece locked'daysa)
# ─────────────────────────────────────────────────────────────────────
section("3. Face lost (target_id=None) after LOCK → UNLOCK")

tracker = DwellTracker(dwell_to_lock_s=1.0)
now = 3000.0

tracker.update("grid-0-2", ts=now)
tracker.update("grid-0-2", ts=now + 1.1)   # LOCK

e = tracker.update(None, ts=now + 1.5)     # yüz kayboldu
_check("Face lost → UNLOCK", e is not None and e.type == "UNLOCK", str(e))

# ─────────────────────────────────────────────────────────────────────
# 4. Ambiguity: hızlı geçişler AMBIGUOUS üretmeli
# ─────────────────────────────────────────────────────────────────────
section("4. Rapid target switches → AMBIGUOUS")

tracker = DwellTracker(dwell_to_lock_s=1.0, ambiguous_switch_count=3)
now = 4000.0

events = []
# Kısa aralıklarla 3 farklı hedefe bak
targets = ["grid-0-0", "grid-0-1", "grid-0-2", "grid-0-0"]
ts = now
for t in targets:
    ev = tracker.update(t, ts=ts)
    if ev:
        events.append(ev)
    ts += 0.15  # 150 ms aralık — dwell süresinden kısa

ambiguous_events = [ev for ev in events if ev.type == "AMBIGUOUS"]
_check("Rapid switches → AMBIGUOUS emitted", len(ambiguous_events) >= 1,
       f"events={[e.type for e in events]}")
if ambiguous_events:
    cands = ambiguous_events[0].payload.get("candidates", [])
    _check("AMBIGUOUS has candidates list", len(cands) >= 2, str(cands))

# ─────────────────────────────────────────────────────────────────────
# 5. AMBIGUOUS'tan LOCK'a geçiş (tek hedefe odaklanma)
# ─────────────────────────────────────────────────────────────────────
section("5. Resolve AMBIGUOUS by holding one target")

tracker = DwellTracker(dwell_to_lock_s=1.0, ambiguous_switch_count=3)
now = 5000.0

# Önce ambiguous tetikle
for t, offset in [("grid-0-0", 0), ("grid-0-1", 0.15), ("grid-0-2", 0.30), ("grid-0-0", 0.45)]:
    tracker.update(t, ts=now + offset)

# Şimdi bir hedefte 1 s dwell
tracker.update("grid-1-1", ts=now + 0.6)
e = tracker.update("grid-1-1", ts=now + 1.7)

_check("Resolved AMBIGUOUS → LOCK", e is not None and e.type == "LOCK", str(e))
if e and e.type == "LOCK":
    _check("Resolved LOCK target correct", e.payload.get("target_id") == "grid-1-1")
    _check("LOCK payload marks resolved_from_ambiguous",
           e.payload.get("resolved_from_ambiguous") is True)

# ─────────────────────────────────────────────────────────────────────
# 6. Henüz kilitli değilken None gelmesi — olay üretmemeli
# ─────────────────────────────────────────────────────────────────────
section("6. Face lost before LOCK → no UNLOCK event")

tracker = DwellTracker(dwell_to_lock_s=1.0)
now = 6000.0

tracker.update("grid-0-0", ts=now)       # başladı ama kilitlenmedi
e = tracker.update(None, ts=now + 0.3)  # yüz kayboldu
_check("No face before LOCK → no event", e is None, str(e))
_check("State remains idle", tracker.state == "idle")

# ─────────────────────────────────────────────────────────────────────
# 7. Reset — tüm state sıfırlanmalı
# ─────────────────────────────────────────────────────────────────────
section("7. reset() clears all state")

tracker = DwellTracker(dwell_to_lock_s=1.0)
now = 7000.0

tracker.update("grid-0-0", ts=now)
tracker.update("grid-0-0", ts=now + 1.1)  # LOCK
_check("State is locked before reset", tracker.state == "locked")

tracker.reset()
_check("State is idle after reset", tracker.state == "idle")
_check("locked_target is None after reset", tracker.locked_target is None)

# ─────────────────────────────────────────────────────────────────────
# 8. locked_target property
# ─────────────────────────────────────────────────────────────────────
section("8. locked_target property")

tracker = DwellTracker(dwell_to_lock_s=1.0)
now = 8000.0

_check("locked_target is None when idle", tracker.locked_target is None)
tracker.update("grid-0-2", ts=now)
tracker.update("grid-0-2", ts=now + 1.1)
_check("locked_target is set after LOCK", tracker.locked_target == "grid-0-2")
tracker.update(None, ts=now + 2)
_check("locked_target is None after UNLOCK", tracker.locked_target is None)

# ─────────────────────────────────────────────────────────────────────
# Sonuç
# ─────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}Result: {GREEN}{_passed} passed{RESET}{BOLD}, "
      f"{RED}{_failed} failed{RESET}")
if _failed:
    sys.exit(1)
