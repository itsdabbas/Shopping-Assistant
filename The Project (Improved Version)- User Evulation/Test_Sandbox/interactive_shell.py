"""
Interactive Multimodal Sandbox
==============================

Bu interaktif konsol aracı, GazeShop MMUI toolkit mimarisini 
kendi ellerinizle test etmeniz için tasarlanmıştır.

Kullanımı:
  - Bakış kilidi (gaze lock) oluşturabilirsiniz.
  - Metin tabanlı komut yollayabilirsiniz (isim / niyetler).
  - Mikrofonunuzdan ses kaydederek gerçek ASR ile sistemi test edebilirsiniz.
  - DM (Dialog Manager) ve Fusion Motorunun ürettiği olayları anlık görürsünüz.
"""

import sys
import os
import time
from pathlib import Path

# Proje içindeki Toolkit modüllerini bulabilmesi için sys.path eklemesi:
PROJECT_DIR = Path(__file__).resolve().parent.parent / "Speech_Detection-main"
sys.path.insert(0, str(PROJECT_DIR))

try:
    from gazeshop.toolkit.config import Config
    from gazeshop.toolkit.event_bus import EventBus
    from gazeshop.toolkit.adapters.gaze_adapter import GazeAdapterStub
    from gazeshop.toolkit.adapters.speech_adapter import SpeechAdapter
    from gazeshop.toolkit.fusion_engine import FusionEngine
    from gazeshop.toolkit.dialogue_manager import DialogueManager
except ImportError as e:
    print(f"Modüller yüklenemedi. Klasör yollarını kontrol edin: {e}")
    sys.exit(1)


def print_help():
    print("\n\033[93m" + "="*50)
    print(" 🛠️ GazeShop - INTERAKTIF TEST ORTAMI")
    print("="*50 + "\033[0m")
    print(" \033[1m[ GAZE KOMUTLARI ]\033[0m")
    print("  g lock <hedef_id>   : Hedefe bakış kilitler (Örn: g lock laptop_1)")
    print("  g unlock            : Bakışı hedeften çeker")
    print("  g ambig <c1> <c2>   : Kararsız bakış simüle eder (Örn: g ambig sag_urun sol_urun)")
    print("\n \033[1m[ SPEECH KOMUTLARI ]\033[0m")
    print("  s text <komut>      : Metin ile ses komutu yollar (Örn: s text add to cart)")
    print("  s mic               : Bas-konuş Mikrofonu AÇAR / KAPATIR")
    print("\n \033[1m[ SISTEM KONTROL ]\033[0m")
    print("  state               : Diyalog ve Fusion anlık durumunu gösterir")
    print("  help                : Yardım menüsü")
    print("  exit                : Çıkış")
    print("\033[93m" + "="*50 + "\033[0m\n")


def main():
    print("\033[36mSistem bileşenleri başlatılıyor...\033[0m")
    bus = EventBus()
    config = Config(FUSION_TIME_WINDOW_S=2.0, GAZE_DWELL_TO_LOCK_S=1.0, LOCK_TTL_S=4.0)

    # Bağdaştırıcılar
    gaze = GazeAdapterStub(event_bus=bus)
    speech = SpeechAdapter(event_bus=bus, config=config)
    engine = FusionEngine(event_bus=bus, config=config)
    dm = DialogueManager(event_bus=bus, config=config)

    # Kullanıcı Arayüzü İçin Event Çıktılayıcı
    def log_event(color, tag, msg):
        print(f"\n\033[{color}m[{tag}]\033[0m \033[1m{msg}\033[0m\n> ", end="", flush=True)

    bus.subscribe("MultimodalCommandEvent", lambda e: log_event("42;97", "🎯 BAŞARILI! (EXECUTE)", f"{e.intent} on target: {e.target_id}"))
    bus.subscribe("PromptEvent", lambda e: log_event("95", "🤖 SISTEM", e.message))
    bus.subscribe("DisambiguationPromptEvent", lambda e: log_event("96", "A/B SEÇİMİ", e.message))
    bus.subscribe("ConfirmationPromptEvent", lambda e: log_event("96", "ONAY (E/H)", e.message))
    bus.subscribe("ActionCancelledEvent", lambda e: log_event("91", "İPTAL EDİLDİ", e.message))
    bus.subscribe("TargetLockedEvent", lambda e: log_event("90", "👁️ GAZE", f"Kilitlendi: {e.target_id}"))
    bus.subscribe("TargetUnlockedEvent", lambda e: log_event("90", "👁️ GAZE", f"Kilit Kalktı"))
    bus.subscribe("TargetExpiredEvent", lambda e: log_event("91", "⏳ ZAMAN AŞIMI", f"Target TTL Bitti: {e.target_id}"))

    gaze.start()
    time.sleep(0.5)
    
    print_help()
    
    mic_active = False

    while True:
        try:
            cmd = input("> ").strip()
            if not cmd:
                continue
                
            parts = cmd.split()
            base = parts[0].lower()
            
            if base in ["exit", "quit", "q"]:
                print("Çıkılıyor...")
                speech.stop()
                gaze.stop()
                break
                
            elif base == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_help()
                
            elif base == "help":
                print_help()
                
            elif base == "state":
                print(f"\n  \033[44;97m[SİSTEM DURUMU]\033[0m")
                print(f"  Dialogue Manager : {dm.state}")
                print(f"  Fusion Gaze Lock : {engine._gaze_state} (Target: {engine._locked_target})\n")
                
            elif base == "g":
                if len(parts) < 2:
                    print("⚠️ Eksik komut. Örn: g lock item_1")
                    continue
                action = parts[1].lower()
                
                if action == "lock" and len(parts) >= 3:
                    target = " ".join(parts[2:])
                    gaze.simulate_lock(target)
                elif action == "unlock":
                    gaze.simulate_unlock()
                elif action == "ambig" and len(parts) >= 4:
                    gaze.simulate_ambiguous(parts[2:])
                else:
                    print("⚠️ Gaze komutu anlaşılamadı.")
                    
            elif base == "s":
                if len(parts) < 2:
                    print("⚠️ Eksik komut. Örn: s text add to cart")
                    continue
                action = parts[1].lower()
                
                if action == "text" and len(parts) >= 3:
                    text = " ".join(parts[2:])
                    # Eğer DM aktifse, speech adapter'ın intent resolver'ını repair öncelikli hale getir.
                    speech.set_dialog_active(dm.state != "IDLE")
                    speech.process_text(text)
                    
                elif action == "mic":
                    if not mic_active:
                        speech.on_ptt_press()
                        mic_active = True
                        print("\n🎙️ \033[91mMikrofon DİNLİYOR...\033[0m (Kapatmak için tekrar 's mic' yazın)\n> ", end="")
                    else:
                        speech.on_ptt_release()
                        mic_active = False
                        print("\n⏳ Ses işleniyor...\n> ", end="")
                else:
                    print("⚠️ Speech komutu anlaşılamadı.")
            else:
                print("⚠️ Bilinmeyen komut. Yardım için 'help' yazabilirsiniz.")
                
        except KeyboardInterrupt:
            print("\nÇıkılıyor...")
            speech.stop()
            break
        except Exception as e:
            print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    main()
