"""
Speech Adapter
==============

Bridges the microphone hardware and ASR engine to the toolkit's
``EventBus`` via standardised ``SpeechEvent`` messages.

PTT Toggle State Machine
------------------------
IDLE ──(begin_listening)──► LISTENING ──(end_listening)──► PROCESSING ──► IDLE
                                                                     └──► ERROR ──► IDLE

Design:
* Toggle only: first M-press = begin_listening(), second M-press = end_listening().
* No hold detection, no tail-silence sleep (user taps key intentionally).
* Stream is stopped synchronously in end_listening() before buffer snapshot
  → no race between callback appends and buffer read.
* Single persistent worker thread drains a queue → no concurrent ASR.
* PTT press rejected when pipeline is PROCESSING (not IDLE).
* 15-second ASR hard-timeout.
* Full timestamp logging for every pipeline stage.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time as _time
from enum import Enum
from typing import Any

import numpy as np

from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.intent_parser import IntentParser

logger = logging.getLogger(__name__)

# ── Vosk grammar vocabulary (grammar mode restricts recognition to these phrases)
_VOSK_VOCAB_JSON: str = json.dumps([
    "add to cart",
    "add",
    "cart",
    "card",
    "two cart",
    "two card",
    "to cart",
    "to card",
    "show details",
    "find similar",
    "compare items",
    "next item",
    "go back",
    "zoom in please",
    "read summary",
    "get help",
    "confirm yes",
    "cancel no",
    "select item",
    "[unk]",
])

# ── PTT State Machine ─────────────────────────────────────────────────────────

class PipelineState(str, Enum):
    IDLE       = "IDLE"        # ready for PTT press
    LISTENING  = "LISTENING"   # recording audio
    PROCESSING = "PROCESSING"  # ASR + intent running
    ERROR      = "ERROR"       # transient error; worker recovers → IDLE


# ── SpeechAdapter ─────────────────────────────────────────────────────────────

class SpeechAdapter(ModalityAdapter):
    """Speech modality adapter: PTT → Audio → ASR → Intent → Event.

    The audio processing pipeline runs in a **dedicated worker thread** so the
    UI / render thread is NEVER blocked.  PTT release captures audio, drops
    it into a queue, and returns immediately.

    Parameters
    ----------
    event_bus:
        Shared event dispatcher.
    config:
        Toolkit configuration.
    intent_parser:
        Optional pre-configured IntentParser.
    """

    # Maximum seconds we wait for the ASR engine before force-aborting.
    _ASR_TIMEOUT_S: float = 15.0

    def __init__(
        self,
        event_bus: EventBus,
        config: Config | None = None,
        intent_parser: IntentParser | None = None,
    ) -> None:
        super().__init__(event_bus)
        self.config = config or Config()
        self.parser = intent_parser or IntentParser(self.config)

        # ── Pipeline state machine ─────────────────────────────────────
        self._pipeline_state: PipelineState = PipelineState.IDLE
        self._state_lock = threading.Lock()

        # ── Audio capture ──────────────────────────────────────────────
        self._audio_buffer: list[np.ndarray] = []
        self._record_start: float = 0.0
        self._recording_active: bool = False   # checked by audio callback
        self._stream: Any = None
        self._stream_lock = threading.Lock()   # protects _stream

        # ── Worker thread / queue ──────────────────────────────────────
        # maxsize=1: if a job is already queued we drop the new one
        self._audio_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker_thread: threading.Thread | None = None

        # ── ASR engine ─────────────────────────────────────────────────
        self._asr_engine: Any = None
        self._vosk_model: Any = None        # keep Model; recreate Recognizer per session
        self._asr_initialised: bool = False

        # ── Silero VAD (lazy) ──────────────────────────────────────────
        self._silero_model = None
        self._silero_get_speech_ts = None
        self._silero_collect_chunks = None
        self._silero_available: bool | None = None

        # ── PTT key listener ───────────────────────────────────────────
        self._key_listener: Any = None

        # ── Dialog context ─────────────────────────────────────────────
        self._dialog_active: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            logger.warning("SpeechAdapter already running.")
            return

        self._init_asr()
        self._start_key_listener()
        self._running = True

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="SpeechWorker",
        )
        self._worker_thread.start()

        logger.info(
            "SpeechAdapter started  PTT_MODE=%s  ASR=%s  worker=%s",
            self.config.PTT_MODE, self.config.ASR_ENGINE,
            self._worker_thread.name,
        )

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        # Unblock worker with sentinel
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass

        # Stop key listener
        if self._key_listener is not None:
            try:
                self._key_listener.stop()
            except Exception:
                pass
            self._key_listener = None

        # Stop audio stream
        self._force_stop_stream()

        # Wait for worker (3 s max)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

        logger.info("SpeechAdapter stopped.")

    # ── State helpers ──────────────────────────────────────────────────────────

    def _set_state(self, new_state: PipelineState) -> None:
        old = self._pipeline_state
        self._pipeline_state = new_state
        if old != new_state:
            logger.debug("[PTT-SM] %s → %s  (t=%.3f)", old.value, new_state.value, _time.time())

    @property
    def pipeline_state(self) -> PipelineState:
        return self._pipeline_state

    # ── PTT press / release ────────────────────────────────────────────────────

    # ── Toggle PTT public API ──────────────────────────────────────────────────

    def begin_listening(self) -> None:
        """Toggle ON: start recording.  Call when user taps PTT key the first time."""
        ts = _time.time()
        with self._state_lock:
            if self._pipeline_state != PipelineState.IDLE:
                logger.warning(
                    "[PTT TOGGLE ON  %.3f] Ignored – state=%s",
                    ts, self._pipeline_state.value,
                )
                return
            logger.info("[PTT TOGGLE ON  %.3f] -> LISTENING", ts)
            self._set_state(PipelineState.LISTENING)
            self._audio_buffer.clear()
            self._record_start = ts
            self._recording_active = True

        # Outside lock: emit event then open audio stream
        self.event_bus.emit(SpeechEvent(
            timestamp=ts, type=SpeechEventType.LISTENING, payload={},
        ))
        self._start_audio_stream()

    def end_listening(self) -> None:
        """Toggle OFF: stop recording, snapshot buffer, enqueue for ASR.

        Steps (no tail-silence sleep; user tapped the key deliberately):
        1. Set _recording_active=False under lock  → callback stops appending.
        2. Stop stream synchronously outside lock   → last callback completes.
        3. Snapshot + clear _audio_buffer           → no stale frames next time.
        4. Transition PROCESSING and enqueue audio.
        Always returns to IDLE even on empty/short audio.
        """
        ts = _time.time()
        with self._state_lock:
            if self._pipeline_state != PipelineState.LISTENING:
                logger.warning(
                    "[PTT TOGGLE OFF %.3f] Ignored – state=%s",
                    ts, self._pipeline_state.value,
                )
                return
            duration = ts - self._record_start
            logger.info(
                "[PTT TOGGLE OFF %.3f] Recorded %.2fs -> PROCESSING", ts, duration
            )
            self._recording_active = False  # step 1: stop callback appending

        # Step 2: stop stream synchronously (waits for in-flight callback)
        self._force_stop_stream()

        # Step 3: snapshot + clear buffer (safe – no more callbacks running)
        buf_snapshot = list(self._audio_buffer)
        self._audio_buffer.clear()

        # Build audio array
        if buf_snapshot:
            audio = np.concatenate(buf_snapshot, axis=0).flatten()
        else:
            audio = np.array([], dtype=np.int16)

        logger.info(
            "[PTT AUDIO %.3f] Buffer=%d samples  duration=%.2fs",
            _time.time(), len(audio),
            len(audio) / max(self.config.SAMPLE_RATE, 1),
        )

        # Transition to PROCESSING
        with self._state_lock:
            self._set_state(PipelineState.PROCESSING)

        # Enqueue (drop + recover gracefully if worker busy)
        try:
            self._audio_queue.put_nowait(audio)
        except queue.Full:
            logger.error("[PTT] Audio queue full – dropping utterance.")
            with self._state_lock:
                self._set_state(PipelineState.IDLE)
            self._emit_stopped()  # reset UI mic indicator

    # ── Legacy shims (backward compat) ────────────────────────────────────────

    def on_ptt_press(self) -> None:
        """Shim: forwards to begin_listening() (toggle mode always on)."""
        self.begin_listening()

    def on_ptt_release(self) -> None:
        """Shim: no-op in toggle mode (stop is triggered by second press)."""
        pass  # end_listening() is called by second key-press in CV demos

    # ── Audio stream ───────────────────────────────────────────────────────────

    def _start_audio_stream(self) -> None:
        try:
            import sounddevice as sd

            with self._stream_lock:
                stream = sd.InputStream(
                    samplerate=self.config.SAMPLE_RATE,
                    channels=self.config.CHANNELS,
                    dtype="int16",
                    callback=self._audio_callback,
                    blocksize=int(self.config.SAMPLE_RATE * 0.1),
                )
                stream.start()
                self._stream = stream
            logger.debug("[STREAM] Started  t=%.3f", _time.time())

        except Exception as exc:
            logger.error("[STREAM] Failed to start: %s", exc)
            self.event_bus.emit(SpeechEvent(
                timestamp=_time.time(), type=SpeechEventType.ERROR,
                payload={"error": f"Audio stream error: {exc}"},
            ))

    def _force_stop_stream(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                logger.debug("[STREAM] Stopped  t=%.3f", _time.time())

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.warning("[STREAM CB] %s", status)

        if not self._recording_active:
            return

        self._audio_buffer.append(indata.copy())

        # Max utterance guard
        elapsed_ms = (_time.time() - self._record_start) * 1000
        if elapsed_ms > self.config.MAX_UTTERANCE_MS:
            logger.warning("[STREAM CB] Max utterance exceeded – auto-stopping.")
            self._recording_active = False
            # queue stop in a thread (can't call complex logic from sd callback)
            threading.Thread(
                target=self._tail_then_queue,
                args=(list(self._audio_buffer), self._record_start),
                daemon=True,
            ).start()

    # ── Worker thread ──────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Persistent worker: drain queue, run full pipeline, return to IDLE."""
        logger.info("[WORKER] Started.")
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio is None:   # sentinel → shutdown
                break

            logger.debug("[WORKER] Job received  size=%d samples", len(audio))
            try:
                self._run_pipeline(audio)
            except Exception:
                logger.exception("[WORKER] Unhandled exception in pipeline.")
            finally:
                with self._state_lock:
                    if self._pipeline_state == PipelineState.PROCESSING:
                        self._set_state(PipelineState.IDLE)
                self._emit_stopped()
                self._audio_queue.task_done()

        logger.info("[WORKER] Exiting.")

    # ── Audio processing pipeline ──────────────────────────────────────────────

    def _run_pipeline(self, audio: np.ndarray) -> None:
        """Full pipeline: validate → VAD → ASR → intent → emit."""
        t0 = _time.time()

        # Duration check
        duration_s = len(audio) / max(self.config.SAMPLE_RATE, 1)
        if duration_s * 1000 < self.config.MIN_UTTERANCE_MS:
            logger.debug("[PIPELINE] Too short (%.0f ms) – discarded.", duration_s * 1000)
            return

        # Energy gate
        if len(audio) > 0:
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            logger.debug("[PIPELINE] RMS=%.1f", rms)
            if rms < self.config.ENERGY_THRESHOLD:
                logger.debug("[PIPELINE] Below energy threshold – discarded.")
                return

        # VAD
        if self.config.VAD_ENABLED:
            audio = self._apply_vad(audio)
            if len(audio) == 0:
                logger.debug("[PIPELINE] VAD removed all audio – discarded.")
                return

        # ASR with hard timeout
        logger.info("[ASR] Start  t=%.3f", _time.time())
        transcript, conf = self._transcribe_with_timeout(audio)
        logger.info("[ASR] End    t=%.3f  transcript=%r  conf=%.2f",
                    _time.time(), transcript, conf)

        if not transcript or not transcript.strip():
            self.event_bus.emit(SpeechEvent(
                timestamp=_time.time(), type=SpeechEventType.ERROR,
                payload={"error": "ASR returned empty transcript."},
                transcript="", confidence=0.0,
            ))
            return

        # Intent parse
        logger.info("[INTENT] Parse start  t=%.3f", _time.time())
        event = self.parser.parse(
            transcript,
            asr_confidence=conf,
            dialog_active=self._dialog_active,
        )
        logger.info("[INTENT] Parse end    t=%.3f  intent=%s",
                    _time.time(), event.payload.get("intent"))

        # EventBus emission
        logger.info("[BUS] Emit SpeechEvent  t=%.3f", _time.time())
        self.event_bus.emit(event)

        logger.info("[PIPELINE] Total=%.2fs", _time.time() - t0)

    def _transcribe_with_timeout(self, audio: np.ndarray) -> tuple[str, float]:
        """Run ASR in a sub-thread with a hard timeout."""
        result: list = [None, None, None]   # [transcript, conf, exception]

        def _run():
            try:
                t, c = self._transcribe(audio)
                result[0], result[1] = t, c
            except Exception as exc:
                result[2] = exc

        t = threading.Thread(target=_run, daemon=True, name="ASR")
        t.start()
        t.join(timeout=self._ASR_TIMEOUT_S)

        if t.is_alive():
            logger.error("[ASR] Timeout after %.0fs – aborting.", self._ASR_TIMEOUT_S)
            raise TimeoutError("ASR exceeded timeout.")

        if result[2] is not None:
            raise result[2]

        return result[0] or "", result[1] or 0.0

    def _emit_stopped(self) -> None:
        self.event_bus.emit(SpeechEvent(
            timestamp=_time.time(), type=SpeechEventType.STOPPED, payload={},
        ))

    # ── ASR engines ───────────────────────────────────────────────────────────

    def _init_asr(self) -> None:
        if self._asr_initialised:
            return
        engine = self.config.ASR_ENGINE
        if engine == "vosk":
            self._init_vosk()
        elif engine == "whisper":
            self._init_whisper()
        else:
            raise ValueError(f"Unsupported ASR engine: {engine!r}")

    def _init_vosk(self) -> None:
        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)
            logger.info("Loading Vosk model from: %s", self.config.VOSK_MODEL_PATH)
            self._vosk_model = Model(self.config.VOSK_MODEL_PATH)
            # Create recognizer (will be recreated each session via _reset_vosk)
            self._asr_engine = KaldiRecognizer(self._vosk_model, self.config.SAMPLE_RATE, _VOSK_VOCAB_JSON)
            self._asr_initialised = True
            logger.info("Vosk ASR ready.")
        except ImportError:
            logger.error("vosk not installed.  pip install vosk")
            raise
        except Exception as exc:
            logger.error("Failed to load Vosk model: %s", exc)
            raise

    def _init_whisper(self) -> None:
        try:
            import whisper
            logger.info("Loading Whisper model: %s", self.config.WHISPER_MODEL_SIZE)
            self._asr_engine = whisper.load_model(self.config.WHISPER_MODEL_SIZE)
            self._asr_initialised = True
            logger.info("Whisper ASR ready.")
        except ImportError:
            logger.error("openai-whisper not installed.")
            raise

    def _reset_vosk_recognizer(self) -> None:
        """Recreate the Vosk recognizer to clear internal state between sessions."""
        try:
            from vosk import KaldiRecognizer
            self._asr_engine = KaldiRecognizer(self._vosk_model, self.config.SAMPLE_RATE, _VOSK_VOCAB_JSON)
            logger.debug("[VOSK] Recognizer reset.")
        except Exception as exc:
            logger.error("[VOSK] Reset failed: %s", exc)

    def _transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        engine = self.config.ASR_ENGINE
        if engine == "vosk":
            return self._transcribe_vosk(audio)
        elif engine == "whisper":
            return self._transcribe_whisper(audio)
        raise ValueError(f"Unsupported ASR engine: {engine!r}")

    def _transcribe_vosk(self, audio: np.ndarray) -> tuple[str, float]:
        # *** CRITICAL FIX: reset recognizer before each new utterance ***
        self._reset_vosk_recognizer()
        rec = self._asr_engine
        rec.AcceptWaveform(audio.tobytes())
        result = json.loads(rec.FinalResult())
        transcript = result.get("text", "").strip()
        word_results = result.get("result", [])
        if word_results:
            mean_conf = sum(w.get("conf", 0.0) for w in word_results) / len(word_results)
        else:
            mean_conf = 0.5
        return transcript, mean_conf

    def _transcribe_whisper(self, audio: np.ndarray) -> tuple[str, float]:
        import math, whisper
        audio_f32 = audio.astype(np.float32) / 32768.0
        result = self._asr_engine.transcribe(
            audio_f32,
            language="en",
            fp16=False,
            initial_prompt=self.config.WHISPER_INITIAL_PROMPT,
            temperature=self.config.WHISPER_TEMPERATURE,
            condition_on_previous_text=self.config.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            no_speech_threshold=self.config.WHISPER_NO_SPEECH_THRESHOLD,
            word_timestamps=False,
        )
        segments = result.get("segments", [])
        speech_segs = [s for s in segments
                       if s.get("no_speech_prob", 0.0) < self.config.WHISPER_NO_SPEECH_THRESHOLD]
        if not speech_segs and segments:
            return "", 0.0
        active = speech_segs or segments
        transcript = " ".join(s.get("text", "").strip() for s in active).strip() if active \
            else result.get("text", "").strip()
        if active:
            avg_lp = sum(s.get("avg_logprob", -1.0) for s in active) / len(active)
            conf = max(0.0, min(1.0, math.exp(avg_lp)))
        else:
            conf = 0.5
        return transcript, conf

    # ── Silero VAD ─────────────────────────────────────────────────────────────

    def _init_silero_vad(self) -> bool:
        if self._silero_available is not None:
            return self._silero_available
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks
            self._silero_model = load_silero_vad()
            self._silero_get_speech_ts = get_speech_timestamps
            self._silero_collect_chunks = collect_chunks
            self._silero_available = True
            logger.info("Silero VAD loaded.")
        except ImportError:
            logger.warning("silero-vad not installed – falling back to WebRTC VAD.")
            self._silero_available = False
        except Exception as exc:
            logger.warning("Silero VAD unavailable (%s).", exc)
            self._silero_available = False
        return bool(self._silero_available)

    def _vad_trim_silero(self, audio: np.ndarray) -> np.ndarray:
        if not self._init_silero_vad():
            return audio
        if len(audio) == 0:
            return audio
        try:
            import torch
            audio_f32 = audio.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio_f32)
            ts = self._silero_get_speech_ts(
                tensor, self._silero_model,
                sampling_rate=self.config.SAMPLE_RATE,
                threshold=self.config.SILERO_VAD_THRESHOLD,
                min_speech_duration_ms=self.config.SILERO_VAD_MIN_SPEECH_MS,
                speech_pad_ms=self.config.SILERO_VAD_PADDING_MS,
            )
            if not ts:
                return audio
            chunks = self._silero_collect_chunks(ts, tensor)
            trimmed = np.clip(chunks.numpy() * 32768.0, -32768, 32767).astype(np.int16)
            logger.debug("Silero VAD: %d→%d samples.", len(audio), len(trimmed))
            return trimmed
        except Exception as exc:
            logger.warning("Silero VAD trim failed (%s).", exc)
            return audio

    def _vad_trim_webrtc(self, audio: np.ndarray) -> np.ndarray:
        try:
            import webrtcvad
        except ImportError:
            return audio
        if len(audio) == 0:
            return audio
        try:
            vad = webrtcvad.Vad(self.config.VAD_AGGRESSIVENESS)
            frame_samples = int(self.config.SAMPLE_RATE * 30 / 1000)
            frames = [audio[i:i + frame_samples]
                      for i in range(0, len(audio) - frame_samples + 1, frame_samples)]
            if not frames:
                return audio
            speech = [f for f in frames
                      if vad.is_speech(f.tobytes(), self.config.SAMPLE_RATE)]
            return np.concatenate(speech) if speech else audio
        except Exception as exc:
            logger.warning("WebRTC VAD failed (%s).", exc)
            return audio

    def _apply_vad(self, audio: np.ndarray) -> np.ndarray:
        backend = getattr(self.config, "VAD_BACKEND", "silero")
        if backend == "none":
            return audio
        if backend == "silero":
            return self._vad_trim_silero(audio) if self._init_silero_vad() \
                else self._vad_trim_webrtc(audio)
        if backend == "webrtc":
            return self._vad_trim_webrtc(audio)
        return audio

    # ── PTT key listener (pynput) ──────────────────────────────────────────────

    def _start_key_listener(self) -> None:
        # CV demos set PTT_KEY="" to drive PTT via cv2.waitKey instead.
        if not self.config.PTT_KEY:
            logger.info("PTT_KEY is empty – pynput listener skipped (CV demo mode).")
            return
        try:
            from pynput import keyboard
            ptt_key_name = self.config.PTT_KEY
            try:
                target_key = getattr(keyboard.Key, ptt_key_name)
            except AttributeError:
                target_key = keyboard.KeyCode.from_char(ptt_key_name)

            def on_press(key):
                if key == target_key:
                    self.on_ptt_press()

            def on_release(key):
                if key == target_key:
                    self.on_ptt_release()

            self._key_listener = keyboard.Listener(
                on_press=on_press, on_release=on_release,
            )
            self._key_listener.daemon = True
            self._key_listener.start()
            logger.info("PTT key listener started (key=%s).", ptt_key_name)
        except ImportError:
            logger.warning(
                "pynput not installed – PTT key listener disabled. "
                "Call on_ptt_press() / on_ptt_release() programmatically."
            )

    # ── Dialog context ─────────────────────────────────────────────────────────

    def set_dialog_active(self, active: bool) -> None:
        self._dialog_active = active
        logger.debug("Dialog context: %s", active)

    # ── Programmatic trigger (tests / demos) ───────────────────────────────────

    def process_text(self, transcript: str, asr_confidence: float = 0.9) -> SpeechEvent:
        """Bypass audio capture and parse a transcript string directly."""
        event = self.parser.parse(
            transcript,
            asr_confidence=asr_confidence,
            dialog_active=self._dialog_active,
        )
        self.event_bus.emit(event)
        return event

    # ── Legacy shim (kept for backward compat) ─────────────────────────────────
    # Old code checked self._is_recording; map to new flag.
    @property
    def _is_recording(self) -> bool:
        return self._pipeline_state == PipelineState.LISTENING

    @_is_recording.setter
    def _is_recording(self, val: bool) -> None:
        pass   # state machine manages this now
