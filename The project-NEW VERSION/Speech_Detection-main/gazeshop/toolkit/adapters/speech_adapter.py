"""
Speech Adapter
==============

Bridges the microphone hardware and ASR engine to the toolkit's
``EventBus`` via standardised ``SpeechEvent`` messages.

Responsibilities
----------------
1. **Push-to-Talk management** — listen for PTT key press/release via
   ``pynput`` (or a programmatic trigger for testing).
2. **Audio capture** — record 16 kHz mono PCM via ``sounddevice``.
3. **Endpointing** — apply tail silence, min/max duration filters.
4. **ASR transcription** — run Vosk (default) or Whisper on the
   captured audio buffer.
5. **Intent parsing** — delegate to ``IntentParser`` and emit the
   resulting ``SpeechEvent`` on the ``EventBus``.

Integration notes
-----------------
* The adapter is designed to be **gaze-agnostic**: it knows nothing
  about gaze targets.  The ``FusionEngine`` (a later module) is
  responsible for binding speech intents to gaze targets.
* The ``SpeechEvent.payload.target_required`` field tells the
  FusionEngine whether a gaze target is needed.

Usage
-----
    from gazeshop.toolkit import Config, EventBus
    from gazeshop.toolkit.adapters import SpeechAdapter

    bus = EventBus()
    cfg = Config()
    adapter = SpeechAdapter(bus, cfg)
    adapter.start()     # begins listening for PTT key
    # ... application runs ...
    adapter.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time as _time
from typing import Any

import numpy as np

from gazeshop.toolkit.adapters.base_adapter import ModalityAdapter
from gazeshop.toolkit.config import Config
from gazeshop.toolkit.event_bus import EventBus, SpeechEvent, SpeechEventType
from gazeshop.toolkit.intent_parser import IntentParser

logger = logging.getLogger(__name__)


class SpeechAdapter(ModalityAdapter):
    """Speech modality adapter: PTT → Audio → ASR → Intent → Event.

    Parameters
    ----------
    event_bus:
        Shared event dispatcher.
    config:
        Toolkit configuration (sample rate, PTT mode, ASR engine, …).
    intent_parser:
        Optional pre-configured ``IntentParser``.  If ``None`` a
        default parser is created from *config*.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Config | None = None,
        intent_parser: IntentParser | None = None,
    ) -> None:
        super().__init__(event_bus)
        self.config = config or Config()
        self.parser = intent_parser or IntentParser(self.config)

        # Audio state
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording: bool = False
        self._record_start_time: float = 0.0
        self._stream: Any = None

        # ASR engine (lazy-initialised)
        self._asr_engine: Any = None
        self._asr_initialised: bool = False

        # Silero VAD (lazy-initialised on first use)
        self._silero_model: Any = None
        self._silero_get_speech_ts: Any = None
        self._silero_collect_chunks: Any = None
        self._silero_read_audio: Any = None
        self._silero_available: bool | None = None  # None = not yet attempted

        # PTT key listener
        self._key_listener: Any = None

        # Dialog context (set by DialogueManager when a dialog is open)
        self._dialog_active: bool = False

        # Lock for thread safety
        self._lock = threading.Lock()

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the PTT key listener and prepare the ASR engine.

        This does NOT start audio recording — recording is gated by
        the push-to-talk key.
        """
        if self._running:
            logger.warning("SpeechAdapter is already running.")
            return

        self._init_asr()
        self._start_key_listener()
        self._running = True
        logger.info(
            "SpeechAdapter started (PTT_MODE=%s, ASR=%s)",
            self.config.PTT_MODE,
            self.config.ASR_ENGINE,
        )

    def stop(self) -> None:
        """Stop listening, release audio resources, and join threads."""
        if not self._running:
            return

        self._running = False

        # Stop any in-progress recording
        if self._is_recording:
            self._stop_recording()

        # Stop key listener
        if self._key_listener is not None:
            try:
                self._key_listener.stop()
            except Exception:
                logger.debug("Key listener already stopped.")
            self._key_listener = None

        # Close audio stream
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        logger.info("SpeechAdapter stopped.")

    # ── PTT handling ────────────────────────────────────────────────

    def on_ptt_press(self) -> None:
        """Called when the PTT key is pressed (or toggled on).

        Starts audio capture and emits a ``LISTENING`` event.
        """
        with self._lock:
            if self._is_recording:
                if self.config.PTT_MODE == "toggle":
                    # Toggle mode: second press stops recording
                    self._on_ptt_release_internal()
                    return
                else:
                    return  # Already recording in hold mode

            logger.debug("PTT pressed — starting recording.")
            self._audio_buffer.clear()
            self._is_recording = True
            self._record_start_time = _time.time()

        # Emit LISTENING event
        self.event_bus.emit(SpeechEvent(
            timestamp=_time.time(),
            type=SpeechEventType.LISTENING,
            payload={},
        ))

        # Start the audio stream
        self._start_audio_stream()

    def on_ptt_release(self) -> None:
        """Called when the PTT key is released.

        In 'hold' mode this triggers processing.  In 'toggle' mode
        this is a no-op (the second press stops recording).
        """
        if self.config.PTT_MODE == "toggle":
            return  # Toggle mode: release does nothing

        with self._lock:
            self._on_ptt_release_internal()

    def _on_ptt_release_internal(self) -> None:
        """Internal release logic (called under lock)."""
        if not self._is_recording:
            return

        logger.debug("PTT released — processing audio.")
        self._is_recording = False

        # Tail silence: keep recording for a short duration
        tail_ms = self.config.TAIL_SILENCE_MS
        if tail_ms > 0:
            _time.sleep(tail_ms / 1000.0)

        # Stop the audio stream
        self._stop_recording()

        # Process in a separate thread to avoid blocking
        threading.Thread(
            target=self._process_audio,
            daemon=True,
        ).start()

    # ── Audio capture ───────────────────────────────────────────────

    def _start_audio_stream(self) -> None:
        """Open a ``sounddevice`` input stream."""
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self.config.SAMPLE_RATE,
                channels=self.config.CHANNELS,
                dtype="int16",
                callback=self._audio_callback,
                blocksize=int(self.config.SAMPLE_RATE * 0.1),  # 100 ms blocks
            )
            self._stream.start()
            logger.debug("Audio stream started.")
        except Exception as exc:
            logger.error("Failed to start audio stream: %s", exc)
            self.event_bus.emit(SpeechEvent(
                timestamp=_time.time(),
                type=SpeechEventType.ERROR,
                payload={"error": f"Audio stream error: {exc}"},
            ))

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Called by sounddevice for each audio block.

        Appends incoming audio to the buffer while recording is active.
        Also enforces the MAX_UTTERANCE guard.
        """
        if status:
            logger.warning("Audio stream status: %s", status)

        if self._is_recording:
            self._audio_buffer.append(indata.copy())

            # Max utterance guard
            elapsed_ms = (_time.time() - self._record_start_time) * 1000
            if elapsed_ms > self.config.MAX_UTTERANCE_MS:
                logger.warning(
                    "Max utterance length (%d ms) exceeded — auto-stopping.",
                    self.config.MAX_UTTERANCE_MS,
                )
                self._is_recording = False
                threading.Thread(
                    target=self._process_audio,
                    daemon=True,
                ).start()

    def _stop_recording(self) -> None:
        """Stop and close the active audio stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _finalize_audio(self) -> np.ndarray:
        """Concatenate buffered audio blocks into a single array."""
        if not self._audio_buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._audio_buffer, axis=0).flatten()

    # ── Silero VAD (neural, primary backend) ────────────────────────

    def _init_silero_vad(self) -> bool:
        """Lazy-load the Silero VAD model via the ``silero-vad`` pip package.

        Returns ``True`` on success, ``False`` if ``silero-vad`` or
        ``torch`` is unavailable (callers fall back to WebRTC VAD).

        The model weights are shipped inside the ``silero-vad`` package
        itself — no network download is required at runtime.  Subsequent
        calls are near-instantaneous (model is cached in instance attrs).

        Install: ``pip install silero-vad``
        """
        if self._silero_available is not None:
            return self._silero_available  # already attempted

        try:
            from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks

            model = load_silero_vad()

            self._silero_model = model
            self._silero_get_speech_ts = get_speech_timestamps
            self._silero_collect_chunks = collect_chunks
            self._silero_available = True
            logger.info("Silero VAD model loaded successfully (silero-vad pip package).")

        except ImportError:
            logger.warning(
                "silero-vad package not installed — falling back to WebRTC VAD. "
                "Install with: pip install silero-vad"
            )
            self._silero_available = False
        except Exception as exc:
            logger.warning(
                "Silero VAD unavailable (%s) — will fall back to WebRTC VAD.",
                exc,
            )
            self._silero_available = False

        return bool(self._silero_available)

    def _vad_trim_silero(self, audio: np.ndarray) -> np.ndarray:
        """Trim silence using the Silero VAD neural network model.

        Converts int16 audio to float32, runs the Silero model to obtain
        speech timestamps, adds ``SILERO_VAD_PADDING_MS`` of context
        around each segment, and returns only the speech portions
        concatenated together.

        Falls back to the original audio if:

        * The model is not available / failed to load.
        * No speech is detected (preserves original to avoid discarding
          valid but very quiet speech).
        * Any unexpected runtime error occurs.

        Parameters
        ----------
        audio:
            1-D **int16** NumPy array sampled at ``SAMPLE_RATE`` Hz.

        Returns
        -------
        np.ndarray
            Trimmed 1-D int16 array (may equal *audio* on fallback).
        """
        if not self._init_silero_vad():
            return audio  # silero unavailable — caller will fall back

        if len(audio) == 0:
            return audio

        try:
            import torch

            # Silero expects float32 in [-1, 1]
            audio_f32 = audio.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio_f32)

            padding_samples = int(
                self.config.SAMPLE_RATE * self.config.SILERO_VAD_PADDING_MS / 1000
            )
            min_speech_samples = int(
                self.config.SAMPLE_RATE * self.config.SILERO_VAD_MIN_SPEECH_MS / 1000
            )

            speech_timestamps = self._silero_get_speech_ts(
                tensor,
                self._silero_model,
                sampling_rate=self.config.SAMPLE_RATE,
                threshold=self.config.SILERO_VAD_THRESHOLD,
                min_speech_duration_ms=self.config.SILERO_VAD_MIN_SPEECH_MS,
                # Keep some context around each segment to avoid clipping
                speech_pad_ms=self.config.SILERO_VAD_PADDING_MS,
            )

            if not speech_timestamps:
                logger.debug(
                    "Silero VAD: no speech detected — keeping original audio "
                    "(threshold=%.2f).",
                    self.config.SILERO_VAD_THRESHOLD,
                )
                return audio

            # Concatenate all detected speech segments
            chunks = self._silero_collect_chunks(speech_timestamps, tensor)
            trimmed_f32 = chunks.numpy()

            # Convert back to int16
            trimmed = np.clip(
                trimmed_f32 * 32768.0, -32768, 32767
            ).astype(np.int16)

            retained_pct = 100.0 * len(trimmed) / max(len(audio), 1)
            logger.debug(
                "Silero VAD: %d → %d samples (%.0f%% retained, %d segment(s)).",
                len(audio), len(trimmed), retained_pct, len(speech_timestamps),
            )
            return trimmed

        except Exception as exc:
            logger.warning(
                "Silero VAD trim failed (%s) — using original audio.", exc
            )
            return audio

    # ── WebRTC VAD (rule-based, fallback backend) ────────────────────

    def _vad_trim_webrtc(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading/trailing silence using WebRTC VAD.

        Splits the audio into 30 ms frames and discards frames the VAD
        classifies as non-speech.  The surviving speech frames are
        concatenated and returned.

        Falls back to the original audio if:

        * ``webrtcvad`` is not installed.
        * No speech frames are found (avoids returning an empty buffer).
        * Any unexpected error occurs.

        Parameters
        ----------
        audio:
            1-D int16 NumPy array at ``SAMPLE_RATE`` Hz.

        Returns
        -------
        np.ndarray
            Trimmed 1-D int16 array (may equal *audio* on fallback).
        """
        try:
            import webrtcvad  # optional dependency
        except ImportError:
            logger.debug(
                "webrtcvad not installed — WebRTC VAD trim skipped. "
                "Install with: pip install webrtcvad"
            )
            return audio

        if len(audio) == 0:
            return audio

        try:
            vad = webrtcvad.Vad(self.config.VAD_AGGRESSIVENESS)

            # WebRTC VAD requires exactly 10 / 20 / 30 ms frames at
            # 8 000 / 16 000 / 32 000 / 48 000 Hz — we use 30 ms.
            frame_ms = 30
            frame_samples = int(self.config.SAMPLE_RATE * frame_ms / 1000)  # 480 @ 16 kHz

            # Build complete frames only (discard trailing incomplete frame)
            frames = [
                audio[i : i + frame_samples]
                for i in range(0, len(audio) - frame_samples + 1, frame_samples)
            ]

            if not frames:
                return audio

            speech_frames: list[np.ndarray] = []
            for frame in frames:
                try:
                    if vad.is_speech(frame.tobytes(), self.config.SAMPLE_RATE):
                        speech_frames.append(frame)
                except Exception as exc:  # pragma: no cover
                    logger.debug("WebRTC VAD frame error: %s — keeping frame.", exc)
                    speech_frames.append(frame)

            if not speech_frames:
                logger.debug("WebRTC VAD: no speech frames detected — keeping original audio.")
                return audio

            trimmed = np.concatenate(speech_frames)
            retained_pct = 100.0 * len(trimmed) / max(len(audio), 1)
            logger.debug(
                "WebRTC VAD trim: %d → %d samples (%.0f%% retained).",
                len(audio), len(trimmed), retained_pct,
            )
            return trimmed

        except Exception as exc:  # pragma: no cover
            logger.warning("WebRTC VAD trim failed (%s) — using original audio.", exc)
            return audio

    def _apply_vad(self, audio: np.ndarray) -> np.ndarray:
        """Dispatch to the configured VAD backend with automatic fallback.

        Chain: Silero VAD → WebRTC VAD → no trimming.

        The active backend is determined by ``config.VAD_BACKEND``:

        * ``'silero'``  — try Silero first, fall back to WebRTC if torch
          is unavailable, then no trimming.
        * ``'webrtc'``  — use WebRTC VAD only, no Silero attempt.
        * ``'none'``    — skip VAD entirely (same as VAD_ENABLED=False).

        Parameters
        ----------
        audio:
            1-D int16 NumPy array at ``SAMPLE_RATE`` Hz.

        Returns
        -------
        np.ndarray
            VAD-trimmed audio (may equal *audio* if no trimming applied).
        """
        backend = getattr(self.config, "VAD_BACKEND", "silero")

        if backend == "none":
            return audio

        if backend == "silero":
            if self._init_silero_vad():
                logger.debug("VAD: using Silero backend.")
                return self._vad_trim_silero(audio)
            else:
                logger.debug("VAD: Silero unavailable — falling back to WebRTC.")
                return self._vad_trim_webrtc(audio)

        if backend == "webrtc":
            logger.debug("VAD: using WebRTC backend.")
            return self._vad_trim_webrtc(audio)

        # Unknown backend — log and skip
        logger.warning("Unknown VAD_BACKEND %r — skipping VAD.", backend)
        return audio


    # ── Audio processing pipeline ───────────────────────────────────

    def _process_audio(self) -> None:
        """Full pipeline: finalise audio → validate → transcribe → parse → emit."""
        try:
            audio = self._finalize_audio()

            # Duration check
            duration_s = len(audio) / self.config.SAMPLE_RATE
            duration_ms = duration_s * 1000

            if duration_ms < self.config.MIN_UTTERANCE_MS:
                logger.debug(
                    "Utterance too short (%.0f ms < %d ms) — discarded.",
                    duration_ms,
                    self.config.MIN_UTTERANCE_MS,
                )
                return

            # Energy gate
            if len(audio) > 0:
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                if rms < self.config.ENERGY_THRESHOLD:
                    logger.debug("Audio below energy threshold (RMS=%.1f) — discarded.", rms)
                    return

            # VAD trim — strip silence before sending to ASR
            if self.config.VAD_ENABLED:
                audio = self._apply_vad(audio)
                if len(audio) == 0:
                    logger.debug("VAD trimmed entire buffer — no speech detected, discarding.")
                    return

            # ASR transcription
            transcript, asr_confidence = self._transcribe(audio)

            if not transcript or not transcript.strip():
                self.event_bus.emit(SpeechEvent(
                    timestamp=_time.time(),
                    type=SpeechEventType.ERROR,
                    payload={"error": "ASR returned empty transcript."},
                    transcript="",
                    confidence=0.0,
                ))
                return

            logger.info("Transcript: %r (ASR conf=%.2f)", transcript, asr_confidence)

            # Intent parsing
            event = self.parser.parse(
                transcript,
                asr_confidence=asr_confidence,
                dialog_active=self._dialog_active,
            )
            self.event_bus.emit(event)

        except Exception as exc:
            logger.exception("Error processing audio.")
            self.event_bus.emit(SpeechEvent(
                timestamp=_time.time(),
                type=SpeechEventType.ERROR,
                payload={"error": str(exc)},
            ))

        finally:
            self._emit_stopped()

    def _emit_stopped(self) -> None:
        """Emit a STOPPED event to signal the mic is off."""
        self.event_bus.emit(SpeechEvent(
            timestamp=_time.time(),
            type=SpeechEventType.STOPPED,
            payload={},
        ))

    # ── ASR engines ─────────────────────────────────────────────────

    def _init_asr(self) -> None:
        """Lazy-initialise the configured ASR engine."""
        if self._asr_initialised:
            return

        engine = self.config.ASR_ENGINE

        if engine == "vosk":
            try:
                from vosk import Model, KaldiRecognizer, SetLogLevel

                SetLogLevel(-1)  # Suppress Vosk logs
                logger.info(
                    "Loading Vosk model from: %s",
                    self.config.VOSK_MODEL_PATH,
                )
                model = Model(self.config.VOSK_MODEL_PATH)
                self._asr_engine = KaldiRecognizer(
                    model, self.config.SAMPLE_RATE,
                )
                self._asr_initialised = True
                logger.info("Vosk ASR engine ready.")
            except ImportError:
                logger.error(
                    "Vosk is not installed. Install with: pip install vosk"
                )
                raise
            except Exception as exc:
                logger.error("Failed to load Vosk model: %s", exc)
                raise

        elif engine == "whisper":
            try:
                import whisper

                logger.info(
                    "Loading Whisper model: %s",
                    self.config.WHISPER_MODEL_SIZE,
                )
                self._asr_engine = whisper.load_model(
                    self.config.WHISPER_MODEL_SIZE,
                )
                self._asr_initialised = True
                logger.info("Whisper ASR engine ready.")
            except ImportError:
                logger.error(
                    "Whisper is not installed. Install with: pip install openai-whisper"
                )
                raise

        else:
            raise ValueError(f"Unsupported ASR engine: {engine!r}")

    def _transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Run ASR on *audio*.  Returns ``(transcript, confidence)``.

        Parameters
        ----------
        audio:
            1-D int16 NumPy array at ``SAMPLE_RATE`` Hz.

        Returns
        -------
        tuple[str, float]
            The recognised text and mean word-level confidence (0–1).
        """
        engine = self.config.ASR_ENGINE

        if engine == "vosk":
            return self._transcribe_vosk(audio)
        elif engine == "whisper":
            return self._transcribe_whisper(audio)
        else:
            raise ValueError(f"Unsupported ASR engine: {engine!r}")

    def _transcribe_vosk(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe using Vosk/Kaldi recogniser."""
        rec = self._asr_engine
        rec.AcceptWaveform(audio.tobytes())
        result = json.loads(rec.FinalResult())

        transcript = result.get("text", "").strip()

        # Compute mean word-level confidence
        word_results = result.get("result", [])
        if word_results:
            confidences = [w.get("conf", 0.0) for w in word_results]
            mean_conf = sum(confidences) / len(confidences)
        else:
            mean_conf = 0.5  # Default if no word-level info

        return transcript, mean_conf

    def _transcribe_whisper(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe using OpenAI Whisper with command-optimised settings.

        Key improvements over the vanilla call:

        * **initial_prompt** — seeds the decoder with the expected command
          vocabulary so Whisper's beam search strongly prefers those tokens.
          This is the primary fix for hallucinations on short utterances
          (e.g. "Ed Python" instead of "add to cart").
        * **temperature=0** — greedy / deterministic decoding; eliminates
          random sampling artefacts on short clips.
        * **condition_on_previous_text=False** — each PTT press is treated as
          a fresh, independent utterance.
        * **no_speech filtering** — segments whose ``no_speech_prob`` exceeds
          ``WHISPER_NO_SPEECH_THRESHOLD`` are discarded before parsing,
          preventing empty-room noise from triggering phantom intents.
        """
        import math
        import whisper

        # Whisper expects float32 audio normalised to [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        result = self._asr_engine.transcribe(
            audio_f32,
            language="en",
            fp16=False,
            initial_prompt=self.config.WHISPER_INITIAL_PROMPT,
            temperature=self.config.WHISPER_TEMPERATURE,
            condition_on_previous_text=self.config.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            no_speech_threshold=self.config.WHISPER_NO_SPEECH_THRESHOLD,
            # word_timestamps costs ~10 % extra compute but gives better
            # segment boundaries on short clips — keep False for now.
            word_timestamps=False,
        )

        # ── Filter out no-speech segments ───────────────────────────
        segments = result.get("segments", [])
        speech_segments = [
            s for s in segments
            if s.get("no_speech_prob", 0.0) < self.config.WHISPER_NO_SPEECH_THRESHOLD
        ]

        if not speech_segments and segments:
            # All segments were silence — return empty transcript
            logger.debug(
                "Whisper: all segments below no_speech_threshold (%.2f) — discarded.",
                self.config.WHISPER_NO_SPEECH_THRESHOLD,
            )
            return "", 0.0

        # Re-assemble transcript from surviving segments only
        if speech_segments:
            transcript = " ".join(s.get("text", "").strip() for s in speech_segments).strip()
        else:
            # No segment metadata (older Whisper API) — fall back to top-level text
            transcript = result.get("text", "").strip()

        # ── Confidence from avg_logprob ──────────────────────────────
        # Convert log-probability to a 0-1 proxy confidence score.
        # avg_logprob is typically in [-1, 0]; exp maps it to (0, 1].
        active_segs = speech_segments or segments
        if active_segs:
            avg_logprob = sum(s.get("avg_logprob", -1.0) for s in active_segs) / len(active_segs)
            mean_conf = max(0.0, min(1.0, math.exp(avg_logprob)))
        else:
            mean_conf = 0.5

        logger.debug(
            "Whisper transcript: %r  conf=%.3f  segments=%d",
            transcript, mean_conf, len(active_segs),
        )
        return transcript, mean_conf


    # ── PTT key listener ────────────────────────────────────────────

    def _start_key_listener(self) -> None:
        """Start a ``pynput`` keyboard listener for the PTT key."""
        try:
            from pynput import keyboard

            ptt_key_name = self.config.PTT_KEY

            # Resolve the target key
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
                on_press=on_press,
                on_release=on_release,
            )
            self._key_listener.daemon = True
            self._key_listener.start()
            logger.info("PTT key listener started (key=%s).", ptt_key_name)

        except ImportError:
            logger.warning(
                "pynput is not installed — PTT key listener disabled.  "
                "Call on_ptt_press() / on_ptt_release() programmatically."
            )

    # ── Dialog context ──────────────────────────────────────────────

    def set_dialog_active(self, active: bool) -> None:
        """Called by the DialogueManager to switch parsing context.

        When *active* is ``True``, the parser prioritises REPAIR /
        CONFIRM / DENY patterns over regular intents.
        """
        self._dialog_active = active
        logger.debug("Dialog context set to: %s", active)

    # ── Programmatic trigger (for testing / demo) ───────────────────

    def process_text(self, transcript: str, asr_confidence: float = 0.9) -> SpeechEvent:
        """Bypass audio capture and directly parse a transcript string.

        This is useful for unit testing and CLI demos where no
        microphone is available.

        Parameters
        ----------
        transcript:
            The text to parse as if it came from the ASR engine.
        asr_confidence:
            Simulated ASR confidence score.

        Returns
        -------
        SpeechEvent
            The parsed event (also emitted on the EventBus).
        """
        event = self.parser.parse(
            transcript,
            asr_confidence=asr_confidence,
            dialog_active=self._dialog_active,
        )
        self.event_bus.emit(event)
        return event
