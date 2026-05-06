"""
Configuration Module
====================

Centralised configuration for all GazeShop toolkit parameters.
All thresholds, time windows, and hardware settings are defined here.

Usage
-----
    from gazeshop.toolkit.config import Config

    cfg = Config()                       # defaults
    cfg = Config(PTT_MODE="toggle")      # override at creation
    cfg.CONFIDENCE_THRESHOLD = 0.70      # override later
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Immutable-by-convention configuration container.

    Every public attribute corresponds to a tunable parameter
    documented in the GazeShop Speech Module Specification.
    """

    # ── Push-to-Talk ────────────────────────────────────────────────
    PTT_MODE: str = "hold"
    """'hold' = hold-to-talk (press & hold Space); 'toggle' = click once
    to start, click again to stop."""

    PTT_KEY: str = "space"
    """Keyboard key that activates push-to-talk.  Must be a valid
    ``pynput.keyboard.Key`` name or a single character."""

    # ── Audio Capture ───────────────────────────────────────────────
    SAMPLE_RATE: int = 16_000
    """Audio sample rate in Hz.  16 kHz is the standard for most ASR
    engines (Vosk, Whisper)."""

    CHANNELS: int = 1
    """Number of audio channels.  Mono (1) is required by ASR models."""

    TAIL_SILENCE_MS: int = 500
    """Milliseconds of audio to continue recording after the PTT key
    is released, capturing the tail end of the utterance.
    500 ms is safer than 300 ms for commands that end with short vowels."""

    MIN_UTTERANCE_MS: int = 300
    """Utterances shorter than this are discarded as accidental taps.
    300 ms avoids false-positive triggers from key-noise or breath."""

    MAX_UTTERANCE_MS: int = 15_000
    """Utterances longer than this trigger an automatic stop + warning."""

    ENERGY_THRESHOLD: float = 200.0
    """Root-mean-square energy threshold.  Buffers below this are
    treated as silence and discarded.  200 is a reasonable floor for
    a typical laptop mic in a quiet room; raise to 400–600 in noisy
    environments."""

    # ── ASR Engine ──────────────────────────────────────────────────
    ASR_ENGINE: str = "vosk"
    """Active ASR backend.  Supported: 'vosk', 'whisper'."""

    VOSK_MODEL_PATH: str = "models/vosk-model-small-en-us-0.15"
    """Path to the Vosk language model directory."""

    WHISPER_MODEL_SIZE: str = "small"
    """Whisper model variant when ASR_ENGINE='whisper'.
    Options: 'tiny', 'base', 'small', 'medium', 'large'.
    'small' strongly preferred over 'base' for short command phrases —
    'base' hallucinates on 1-3 word utterances."""

    WHISPER_INITIAL_PROMPT: str = (
        "Add to cart. Show details. Find similar. Compare. Show alternatives. "
        "Pin this. Remove this. Scroll up. Scroll down. Open cart. "
        "Help. Cancel. Never mind. Undo. Go back. "
        "Yes. No. Left. Right. First. Second."
    )
    """Prompt prepended to every Whisper decode pass.
    Providing the expected command vocabulary steers Whisper's beam search
    toward these words and prevents hallucinations (e.g. 'Ed Python' for
    'add to cart').  Update this list if new intents are added."""

    WHISPER_TEMPERATURE: float = 0.0
    """Decoding temperature for Whisper (0.0 = greedy / deterministic).
    Values > 0 introduce sampling randomness; use 0.0 for commands."""

    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    """Probability threshold above which a segment is considered silence.
    Segments classified as no-speech are discarded before intent parsing."""

    WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = False
    """Whether each Whisper decode pass should be conditioned on the
    previous transcript.  Set False for PTT — every press is independent."""

    # ── Intent Parsing ──────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.60
    """Intent confidence below this value triggers a confirmation
    dialog rather than direct execution."""

    CONFIDENCE_WEIGHTS: dict = field(default_factory=lambda: {
        "pattern_match": 0.4,
        "asr_confidence": 0.4,
        "word_coverage": 0.2,
    })
    """Weights for the three signals that compose the heuristic
    confidence score (see §D.3 of the specification)."""

    # ── Fusion (reference values for integration) ───────────────────
    FUSION_TIME_WINDOW_S: float = 2.0
    """Maximum seconds between a gaze LOCK timestamp and a speech
    INTENT timestamp for the two to be fused."""

    GAZE_DWELL_TO_LOCK_S: float = 1.0
    """Seconds the user must gaze at an item before it locks.
    (Defined in GazeAdapter; repeated here for cross-reference.)"""

    LOCK_TTL_S: float = 4.0
    """Seconds a gaze lock remains valid before expiring ('stale').
    If a speech command arrives after this TTL, it will be rejected
    and a 'lock_expired' prompt will be issued."""

    # ── Dialogue Manager ────────────────────────────────────────────
    DISAMBIGUATION_TIMEOUT_S: float = 8.0
    """Seconds the system waits for a repair / confirmation response
    before auto-cancelling."""

    MAX_REPAIR_ATTEMPTS: int = 2
    """After this many failed repair rounds the action is cancelled."""

    # ── Logging / Telemetry ─────────────────────────────────────────
    ENABLE_TELEMETRY: bool = True
    """Whether to export telemetry data (JSONL)."""

    TELEMETRY_EXPORT_PATH: str = "logs/telemetry.jsonl"
    """Path to write telemetry logs (will be appended to)."""

    # ── Voice Activity Detection (VAD) ──────────────────────────────
    VAD_ENABLED: bool = True
    """Whether to apply VAD trimming before sending audio to ASR.
    When True the adapter tries Silero VAD first, then falls back to
    WebRTC VAD, then falls back to no trimming."""

    VAD_BACKEND: str = "silero"
    """Preferred VAD backend.  Options:
    * ``'silero'``  — neural-network VAD (recommended; needs torch).
    * ``'webrtc'``  — rule-based WebRTC VAD (needs webrtcvad / webrtcvad-wheels).
    * ``'none'``    — disable VAD trimming entirely (same as VAD_ENABLED=False).
    The adapter falls back automatically if the chosen backend is unavailable."""

    VAD_AGGRESSIVENESS: int = 2
    """WebRTC VAD aggressiveness level (0–3).
    0 = least aggressive (more false speech), 3 = most aggressive (may
    clip soft consonants).  Level 2 is a good default for close-mic PTT.
    Only used when VAD_BACKEND='webrtc'."""

    # ── Silero VAD ──────────────────────────────────────────────────
    SILERO_VAD_THRESHOLD: float = 0.4
    """Speech-probability threshold for Silero VAD (0.0–1.0).
    A frame is considered speech when the model outputs a probability
    ≥ this value.  Lower → more permissive (keeps more audio);
    higher → more aggressive (strips more).  0.4 is a safe default for
    close-mic push-to-talk; try 0.3 in noisy rooms."""

    SILERO_VAD_PADDING_MS: int = 100
    """Milliseconds of audio to keep *before* the first detected speech
    frame and *after* the last, to avoid clipping onset consonants (e.g.
    the 'a' in 'add').  100 ms is a good default."""

    SILERO_VAD_MIN_SPEECH_MS: int = 200
    """Minimum duration (ms) for a detected speech segment to be kept.
    Shorter blips are discarded as noise.  200 ms avoids false positives
    from keyboard clicks or brief breath pops."""

    # Note: no SILERO_VAD_MODEL_REPO needed — the silero-vad pip package
    # bundles the model weights directly (pip install silero-vad).

    def __post_init__(self) -> None:
        """Basic validation."""
        assert self.PTT_MODE in ("hold", "toggle"), (
            f"PTT_MODE must be 'hold' or 'toggle', got '{self.PTT_MODE}'"
        )
        assert self.ASR_ENGINE in ("vosk", "whisper"), (
            f"ASR_ENGINE must be 'vosk' or 'whisper', got '{self.ASR_ENGINE}'"
        )
        assert 0.0 <= self.CONFIDENCE_THRESHOLD <= 1.0, (
            "CONFIDENCE_THRESHOLD must be between 0.0 and 1.0"
        )
        assert self.VAD_AGGRESSIVENESS in (0, 1, 2, 3), (
            f"VAD_AGGRESSIVENESS must be 0, 1, 2 or 3, got {self.VAD_AGGRESSIVENESS}"
        )
        assert self.VAD_BACKEND in ("silero", "webrtc", "none"), (
            f"VAD_BACKEND must be 'silero', 'webrtc', or 'none', got '{self.VAD_BACKEND}'"
        )
        assert 0.0 <= self.SILERO_VAD_THRESHOLD <= 1.0, (
            "SILERO_VAD_THRESHOLD must be between 0.0 and 1.0"
        )

