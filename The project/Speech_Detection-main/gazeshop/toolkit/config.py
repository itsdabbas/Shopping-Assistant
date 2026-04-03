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

    TAIL_SILENCE_MS: int = 300
    """Milliseconds of audio to continue recording after the PTT key
    is released, capturing the tail end of the utterance."""

    MIN_UTTERANCE_MS: int = 200
    """Utterances shorter than this are discarded as accidental taps."""

    MAX_UTTERANCE_MS: int = 15_000
    """Utterances longer than this trigger an automatic stop + warning."""

    ENERGY_THRESHOLD: float = 300.0
    """Root-mean-square energy threshold.  Buffers below this are
    treated as silence and discarded."""

    # ── ASR Engine ──────────────────────────────────────────────────
    ASR_ENGINE: str = "vosk"
    """Active ASR backend.  Supported: 'vosk', 'whisper'."""

    VOSK_MODEL_PATH: str = "models/vosk-model-small-en-us-0.15"
    """Path to the Vosk language model directory."""

    WHISPER_MODEL_SIZE: str = "base"
    """Whisper model variant when ASR_ENGINE='whisper'.
    Options: 'tiny', 'base', 'small', 'medium', 'large'."""

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
