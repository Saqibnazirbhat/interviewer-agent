"""Terminal voice mode — TTS for questions, STT for answers.

Uses pyttsx3 for text-to-speech and sounddevice+scipy for recording.
All imports are lazy so voice mode is optional — the rest of the app
works without these dependencies installed.
"""

import logging
import re
import tempfile
import threading

logger = logging.getLogger("interviewer.voice")

_tts_engine = None
_tts_lock = threading.Lock()


def is_available() -> bool:
    """Check if voice dependencies are installed."""
    try:
        import pyttsx3
        return True
    except ImportError:
        return False


def _clean_text_for_speech(text: str) -> str:
    """Strip punctuation and special characters so TTS only speaks natural words."""
    text = re.sub(r'["""\'\'`]', '', text)
    text = re.sub(r'[#*_~^|\\{}\[\]<>@]', '', text)
    text = re.sub(r'\s*[—–]\s*', ', ', text)
    text = re.sub(r'[()]', ', ', text)
    text = re.sub(r'\s*[/:;]\s*', ', ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def speak(text: str, rate: int = 175):
    """Speak text using pyttsx3. Blocks until speech is complete."""
    global _tts_engine
    try:
        import pyttsx3
    except ImportError:
        logger.warning("pyttsx3 not installed — voice mode unavailable")
        return

    clean = _clean_text_for_speech(text)
    with _tts_lock:
        try:
            if _tts_engine is None:
                _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", rate)
            _tts_engine.say(clean)
            _tts_engine.runAndWait()
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)
            _tts_engine = None


def record_audio(
    duration: float = 120.0,
    sample_rate: int = 16000,
    silence_timeout: float = 3.0,
    silence_threshold: int = 300,
) -> str | None:
    """Record audio from microphone and save to a temp WAV file.

    Recording stops after *silence_timeout* seconds of continuous silence
    (amplitude below *silence_threshold*) or when *duration* is reached.
    Short pauses (< 6 s) are treated as natural thinking time and do NOT
    stop the recording.

    Args:
        duration: Maximum recording duration in seconds.
        sample_rate: Audio sample rate.
        silence_timeout: Seconds of continuous silence before auto-stop.
        silence_threshold: Amplitude below which audio counts as silence.

    Returns:
        Path to the temporary WAV file, or None if recording failed.
    """
    try:
        import sounddevice as sd
        from scipy.io import wavfile
        import numpy as np
    except ImportError:
        logger.warning("sounddevice/scipy not installed — recording unavailable")
        return None

    try:
        chunk_duration = 0.5  # check silence every 0.5 s
        chunk_samples = int(chunk_duration * sample_rate)
        max_chunks = int(duration / chunk_duration)
        chunks: list = []
        silent_seconds = 0.0

        print(f"  Recording... (up to {int(duration)}s, {int(silence_timeout)}s silence to stop, Ctrl+C to stop)")

        try:
            for _ in range(max_chunks):
                chunk = sd.rec(chunk_samples, samplerate=sample_rate, channels=1, dtype="int16")
                sd.wait()
                chunks.append(chunk)

                peak = int(np.max(np.abs(chunk)))
                if peak < silence_threshold:
                    silent_seconds += chunk_duration
                    if silent_seconds >= silence_timeout:
                        logger.debug("Silence timeout reached (%.1fs)", silence_timeout)
                        break
                else:
                    silent_seconds = 0.0
        except KeyboardInterrupt:
            pass

        if not chunks:
            return None

        audio = np.concatenate(chunks)

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, sample_rate, audio)
        return tmp.name

    except Exception as exc:
        logger.warning("Recording failed: %s", exc)
        return None


def set_persona_voice(persona_name: str):
    """Try to set a TTS voice that matches the persona character."""
    global _tts_engine
    try:
        import pyttsx3
    except ImportError:
        return

    with _tts_lock:
        try:
            if _tts_engine is None:
                _tts_engine = pyttsx3.init()

            voices = _tts_engine.getProperty("voices")
            if not voices:
                return

            # Simple heuristic: pick a different voice for different personas
            persona_lower = persona_name.lower()
            target_idx = 0

            if "hr" in persona_lower or "manager" in persona_lower:
                target_idx = 1 if len(voices) > 1 else 0
            elif "executive" in persona_lower or "panel" in persona_lower:
                target_idx = min(2, len(voices) - 1)

            _tts_engine.setProperty("voice", voices[target_idx].id)

        except Exception as exc:
            logger.debug("Failed to set persona voice: %s", exc)
