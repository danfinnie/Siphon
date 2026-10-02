"""Whisper transcription utility using faster-whisper."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Singleton model management — avoids reloading on every transcription
# ------------------------------------------------------------------ #

_model = None
_model_lock = threading.Lock()
_model_config: tuple = (None, None, None, None)  # (model_size, device, compute_type, num_workers)


def _ensure_cuda_dlls():
    """Add pip-installed NVIDIA DLL paths to os.environ['PATH'] on Windows."""
    import sys
    if sys.platform != "win32":
        return
    try:
        import importlib.util
        for pkg in ("nvidia.cublas", "nvidia.cudnn"):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    bin_dir = os.path.join(loc, "bin")
                    if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                        logger.info("Added CUDA DLL path: %s", bin_dir)
    except Exception as e:
        logger.warning("Could not add CUDA DLL paths: %s", e)


def _get_model(model_size: str, device: str, num_workers: int = 1):
    """Return a shared WhisperModel instance, loading only when config changes."""
    global _model, _model_config
    compute_type = "float16" if device == "cuda" else "int8"
    # CUDA doesn't support multiple workers — force to 1
    if device == "cuda" and num_workers > 1:
        logger.warning("CUDA does not support num_workers > 1, forcing to 1")
        num_workers = 1
    config = (model_size, device, compute_type, num_workers)
    with _model_lock:
        if _model is None or _model_config != config:
            if device == "cuda":
                _ensure_cuda_dlls()

            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model %s on %s (num_workers=%d)",
                model_size, device, num_workers,
            )
            _model = WhisperModel(
                model_size, device=device, compute_type=compute_type,
                cpu_threads=4, num_workers=num_workers,
            )
            _model_config = config
        return _model


def transcribe(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    word_timestamps: bool = True,
    num_workers: int = 1,
    vad_filter: bool = False,
) -> dict[str, Any]:
    """Transcribe an audio file using faster-whisper.

    Returns a dict with:
        {
            "segments": [
                {"start": 0.0, "end": 5.2, "text": "Hello world..."}
            ],
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.3},
                {"word": "world", "start": 0.4, "end": 0.7},
            ],
            "text": "Full transcript text...",
            "language": "en",
            "duration": 300.0,
        }
    """
    model = _get_model(model_size, device, num_workers)

    logger.info("Transcribing %s (word_timestamps=%s, vad_filter=%s)", audio_path, word_timestamps, vad_filter)
    segments_iter, info = model.transcribe(
        audio_path, beam_size=5, word_timestamps=word_timestamps, vad_filter=vad_filter,
    )

    segments = []
    words = []
    full_text_parts = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

        # Extract word-level timestamps when available
        if word_timestamps and hasattr(seg, "words") and seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                })

    full_text = " ".join(full_text_parts)
    logger.info(
        "Transcribed %s: %d segments, %d words, %.1f seconds, language=%s",
        audio_path, len(segments), len(words), info.duration, info.language,
    )

    return {
        "segments": segments,
        "words": words,
        "text": full_text,
        "language": info.language,
        "duration": info.duration,
    }
