"""LLM trim orchestrator — ties together transcription, ad detection, and cutting."""

from __future__ import annotations

import json
import logging
import os
import tempfile

from siphon.ad_detect import build_transcript_for_claude, detect_ads, filter_segments, resolve_prompt
from siphon.config import LLMConfig, ResolvedFeed
from siphon.cutter import cut_segments, extract_audio
from siphon.transcribe import transcribe

logger = logging.getLogger(__name__)


def run_llm_trim(
    file_path: str,
    feed: ResolvedFeed,
    llm_config: LLMConfig,
) -> dict:
    """Run the full LLM trim pipeline on a media file.

    Steps:
    1. Extract audio (if video) for Whisper
    2. Transcribe with Whisper
    3. Detect ads with Claude CLI
    4. Filter segments by confidence/duration
    5. Cut segments with ffmpeg

    Returns a dict with:
        llm_trim_status: "done" | "error"
        llm_segments_json: JSON string of all detected segments
        llm_cuts_applied: number of segments actually cut
        error: error message if status is "error"
    """
    try:
        return _run_pipeline(file_path, feed, llm_config)
    except Exception as exc:
        logger.error("LLM trim failed for %s: %s", file_path, exc)
        return {
            "llm_trim_status": "error",
            "llm_segments_json": "{}",
            "llm_cuts_applied": 0,
            "error": str(exc),
        }


def _run_pipeline(
    file_path: str,
    feed: ResolvedFeed,
    llm_config: LLMConfig,
) -> dict:
    """Internal pipeline — raises on error."""

    # Step 1: Get audio for Whisper
    is_audio = file_path.lower().endswith((".mp3", ".m4a", ".ogg", ".wav", ".flac", ".aac"))

    if is_audio:
        whisper_input = file_path
        temp_audio = None
    else:
        # Extract audio from video to a temp file
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        try:
            extract_audio(file_path, temp_audio.name)
            whisper_input = temp_audio.name
        except Exception:
            os.unlink(temp_audio.name)
            raise

    try:
        # Step 2: Transcribe
        logger.info("Transcribing %s with model=%s", file_path, llm_config.whisper_model)
        transcript = transcribe(
            whisper_input,
            model_size=llm_config.whisper_model,
            device=llm_config.whisper_device,
            word_timestamps=llm_config.whisper_word_timestamps,
            vad_filter=llm_config.whisper_vad_filter,
        )
    finally:
        # Clean up temp audio
        if temp_audio is not None:
            try:
                os.unlink(temp_audio.name)
            except OSError:
                pass

    transcript_text = transcript["text"]
    if not transcript_text.strip():
        logger.info("Empty transcript for %s, skipping ad detection", file_path)
        return {
            "llm_trim_status": "done",
            "llm_segments_json": json.dumps({"segments": []}),
            "llm_cuts_applied": 0,
        }

    # Step 3: Detect ads with Claude
    # Skip word timestamps for long episodes to avoid context window limits
    words = transcript.get("words")
    duration_minutes = transcript.get("duration", 0) / 60
    if duration_minutes > llm_config.word_timestamps_max_minutes:
        logger.info(
            "Episode is %.0f min (> %d min limit), using segment-only timestamps",
            duration_minutes, llm_config.word_timestamps_max_minutes,
        )
        words = None

    prompt = resolve_prompt(feed, llm_config)
    raw_result = detect_ads(
        transcript_text,
        prompt,
        model=llm_config.claude_model,
        effort=llm_config.claude_effort,
        words=words,
        segments=transcript.get("segments"),
    )

    all_segments = raw_result.get("segments", [])
    logger.info("Claude detected %d potential ad segments", len(all_segments))

    # Step 4: Filter
    high_confidence, marginal = filter_segments(
        all_segments,
        confidence_threshold=llm_config.confidence_threshold,
        min_duration=llm_config.min_segment_duration,
        max_duration=llm_config.max_segment_duration,
        merge_gap=llm_config.merge_gap_seconds,
    )

    logger.info(
        "After filtering: %d to cut, %d marginal",
        len(high_confidence), len(marginal),
    )

    # Step 5: Cut
    if high_confidence:
        cut_segments(
            file_path, high_confidence,
            snap_to_silence=llm_config.snap_cuts_to_silence,
            snap_window=llm_config.snap_window_seconds,
        )
        logger.info("Applied %d cuts to %s", len(high_confidence), file_path)

    # Build audit data
    audit = {
        "segments": all_segments,
        "high_confidence": [s.get("label", "") for s in high_confidence],
        "marginal": [s.get("label", "") for s in marginal],
    }

    return {
        "llm_trim_status": "done",
        "llm_segments_json": json.dumps(audit),
        "llm_cuts_applied": len(high_confidence),
    }
