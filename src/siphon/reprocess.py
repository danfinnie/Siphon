"""Put a finished episode back in the queue so it is processed again with the current settings.

Two ways, both starting from the kept original audio (``storage.keep_inspection_data``):

* ``detect``     reuse the saved transcript and run ad detection and cutting again
                 (new prompt, effort, merge gap, snapping, ...). Fast: no Whisper.
* ``transcribe`` throw the transcript away and run Whisper too (new model, voice filter, ...).

The episode is restored to its original audio and sent back to the Claude or Whisper queue; the
normal workers do the rest. It drops out of the RSS feed until it is done again.
"""

from __future__ import annotations

import os

from siphon import inspector
from siphon.config import SiphonConfig, resolve_feed
from siphon.db import Database
from siphon.episode_files import original_path, restore_original, transcript_path

MODES = ("detect", "transcribe")

QUEUED_MESSAGES = {
    "detect": "Queued for ad detection again with your current settings. The episode leaves your feed until it is done.",
    "transcribe": "Queued to be transcribed and checked again with your current settings. The episode leaves your feed until it is done.",
}

STATUS_TEXT = {
    "pending_whisper": "Waiting to be transcribed (Whisper)",
    "pending_claude": "Waiting for ad detection (Claude)",
    "downloading": "Downloading",
    "eligible": "Waiting to download",
    "pending": "Waiting to download",
}


class ReprocessError(Exception):
    """The episode cannot be reprocessed; the message says why, in plain words."""


def check_reprocess(config: SiphonConfig, episode: dict, mode: str) -> str | None:
    """Return why ``episode`` cannot be reprocessed in ``mode``, or None if it can."""
    if mode not in MODES:
        return "Unknown way to reprocess."
    if episode.get("status") != "done":
        return "This episode is still being processed. Try again when it has finished."

    feed_config = next((f for f in config.feeds if f.name == episode["feed_name"]), None)
    if feed_config is None:
        return "This episode's feed is no longer in the config."
    if not resolve_feed(feed_config, config.defaults).llm_trim:
        return "Ad removal (llm_trim) is turned off for this feed, so there is nothing to redo."

    file_path = episode.get("file_path")
    if not file_path:
        return "This episode has no audio file."
    kept = os.path.isfile(original_path(file_path))
    audit = inspector.parse_audit(episode.get("llm_segments_json"))
    had_cuts = bool(inspector.removed_ranges(audit)) or (episode.get("llm_cuts_applied") or 0) > 0
    if not kept:
        if had_cuts:
            return (
                "The original audio was not kept for this episode, so it cannot be processed again. "
                "Turn on storage.keep_inspection_data to keep originals for new episodes."
            )
        if not os.path.isfile(file_path):
            return "The episode's audio file is missing."

    if mode == "detect" and not os.path.isfile(
        transcript_path(config.storage.download_dir, episode["feed_name"], episode["video_id"])
    ):
        return "The transcript was not kept for this episode. Use \"Re-transcribe and redo\" instead."
    return None


def reprocess_episode(config: SiphonConfig, db: Database, episode: dict, mode: str) -> str:
    """Restore the original audio and queue the episode again. Returns a message for the user."""
    problem = check_reprocess(config, episode, mode)
    if problem:
        raise ReprocessError(problem)

    file_path = episode["file_path"]
    try:
        restore_original(file_path)  # does nothing when the file already is the original
    except OSError as exc:
        raise ReprocessError(f"Could not restore the original audio ({exc}). Try again in a moment.") from exc

    if mode == "transcribe":
        try:
            os.remove(transcript_path(config.storage.download_dir, episode["feed_name"], episode["video_id"]))
        except OSError:
            pass

    db.update_episode_status(
        episode["video_id"], episode["feed_name"],
        "pending_claude" if mode == "detect" else "pending_whisper",
        llm_trim_status="pending",
        llm_cuts_applied=0,
        llm_retry_count=0,
        llm_segments_json=None,
        error=None,
        file_size=os.path.getsize(file_path),
    )
    return QUEUED_MESSAGES[mode]
