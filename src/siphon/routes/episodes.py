"""Episodes page: see an episode's transcript, what was cut, and play the original and cleaned audio."""

from __future__ import annotations

import logging
import os
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from siphon import inspector
from siphon.episode_files import original_path, transcript_path
from siphon.reprocess import (
    BUSY_STATUSES, QUEUED_MESSAGES, STATUS_TEXT, ReprocessError, check_reprocess, reprocess_episode,
)
from siphon.routes.ui import _flash, _get_background_tasks, _get_messages, _redirect, _render, templates

# Same prefix as the rest of the web UI, so the localhost-only rule in the security
# middleware covers these pages and the audio they stream.
router = APIRouter(prefix="/ui")
logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}
VIDEO_EXTENSIONS = {".mp4", ".webm"}

templates.env.filters["clock"] = inspector.format_clock
templates.env.filters["urlq"] = lambda value: quote(str(value), safe="")


def _audio_paths(episode: dict, audit: dict) -> dict:
    """Where the cleaned and original audio are, or None when a file is not available."""
    clean = episode.get("file_path")
    clean = clean if clean and os.path.isfile(clean) else None
    original = None
    if clean:
        kept = original_path(clean)
        if os.path.isfile(kept):
            original = kept
        elif not inspector.removed_ranges(audit):
            original = clean  # nothing was cut, so the file is the original
    return {"clean": clean, "original": original}


def _episode_url(feed_name: str, video_id: str) -> str:
    return f"/ui/episode?feed={quote(feed_name, safe='')}&id={quote(video_id, safe='')}"


def _upload_date(value: str | None) -> str:
    """YYYYMMDD -> YYYY-MM-DD (anything else is shown as it is)."""
    value = (value or "").strip()
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}" if len(value) == 8 and value.isdigit() else value


@router.get("/episodes", response_class=HTMLResponse)
async def episodes_page(request: Request, feed: str = "", show: str = ""):
    db = request.app.state.db
    config = request.app.state.config
    show_all = show == "all"
    if show_all:
        rows = db.get_recent_episodes(limit=500, feed_name=feed or None)
    else:
        rows = db.get_processed_episodes(limit=300, feed_name=feed or None)
    items = []
    for ep in rows:
        audit = inspector.parse_audit(ep.get("llm_segments_json"))
        cuts = inspector.build_cut_list(audit)
        paths = _audio_paths(ep, audit)
        items.append({
            "title": ep.get("title") or ep["video_id"],
            "feed_name": ep["feed_name"],
            "when": _upload_date(ep.get("upload_date")) if show_all else (ep.get("updated_at") or "")[:16],
            "status": ep.get("status"),
            "cut_count": len(cuts),
            "removed": sum(c["duration"] for c in cuts),
            "has_transcript": os.path.isfile(transcript_path(config.storage.download_dir, ep["feed_name"], ep["video_id"])),
            "has_original": paths["original"] is not None,
            "url": _episode_url(ep["feed_name"], ep["video_id"]),
        })
    return _render(request, "episodes.html", {
        "active_page": "episodes",
        "items": items,
        "feeds": sorted(f.name for f in config.feeds),
        "selected_feed": feed,
        "show_all": show_all,
        "keep_data": config.storage.keep_inspection_data,
        "messages": _get_messages(request),
    })


def _render_episode(request: Request, episode: dict, status_code: int = 200):
    """Build the inspector page for ``episode`` (used by the GET page and by failed reprocess requests)."""
    config = request.app.state.config
    feed, video_id = episode["feed_name"], episode["video_id"]

    audit = inspector.parse_audit(episode.get("llm_segments_json"))
    cuts = inspector.build_cut_list(audit)
    paths = _audio_paths(episode, audit)

    segments = inspector.load_transcript_segments(transcript_path(config.storage.download_dir, feed, video_id))
    lines = inspector.classify_segments(segments, audit) if segments else []

    ext = os.path.splitext(episode.get("file_path") or "")[1].lower()
    base_audio_url = f"/ui/episode/audio?feed={quote(feed, safe='')}&id={quote(video_id, safe='')}"
    response = _render(request, "episode.html", {
        "active_page": "episodes",
        "episode": episode,
        "feed_name": feed,
        "cuts": cuts,
        "summary": inspector.summarize(audit, episode, cuts),
        "flagged": inspector.flagged_not_cut(audit),
        "lines": lines,
        "has_transcript": bool(lines),
        "media_tag": "video" if ext in VIDEO_EXTENSIONS else "audio",
        "original_url": f"{base_audio_url}&which=original" if paths["original"] else None,
        "clean_url": f"{base_audio_url}&which=clean" if paths["clean"] else None,
        "original_is_clean_file": bool(paths["original"]) and paths["original"] == paths["clean"],
        "keep_data": config.storage.keep_inspection_data,
        "time_map": inspector.clean_time_map(cuts),
        "chime_seconds": inspector.chime_seconds(audit),
        "reprocess": {
            "detect_problem": check_reprocess(config, episode, "detect"),
            "transcribe_problem": check_reprocess(config, episode, "transcribe"),
            "redownload_problem": check_reprocess(config, episode, "redownload"),
            "redownload_label": "Re-download and redo everything" if episode.get("status") == "done" else "Download and process",
            "waiting_for_download": bool(episode.get("redownload_requested")),
            "busy": episode.get("status") in BUSY_STATUSES or bool(episode.get("redownload_requested")),
            "status_text": STATUS_TEXT.get(episode.get("status"), episode.get("status")),
        },
        "messages": _get_messages(request),
    })
    response.status_code = status_code
    return response


@router.get("/episode", response_class=HTMLResponse)
async def episode_page(request: Request, feed: str, id: str, queued: str = ""):
    episode = request.app.state.db.get_episode(id, feed)
    if episode is None:
        return HTMLResponse("Episode not found", status_code=404)
    if queued in QUEUED_MESSAGES:
        _flash(request, QUEUED_MESSAGES[queued], "info")
    return _render_episode(request, episode)


def _kick_downloads(app) -> None:
    """Run the download worker in the background (it skips itself if it is already running)."""
    import asyncio
    from siphon import pipeline

    config, db = app.state.config, app.state.db

    async def run() -> None:
        try:
            await pipeline.process_downloads(config, db)
        except Exception:
            logger.exception("Download run started from the Episodes page failed")

    task = asyncio.create_task(run())
    tasks = _get_background_tasks(app)
    tasks.add(task)
    task.add_done_callback(tasks.discard)


@router.post("/episode/reprocess")
async def episode_reprocess(request: Request, feed: str = Form(...), id: str = Form(...), mode: str = Form(...)):
    """Send a finished episode back through ad detection (and optionally Whisper) with current settings."""
    state = request.app.state
    episode = state.db.get_episode(id, feed)
    if episode is None:
        return HTMLResponse("Episode not found", status_code=404)
    try:
        reprocess_episode(state.config, state.db, episode, mode)
    except ReprocessError as exc:
        _flash(request, str(exc), "error")
        return _render_episode(request, episode, status_code=400)
    if mode == "redownload":
        _kick_downloads(request.app)   # start now instead of waiting for the next scheduled run
    return _redirect(request, f"{_episode_url(feed, id)}&queued={quote(mode, safe='')}")


@router.get("/episode/audio")
async def episode_audio(request: Request, feed: str, id: str, which: str):
    if which not in ("original", "clean"):
        return Response("Unknown audio", status_code=400)
    episode = request.app.state.db.get_episode(id, feed)
    if episode is None:
        return Response("Episode not found", status_code=404)
    audit = inspector.parse_audit(episode.get("llm_segments_json"))
    path = _audio_paths(episode, audit)[which]
    if path is None:
        return Response("Audio not available", status_code=404)
    media_type = MEDIA_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
    # FileResponse answers HTTP range requests, so seeking works in the players
    return FileResponse(path, media_type=media_type)
