"""Pipeline orchestration for Siphon.

Three independent workers form a queue-based pipeline:

    Download Queue -> Whisper Queue -> Claude Queue
       (5 min)         (30 sec)         (30 sec)

The download worker fetches media files and sets status to
pending_whisper (if llm_trim is enabled) or done (if not).

The Whisper worker transcribes one episode at a time, saves the
transcript JSON to disk, and moves the episode to pending_claude.

The Claude worker reads transcript JSON, runs ad detection with
Claude, applies ffmpeg cuts, and sets the episode to done.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from siphon.activity import (
    check_paused,
    clear_alert,
    log_activity,
    set_alert,
    worker_done,
    worker_start,
)
from siphon.ad_detect import ClaudeAuthError, ClaudeTransientError
from siphon.config import SiphonConfig, resolve_feed
from siphon.db import Database
from siphon.downloader import (
    download_video,
    extract_feed_metadata,
    find_downloaded_file,
)
from siphon.filters import apply_filters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Interrupt recovery
# ---------------------------------------------------------------------- #


def recover_interrupted(config: SiphonConfig, db: Database) -> None:
    """Reset episodes stuck in intermediate pipeline states after a restart.

    - pending_whisper: leave as is (Whisper worker will pick them up)
    - pending_claude: check if transcript JSON exists on disk; if not,
      reset to pending_whisper so they get re-transcribed.
    - downloading: handled by existing reset_stale_downloads
    """
    pending_claude = db.get_pending_claude(limit=100)
    reset_count = 0
    for ep in pending_claude:
        transcript_path = _transcript_path(config, ep["feed_name"], ep["video_id"])
        if not os.path.exists(transcript_path):
            db.update_episode_status(ep["video_id"], ep["feed_name"], "pending_whisper")
            reset_count += 1
            logger.info(
                "Reset %s/%s from pending_claude to pending_whisper (transcript missing)",
                ep["feed_name"], ep["video_id"],
            )
    if reset_count:
        log_activity(f"Recovery: reset {reset_count} episodes to pending_whisper")


def _transcript_path(config: SiphonConfig, feed_name: str, video_id: str) -> str:
    """Return the expected transcript JSON path for an episode."""
    import re
    safe_id = re.sub(r'[^\w-]', '_', video_id)[:80]
    return os.path.join(
        config.storage.download_dir, feed_name, f"{safe_id}_transcript.json"
    )


async def _get_file_duration(loop, path: str) -> int | None:
    """Get duration of a media file via ffprobe. Returns seconds or None."""
    try:
        from siphon.cutter import get_duration
        secs = await loop.run_in_executor(None, get_duration, path)
        return int(secs)
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return None


def _post_download_filter(duration: int | None, resolved) -> str | None:
    """Check if a downloaded episode should be filtered by duration."""
    if duration is None:
        return None
    if resolved.block_shorts and duration < 60:
        return "short"
    if duration < resolved.min_duration_seconds:
        return "too_short"
    return None


def _delete_file(path: str) -> None:
    """Delete a file, ignoring errors."""
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------- #
# Feed checking
# ---------------------------------------------------------------------- #


async def check_feeds(config: SiphonConfig, db: Database) -> None:
    """Discover new episodes from the next batch of feeds."""
    worker_start("checking_feeds")
    log_activity("Starting feed check")

    for feed_type, limit in (
        ("youtube", config.schedule.youtube_feeds_per_check),
        ("podcast", config.schedule.podcast_feeds_per_check),
    ):
        feeds_to_check = db.get_feeds_to_check(limit, feed_type=feed_type)

        for feed_idx, feed_db in enumerate(feeds_to_check):
            if feed_idx > 0:
                await asyncio.sleep(5 if feed_type == "youtube" else 1)

            feed_config = None
            for fc in config.feeds:
                if fc.name == feed_db["name"]:
                    feed_config = fc
                    break

            if feed_config is None:
                logger.warning("Feed %r not found in config, skipping", feed_db["name"])
                continue

            resolved = resolve_feed(feed_config, config.defaults)

            try:
                log_activity(f"Checking {feed_db['name']}", feed=feed_db["name"])
                if resolved.type == "podcast":
                    await _check_podcast_feed(resolved, db)
                else:
                    await _check_youtube_feed(resolved, config, db)

                db.update_feed_checked(feed_db["name"])

            except Exception as exc:
                from siphon.youtube import YouTubeQuotaExceeded
                if isinstance(exc, YouTubeQuotaExceeded):
                    log_activity(str(exc), level="warning")
                    break  # Stop checking YouTube feeds entirely
                logger.error("Error checking feed %s: %s", feed_db["name"], exc)
                db.update_feed_checked(feed_db["name"], error=str(exc))

    # Clean up orphaned temp files from interrupted downloads/cuts
    _cleanup_temp_files(config)

    # Auto-prune completed Pocket Casts episodes
    if config.pocketcasts.auto_prune and config.pocketcasts.email:
        try:
            await _auto_prune_pocketcasts(config, db)
        except Exception as exc:
            logger.warning("Pocket Casts auto-prune failed: %s", exc)

    worker_done("checking_feeds")
    log_activity("Feed check complete")


async def _reevaluate_duration_unknown(resolved, config, db) -> None:
    """Re-check episodes filtered as duration_unknown (likely livestreams).

    If the video now has a real duration, run filters again and either
    unfilter it or reclassify. Give up after 72 hours.
    """
    from siphon.youtube import _api_get, _parse_iso8601_duration, VIDEOS_URL

    episodes = db.conn.execute(
        "SELECT video_id, discovered_at FROM episodes "
        "WHERE feed_name = ? AND status = 'filtered' AND filter_reason = 'duration_unknown'",
        (resolved.name,),
    ).fetchall()

    if not episodes:
        return

    api_key = config.youtube.api_key
    cooldown_hours = config.youtube.quota_cooldown_hours
    loop = asyncio.get_event_loop()

    # Check age — give up after 72 hours
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
    stale = [ep for ep in episodes if (ep["discovered_at"] or "") < cutoff]
    fresh = [ep for ep in episodes if ep not in stale]

    # Mark stale ones as permanently filtered
    for ep in stale:
        db.update_episode_status(ep["video_id"], resolved.name, "filtered", filter_reason="stale_livestream")
        logger.info("Gave up on duration_unknown %s/%s after 72h", resolved.name, ep["video_id"])

    if not fresh:
        return

    # Build title lookup from DB for re-filtering
    all_eps = {ep["video_id"]: ep for ep in db.get_episodes_by_feed(resolved.name)}

    # Batch-fetch durations for fresh ones
    video_ids = [ep["video_id"] for ep in fresh]
    try:
        data = await loop.run_in_executor(None, _api_get, VIDEOS_URL, {
            "key": api_key,
            "id": ",".join(video_ids),
            "part": "contentDetails",
        }, cooldown_hours)
    except Exception as exc:
        logger.warning("Failed to re-check duration_unknown: %s", exc)
        return

    for item in data.get("items", []):
        vid = item["id"]
        raw_dur = item.get("contentDetails", {}).get("duration", "")
        duration = _parse_iso8601_duration(raw_dur)

        if duration == 0:
            continue  # Still live or no duration, try again next cycle

        # Re-run filters with the real duration and original title
        db_ep = all_eps.get(vid, {})
        entry = {"id": vid, "duration": duration, "url": f"https://www.youtube.com/watch?v={vid}",
                 "title": db_ep.get("title", "")}
        reason = apply_filters(
            entry, resolved.block_shorts, resolved.title_exclude,
            resolved.min_duration_seconds, None,  # skip date_cutoff, already passed
        )

        if reason:
            db.update_episode_status(vid, resolved.name, "filtered", filter_reason=reason, duration=duration)
            logger.info("Re-evaluated %s/%s: filtered (%s, %ds)", resolved.name, vid, reason, duration)
        else:
            # Unfilter — set to pending for download
            eligible_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            db.update_episode_status(vid, resolved.name, "pending", duration=duration, eligible_at=eligible_at)
            logger.info("Re-evaluated %s/%s: now eligible (%ds)", resolved.name, vid, duration)
            log_activity(f"Livestream VOD now available: {vid}", feed=resolved.name)


def _normalize_youtube_url(url: str) -> str:
    """Ensure a YouTube channel URL points to the /videos tab."""
    # Strip trailing slash
    url = url.rstrip("/")
    # If it's a channel URL without a tab, add /videos
    if ("youtube.com/@" in url or "youtube.com/c/" in url or "youtube.com/channel/" in url):
        if not any(url.endswith(f"/{tab}") for tab in ("videos", "shorts", "streams", "playlists", "community")):
            url += "/videos"
    return url


async def _check_youtube_feed(resolved, config, db) -> None:
    """Check a YouTube feed for new episodes via YouTube Data API.

    First check: resolve channel ID, get metadata, paginate backwards
    until date_cutoff.
    Subsequent checks: paginate backwards until we find a known video
    or hit date_cutoff.
    """
    from siphon.youtube import get_channel_metadata, list_videos, resolve_channel_id

    api_key = config.youtube.api_key
    cooldown_hours = config.youtube.quota_cooldown_hours
    feed_db = db.get_feed(resolved.name)
    channel_id = feed_db.get("channel_id") if feed_db else None

    # Resolve channel ID if we don't have one
    if not channel_id:
        log_activity("Resolving channel ID...", feed=resolved.name)
        loop = asyncio.get_event_loop()
        channel_id = await loop.run_in_executor(
            None, resolve_channel_id, resolved.url, api_key, cooldown_hours,
        )
        if not channel_id:
            raise Exception(f"Could not resolve channel ID for {resolved.url}")
        db.update_feed_channel_id(resolved.name, channel_id)

        # Get channel metadata (thumbnail, title)
        meta = await loop.run_in_executor(
            None, get_channel_metadata, channel_id, api_key, cooldown_hours,
        )
        if meta.get("image_url"):
            db.update_feed_image(resolved.name, meta["image_url"])

    # Re-evaluate duration_unknown episodes (livestreams that may now be VODs)
    await _reevaluate_duration_unknown(resolved, config, db)

    # Get known video IDs for this feed to detect where to stop
    # Exclude filtered episodes — they shouldn't block paging to older content
    existing = db.get_episodes_by_feed(resolved.name)
    known_ids = {ep["video_id"] for ep in existing if ep["status"] != "filtered"}

    # Fetch videos from API — stops at cutoff date or known video
    loop = asyncio.get_event_loop()
    entries = await loop.run_in_executor(
        None, list_videos, channel_id, api_key,
        resolved.date_cutoff, known_ids, 200, cooldown_hours,
        config.youtube.country,
    )

    new_count = _insert_youtube_entries(entries, resolved, db)

    if new_count > 0:
        log_activity(f"Found {new_count} new episodes", feed=resolved.name)


def _insert_youtube_entries(entries: list, resolved, db) -> int:
    """Insert YouTube entries into DB with filters. Returns new episode count."""
    new_count = 0
    for entry in entries:
        video_id = entry["id"]

        if db.get_episode(video_id, resolved.name) is not None:
            continue

        reason = apply_filters(
            entry,
            resolved.block_shorts,
            resolved.title_exclude,
            resolved.min_duration_seconds,
            resolved.date_cutoff,
        )

        if reason is not None:
            logger.info("Filtered %s from %s: %s", video_id, resolved.name, reason)
            db.insert_episode(
                video_id=video_id,
                feed_name=resolved.name,
                title=entry.get("title", ""),
                description=entry.get("description"),
                thumbnail_url=entry.get("thumbnail"),
                channel_name=entry.get("channel"),
                duration=entry.get("duration"),
                upload_date=entry.get("upload_date"),
                eligible_at=None,
                status="filtered",
                filter_reason=reason,
            )
        else:
            # SponsorBlock delay based on publish date, not discovery time
            # Prefer full ISO 8601 (published_at) over YYYYMMDD (upload_date)
            published_at = entry.get("published_at")
            upload_str = entry.get("upload_date")
            pub_dt = None
            if published_at and resolved.sponsorblock_delay_minutes:
                try:
                    pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            if pub_dt is None and upload_str and resolved.sponsorblock_delay_minutes:
                try:
                    pub_dt = datetime.strptime(upload_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            if pub_dt is not None and resolved.sponsorblock_delay_minutes:
                eligible_dt = pub_dt + timedelta(minutes=resolved.sponsorblock_delay_minutes)
                if eligible_dt <= datetime.now(timezone.utc):
                    eligible_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    eligible_at = eligible_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                eligible_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            db.insert_episode(
                video_id=video_id,
                feed_name=resolved.name,
                title=entry.get("title", ""),
                description=entry.get("description"),
                thumbnail_url=entry.get("thumbnail"),
                channel_name=entry.get("channel"),
                duration=entry.get("duration"),
                upload_date=entry.get("upload_date"),
                eligible_at=eligible_at,
                status="pending",
            )
            new_count += 1

    return new_count


async def _check_podcast_feed(resolved, db) -> None:
    """Check a podcast feed for new episodes."""
    from siphon.podcast import fetch_podcast_rss, parse_podcast_feed

    loop = asyncio.get_event_loop()
    xml_bytes = await loop.run_in_executor(None, fetch_podcast_rss, resolved.url)
    feed_data = parse_podcast_feed(xml_bytes)

    # Store podcast artwork in DB
    feed_image_url = feed_data.get("image_url")
    if feed_image_url:
        db.update_feed_image(resolved.name, feed_image_url)

    episodes = feed_data.get("episodes") or []
    new_count = 0

    for ep in episodes:
        guid = ep["guid"]

        if db.get_episode(guid, resolved.name) is not None:
            continue

        # Build a filter-compatible entry dict
        entry = {
            "id": guid,
            "title": ep.get("title", ""),
            "url": ep.get("audio_url", ""),
            "duration": ep.get("duration"),
            "upload_date": ep.get("pub_date"),
        }

        reason = apply_filters(
            entry,
            block_shorts=False,  # No shorts concept for podcasts
            title_exclude=resolved.title_exclude,
            min_duration_seconds=resolved.min_duration_seconds,
            date_cutoff=resolved.date_cutoff,
        )

        if reason is not None:
            logger.info("Filtered %s from %s: %s", guid, resolved.name, reason)
            db.insert_episode(
                video_id=guid,
                feed_name=resolved.name,
                title=ep.get("title", ""),
                description=ep.get("description"),
                thumbnail_url=ep.get("thumbnail_url"),
                channel_name=feed_data.get("title"),
                duration=ep.get("duration"),
                upload_date=ep.get("pub_date"),
                eligible_at=None,
                status="filtered",
                filter_reason=reason,
            )
        else:
            # Podcasts don't need SponsorBlock delay — eligible immediately
            eligible_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            db.insert_episode(
                video_id=guid,
                feed_name=resolved.name,
                title=ep.get("title", ""),
                description=ep.get("description"),
                thumbnail_url=ep.get("thumbnail_url"),
                channel_name=feed_data.get("title"),
                duration=ep.get("duration"),
                upload_date=ep.get("pub_date"),
                eligible_at=eligible_at,
                status="pending",
            )
            new_count += 1

    if new_count > 0:
        log_activity(f"Found {new_count} new episodes", feed=resolved.name)
    logger.info("Checked podcast feed %s: %d episodes, %d new", resolved.name, len(episodes), new_count)


# ---------------------------------------------------------------------- #
# Worker 1: Download processing
# ---------------------------------------------------------------------- #


def _get_schedule_params(config: SiphonConfig, feed_type: str) -> tuple[int, int, int]:
    """Return (max_downloads_per_hour, download_workers, download_delay_seconds)
    for the given feed type."""
    sched = config.schedule
    if feed_type == "podcast":
        return (
            sched.podcast_max_downloads_per_hour,
            sched.podcast_download_workers,
            sched.podcast_download_delay_seconds,
        )
    return (
        sched.youtube_max_downloads_per_hour,
        sched.youtube_download_workers,
        sched.youtube_download_delay_seconds,
    )


def _get_keep_per_feed(config: SiphonConfig, feed_type: str) -> int:
    """Return the keep_per_feed limit for the given feed type."""
    if feed_type == "podcast":
        return config.storage.podcast_keep_per_feed
    return config.storage.youtube_keep_per_feed


_download_lock = asyncio.Lock()


async def process_downloads(config: SiphonConfig, db: Database) -> None:
    """Download eligible episodes and prune disk when needed."""
    if _download_lock.locked():
        logger.info("process_downloads already running, skipping")
        return
    async with _download_lock:
        await _process_downloads_inner(config, db)


async def _process_downloads_inner(config: SiphonConfig, db: Database) -> None:
    """Inner download processing — runs under _download_lock."""
    worker_start("checking_downloads")

    _downloaded: set = set()  # track episodes downloaded this cycle

    promoted = db.promote_eligible_episodes()
    if promoted and promoted > 0:
        log_activity(f"Promoted {promoted} episodes to eligible")
    db.reset_stale_downloads()
    db.retry_failed_episodes()

    for feed_type in ("youtube", "podcast"):
        max_per_hour, workers, delay = _get_schedule_params(config, feed_type)

        while True:  # Keep downloading until queue empty or rate limited
            if check_paused():
                log_activity("Paused — stopping downloads")
                return

            recent = db.get_recent_download_count(hours=1, feed_type=feed_type)
            remaining = max(0, max_per_hour - recent)
            limit = min(workers, remaining)

            if limit == 0:
                if remaining == 0:
                    log_activity(
                        f"Rate limit reached for {feed_type} ({recent}/{max_per_hour}/hr)",
                        level="warning",
                    )
                    logger.info(
                        "Hourly %s download cap reached (%d/%d), skipping",
                        feed_type, recent, max_per_hour,
                    )
                break

            episodes = db.get_eligible_episodes(limit, feed_type=feed_type)
            if not episodes:
                break

            for i, episode in enumerate(episodes):
                if i > 0 and delay > 0:
                    logger.debug("Throttling %s: waiting %ds", feed_type, delay)
                    await asyncio.sleep(delay)

                video_id = episode["video_id"]
                feed_name = episode["feed_name"]
                title = episode.get("title", video_id)

                if (video_id, feed_name) in _downloaded:
                    continue
                _downloaded.add((video_id, feed_name))

                feed_config = None
                for fc in config.feeds:
                    if fc.name == feed_name:
                        feed_config = fc
                        break

                if feed_config is None:
                    logger.warning("Feed config %r not found for %s, skipping", feed_name, video_id)
                    continue

                resolved = resolve_feed(feed_config, config.defaults)
                db.update_episode_status(video_id, feed_name, "downloading")
                log_activity(f"Downloading {title[:50]}", feed=feed_name)
                worker_start("download")

                try:
                    if resolved.type == "podcast":
                        await _download_podcast_episode(episode, resolved, config, db)
                    else:
                        await _download_youtube_episode(episode, resolved, config, db)

                    log_activity(f"Downloaded {title[:50]}", feed=feed_name)

                except Exception as exc:
                    logger.error("Download failed for %s/%s: %s", feed_name, video_id, exc)
                    log_activity(f"Download failed: {str(exc)[:80]}", feed=feed_name, level="error")
                    db.update_episode_status(
                        video_id, feed_name, "failed",
                        error=str(exc),
                        retry_count=episode["retry_count"] + 1,
                    )
                finally:
                    worker_done("download")

    await _prune_disk(config, db)
    worker_done("checking_downloads")


async def _download_youtube_episode(episode, resolved, config, db) -> None:
    """Download a YouTube episode."""
    video_id = episode["video_id"]
    feed_name = episode["feed_name"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, download_video, video_url, resolved,
        config.cookies, config.storage.download_dir,
        config.youtube.player_clients,
    )

    result = find_downloaded_file(config.storage.download_dir, resolved.name, video_id)

    if result is not None:
        path, size = result
        mime = "audio/mpeg" if resolved.mode == "audio" else "video/mp4"

        # Validate downloaded file
        from siphon.cutter import validate_file, has_real_video_stream
        is_valid = await loop.run_in_executor(None, validate_file, path)
        if not is_valid:
            logger.error("Downloaded file failed validation: %s/%s", feed_name, video_id)
            log_activity(f"Download corrupt: {episode.get('title', video_id)[:50]}", feed=feed_name, level="error")
            _delete_file(path)
            db.update_episode_status(video_id, feed_name, "failed", error="Downloaded file corrupt (ffprobe validation failed)")
            return

        # For video feeds, verify the file has a real video stream (not just an embedded thumbnail).
        # yt-dlp's /best fallback can silently produce audio-only files with the thumbnail attached
        # as a fake PNG video track.
        if resolved.mode == "video":
            has_video = await loop.run_in_executor(None, has_real_video_stream, path)
            if not has_video:
                logger.error("Downloaded file has no real video stream: %s/%s", feed_name, video_id)
                log_activity(f"Download audio-only (expected video): {episode.get('title', video_id)[:50]}", feed=feed_name, level="error")
                _delete_file(path)
                db.update_episode_status(
                    video_id, feed_name, "failed",
                    error="Downloaded file has no real video stream (got audio-only with embedded thumbnail)",
                    retry_count=episode["retry_count"] + 1,
                )
                return

        # Get actual duration via ffprobe
        duration = await _get_file_duration(loop, path)

        # Post-download filter: check duration now that we know it
        filter_reason = _post_download_filter(duration, resolved)
        if filter_reason:
            logger.info("Post-download filtered %s/%s: %s (duration=%s)", feed_name, video_id, filter_reason, duration)
            log_activity(f"Filtered after download: {episode.get('title', video_id)[:50]} ({filter_reason})", feed=feed_name)
            _delete_file(path)
            db.update_episode_status(video_id, feed_name, "filtered", filter_reason=filter_reason, duration=duration)
            return

        # Query SponsorBlock for segment info if enabled
        sb_kwargs: dict = {}
        if resolved.sponsorblock:
            from siphon.sponsorblock import get_segment_info
            sb_cuts, sb_secs = await loop.run_in_executor(
                None, get_segment_info, video_id, resolved.sponsorblock_categories,
            )
            logger.info("SponsorBlock segments for %s: %d (%.1fs removed)", video_id, sb_cuts, sb_secs)
            sb_kwargs = {"sb_cuts_applied": sb_cuts, "sb_seconds_removed": sb_secs}

        # Normalize timestamps (fixes SB stream-copy desync)
        from siphon.cutter import normalize_timestamps
        await loop.run_in_executor(None, normalize_timestamps, path)

        # Update file size after normalization
        if os.path.exists(path):
            size = os.path.getsize(path)

        next_status = "pending_whisper" if resolved.llm_trim else "done"
        db.update_episode_status(
            video_id, feed_name, next_status,
            file_path=path, file_size=size, mime_type=mime, duration=duration,
            **sb_kwargs,
        )
        logger.info("Downloaded YouTube %s for feed %s → %s", video_id, feed_name, next_status)
    else:
        db.update_episode_status(
            video_id, feed_name, "failed",
            error="File not found after download",
        )


async def _download_podcast_episode(episode, resolved, config, db) -> None:
    """Download a podcast episode audio file."""
    from siphon.podcast import download_podcast_audio, episode_filename

    video_id = episode["video_id"]  # This is the GUID for podcasts
    feed_name = episode["feed_name"]

    from siphon.podcast import fetch_podcast_rss, parse_podcast_feed

    loop = asyncio.get_event_loop()
    xml_bytes = await loop.run_in_executor(None, fetch_podcast_rss, resolved.url)
    feed_data = parse_podcast_feed(xml_bytes)

    # Find the episode by GUID
    audio_url = None
    for ep in feed_data.get("episodes", []):
        if ep["guid"] == video_id:
            audio_url = ep["audio_url"]
            break

    if audio_url is None:
        db.update_episode_status(
            video_id, feed_name, "failed",
            error=f"Audio URL not found for GUID {video_id}",
        )
        return

    filename = episode_filename(video_id, audio_url)
    output_path = os.path.join(config.storage.download_dir, feed_name, filename)

    file_size = await loop.run_in_executor(
        None, download_podcast_audio, audio_url, output_path,
    )

    # Validate downloaded file
    loop = asyncio.get_event_loop()
    from siphon.cutter import validate_file
    is_valid = await loop.run_in_executor(None, validate_file, output_path)
    if not is_valid:
        logger.error("Downloaded podcast file failed validation: %s/%s", feed_name, video_id)
        log_activity(f"Download corrupt: {episode.get('title', video_id)[:50]}", feed=feed_name, level="error")
        _delete_file(output_path)
        db.update_episode_status(video_id, feed_name, "failed", error="Downloaded file corrupt (ffprobe validation failed)")
        return

    # Get actual duration via ffprobe
    duration = await _get_file_duration(loop, output_path)

    # Post-download filter: check duration
    filter_reason = _post_download_filter(duration, resolved)
    if filter_reason:
        logger.info("Post-download filtered %s/%s: %s (duration=%s)", feed_name, video_id, filter_reason, duration)
        log_activity(f"Filtered after download: {episode.get('title', video_id)[:50]} ({filter_reason})", feed=feed_name)
        _delete_file(output_path)
        db.update_episode_status(video_id, feed_name, "filtered", filter_reason=filter_reason, duration=duration)
        return

    # Normalize timestamps if needed
    from siphon.cutter import normalize_timestamps
    await loop.run_in_executor(None, normalize_timestamps, output_path)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)

    next_status = "pending_whisper" if resolved.llm_trim else "done"
    db.update_episode_status(
        video_id, feed_name, next_status,
        file_path=output_path,
        file_size=file_size,
        mime_type="audio/mpeg",
        duration=duration,
    )
    logger.info("Downloaded podcast %s for feed %s", video_id, feed_name)


# ---------------------------------------------------------------------- #
# Worker 2: Whisper transcription
# ---------------------------------------------------------------------- #

# Hard ceiling on a single transcription. A healthy GPU base/large transcribe
# finishes well inside this even for multi-hour episodes; blowing past it means
# the call is hung (the leading suspect for the "everything skipped after a
# week" degradation). NOTE: run_in_executor threads cannot be cancelled, so a
# true hang still leaks one worker thread — the resource snapshot's thread count
# is what reveals that. The timeout's job here is to surface the hang in the log
# and let the episode retry instead of stalling the whole queue forever.
WHISPER_TIMEOUT_SECONDS = 7200  # 2 hours

_whisper_lock = asyncio.Lock()


async def process_whisper(config: SiphonConfig, db: Database) -> None:
    """Process one episode through Whisper transcription."""
    if _whisper_lock.locked():
        logger.debug("process_whisper already running, skipping")
        return
    async with _whisper_lock:
        await _process_whisper_inner(config, db)


async def _process_whisper_inner(config: SiphonConfig, db: Database) -> None:
    """Inner Whisper processing — loops until queue empty or paused."""
    num_workers = config.llm.whisper_workers

    _attempted: set = set()
    while True:
        if check_paused():
            log_activity("Paused — stopping Whisper")
            return

        episodes = db.get_pending_whisper(limit=num_workers)
        # Filter out episodes we already attempted this cycle
        episodes = [ep for ep in episodes if ep["video_id"] not in _attempted]
        if not episodes:
            return

        for ep in episodes:
            _attempted.add(ep["video_id"])

        if len(episodes) > 1:
            tasks = [asyncio.create_task(_process_one_whisper(ep, config, db)) for ep in episodes]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await _process_one_whisper(episodes[0], config, db)


async def _process_one_whisper(ep: dict, config: SiphonConfig, db: Database) -> None:
    """Process a single episode through Whisper transcription."""
    video_id = ep["video_id"]
    feed_name = ep["feed_name"]
    file_path = ep.get("file_path")
    title = ep.get("title", video_id)

    if not file_path or not os.path.exists(file_path):
        logger.error("File missing for %s/%s, marking failed", feed_name, video_id)
        db.update_episode_status(
            video_id, feed_name, "failed",
            error="Media file missing for Whisper",
        )
        return

    log_activity(f"Whisper: transcribing {title[:50]}", feed=feed_name)
    worker_start("whisper")

    t0 = time.time()
    loop = asyncio.get_event_loop()

    try:
        from siphon.transcribe import transcribe
        from siphon.cutter import extract_audio
        import tempfile

        # Step 1: Extract audio if needed (video files)
        is_audio = file_path.lower().endswith((".mp3", ".m4a", ".ogg", ".wav", ".flac", ".aac"))
        temp_audio = None

        if is_audio:
            whisper_input = file_path
        else:
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()
            try:
                await loop.run_in_executor(None, extract_audio, file_path, temp_audio.name)
                whisper_input = temp_audio.name
            except Exception:
                os.unlink(temp_audio.name)
                raise

        try:
            # Step 2: Transcribe (hard timeout surfaces a hung GPU/CTranslate2
            # call instead of letting it stall the queue indefinitely)
            transcript = await asyncio.wait_for(
                loop.run_in_executor(
                    None, transcribe, whisper_input,
                    config.llm.whisper_model, config.llm.whisper_device,
                    config.llm.whisper_word_timestamps, config.llm.whisper_workers,
                ),
                timeout=WHISPER_TIMEOUT_SECONDS,
            )
        finally:
            if temp_audio is not None:
                try:
                    os.unlink(temp_audio.name)
                except OSError:
                    pass

        whisper_duration = time.time() - t0

        # Step 3: Save transcript JSON to disk
        transcript_file = _transcript_path(config, feed_name, video_id)
        os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
        transcript_json = json.dumps(transcript)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript_json)

        # Step 4: Record metrics and move to pending_claude
        word_count = len(transcript.get("words", []))
        segment_count = len(transcript.get("segments", []))

        db.update_episode_status(
            video_id, feed_name, "pending_claude",
            whisper_duration_seconds=round(whisper_duration, 2),
            whisper_word_count=word_count,
            whisper_segment_count=segment_count,
            transcript_size_bytes=len(transcript_json),
            whisper_model=config.llm.whisper_model,
            whisper_device=config.llm.whisper_device,
        )

        _fmt_time = _format_duration(whisper_duration)
        log_activity(
            f"Whisper: {word_count} words ({_fmt_time})",
            feed=feed_name,
        )
        logger.info(
            "Whisper done for %s/%s: %d words, %d segments in %.1fs",
            feed_name, video_id, word_count, segment_count, whisper_duration,
        )

    except Exception as exc:
        # logger.exception captures the full traceback to the file log — the
        # exception *type* (CUDA OOM vs subprocess spawn fail vs timeout) is the
        # key clue for the "everything skipped" hunt.
        logger.exception("Whisper failed for %s/%s", feed_name, video_id)
        log_activity(f"Whisper failed: {str(exc)[:80]}", feed=feed_name, level="error")
        retry_count = (ep.get("llm_retry_count") or 0) + 1
        if retry_count >= 3:
            db.update_episode_status(
                video_id, feed_name, "done",
                llm_trim_status="skipped",
                llm_retry_count=retry_count,
                error=f"Whisper skipped after {retry_count} failures: {exc}",
            )
            log_activity(f"Whisper: skipped after {retry_count} failures", feed=feed_name, level="warning")
        else:
            db.update_episode_status(
                video_id, feed_name, "pending_whisper",
                llm_retry_count=retry_count,
                error=str(exc),
            )
    finally:
        worker_done("whisper")


# ---------------------------------------------------------------------- #
# Worker 3: Claude ad detection
# ---------------------------------------------------------------------- #

# Claude CLI back-off.  Systemic failures — a logged-out CLI, API overload,
# usage limits — pause the whole worker instead of burning each episode's
# three strikes (which is how a two-day OAuth outage in Aug 2026 turned into
# ~110 permanently skipped episodes).  Module state survives scheduler ticks.
CLAUDE_AUTH_COOLDOWN_SECONDS = 600
CLAUDE_TRANSIENT_BACKOFF_SECONDS = (120, 300, 900, 1800)
CLAUDE_AUTH_ALERT_KEY = "claude_auth"
CLAUDE_AUTH_ALERT_MESSAGE = (
    "Claude CLI is logged out — run `claude /login` on the host. "
    "Ad detection is paused; queued episodes resume automatically."
)

_claude_cooldown_until: float = 0.0
_claude_cooldown_reason: str = ""
_claude_transient_failures: int = 0


def claude_cooldown_remaining() -> float:
    """Seconds until the Claude worker may run again (0 when not paused)."""
    return max(0.0, _claude_cooldown_until - time.time())


def reset_claude_backoff() -> None:
    """Forget any cooldown/backoff state (after a success, and in tests)."""
    global _claude_cooldown_until, _claude_cooldown_reason, _claude_transient_failures
    _claude_cooldown_until = 0.0
    _claude_cooldown_reason = ""
    _claude_transient_failures = 0


def _pause_claude_worker(exc: Exception, feed_name: str) -> None:
    """React to a systemic Claude failure: pause the worker, tell the operator once."""
    global _claude_cooldown_until, _claude_cooldown_reason, _claude_transient_failures
    already_paused = claude_cooldown_remaining() > 0
    if isinstance(exc, ClaudeAuthError):
        seconds = CLAUDE_AUTH_COOLDOWN_SECONDS
        reason = "Claude CLI logged out"
        if set_alert(CLAUDE_AUTH_ALERT_KEY, CLAUDE_AUTH_ALERT_MESSAGE):
            log_activity(CLAUDE_AUTH_ALERT_MESSAGE, level="error")
            logger.error("%s (%s)", CLAUDE_AUTH_ALERT_MESSAGE, exc)
    else:
        # A batch of concurrent failures is one incident, not several.
        if not already_paused:
            _claude_transient_failures += 1
        idx = min(_claude_transient_failures, len(CLAUDE_TRANSIENT_BACKOFF_SECONDS)) - 1
        seconds = CLAUDE_TRANSIENT_BACKOFF_SECONDS[max(idx, 0)]
        reason = str(exc)[:80]
        if not already_paused:
            log_activity(f"Claude paused {seconds // 60} min: {reason}", feed=feed_name, level="warning")
            logger.warning("Claude worker paused for %ds after transient failure: %s", seconds, exc)
    _claude_cooldown_until = max(_claude_cooldown_until, time.time() + seconds)
    _claude_cooldown_reason = reason


def _note_claude_success() -> None:
    """A CLI run succeeded: lift any cooldown and clear the logged-out alert."""
    reset_claude_backoff()
    if clear_alert(CLAUDE_AUTH_ALERT_KEY):
        log_activity("Claude CLI is working again", level="info")
        logger.info("Claude CLI recovered; alert cleared")


_claude_lock = asyncio.Lock()


async def process_claude(config: SiphonConfig, db: Database) -> None:
    """Process episodes through Claude ad detection, up to claude_concurrency."""
    if _claude_lock.locked():
        logger.debug("process_claude already running, skipping")
        return
    async with _claude_lock:
        await _process_claude_inner(config, db)


async def _process_claude_inner(config: SiphonConfig, db: Database) -> None:
    """Inner Claude processing — loops until queue empty or paused."""
    concurrency = config.llm.claude_concurrency

    _attempted: set = set()
    while True:
        if check_paused():
            log_activity("Paused — stopping Claude")
            return

        remaining = claude_cooldown_remaining()
        if remaining > 0:
            logger.debug("Claude worker in cooldown for %.0fs more (%s)", remaining, _claude_cooldown_reason)
            return

        episodes = db.get_pending_claude(limit=concurrency)
        episodes = [ep for ep in episodes if ep["video_id"] not in _attempted]
        if not episodes:
            return

        for ep in episodes:
            _attempted.add(ep["video_id"])

        sem = asyncio.Semaphore(concurrency)
        tasks = []

        for ep in episodes:
            async def _do(ep=ep):
                async with sem:
                    await _process_one_claude(ep, config, db)
            tasks.append(asyncio.create_task(_do()))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Claude task failed: %s", r)


async def _process_one_claude(ep: dict, config: SiphonConfig, db: Database) -> None:
    """Run Claude ad detection + ffmpeg cut on a single episode."""
    from siphon.ad_detect import build_transcript_for_claude, detect_ads, filter_segments, resolve_prompt
    from siphon.cutter import cut_segments

    video_id = ep["video_id"]
    feed_name = ep["feed_name"]
    file_path = ep.get("file_path")
    title = ep.get("title", video_id)
    retry_count = ep.get("llm_retry_count") or 0

    # Load transcript from disk
    transcript_file = _transcript_path(config, feed_name, video_id)
    if not os.path.exists(transcript_file):
        logger.error("Transcript missing for %s/%s, resetting to pending_whisper", feed_name, video_id)
        db.update_episode_status(video_id, feed_name, "pending_whisper")
        return

    feed_config = None
    for fc in config.feeds:
        if fc.name == feed_name:
            feed_config = fc
            break
    if feed_config is None:
        logger.warning("Feed config %r not found for %s, skipping", feed_name, video_id)
        return

    resolved = resolve_feed(feed_config, config.defaults)

    log_activity(f"Claude: processing {title[:50]}", feed=feed_name)
    db.update_episode_status(video_id, feed_name, "pending_claude", llm_trim_status="pending")
    worker_start("claude")

    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        transcript_text = transcript.get("text", "")

        if not transcript_text.strip():
            logger.info("Empty transcript for %s/%s, skipping ad detection", feed_name, video_id)
            # Clean up transcript file
            try:
                os.remove(transcript_file)
            except OSError:
                pass
            db.update_episode_status(
                video_id, feed_name, "done",
                llm_trim_status="done",
                llm_segments_json=json.dumps({"segments": []}),
                llm_cuts_applied=0,
            )
            return

        # Skip word timestamps for long episodes
        words = transcript.get("words")
        duration_minutes = transcript.get("duration", 0) / 60
        if duration_minutes > config.llm.word_timestamps_max_minutes:
            logger.info(
                "Episode is %.0f min (> %d min limit), using segment-only timestamps",
                duration_minutes, config.llm.word_timestamps_max_minutes,
            )
            words = None

        t0_claude = time.time()
        loop = asyncio.get_event_loop()

        prompt = resolve_prompt(resolved, config.llm)
        raw_result = await loop.run_in_executor(
            None, detect_ads,
            transcript_text, prompt, config.llm.claude_model,
            config.llm.claude_effort, words, transcript.get("segments"),
            title, resolved.display_name or feed_name,
        )

        claude_duration = time.time() - t0_claude
        _note_claude_success()

        all_segments = raw_result.get("segments", [])
        logger.info("Claude detected %d potential ad segments", len(all_segments))

        # Filter
        high_confidence, marginal = filter_segments(
            all_segments,
            confidence_threshold=config.llm.confidence_threshold,
            min_duration=config.llm.min_segment_duration,
            max_duration=config.llm.max_segment_duration,
            merge_gap=config.llm.merge_gap_seconds,
        )

        logger.info(
            "After filtering: %d to cut, %d marginal",
            len(high_confidence), len(marginal),
        )

        # Cut with ffmpeg — validate output, retry once, fall back to uncut if still bad
        ffmpeg_duration = 0.0
        cut_failed = False
        if high_confidence and file_path:
            t0_ffmpeg = time.time()
            for attempt in range(2):
                try:
                    await loop.run_in_executor(None, cut_segments, file_path, high_confidence)
                    ffmpeg_duration = time.time() - t0_ffmpeg
                    logger.info("Applied %d cuts to %s in %.1fs", len(high_confidence), file_path, ffmpeg_duration)
                    break
                except RuntimeError as exc:
                    if attempt == 0:
                        logger.warning("Cut failed for %s, retrying: %s", file_path, exc)
                    else:
                        logger.error("Cut failed twice for %s, serving uncut: %s", file_path, exc)
                        log_activity(f"LLM cut failed, serving uncut: {title[:40]}", feed=feed_name, level="warning")
                        cut_failed = True
                        high_confidence = []  # Record 0 cuts applied

        # Normalize timestamps after cuts
        if file_path and not cut_failed:
            from siphon.cutter import normalize_timestamps
            await loop.run_in_executor(None, normalize_timestamps, file_path)

        # Delete transcript file
        try:
            os.remove(transcript_file)
        except OSError:
            pass

        # Update file_size after cuts
        new_size = None
        if file_path and os.path.exists(file_path):
            new_size = os.path.getsize(file_path)

        # Build audit data (full segment dicts for Time Saved calculation)
        audit = {
            "segments": all_segments,
            "high_confidence": high_confidence,
            "marginal": marginal,
        }

        update_kwargs = {
            "llm_trim_status": "done",
            "llm_segments_json": json.dumps(audit),
            "llm_cuts_applied": len(high_confidence),
            "claude_duration_seconds": round(claude_duration, 2),
            "ffmpeg_duration_seconds": round(ffmpeg_duration, 2),
        }
        if new_size is not None:
            update_kwargs["file_size"] = new_size

        db.update_episode_status(video_id, feed_name, "done", **update_kwargs)

        cuts = len(high_confidence)
        _fmt_claude = _format_duration(claude_duration)
        _fmt_ffmpeg = _format_duration(ffmpeg_duration) if ffmpeg_duration > 0 else "0:00"
        log_activity(f"Claude: {cuts} cuts ({_fmt_claude})", feed=feed_name)
        if ffmpeg_duration > 0:
            log_activity(f"ffmpeg: cut applied ({_fmt_ffmpeg})", feed=feed_name)

    except (ClaudeAuthError, ClaudeTransientError) as exc:
        # Systemic, not this episode's fault: leave it queued with its retry
        # count untouched and pause the worker instead.
        logger.error("Claude unavailable for %s/%s (%s): %s", feed_name, video_id, type(exc).__name__, exc)
        db.update_episode_status(
            video_id, feed_name, "pending_claude",
            llm_trim_status="error",
            llm_segments_json=json.dumps({"error": str(exc)}),
        )
        _pause_claude_worker(exc, feed_name)
    except Exception as exc:
        # Full traceback to the file log — the exception type pins down whether
        # the Claude CLI subprocess failed to spawn, timed out, or returned junk.
        logger.exception("Claude failed for %s/%s", feed_name, video_id)
        log_activity(f"Claude failed: {str(exc)[:80]}", feed=feed_name, level="error")
        new_retry = retry_count + 1
        if new_retry >= 3:
            # Clean up transcript file on give-up
            try:
                os.remove(transcript_file)
            except OSError:
                pass
            db.update_episode_status(
                video_id, feed_name, "done",
                llm_trim_status="skipped",
                llm_retry_count=new_retry,
                error=f"Claude skipped after {new_retry} failures: {exc}",
            )
            log_activity(f"Claude: skipped after {new_retry} failures", feed=feed_name, level="warning")
        else:
            db.update_episode_status(
                video_id, feed_name, "pending_claude",
                llm_trim_status="error",
                llm_segments_json=json.dumps({"error": str(exc)}),
                llm_retry_count=new_retry,
            )
            log_activity(f"Claude: failed (attempt {new_retry}/3)", feed=feed_name, level="error")
    finally:
        worker_done("claude")


# ---------------------------------------------------------------------- #
# Disk pruning
# ---------------------------------------------------------------------- #


async def _prune_disk(config: SiphonConfig, db: Database) -> None:
    """Remove old episodes to stay within per-feed and global disk limits."""

    for feed in config.feeds:
        keep_limit = _get_keep_per_feed(config, feed.type)
        count = db.get_feed_episode_count(feed.name)
        if count > keep_limit:
            excess = count - keep_limit
            oldest = db.get_oldest_done_episodes(excess, feed_name=feed.name)
            for ep in oldest:
                if ep.get("file_path"):
                    try:
                        os.remove(ep["file_path"])
                    except OSError:
                        pass
                db.update_episode_status(ep["video_id"], ep["feed_name"], "pruned")
                logger.info("Pruned %s from %s (per-feed limit)", ep["video_id"], ep["feed_name"])

    max_bytes = config.storage.max_disk_gb * (1024 ** 3)
    usage = db.get_disk_usage()

    if usage > max_bytes:
        oldest = db.get_oldest_done_episodes(50)
        for ep in oldest:
            if usage <= max_bytes:
                break
            file_size = ep.get("file_size") or 0
            if ep.get("file_path"):
                try:
                    os.remove(ep["file_path"])
                except OSError:
                    pass
            db.update_episode_status(ep["video_id"], ep["feed_name"], "pruned")
            usage -= file_size
            logger.info("Pruned %s from %s (disk limit)", ep["video_id"], ep["feed_name"])


# ---------------------------------------------------------------------- #
# Temp file cleanup
# ---------------------------------------------------------------------- #


def _cleanup_temp_files(config: SiphonConfig) -> None:
    """Remove orphaned temp/keyframes files from media directories."""
    import glob
    download_dir = config.storage.download_dir
    if not os.path.isdir(download_dir):
        return
    patterns = ["*.temp.*", "*.keyframes.*", "*.part", "*.ytdl"]
    removed = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(download_dir, "**", pattern), recursive=True):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info("Cleaned up %d orphaned temp files", removed)
        log_activity(f"Cleaned up {removed} orphaned temp files")


# ---------------------------------------------------------------------- #
# Pocket Casts auto-prune
# ---------------------------------------------------------------------- #


async def _auto_prune_pocketcasts(config: SiphonConfig, db: Database) -> None:
    """Prune episodes that have been completed or archived in Pocket Casts."""
    from siphon.pocketcasts import (
        _extract_uuid_from_pc_url, _login, get_completed_video_ids, clear_token,
    )

    pc = config.pocketcasts
    loop = asyncio.get_event_loop()

    # Get feeds with pc_url set, prioritize those without pc_uuid (need mapping)
    feeds_needing_uuid = []
    feeds_ready = []
    for fc in config.feeds:
        resolved = resolve_feed(fc, config.defaults)
        if not resolved.pc_url:
            continue
        feed_db = db.get_feed(fc.name)
        if not feed_db:
            continue
        pc_uuid = feed_db.get("pc_uuid")
        if not pc_uuid:
            # Try to extract from pc_url
            pc_uuid = _extract_uuid_from_pc_url(resolved.pc_url)
            if pc_uuid:
                db.conn.execute(
                    "UPDATE feeds SET pc_uuid = ? WHERE name = ?",
                    (pc_uuid, fc.name),
                )
                db.conn.commit()
                logger.info("Mapped PC UUID %s for %s", pc_uuid, fc.name)
            else:
                feeds_needing_uuid.append(fc)
                continue
        # Smart skip: need at least 2 done episodes to prune (always keep 1)
        done_count = db.get_feed_episode_count(fc.name)
        if done_count < 2:
            continue

        feeds_ready.append((fc, pc_uuid))

    if not feeds_ready:
        return

    # Filter out feeds checked within the interval window
    interval_hours = pc.auto_prune_interval_hours
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=interval_hours)).strftime("%Y-%m-%d %H:%M:%S")
    feeds_due = [
        (fc, uuid) for fc, uuid in feeds_ready
        if ((db.get_feed(fc.name) or {}).get("pc_last_checked") or "") < cutoff
    ]

    if not feeds_due:
        return

    # Check N feeds per cycle, oldest PC check first
    feeds_due.sort(key=lambda x: (db.get_feed(x[0].name) or {}).get("pc_last_checked") or "")
    batch = feeds_due[:pc.feeds_per_check]

    pruned_total = 0
    for fc, pc_uuid in batch:
        try:
            completed_ids = await loop.run_in_executor(
                None, get_completed_video_ids, pc.email, pc.password, pc_uuid,
            )
        except Exception as exc:
            logger.warning("PC prune failed for %s: %s", fc.name, exc)
            clear_token()  # Force re-login on next attempt
            continue

        # Mark this feed as checked for PC prune
        db.conn.execute(
            "UPDATE feeds SET pc_last_checked = datetime('now') WHERE name = ?",
            (fc.name,),
        )
        db.conn.commit()

        if not completed_ids:
            continue

        # Get done episodes for this feed
        episodes = db.get_episodes_by_feed(fc.name)
        done_episodes = [
            ep for ep in episodes
            if ep["status"] == "done"
        ]

        if not done_episodes:
            continue

        # Find the latest done episode to keep (never prune the newest)
        done_episodes.sort(key=lambda e: e.get("upload_date") or "", reverse=True)
        keep_video_id = done_episodes[0]["video_id"]

        for ep in done_episodes:
            if ep["video_id"] == keep_video_id:
                continue  # Always keep latest
            if ep["video_id"] not in completed_ids:
                continue  # Not completed in PC

            # Prune this episode
            if ep.get("file_path"):
                try:
                    os.remove(ep["file_path"])
                except OSError:
                    pass

            # Delete transcript
            transcript = os.path.join(
                config.storage.download_dir, fc.name,
                f"{ep['video_id']}_transcript.json",
            )
            try:
                os.remove(transcript)
            except OSError:
                pass

            db.update_episode_status(ep["video_id"], fc.name, "pruned")
            pruned_total += 1
            logger.info("Auto-pruned %s from %s (completed in Pocket Casts)",
                        ep["video_id"], fc.name)

        await asyncio.sleep(20)  # Be nice to PC API

    if pruned_total:
        log_activity(f"Auto-pruned {pruned_total} completed episodes via Pocket Casts")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
