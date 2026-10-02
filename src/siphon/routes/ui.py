"""Web UI routes for feed management."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from fastapi.staticfiles import StaticFiles
from siphon.config import FeedConfig, SiphonConfig, resolve_feed
from siphon.db import Database

router = APIRouter(prefix="/ui")

def get_img_dir() -> str | None:
    """Find the img directory for static logo files."""
    for d in [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "img"),
        os.path.join(os.getcwd(), "img"),
    ]:
        if os.path.isdir(d):
            return d
    return None

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
)


def _format_number(value):
    """Jinja2 filter: add commas to large numbers."""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value


templates.env.filters["commafy"] = _format_number


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _render(request: Request, template_name: str, context: dict):
    """Render template — full page for normal requests, content-only for htmx."""
    context["request"] = request
    context["is_htmx"] = _is_htmx(request)
    return templates.TemplateResponse(template_name, context)


def _redirect(request: Request, url: str):
    """Redirect — htmx-aware (uses HX-Redirect header instead of 303)."""
    if _is_htmx(request):
        return HTMLResponse("", headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


def _render_feed_card(request: Request, feed_name: str):
    """Render a single feed card partial for htmx swap."""
    config = request.app.state.config
    feeds = _get_feed_display(request)
    feed = next((f for f in feeds if f["name"] == feed_name), None)
    if feed is None:
        return HTMLResponse("")

    from urllib.parse import urlparse
    parsed = urlparse(config.server.base_url)
    auth_base_url = f"{parsed.scheme}://{config.auth.username}:{config.auth.password}@{parsed.netloc}"

    tmpl = templates.env.get_template("_feed_card.html")
    html = tmpl.render(feed=feed, auth_base_url=auth_base_url)
    return HTMLResponse(html)


def _flash(request: Request, text: str, msg_type: str = "info") -> None:
    if not hasattr(request.state, "messages"):
        request.state.messages = []
    request.state.messages.append({"text": text, "type": msg_type})


def _get_messages(request: Request) -> list[dict]:
    return getattr(request.state, "messages", [])


def _get_background_tasks(app) -> set:
    """Return (and lazily create) the set of strong task references on app.state."""
    if not hasattr(app.state, "_background_tasks"):
        app.state._background_tasks = set()
    return app.state._background_tasks


def _normalize_date_cutoff(val: str) -> str:
    """Convert YYYY-MM-DD (from date picker) to YYYYMMDD."""
    return val.replace("-", "") if val else val


def _slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")[:60]


def _get_feed_display(request: Request) -> list[dict]:
    config = request.app.state.config
    db = request.app.state.db

    feeds_display = []
    for fc in config.feeds:
        resolved = resolve_feed(fc, config.defaults)
        db_feed = db.get_feed(fc.name) or {}
        episodes = db.get_episodes_by_feed(fc.name)

        status_counts = {}
        in_rss = 0
        whisper_pending = 0
        claude_pending = 0
        sb_waiting = 0
        dl_ready = 0
        dl_active = 0
        sb_cuts_total = 0
        llm_cuts_total = 0
        disk_bytes = 0
        crap_filtered = 0
        time_saved_secs = 0.0
        source_count = len(episodes)
        latest_done_date = None
        for ep in episodes:
            s = ep["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
            if s == "done":
                in_rss += 1
                # Track latest episode date among done episodes
                ud = ep.get("upload_date")
                if ud and (latest_done_date is None or ud > latest_done_date):
                    latest_done_date = ud
            if s == "pending_whisper":
                whisper_pending += 1
            if s == "pending_claude":
                claude_pending += 1
            if s == "pending":
                sb_waiting += 1
            if s == "eligible":
                dl_ready += 1
            if s == "downloading":
                dl_active += 1
            sb_cuts_total += ep.get("sb_cuts_applied") or 0
            llm_cuts_total += ep.get("llm_cuts_applied") or 0
            # Disk usage: sum file_size for non-pruned, non-filtered episodes
            if s not in ("filtered", "pruned"):
                disk_bytes += ep.get("file_size") or 0
            # Crap filtered: filtered by anything except date
            if s == "filtered" and ep.get("filter_reason") not in (None, "too_old", "unknown_date"):
                crap_filtered += 1
                # Filtered episode duration counts as time saved
                if ep.get("duration"):
                    time_saved_secs += ep["duration"]
            # SB seconds removed
            time_saved_secs += ep.get("sb_seconds_removed") or 0
            # LLM cuts time saved
            llm_json = ep.get("llm_segments_json")
            if llm_json and (ep.get("llm_cuts_applied") or 0) > 0:
                try:
                    import json as _json
                    data = _json.loads(llm_json)
                    for seg in data.get("high_confidence", data.get("segments", [])):
                        if "start" in seg and "end" in seg:
                            time_saved_secs += seg["end"] - seg["start"]
                except (ValueError, TypeError, KeyError):
                    pass

        # Compute disk usage display
        disk_usage_mb = round(disk_bytes / (1024 * 1024), 1)
        if disk_usage_mb >= 1024:
            disk_usage_display = f"{round(disk_usage_mb / 1024, 1)} GB"
        else:
            disk_usage_display = f"{disk_usage_mb} MB"

        # Compute latest episode date as relative string
        if latest_done_date:
            try:
                latest_dt = datetime.strptime(latest_done_date, "%Y%m%d")
                days_ago = (datetime.now() - latest_dt).days
                latest_episode_date = f"{days_ago}d ago"
            except (ValueError, TypeError):
                latest_episode_date = "\u2014"
        else:
            latest_episode_date = None

        # Compute relative checked time
        last_checked_raw = db_feed.get("last_checked_at")
        checked_relative = None
        if last_checked_raw:
            try:
                checked_dt = datetime.strptime(last_checked_raw, "%Y-%m-%d %H:%M:%S")
                delta = datetime.now() - checked_dt
                total_mins = int(delta.total_seconds() / 60)
                if total_mins < 1:
                    checked_relative = "just now"
                elif total_mins < 60:
                    checked_relative = f"{total_mins}m ago"
                elif total_mins < 1440:
                    checked_relative = f"{total_mins // 60}h ago"
                else:
                    checked_relative = f"{total_mins // 1440}d ago"
            except (ValueError, TypeError):
                checked_relative = last_checked_raw

        # Compute time saved display
        ts = int(time_saved_secs)
        if ts >= 86400:
            time_saved_display = f"{ts // 86400}d saved"
        elif ts >= 3600:
            time_saved_display = f"{ts // 3600}h saved"
        elif ts >= 60:
            time_saved_display = f"{ts // 60}m saved"
        elif ts > 0:
            time_saved_display = f"{ts}s saved"
        else:
            time_saved_display = None

        feeds_display.append({
            "name": fc.name,
            "url": fc.url,
            "feed_type": fc.type,
            "mode": resolved.mode,
            "quality": resolved.quality,
            "sponsorblock": resolved.sponsorblock,
            "sponsorblock_categories": resolved.sponsorblock_categories,
            "sponsorblock_delay_minutes": resolved.sponsorblock_delay_minutes,
            "block_shorts": resolved.block_shorts,
            "min_duration_seconds": resolved.min_duration_seconds,
            "date_cutoff": resolved.date_cutoff,
            "title_exclude": resolved.title_exclude,
            "llm_trim": resolved.llm_trim,
            "claude_prompt_extra": resolved.claude_prompt_extra,
            "claude_prompt_override": resolved.claude_prompt_override,
            "display_name": resolved.display_name,
            "pc_url": resolved.pc_url,
            "image_url": db_feed.get("image_url"),
            "last_checked_at": db_feed.get("last_checked_at"),
            "checked_relative": checked_relative,
            "last_error": db_feed.get("last_error"),
            "episode_counts": status_counts,
            "in_rss": in_rss,
            "whisper_pending": whisper_pending,
            "claude_pending": claude_pending,
            "sb_waiting": sb_waiting,
            "dl_ready": dl_ready,
            "dl_active": dl_active,
            "sb_cuts_total": sb_cuts_total,
            "llm_cuts_total": llm_cuts_total,
            "disk_usage_mb": disk_usage_mb,
            "disk_usage_display": disk_usage_display,
            "crap_filtered": crap_filtered,
            "source_count": source_count,
            "latest_episode_date": latest_episode_date,
            "latest_episode_raw": latest_done_date or "",
            "time_saved_secs": int(time_saved_secs),
            "time_saved_display": time_saved_display,
        })

    return feeds_display


def _get_system_status(config: SiphonConfig, db: Database, app=None) -> dict:
    """Build a summary dict of system-wide status for the dashboard."""
    from siphon.activity import get_active_counts, get_alerts, get_pause_state

    sched = config.schedule
    yt_recent = db.get_recent_download_count(hours=1, feed_type="youtube")
    pod_recent = db.get_recent_download_count(hours=1, feed_type="podcast")
    active = get_active_counts()

    row = db.conn.execute(
        "SELECT "
        "  SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS sb_queue, "
        "  SUM(CASE WHEN status = 'eligible' THEN 1 ELSE 0 END) AS dl_queue, "
        "  SUM(CASE WHEN status = 'pending_whisper' THEN 1 ELSE 0 END) AS whisper_queue, "
        "  SUM(CASE WHEN status = 'pending_claude' THEN 1 ELSE 0 END) AS claude_queue, "
        "  SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done_count "
        "FROM episodes"
    ).fetchone()

    # Get next run times from scheduler
    next_feed_check = ""
    next_dl_check = ""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(config.server.timezone)
        if app and hasattr(app.state, 'scheduler'):
            scheduler = app.state.scheduler
            feed_job = scheduler.get_job('check_feeds')
            dl_job = scheduler.get_job('process_downloads')
            if feed_job and feed_job.next_run_time:
                next_feed_check = feed_job.next_run_time.astimezone(tz).strftime("%H:%M")
            if dl_job and dl_job.next_run_time:
                next_dl_check = dl_job.next_run_time.astimezone(tz).strftime("%H:%M")
    except Exception:
        pass

    return {
        "youtube_downloads_this_hour": yt_recent,
        "youtube_downloads_max": sched.youtube_max_downloads_per_hour,
        "podcast_downloads_this_hour": pod_recent,
        "podcast_downloads_max": sched.podcast_max_downloads_per_hour,
        "sb_queue": int(row["sb_queue"] or 0),
        "dl_queue": max(0, int(row["dl_queue"] or 0) - active.get("download", 0)),
        "whisper_queue": max(0, int(row["whisper_queue"] or 0) - active.get("whisper", 0)),
        "claude_queue": max(0, int(row["claude_queue"] or 0) - active.get("claude", 0)),
        "done_count": int(row["done_count"] or 0),
        "alerts": get_alerts(),
        "active_dl": active.get("download", 0),
        "active_whisper": active.get("whisper", 0),
        "active_claude": active.get("claude", 0),
        "checking_feeds": active.get("checking_feeds", 0) > 0,
        "checking_downloads": active.get("checking_downloads", 0) > 0,
        "pause_state": get_pause_state(),
        "next_feed_check": next_feed_check,
        "next_dl_check": next_dl_check,
    }


# ------------------------------------------------------------------ #
# Cookie test (localhost-only, no auth)
# ------------------------------------------------------------------ #

@router.get("/test-cookies")
async def test_cookies_ui(request: Request):
    import asyncio
    from siphon.downloader import test_youtube_cookies

    config = request.app.state.config
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, test_youtube_cookies, config.cookies)
    return JSONResponse(result)


# ------------------------------------------------------------------ #
# Activity log JSON endpoint
# ------------------------------------------------------------------ #

@router.get("/activity")
async def activity_log(request: Request):
    from siphon.activity import get_recent
    return JSONResponse(get_recent(50))


@router.post("/set-whisper-workers")
async def set_whisper_workers(request: Request, workers: int = Form(1)):
    """Dynamically adjust whisper_workers without restart."""
    config = request.app.state.config
    config.llm.whisper_workers = max(1, min(workers, 10))
    from siphon.activity import log_activity
    log_activity(f"Whisper workers set to {config.llm.whisper_workers}")
    return JSONResponse({"workers": config.llm.whisper_workers})


@router.get("/activity-log-data")
async def activity_log_data(request: Request):
    from siphon.activity import get_recent
    config = request.app.state.config
    db = request.app.state.db
    return JSONResponse({
        "system": _get_system_status(config, db, request.app),
        "entries": get_recent(200),
    })


@router.get("/activity-log", response_class=HTMLResponse)
async def activity_log_page(request: Request):
    from siphon.activity import get_recent
    config = request.app.state.config
    db = request.app.state.db
    return templates.TemplateResponse("activity_log.html", {
        "request": request,
        "activity_log": get_recent(50),
        "status": _get_system_status(config, db, request.app),
        "current_status": "See activity footer",
    })


# ------------------------------------------------------------------ #
# Feed list
# ------------------------------------------------------------------ #

@router.get("/", response_class=HTMLResponse)
async def stats_page(request: Request):
    _reload_config(request.app)
    config = request.app.state.config
    db = request.app.state.db

    feeds = _get_feed_display(request)
    disk_usage = db.get_disk_usage()
    total_episodes = sum(
        sum(f["episode_counts"].values()) for f in feeds
    )

    status = _get_system_status(config, db, request.app)

    agg = db.conn.execute(
        "SELECT "
        "  COALESCE(SUM(llm_cuts_applied), 0) AS total_llm_cuts, "
        "  COALESCE(SUM(sb_cuts_applied), 0) AS total_sb_cuts "
        "FROM episodes WHERE status IN ('done', 'pruned')"
    ).fetchone()
    total_llm_cuts = int(agg["total_llm_cuts"])
    total_sb_cuts = int(agg["total_sb_cuts"])

    avg_row = db.conn.execute(
        "SELECT AVG(whisper_duration_seconds + claude_duration_seconds) AS avg_proc "
        "FROM episodes "
        "WHERE whisper_duration_seconds IS NOT NULL AND claude_duration_seconds IS NOT NULL"
    ).fetchone()
    avg_proc = avg_row["avg_proc"]
    if avg_proc is not None:
        avg_minutes = int(avg_proc) // 60
        avg_seconds = int(avg_proc) % 60
        avg_processing_display = f"{avg_minutes}:{avg_seconds:02d}"
    else:
        avg_processing_display = "--:--"

    words_row = db.conn.execute(
        "SELECT COALESCE(SUM(whisper_word_count), 0) AS total_words FROM episodes"
    ).fetchone()
    total_words = int(words_row["total_words"])

    extra_stats = db.conn.execute(
        "SELECT "
        "  SUM(CASE WHEN status = 'pruned' THEN 1 ELSE 0 END) AS purged, "
        "  SUM(CASE WHEN status = 'filtered' AND filter_reason IS NOT NULL "
        "       AND filter_reason NOT IN ('too_old', 'unknown_date') THEN 1 ELSE 0 END) AS crap_filtered "
        "FROM episodes"
    ).fetchone()
    purged_count = int(extra_stats["purged"] or 0)
    crap_filtered = int(extra_stats["crap_filtered"] or 0)

    insights = _compute_insights(db, config)

    podcast_count = sum(1 for f in feeds if f["feed_type"] == "podcast")
    youtube_count = sum(1 for f in feeds if f["feed_type"] == "youtube")

    # Total time saved: LLM segment durations + SB seconds + filtered episode durations
    import json as _json
    total_time_saved = 0.0

    # LLM time saved from segment JSON
    llm_time_rows = db.conn.execute(
        "SELECT llm_segments_json FROM episodes "
        "WHERE llm_cuts_applied > 0 AND llm_segments_json IS NOT NULL"
    ).fetchall()
    for row in llm_time_rows:
        try:
            data = _json.loads(row["llm_segments_json"])
            for seg in data.get("high_confidence", data.get("segments", [])):
                if "start" in seg and "end" in seg:
                    total_time_saved += seg["end"] - seg["start"]
        except (ValueError, TypeError, KeyError):
            pass

    # SB time saved
    sb_time_row = db.conn.execute(
        "SELECT COALESCE(SUM(sb_seconds_removed), 0) AS total FROM episodes"
    ).fetchone()
    total_time_saved += sb_time_row["total"] or 0

    # Filtered episode durations
    filt_time_row = db.conn.execute(
        "SELECT COALESCE(SUM(duration), 0) AS total FROM episodes "
        "WHERE status = 'filtered' AND duration IS NOT NULL "
        "  AND filter_reason NOT IN ('too_old', 'unknown_date')"
    ).fetchone()
    total_time_saved += filt_time_row["total"] or 0

    # Format as Xd Xh Xm
    total_secs = int(total_time_saved)
    days = total_secs // 86400
    hours = (total_secs % 86400) // 3600
    mins = (total_secs % 3600) // 60
    if days > 0:
        total_time_saved_display = f"{days}d {hours}h {mins}m"
    elif hours > 0:
        total_time_saved_display = f"{hours}h {mins}m"
    else:
        total_time_saved_display = f"{mins}m"

    # Total data processed (all episodes including pruned)
    data_row = db.conn.execute(
        "SELECT COALESCE(SUM(file_size), 0) AS total FROM episodes WHERE file_size IS NOT NULL"
    ).fetchone()
    total_data_bytes = data_row["total"] or 0
    total_data_gb = total_data_bytes / (1024 ** 3)
    if total_data_gb >= 1000:
        total_data_display = f"{round(total_data_gb / 1024, 1)} TB"
    else:
        total_data_display = f"{round(total_data_gb, 1)} GB"

    return _render(request, "stats.html", {
        "active_page": "stats",
        "podcast_count": podcast_count,
        "youtube_count": youtube_count,
        "disk_usage_gb": round(disk_usage / (1024 ** 3), 1),
        "max_disk_gb": config.storage.max_disk_gb,
        "total_episodes": total_episodes,
        "total_llm_cuts": total_llm_cuts,
        "total_sb_cuts": total_sb_cuts,
        "total_words": total_words,
        "purged_count": purged_count,
        "crap_filtered": crap_filtered,
        "avg_processing_display": avg_processing_display,
        "status": status,
        "insights": insights,
        "total_time_saved_display": total_time_saved_display,
        "total_data_display": total_data_display,
        "messages": _get_messages(request),
    })


@router.get("/feeds", response_class=HTMLResponse)
async def feeds_page(request: Request):
    _reload_config(request.app)
    config = request.app.state.config

    feeds = _get_feed_display(request)

    from urllib.parse import urlparse
    parsed = urlparse(config.server.base_url)
    auth_base_url = f"{parsed.scheme}://{config.auth.username}:{config.auth.password}@{parsed.netloc}"

    search_query = request.query_params.get("q", "")

    return _render(request, "feeds.html", {
        "active_page": "feeds",
        "feeds": feeds,
        "auth_base_url": auth_base_url,
        "search_query": search_query,
        "messages": _get_messages(request),
    })


# ------------------------------------------------------------------ #
# Check feeds now
# ------------------------------------------------------------------ #

@router.post("/check-now")
async def check_now(request: Request):
    import asyncio
    import logging
    from siphon.pipeline import check_feeds, process_downloads

    logger = logging.getLogger(__name__)

    # Reload config from disk so any changes made via the UI are picked up
    _reload_config(request.app)
    config = request.app.state.config
    db = request.app.state.db

    logger.info("Manual feed check triggered: %d feeds", len(config.feeds))

    async def _check_with_logging():
        try:
            await check_feeds(config, db)
            logger.info("Manual feed check completed, starting downloads")
            await process_downloads(config, db)
            logger.info("Manual download processing completed")
        except Exception as e:
            logger.error("Manual feed check failed: %s", e, exc_info=True)

    # Keep a strong reference so the task isn't garbage-collected mid-flight.
    task = asyncio.create_task(_check_with_logging())
    _get_background_tasks(request.app).add(task)
    task.add_done_callback(_get_background_tasks(request.app).discard)

    if _is_htmx(request):
        return HTMLResponse("", status_code=204)
    return RedirectResponse("/ui/", status_code=303)


def _compute_insights(db: Database, config) -> dict:
    """Compute dashboard insight data from the database."""
    # Build display-name lookup from config feeds
    display_names = {}
    for fc in config.feeds:
        resolved = resolve_feed(fc, config.defaults)
        display_names[fc.name] = resolved.display_name or fc.name

    # Most stale feeds (oldest latest episode)
    stale_rows = db.conn.execute(
        "SELECT feed_name, MAX(upload_date) as latest_date "
        "FROM episodes WHERE status IN ('done', 'pruned') "
        "GROUP BY feed_name "
        "ORDER BY latest_date ASC "
        "LIMIT 3"
    ).fetchall()
    stale = []
    for row in stale_rows:
        if row["latest_date"]:
            try:
                latest_dt = datetime.strptime(row["latest_date"], "%Y%m%d")
                days_ago = (datetime.now() - latest_dt).days
            except (ValueError, TypeError):
                days_ago = "?"
            stale.append({
                "name": display_names.get(row["feed_name"], row["feed_name"]),
                "days_ago": days_ago,
            })

    # Disk hogs (top 3 by total file size)
    disk_rows = db.conn.execute(
        "SELECT feed_name, SUM(file_size) as total_bytes "
        "FROM episodes WHERE status IN ('done', 'pending_whisper', 'pending_claude') "
        "AND file_size IS NOT NULL "
        "GROUP BY feed_name "
        "ORDER BY total_bytes DESC "
        "LIMIT 3"
    ).fetchall()
    disk_hogs = []
    for row in disk_rows:
        total_gb = round(row["total_bytes"] / (1024 ** 3), 1)
        disk_hogs.append({
            "name": display_names.get(row["feed_name"], row["feed_name"]),
            "size_display": f"{total_gb} GB",
        })

    # Longest processing (top 3 episodes by whisper+claude time)
    proc_rows = db.conn.execute(
        "SELECT feed_name, title, "
        "  COALESCE(whisper_duration_seconds, 0) + COALESCE(claude_duration_seconds, 0) as total_time "
        "FROM episodes "
        "WHERE whisper_duration_seconds IS NOT NULL OR claude_duration_seconds IS NOT NULL "
        "ORDER BY total_time DESC "
        "LIMIT 3"
    ).fetchall()
    longest_proc = []
    for row in proc_rows:
        total_secs = int(row["total_time"])
        mins = total_secs // 60
        secs = total_secs % 60
        longest_proc.append({
            "title": row["title"],
            "feed": display_names.get(row["feed_name"], row["feed_name"]),
            "time_display": f"{mins}:{secs:02d}",
        })

    # Most SB cuts per episode (top 3 feeds by avg)
    sb_rows = db.conn.execute(
        "SELECT feed_name, AVG(sb_cuts_applied) as avg_sb, COUNT(*) as ep_count "
        "FROM episodes "
        "WHERE status IN ('done', 'pruned') AND sb_cuts_applied IS NOT NULL AND sb_cuts_applied > 0 "
        "GROUP BY feed_name "
        "ORDER BY avg_sb DESC "
        "LIMIT 3"
    ).fetchall()
    most_sb = []
    for row in sb_rows:
        most_sb.append({
            "name": display_names.get(row["feed_name"], row["feed_name"]),
            "avg": round(row["avg_sb"], 1),
            "count": row["ep_count"],
        })

    # Most LLM cuts per episode (top 3 feeds by avg)
    llm_rows = db.conn.execute(
        "SELECT feed_name, AVG(llm_cuts_applied) as avg_llm, COUNT(*) as ep_count "
        "FROM episodes "
        "WHERE status IN ('done', 'pruned') AND llm_cuts_applied IS NOT NULL AND llm_cuts_applied > 0 "
        "GROUP BY feed_name "
        "ORDER BY avg_llm DESC "
        "LIMIT 3"
    ).fetchall()
    most_llm = []
    for row in llm_rows:
        most_llm.append({
            "name": display_names.get(row["feed_name"], row["feed_name"]),
            "avg": round(row["avg_llm"], 1),
            "count": row["ep_count"],
        })

    # Highest filter rate — only non-date filters (too_short, short, title_match)
    filter_rows = db.conn.execute(
        "SELECT feed_name, "
        "  SUM(CASE WHEN status = 'filtered' AND filter_reason IS NOT NULL "
        "       AND filter_reason NOT IN ('too_old', 'unknown_date') THEN 1 ELSE 0 END) AS filtered, "
        "  COUNT(*) AS total "
        "FROM episodes "
        "GROUP BY feed_name "
        "HAVING total > 0 AND filtered > 0 "
        "ORDER BY CAST(filtered AS REAL) / total DESC "
        "LIMIT 3"
    ).fetchall()
    highest_filter = []
    for row in filter_rows:
        pct = round(100 * row["filtered"] / row["total"])
        highest_filter.append({
            "name": display_names.get(row["feed_name"], row["feed_name"]),
            "pct": pct,
            "filtered": row["filtered"],
            "total": row["total"],
        })

    # Filter breakdown — top 3 filter reasons by count (excluding date filters)
    filter_breakdown_rows = db.conn.execute(
        "SELECT filter_reason, COUNT(*) as cnt "
        "FROM episodes "
        "WHERE status = 'filtered' AND filter_reason IS NOT NULL "
        "  AND filter_reason NOT IN ('too_old', 'unknown_date') "
        "GROUP BY CASE "
        "  WHEN filter_reason LIKE 'title_match:%' THEN 'title_match' "
        "  ELSE filter_reason END "
        "ORDER BY cnt DESC "
        "LIMIT 3"
    ).fetchall()
    total_crap = sum(r["cnt"] for r in filter_breakdown_rows) if filter_breakdown_rows else 0
    filter_breakdown = []
    for row in filter_breakdown_rows:
        reason = row["filter_reason"]
        if reason.startswith("title_match:"):
            reason = "title_match"
        pct = round(100 * row["cnt"] / total_crap) if total_crap else 0
        filter_breakdown.append({
            "reason": reason,
            "count": row["cnt"],
            "pct": pct,
        })

    # Most active — top 3 feeds by episode count in last 7 days
    most_active_rows = db.conn.execute(
        "SELECT feed_name, COUNT(*) as cnt "
        "FROM episodes "
        "WHERE discovered_at >= datetime('now', '-7 days') "
        "  AND status NOT IN ('filtered') "
        "GROUP BY feed_name "
        "ORDER BY cnt DESC "
        "LIMIT 3"
    ).fetchall()
    most_active = []
    for row in most_active_rows:
        most_active.append({
            "name": display_names.get(row["feed_name"], row["feed_name"]),
            "count": row["cnt"],
        })

    # Time saved — top 3 feeds by total time removed (LLM + SB + filtered duration)
    import json as _json
    time_saved_per_feed: dict[str, float] = {}

    # LLM time saved: sum segment durations from llm_segments_json
    llm_time_rows = db.conn.execute(
        "SELECT feed_name, llm_segments_json FROM episodes "
        "WHERE llm_cuts_applied > 0 AND llm_segments_json IS NOT NULL"
    ).fetchall()
    for row in llm_time_rows:
        try:
            data = _json.loads(row["llm_segments_json"])
            for seg in data.get("high_confidence", data.get("segments", [])):
                if "start" in seg and "end" in seg:
                    time_saved_per_feed[row["feed_name"]] = (
                        time_saved_per_feed.get(row["feed_name"], 0)
                        + seg["end"] - seg["start"]
                    )
        except (ValueError, TypeError, KeyError):
            pass

    # SB time saved
    sb_time_rows = db.conn.execute(
        "SELECT feed_name, SUM(sb_seconds_removed) as total_sb "
        "FROM episodes WHERE sb_seconds_removed > 0 "
        "GROUP BY feed_name"
    ).fetchall()
    for row in sb_time_rows:
        time_saved_per_feed[row["feed_name"]] = (
            time_saved_per_feed.get(row["feed_name"], 0)
            + (row["total_sb"] or 0)
        )

    # Filtered episode duration saved
    filtered_time_rows = db.conn.execute(
        "SELECT feed_name, SUM(duration) as total_dur "
        "FROM episodes WHERE status = 'filtered' AND duration IS NOT NULL "
        "  AND filter_reason NOT IN ('too_old', 'unknown_date') "
        "GROUP BY feed_name"
    ).fetchall()
    for row in filtered_time_rows:
        time_saved_per_feed[row["feed_name"]] = (
            time_saved_per_feed.get(row["feed_name"], 0)
            + (row["total_dur"] or 0)
        )

    # Sort and take top 3
    time_saved_sorted = sorted(time_saved_per_feed.items(), key=lambda x: x[1], reverse=True)[:3]
    time_saved = []
    for feed_name, secs in time_saved_sorted:
        mins = int(secs) // 60
        if mins >= 60:
            display = f"{mins // 60}h {mins % 60}m"
        else:
            display = f"{mins}m"
        time_saved.append({
            "name": display_names.get(feed_name, feed_name),
            "display": display,
        })

    # Most data — top 3 feeds by lifetime data processed (including pruned)
    most_data_rows = db.conn.execute(
        "SELECT feed_name, SUM(file_size) as total_bytes "
        "FROM episodes WHERE file_size IS NOT NULL "
        "GROUP BY feed_name "
        "ORDER BY total_bytes DESC "
        "LIMIT 3"
    ).fetchall()
    most_data = []
    for row in most_data_rows:
        total_gb = row["total_bytes"] / (1024 ** 3)
        if total_gb >= 1:
            size_display = f"{round(total_gb, 1)} GB"
        else:
            size_display = f"{round(total_gb * 1024, 1)} MB"
        most_data.append({
            "name": display_names.get(row["feed_name"], row["feed_name"]),
            "size_display": size_display,
        })

    # Error prone — top 3 feeds with most recent errors
    error_rows = db.conn.execute(
        "SELECT f.name, f.last_error, "
        "  (SELECT COUNT(*) FROM episodes e WHERE e.feed_name = f.name AND e.status = 'failed') as fail_count "
        "FROM feeds f "
        "WHERE f.last_error IS NOT NULL AND f.last_error != '' "
        "ORDER BY fail_count DESC, f.last_checked_at DESC "
        "LIMIT 3"
    ).fetchall()
    error_prone = []
    for row in error_rows:
        error_prone.append({
            "name": display_names.get(row["name"], row["name"]),
            "error": row["last_error"][:50],
            "fail_count": row["fail_count"],
        })

    # Queue hogs — top 3 feeds with most episodes in pipeline
    queue_rows = db.conn.execute(
        "SELECT feed_name, "
        "  SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS sb, "
        "  SUM(CASE WHEN status IN ('eligible', 'downloading') THEN 1 ELSE 0 END) AS dl, "
        "  SUM(CASE WHEN status = 'pending_whisper' THEN 1 ELSE 0 END) AS whisper, "
        "  SUM(CASE WHEN status = 'pending_claude' THEN 1 ELSE 0 END) AS claude "
        "FROM episodes "
        "WHERE status IN ('pending', 'eligible', 'downloading', 'pending_whisper', 'pending_claude') "
        "GROUP BY feed_name "
        "ORDER BY (sb + dl + whisper + claude) DESC "
        "LIMIT 3"
    ).fetchall()
    queue_hogs = []
    for row in queue_rows:
        total = row["sb"] + row["dl"] + row["whisper"] + row["claude"]
        if total > 0:
            queue_hogs.append({
                "name": display_names.get(row["feed_name"], row["feed_name"]),
                "total": total,
                "sb": row["sb"],
                "dl": row["dl"],
                "whisper": row["whisper"],
                "claude": row["claude"],
            })

    return {
        "stale": stale,
        "disk_hogs": disk_hogs,
        "longest_proc": longest_proc,
        "most_sb": most_sb,
        "most_llm": most_llm,
        "highest_filter": highest_filter,
        "filter_breakdown": filter_breakdown,
        "most_active": most_active,
        "time_saved": time_saved,
        "most_data": most_data,
        "error_prone": error_prone,
        "queue_hogs": queue_hogs,
    }


def _reload_config(app) -> None:
    """Reload config from disk into app.state so new/renamed feeds are picked up."""
    import logging
    from siphon.config import load_config

    logger = logging.getLogger(__name__)
    config = app.state.config
    config_path = getattr(config, "_config_path", None)
    if not config_path:
        return
    try:
        new_config = load_config(config_path)
        app.state.config = new_config
        # Sync any new feeds to DB
        db = app.state.db
        for feed in new_config.feeds:
            db.upsert_feed(feed.name, feed.url, feed.type)
        logger.info("Config reloaded: %d feeds", len(new_config.feeds))
    except Exception as e:
        logger.error("Config reload failed: %s", e)


# ------------------------------------------------------------------ #
# Add feed
# ------------------------------------------------------------------ #

@router.get("/add", response_class=HTMLResponse)
async def add_feed_page(request: Request):
    config = request.app.state.config
    return _render(request, "add_feed.html", {
        "active_page": "add",
        "prefill": {},
        "defaults": config.defaults,
        "messages": _get_messages(request),
    })


@router.post("/add")
async def add_feed_submit(
    request: Request,
    url: str = Form(...),
    name: str = Form(...),
    type: str = Form(""),
    mode: str = Form(""),
    quality: str = Form(""),
    sponsorblock: str = Form(""),
    sponsorblock_categories: str = Form(""),
    sponsorblock_delay_minutes: str = Form(""),
    block_shorts: str = Form(""),
    min_duration_seconds: str = Form(""),
    llm_trim: str = Form(""),
    date_cutoff: str = Form(""),
    title_exclude: str = Form(""),
    claude_prompt_extra: str = Form(""),
    claude_prompt_override: str = Form(""),
    display_name: str = Form(""),
    pc_url: str = Form(""),
):
    config = request.app.state.config
    db = request.app.state.db

    # Sanitize inputs
    url = url.strip()

    # Auto-detect feed type from URL
    if not type:
        url_lower = url.lower()
        if any(d in url_lower for d in ("youtube.com", "youtu.be", "m.youtube.com")):
            type = "youtube"
        else:
            type = "podcast"

    name = _slugify(name) if name else _slugify(url.split("/")[-1])
    if not name:
        name = f"feed-{len(config.feeds)}"

    # Check for duplicate name
    for fc in config.feeds:
        if fc.name == name:
            _flash(request, f"Feed '{name}' already exists.", "error")
            return _render(request, "add_feed.html", {
                "active_page": "add",
                "prefill": {"url": url, "name": name, "type": type},
                "defaults": config.defaults,
                "messages": _get_messages(request),
            })

    feed_data: dict = {"name": name, "url": url, "type": type}
    if mode:
        feed_data["mode"] = mode
    if quality:
        feed_data["quality"] = quality if quality == "max" else int(quality)
    if sponsorblock:
        feed_data["sponsorblock"] = sponsorblock == "true"
    if sponsorblock_categories:
        feed_data["sponsorblock_categories"] = [
            c.strip() for c in sponsorblock_categories.split(",") if c.strip()
        ]
    if sponsorblock_delay_minutes:
        feed_data["sponsorblock_delay_minutes"] = int(sponsorblock_delay_minutes)
    if block_shorts:
        feed_data["block_shorts"] = block_shorts == "true"
    if min_duration_seconds:
        feed_data["min_duration_seconds"] = int(min_duration_seconds)
    if llm_trim:
        feed_data["llm_trim"] = llm_trim == "true"
    if date_cutoff:
        feed_data["date_cutoff"] = _normalize_date_cutoff(date_cutoff)
    if title_exclude:
        feed_data["title_exclude"] = [t.strip() for t in title_exclude.split(",") if t.strip()]
    if claude_prompt_extra:
        feed_data["claude_prompt_extra"] = claude_prompt_extra
    if claude_prompt_override:
        feed_data["claude_prompt_override"] = claude_prompt_override
    if display_name:
        feed_data["display_name"] = display_name
    if pc_url:
        feed_data["pc_url"] = pc_url

    new_feed = FeedConfig(**feed_data)
    config.feeds.append(new_feed)
    db.upsert_feed(name, url, type)

    _save_config(config)

    return _redirect(request, "/ui/feeds")


# ------------------------------------------------------------------ #
# Feed actions — all use POST with feed_name in form body
# ------------------------------------------------------------------ #

@router.post("/feed-action")
async def feed_action(
    request: Request,
    feed_name: str = Form(...),
    action: str = Form(...),
    # Update fields
    mode: str = Form("video"),
    quality: str = Form("1440"),
    sponsorblock: str = Form("true"),
    sponsorblock_categories: str = Form(""),
    sponsorblock_delay_minutes: int = Form(1440),
    block_shorts: str = Form("true"),
    min_duration_seconds: int = Form(60),
    llm_trim: str = Form("false"),
    date_cutoff: str = Form(""),
    title_exclude: str = Form(""),
    claude_prompt_extra: str = Form(""),
    claude_prompt_override: str = Form(""),
    display_name: str = Form(""),
    pc_url: str = Form(""),
    new_name: str = Form(""),
):
    config = request.app.state.config
    db = request.app.state.db

    if action == "update":
        return _do_update(
            request, config, db, feed_name, mode, quality, sponsorblock,
            sponsorblock_categories, sponsorblock_delay_minutes,
            block_shorts, min_duration_seconds,
            llm_trim, date_cutoff, title_exclude, claude_prompt_extra,
            claude_prompt_override, display_name, pc_url,
        )
    elif action == "rename":
        return _do_rename(request, config, db, feed_name, new_name)
    elif action == "delete":
        return _do_delete(request, config, db, feed_name)
    elif action == "catchup":
        return _do_catchup(request, config, db, feed_name)
    else:
        return _redirect(request, "/ui/feeds")


def _do_update(request, config, db, feed_name, mode, quality, sponsorblock,
               sponsorblock_categories, sponsorblock_delay_minutes,
               block_shorts, min_duration_seconds,
               llm_trim, date_cutoff, title_exclude, claude_prompt_extra,
               claude_prompt_override, display_name, pc_url=""):
    defaults = config.defaults
    for i, fc in enumerate(config.feeds):
        if fc.name == feed_name:
            # Only store values that differ from defaults (avoid baking defaults into per-feed config)
            update: dict = {
                "name": fc.name,
                "url": fc.url,
                "type": fc.type,
            }
            parsed_quality = quality if quality == "max" else int(quality)
            parsed_sb = sponsorblock == "true"
            parsed_sb_cats = (
                [c.strip() for c in sponsorblock_categories.split(",") if c.strip()]
                if sponsorblock_categories else []
            )
            parsed_block_shorts = block_shorts == "true"
            parsed_llm_trim = llm_trim == "true"
            parsed_title_exclude = [t.strip() for t in title_exclude.split(",") if t.strip()]

            if mode != defaults.mode:
                update["mode"] = mode
            if parsed_quality != defaults.quality:
                update["quality"] = parsed_quality
            if parsed_sb != defaults.sponsorblock:
                update["sponsorblock"] = parsed_sb
            if parsed_sb_cats != defaults.sponsorblock_categories:
                update["sponsorblock_categories"] = parsed_sb_cats
            if sponsorblock_delay_minutes != defaults.sponsorblock_delay_minutes:
                update["sponsorblock_delay_minutes"] = sponsorblock_delay_minutes
            if parsed_block_shorts != defaults.block_shorts:
                update["block_shorts"] = parsed_block_shorts
            if min_duration_seconds != defaults.min_duration_seconds:
                update["min_duration_seconds"] = min_duration_seconds
            if parsed_llm_trim != defaults.llm_trim:
                update["llm_trim"] = parsed_llm_trim
            if _normalize_date_cutoff(date_cutoff) if date_cutoff else None:
                update["date_cutoff"] = _normalize_date_cutoff(date_cutoff)
            if parsed_title_exclude:
                update["title_exclude"] = parsed_title_exclude
            if claude_prompt_extra:
                update["claude_prompt_extra"] = claude_prompt_extra
            if claude_prompt_override:
                update["claude_prompt_override"] = claude_prompt_override
            if display_name:
                update["display_name"] = display_name
            if pc_url:
                update["pc_url"] = pc_url

            config.feeds[i] = FeedConfig(**update)
            break
    _save_config(config)
    if _is_htmx(request):
        _reload_config(request.app)
        return _render_feed_card(request, feed_name)
    return _redirect(request, "/ui/feeds")


def _do_rename(request, config, db, feed_name, new_name):
    new_name = _slugify(new_name)
    if not new_name or new_name == feed_name:
        return _redirect(request, "/ui/feeds")

    # Check for duplicate
    if any(fc.name == new_name for fc in config.feeds):
        return _redirect(request, "/ui/feeds")

    # Update config
    for i, fc in enumerate(config.feeds):
        if fc.name == feed_name:
            data = fc.model_dump()
            data["name"] = new_name
            config.feeds[i] = FeedConfig(**data)
            break

    # Update DB — rename feed in place
    old_feed = db.get_feed(feed_name)
    if old_feed:
        db.upsert_feed(new_name, old_feed["url"], old_feed.get("feed_type", "youtube"))
        # Copy metadata that upsert doesn't carry over
        db.conn.execute(
            "UPDATE feeds SET image_url = ?, channel_id = ?, last_checked_at = ? WHERE name = ?",
            (old_feed.get("image_url"), old_feed.get("channel_id"), old_feed.get("last_checked_at"), new_name),
        )
        # Update episodes to point to new feed name
        db.conn.execute(
            "UPDATE episodes SET feed_name = ? WHERE feed_name = ?",
            (new_name, feed_name),
        )
        db.conn.commit()
        db.delete_feed(feed_name)

    # Rename media directory (skip if old name is invalid as a path component)
    download_dir = config.storage.download_dir
    old_dir = os.path.join(download_dir, feed_name)
    new_dir = os.path.join(download_dir, new_name)
    try:
        if os.path.isdir(old_dir) and not os.path.exists(new_dir):
            os.rename(old_dir, new_dir)
            # Update file paths in DB
            db.conn.execute(
                "UPDATE episodes SET file_path = REPLACE(file_path, ?, ?) WHERE feed_name = ?",
                (feed_name, new_name, new_name),
            )
            db.conn.commit()
    except OSError:
        pass  # Old dir didn't exist or had an invalid name — no files to move

    _save_config(config)
    if _is_htmx(request):
        _reload_config(request.app)
        return _render_feed_card(request, new_name)
    return _redirect(request, "/ui/feeds")


def _do_delete(request, config, db, feed_name):
    config.feeds = [f for f in config.feeds if f.name != feed_name]

    download_dir = config.storage.download_dir
    feed_dir = os.path.join(download_dir, feed_name)
    if os.path.isdir(feed_dir):
        for f in os.listdir(feed_dir):
            try:
                os.remove(os.path.join(feed_dir, f))
            except OSError:
                pass
        try:
            os.rmdir(feed_dir)
        except OSError:
            pass

    db.delete_episodes_by_feed(feed_name)
    db.delete_feed(feed_name)
    _save_config(config)
    if _is_htmx(request):
        return HTMLResponse("")
    return _redirect(request, "/ui/feeds")


def _do_catchup(request, config, db, feed_name):
    # Pipeline statuses — never touch these
    pipeline_statuses = ("pending", "eligible", "downloading", "pending_whisper", "pending_claude")

    # Find the newest episode across all statuses (for date cutoff)
    newest = db.conn.execute(
        """SELECT upload_date FROM episodes
           WHERE feed_name = ? AND status NOT IN ('filtered', 'pruned')
           ORDER BY upload_date DESC LIMIT 1""",
        (feed_name,),
    ).fetchone()

    if not newest or not newest["upload_date"]:
        new_cutoff = datetime.now(timezone.utc).strftime("%Y%m%d")
    else:
        from datetime import datetime as dt
        newest_date = dt.strptime(newest["upload_date"], "%Y%m%d")
        new_cutoff = (newest_date - timedelta(days=1)).strftime("%Y%m%d")

    # Update config
    for i, fc in enumerate(config.feeds):
        if fc.name == feed_name:
            update = fc.model_dump()
            update["date_cutoff"] = new_cutoff
            config.feeds[i] = FeedConfig(**update)
            break

    # Find the latest RSS episode to keep (must keep at least 1 in RSS)
    latest_rss = db.conn.execute(
        """SELECT video_id FROM episodes
           WHERE feed_name = ? AND status = 'done'
             AND llm_trim_status IN ('done', 'skipped')
           ORDER BY upload_date DESC LIMIT 1""",
        (feed_name,),
    ).fetchone()
    keep_video_id = latest_rss["video_id"] if latest_rss else None

    # Prune done episodes except the latest RSS one; never touch pipeline episodes
    download_dir = config.storage.download_dir
    all_episodes = db.get_episodes_by_feed(feed_name)

    for ep in all_episodes:
        # Never touch pipeline episodes
        if ep["status"] in pipeline_statuses:
            continue

        # Skip already pruned/filtered
        if ep["status"] in ("filtered", "pruned"):
            continue

        # Keep the latest RSS episode
        if ep["video_id"] == keep_video_id:
            continue

        # Delete file if exists
        if ep.get("file_path"):
            try:
                os.remove(ep["file_path"])
            except OSError:
                pass

        # Also delete the kept original and transcript
        from siphon.episode_files import remove_extras
        remove_extras(download_dir, ep)

        db.update_episode_status(ep["video_id"], feed_name, "pruned")

    _save_config(config)
    if _is_htmx(request):
        _reload_config(request.app)
        return _render_feed_card(request, feed_name)
    return _redirect(request, "/ui/feeds")


# ------------------------------------------------------------------ #
# OPML Import
# ------------------------------------------------------------------ #

@router.get("/import", response_class=HTMLResponse)
async def import_page(request: Request):
    return _render(request, "import_opml.html", {
        "active_page": "import",
        "feeds_to_review": None,
        "messages": _get_messages(request),
    })


@router.post("/import")
async def import_upload(request: Request, opml_file: UploadFile = File(...)):
    content = await opml_file.read()
    feeds = _parse_opml(content)

    if not feeds:
        _flash(request, "No feeds found in OPML file.", "error")
        return templates.TemplateResponse("import_opml.html", {
            "request": request,
            "active_page": "import",
            "feeds_to_review": None,
            "messages": _get_messages(request),
        })

    config = request.app.state.config
    existing_urls = {fc.url for fc in config.feeds}
    existing_names = {fc.name for fc in config.feeds}

    new_feeds = []
    for f in feeds:
        if f["url"] not in existing_urls:
            name = f["name"]
            counter = 2
            while name in existing_names:
                name = f"{f['name']}-{counter}"
                counter += 1
            f["name"] = name
            existing_names.add(name)
            new_feeds.append(f)

    if not new_feeds:
        _flash(request, "All feeds in the OPML file already exist.", "info")
        return templates.TemplateResponse("import_opml.html", {
            "request": request,
            "active_page": "import",
            "feeds_to_review": None,
            "messages": _get_messages(request),
        })

    return _render(request, "import_opml.html", {
        "active_page": "import",
        "feeds_to_review": new_feeds,
        "messages": _get_messages(request),
    })


@router.post("/import/confirm")
async def import_confirm(request: Request):
    config = request.app.state.config
    db = request.app.state.db
    form = await request.form()

    total = int(form.get("total", 0))
    imported = 0

    for i in range(total):
        if not form.get(f"import_{i}"):
            continue

        name = form.get(f"name_{i}", "").strip()
        url = form.get(f"url_{i}", "").strip()
        if not name or not url:
            continue

        name = _slugify(name)
        if not name:
            name = f"podcast-{i}"

        date_cutoff = _normalize_date_cutoff(form.get(f"date_cutoff_{i}", "").strip()) or None
        llm_trim_str = form.get(f"llm_trim_{i}", "").strip()
        title_exclude_str = form.get(f"title_exclude_{i}", "").strip()

        feed_data: dict = {
            "name": name,
            "url": url,
            "type": "podcast",
            "mode": "audio",
        }
        if date_cutoff:
            feed_data["date_cutoff"] = date_cutoff
        if llm_trim_str:
            feed_data["llm_trim"] = llm_trim_str == "true"
        if title_exclude_str:
            feed_data["title_exclude"] = [t.strip() for t in title_exclude_str.split(",") if t.strip()]

        new_feed = FeedConfig(**feed_data)
        config.feeds.append(new_feed)
        db.upsert_feed(name, url, "podcast")
        imported += 1

    _save_config(config)
    return _redirect(request, f"/ui/feeds?imported={imported}")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _parse_opml(content: bytes) -> list[dict]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    feeds = []
    for outline in root.iter("outline"):
        xml_url = outline.get("xmlUrl") or outline.get("xmlurl") or outline.get("url")
        if not xml_url:
            continue

        title = outline.get("title") or outline.get("text") or "unknown"
        name = _slugify(title)
        if not name:
            name = f"feed-{len(feeds)}"

        feeds.append({
            "name": name,
            "url": xml_url,
            "title": title,
        })

    return feeds


def _save_config(config) -> None:
    import yaml
    from pathlib import Path

    data = config.model_dump()

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if v is not None or k in ("date_cutoff",)}
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        return obj

    data = _clean(data)

    config_path = Path(getattr(config, "_config_path", "config.yaml"))
    config_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
