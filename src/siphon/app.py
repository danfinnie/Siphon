"""FastAPI application factory with lifespan, auth, and scheduler."""
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from siphon.config import SiphonConfig
from siphon.db import Database

logger = logging.getLogger(__name__)
security = HTTPBasic()


def verify_credentials(config: SiphonConfig):
    """Return a FastAPI dependency that checks HTTP Basic credentials."""
    def _check(credentials: HTTPBasicCredentials = Depends(security)):
        correct_user = secrets.compare_digest(credentials.username, config.auth.username)
        correct_pass = secrets.compare_digest(credentials.password, config.auth.password)
        if not (correct_user and correct_pass):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials
    return _check


def _is_local_or_tailnet(client_ip: str | None) -> bool:
    """True for localhost or a Tailscale address (100.64.0.0/10 or fd7a::/16)."""
    import ipaddress
    if client_ip in ("127.0.0.1", "::1", "localhost"):
        return True
    try:
        addr = ipaddress.ip_address(client_ip)
    except (TypeError, ValueError):
        return False
    if addr in ipaddress.ip_network("100.64.0.0/10"):
        return True
    return isinstance(addr, ipaddress.IPv6Address) and addr.packed[:2] == b"\xfd\x7a"


def create_app(config: SiphonConfig) -> FastAPI:
    """Build and return the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        db = Database(config.storage.database)
        app.state.db = db
        app.state.config = config

        # Configure activity timezone from config
        from siphon.activity import set_timezone
        set_timezone(config.server.timezone)

        # Sync feeds from config to DB
        for feed in config.feeds:
            db.upsert_feed(feed.name, feed.url, feed.type)

        # Recover episodes stuck in intermediate states from a previous crash
        from siphon.pipeline import recover_interrupted
        recover_interrupted(config, db)

        # Start scheduler
        scheduler = None
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from siphon.activity import check_paused
            from siphon.pipeline import (
                check_feeds, process_downloads,
                process_whisper, process_claude,
            )

            def _reload_config():
                from siphon.config import load_config
                try:
                    config_path = getattr(app.state.config, "_config_path", None)
                    if config_path:
                        app.state.config = load_config(config_path)
                        set_timezone(app.state.config.server.timezone)
                        for feed in app.state.config.feeds:
                            app.state.db.upsert_feed(feed.name, feed.url, feed.type)
                except Exception:
                    pass

            async def _scheduled_check_feeds():
                if check_paused():
                    return
                _reload_config()
                await check_feeds(app.state.config, app.state.db)

            async def _scheduled_process_downloads():
                if check_paused():
                    return
                _reload_config()
                await process_downloads(app.state.config, app.state.db)

            async def _scheduled_process_whisper():
                if check_paused():
                    return
                await process_whisper(app.state.config, app.state.db)

            async def _scheduled_process_claude():
                if check_paused():
                    return
                await process_claude(app.state.config, app.state.db)

            async def _scheduled_resource_snapshot():
                # Always runs (even while paused) — we want the resource trend
                # regardless of queue activity. Best-effort; never raises.
                from siphon.diagnostics import log_resource_snapshot
                log_resource_snapshot()

            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                _scheduled_check_feeds, 'interval',
                minutes=config.schedule.check_interval_minutes,
                id='check_feeds',
                name='Check feeds',
            )
            dl_interval = min(
                config.schedule.youtube_download_interval_minutes,
                config.schedule.podcast_download_interval_minutes,
            )
            scheduler.add_job(
                _scheduled_process_downloads, 'interval',
                minutes=dl_interval,
                id='process_downloads',
                name='Process downloads',
            )
            scheduler.add_job(
                _scheduled_process_whisper, 'interval',
                seconds=30,
                id='process_whisper',
                name='Process Whisper',
            )
            scheduler.add_job(
                _scheduled_process_claude, 'interval',
                seconds=30,
                id='process_claude',
                name='Process Claude',
            )
            scheduler.add_job(
                _scheduled_resource_snapshot, 'interval',
                minutes=5,
                id='resource_snapshot',
                name='Resource snapshot',
                next_run_time=datetime.now(),  # log one immediately at startup
            )
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info("Scheduler started with 5 jobs")
        except Exception as e:
            logger.warning(f"Scheduler failed to start: {e}")

        yield

        # Shutdown
        if scheduler:
            scheduler.shutdown(wait=False)
        db.close()

    auth_dep = verify_credentials(config)

    app = FastAPI(
        title="Siphon",
        description="Self-hosted YouTube-to-podcast bridge",
        lifespan=lifespan,
    )

    from siphon.routes.feeds import router as feeds_router
    from siphon.routes.media import router as media_router
    from siphon.routes.api import router as api_router
    from siphon.routes.ui import router as ui_router, get_img_dir

    # RSS and API routes require auth
    app.include_router(feeds_router, dependencies=[Depends(auth_dep)])
    app.include_router(api_router, dependencies=[Depends(auth_dep)])

    # Media routes — Tailnet-only, no auth needed
    app.include_router(media_router)

    # Web UI — localhost or Tailnet, no auth needed
    app.include_router(ui_router)

    # Static images (logo, icons) — must mount on app, not router
    from starlette.staticfiles import StaticFiles
    img_dir = get_img_dir()
    if img_dir:
        app.mount("/ui/img", StaticFiles(directory=img_dir), name="ui-img")

    # --- Security middleware ---
    # Track bad requests per IP for auto-banning
    _bad_hits: dict[str, list[float]] = {}  # ip -> list of timestamps
    _banned: dict[str, float] = {}  # ip -> ban expiry timestamp
    BAN_THRESHOLD = 20  # bad requests within the window
    BAN_WINDOW = 60  # seconds to count hits
    BAN_DURATION = 3600  # ban for 1 hour

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        import time

        client_ip = request.client.host if request.client else None
        now = time.time()

        # Check if banned
        if client_ip and client_ip in _banned:
            if now < _banned[client_ip]:
                return Response(status_code=403, content="Banned")
            else:
                del _banned[client_ip]

        # Localhost or Tailnet only for /ui/ and /media/
        if request.url.path.startswith(("/ui", "/media/")):
            if not _is_local_or_tailnet(client_ip):
                return Response(status_code=403, content="Forbidden")

        response = await call_next(request)

        # Track 404s and 401s from non-localhost IPs
        if client_ip and client_ip not in ("127.0.0.1", "::1", "localhost"):
            if response.status_code in (404, 401):
                hits = _bad_hits.setdefault(client_ip, [])
                hits.append(now)
                # Trim old hits
                _bad_hits[client_ip] = [t for t in hits if now - t < BAN_WINDOW]
                if len(_bad_hits[client_ip]) >= BAN_THRESHOLD:
                    _banned[client_ip] = now + BAN_DURATION
                    _bad_hits.pop(client_ip, None)
                    logger.warning("Banned IP %s for %ds (%d bad requests)",
                                   client_ip, BAN_DURATION, BAN_THRESHOLD)

        return response

    return app
