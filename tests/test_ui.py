"""Tests for the web UI routes."""

import asyncio
from contextlib import asynccontextmanager
import os
from unittest.mock import patch

import httpx
import pytest

from siphon.app import create_app
from siphon.config import (
    YouTubeConfig,
    AuthConfig,
    CookiesConfig,
    FeedConfig,
    FeedDefaults,
    ScheduleConfig,
    ServerConfig,
    SiphonConfig,
    StorageConfig,
)


@pytest.fixture
def config(tmp_path):
    return SiphonConfig(
        server=ServerConfig(host="127.0.0.1", port=8585, base_url="https://test.example.com"),
        auth=AuthConfig(username="testuser", password="testpass"),
        youtube=YouTubeConfig(api_key="test-key"),
        storage=StorageConfig(
            download_dir=str(tmp_path / "media"),
            database=str(tmp_path / "test.db"),
            max_disk_gb=10,
            youtube_keep_per_feed=5,
            podcast_keep_per_feed=20,
        ),
        schedule=ScheduleConfig(check_interval_minutes=30),
        cookies=CookiesConfig(),
        defaults=FeedDefaults(sponsorblock_delay_minutes=0),
        feeds=[
            FeedConfig(name="test-feed", url="https://www.youtube.com/@TestChannel"),
        ],
    )


@pytest.fixture
async def client(config, tmp_path):
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        with patch.dict("sys.modules", {"apscheduler": None, "apscheduler.schedulers": None, "apscheduler.schedulers.asyncio": None}):
            app = create_app(config)
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                    yield c
    finally:
        os.chdir(old_cwd)


class TestStatsPage:
    async def test_stats_page_no_auth_required(self, client):
        resp = await client.get("/ui/")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text

    async def test_stats_page_shows_disk_usage(self, client):
        resp = await client.get("/ui/")
        assert "Disk /" in resp.text

    async def test_stats_page_has_check_now_button(self, client):
        resp = await client.get("/ui/")
        assert "Check Now" in resp.text


class TestFeedsPage:
    async def test_feeds_page_no_auth_required(self, client):
        resp = await client.get("/ui/feeds")
        assert resp.status_code == 200
        assert "Feeds" in resp.text

    async def test_feeds_page_shows_feed(self, client):
        resp = await client.get("/ui/feeds")
        assert "test-feed" in resp.text
        assert "youtube" in resp.text


class TestAddFeed:
    async def test_add_page_loads(self, client):
        resp = await client.get("/ui/add")
        assert resp.status_code == 200
        assert "Add Feed" in resp.text

    async def test_add_feed_creates_feed(self, client, config):
        resp = await client.post("/ui/add", data={
            "url": "https://www.youtube.com/@NewChannel",
            "name": "new-channel",
            "type": "youtube",
            "mode": "",
            "quality": "",
            "llm_trim": "",
            "date_cutoff": "",
            "sponsorblock_delay_minutes": "",
            "title_exclude": "",
            "claude_prompt_extra": "",
        }, follow_redirects=False)
        assert resp.status_code == 303

        names = [f.name for f in config.feeds]
        assert "new-channel" in names

    async def test_add_feed_with_sponsorblock_delay(self, client, config):
        resp = await client.post("/ui/add", data={
            "url": "https://www.youtube.com/@DelayChannel",
            "name": "delay-channel",
            "type": "youtube",
            "mode": "",
            "quality": "",
            "llm_trim": "",
            "date_cutoff": "",
            "sponsorblock_delay_minutes": "120",
            "title_exclude": "",
            "claude_prompt_extra": "",
        }, follow_redirects=False)
        assert resp.status_code == 303

        feed = next(f for f in config.feeds if f.name == "delay-channel")
        assert feed.sponsorblock_delay_minutes == 120

    async def test_add_podcast_feed(self, client, config):
        resp = await client.post("/ui/add", data={
            "url": "https://example.com/podcast/rss",
            "name": "my-podcast",
            "type": "podcast",
            "mode": "audio",
            "quality": "",
            "llm_trim": "true",
            "date_cutoff": "20240101",
            "sponsorblock_delay_minutes": "",
            "title_exclude": "bonus, trailer",
            "claude_prompt_extra": "Remove Discord promos.",
        }, follow_redirects=False)
        assert resp.status_code == 303

        feed = next(f for f in config.feeds if f.name == "my-podcast")
        assert feed.type == "podcast"
        assert feed.mode == "audio"
        assert feed.llm_trim is True
        assert feed.date_cutoff == "20240101"
        assert feed.title_exclude == ["bonus", "trailer"]
        assert feed.claude_prompt_extra == "Remove Discord promos."

    async def test_add_duplicate_name_rejected(self, client):
        resp = await client.post("/ui/add", data={
            "url": "https://www.youtube.com/@Other",
            "name": "test-feed",
            "type": "youtube",
            "mode": "",
            "quality": "",
            "llm_trim": "",
            "date_cutoff": "",
            "sponsorblock_delay_minutes": "",
            "title_exclude": "",
            "claude_prompt_extra": "",
        })
        assert resp.status_code == 200
        assert "already exists" in resp.text

    async def test_add_feed_slugifies_name(self, client, config):
        resp = await client.post("/ui/add", data={
            "url": "https://www.youtube.com/@Weird",
            "name": "My Cool Channel!",
            "type": "youtube",
            "mode": "",
            "quality": "",
            "llm_trim": "",
            "date_cutoff": "",
            "sponsorblock_delay_minutes": "",
            "title_exclude": "",
            "claude_prompt_extra": "",
        }, follow_redirects=False)
        assert resp.status_code == 303

        names = [f.name for f in config.feeds]
        assert "my-cool-channel" in names


class TestFeedActions:
    async def test_update_feed_settings(self, client, config):
        resp = await client.post("/ui/feed-action", data={
            "feed_name": "test-feed",
            "action": "update",
            "mode": "audio",
            "quality": "720",
            "sponsorblock": "false",
            "llm_trim": "true",
            "block_shorts": "false",
            "min_duration_seconds": "30",
            "date_cutoff": "20240601",
            "sponsorblock_delay_minutes": "0",
            "title_exclude": "live, #shorts",
            "claude_prompt_extra": "Extra stuff",
        }, follow_redirects=False)
        assert resp.status_code == 303

        feed = next(f for f in config.feeds if f.name == "test-feed")
        assert feed.mode == "audio"
        assert feed.quality == 720
        assert feed.sponsorblock is False
        assert feed.llm_trim is True

    async def test_delete_feed(self, client, config):
        await client.post("/ui/add", data={
            "url": "https://www.youtube.com/@Deleteme",
            "name": "delete-me",
            "type": "youtube",
            "mode": "", "quality": "", "llm_trim": "", "date_cutoff": "",
            "sponsorblock_delay_minutes": "", "title_exclude": "", "claude_prompt_extra": "",
        }, follow_redirects=False)

        resp = await client.post("/ui/feed-action", data={
            "feed_name": "delete-me",
            "action": "delete",
        }, follow_redirects=False)
        assert resp.status_code == 303

        names = [f.name for f in config.feeds]
        assert "delete-me" not in names

    async def test_rename_feed(self, client, config):
        resp = await client.post("/ui/feed-action", data={
            "feed_name": "test-feed",
            "action": "rename",
            "new_name": "renamed-feed",
        }, follow_redirects=False)
        assert resp.status_code == 303

        names = [f.name for f in config.feeds]
        assert "renamed-feed" in names
        assert "test-feed" not in names

    async def test_catchup_sets_date_cutoff(self, client, config):
        resp = await client.post("/ui/feed-action", data={
            "feed_name": "test-feed",
            "action": "catchup",
        }, follow_redirects=False)
        assert resp.status_code == 303

        feed = next(f for f in config.feeds if f.name == "test-feed")
        assert feed.date_cutoff is not None
        assert len(feed.date_cutoff) == 8


class TestOPMLImport:
    async def test_import_page_loads(self, client):
        resp = await client.get("/ui/import")
        assert resp.status_code == 200
        assert "OPML" in resp.text

    async def test_import_parses_opml(self, client):
        opml_content = b"""<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head><title>Podcasts</title></head>
  <body>
    <outline text="My Podcast" xmlUrl="https://example.com/feed1.xml" />
    <outline text="Another Show" xmlUrl="https://example.com/feed2.xml" />
  </body>
</opml>"""

        resp = await client.post("/ui/import",
            files={"opml_file": ("podcasts.opml", opml_content, "application/xml")})
        assert resp.status_code == 200
        assert "my-podcast" in resp.text
        assert "another-show" in resp.text

    async def test_import_confirm_adds_feeds(self, client, config):
        resp = await client.post("/ui/import/confirm", data={
            "total": "2",
            "import_0": "1",
            "name_0": "imported-pod",
            "url_0": "https://example.com/imported.xml",
            "date_cutoff_0": "20240101",
            "llm_trim_0": "true",
            "title_exclude_0": "",
            "import_1": "1",
            "name_1": "imported-pod-2",
            "url_1": "https://example.com/imported2.xml",
            "date_cutoff_1": "",
            "llm_trim_1": "",
            "title_exclude_1": "bonus",
        }, follow_redirects=False)
        assert resp.status_code == 303

        names = [f.name for f in config.feeds]
        assert "imported-pod" in names
        assert "imported-pod-2" in names

    async def test_import_skips_unchecked(self, client, config):
        resp = await client.post("/ui/import/confirm", data={
            "total": "2",
            "name_0": "skip-me",
            "url_0": "https://example.com/skip.xml",
            "date_cutoff_0": "", "llm_trim_0": "", "title_exclude_0": "",
            "import_1": "1",
            "name_1": "keep-me",
            "url_1": "https://example.com/keep.xml",
            "date_cutoff_1": "", "llm_trim_1": "", "title_exclude_1": "",
        }, follow_redirects=False)
        assert resp.status_code == 303

        names = [f.name for f in config.feeds]
        assert "skip-me" not in names
        assert "keep-me" in names

    async def test_import_empty_opml(self, client):
        resp = await client.post("/ui/import",
            files={"opml_file": ("empty.opml", b"<opml><body></body></opml>", "application/xml")})
        assert resp.status_code == 200
        assert "No feeds found" in resp.text


class TestCheckNow:
    """Verify that POST /ui/check-now actually runs the background task to completion."""

    @pytest.fixture
    async def app_and_client(self, tmp_path):
        """Yield (app, client) so the test can inspect app.state.db."""
        config = SiphonConfig(
            server=ServerConfig(host="127.0.0.1", port=8585, base_url="https://test.example.com"),
            auth=AuthConfig(username="testuser", password="testpass"),
            youtube=YouTubeConfig(api_key="test-key"),
        storage=StorageConfig(
                download_dir=str(tmp_path / "media"),
                database=str(tmp_path / "test.db"),
                max_disk_gb=10,
                youtube_keep_per_feed=5,
                podcast_keep_per_feed=20,
            ),
            schedule=ScheduleConfig(
                check_interval_minutes=30,
                youtube_feeds_per_check=10,
                podcast_feeds_per_check=30,
            ),
            cookies=CookiesConfig(),
            defaults=FeedDefaults(sponsorblock_delay_minutes=0),
            feeds=[
                FeedConfig(name="test-feed", url="https://www.youtube.com/@TestChannel"),
            ],
        )
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            with patch.dict("sys.modules", {
                "apscheduler": None,
                "apscheduler.schedulers": None,
                "apscheduler.schedulers.asyncio": None,
            }):
                app = create_app(config)
                async with app.router.lifespan_context(app):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                        yield app, c
        finally:
            os.chdir(old_cwd)

    async def test_check_now_runs_background_task(self, app_and_client):
        """POST /ui/check-now must launch a background task that actually inserts episodes."""
        app, client = app_and_client
        db = app.state.db

        fake_entries = [
            {
                "id": "bg-vid-001",
                "title": "Background Check Video",
                "url": "https://www.youtube.com/watch?v=bg-vid-001",
                "duration": 600,
                "upload_date": "20250601",
                "description": "Test",
                "channel": "TestChannel",
                "thumbnail": "https://img.youtube.com/vi/bg-vid-001/0.jpg",
            },
            {
                "id": "bg-vid-002",
                "title": "Second Background Video",
                "url": "https://www.youtube.com/watch?v=bg-vid-002",
                "duration": 900,
                "upload_date": "20250602",
                "description": "Test 2",
                "channel": "TestChannel",
                "thumbnail": "https://img.youtube.com/vi/bg-vid-002/0.jpg",
            },
        ]

        async def _noop_downloads(*a, **kw):
            pass

        with patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST"), \
             patch("siphon.youtube.get_channel_metadata", return_value={}), \
             patch("siphon.youtube.list_videos", return_value=fake_entries), \
             patch("siphon.pipeline.process_downloads", side_effect=_noop_downloads):
            resp = await client.post("/ui/check-now", follow_redirects=False)
            assert resp.status_code == 303

            # Wait for the background task to finish.
            # The task set lives on app.state; drain it.
            tasks = getattr(app.state, "_background_tasks", set())
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Verify episodes were actually inserted
        ep1 = db.get_episode("bg-vid-001", "test-feed")
        ep2 = db.get_episode("bg-vid-002", "test-feed")

        assert ep1 is not None, "Episode bg-vid-001 was not inserted — background task did not run"
        assert ep1["title"] == "Background Check Video"
        assert ep1["status"] == "pending"

        assert ep2 is not None, "Episode bg-vid-002 was not inserted — background task did not run"
        assert ep2["title"] == "Second Background Video"
        assert ep2["status"] == "pending"

    async def test_check_now_task_error_is_logged_not_swallowed(self, app_and_client):
        """If check_feeds raises, the task wrapper must catch it (not crash the server)."""
        app, client = app_and_client

        with patch("siphon.youtube.list_videos", side_effect=RuntimeError("boom")), \
             patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST"), \
             patch("siphon.youtube.get_channel_metadata", return_value={}):
            resp = await client.post("/ui/check-now", follow_redirects=False)
            assert resp.status_code == 303

            tasks = getattr(app.state, "_background_tasks", set())
            if tasks:
                # Should not raise — the wrapper catches the exception
                await asyncio.gather(*tasks, return_exceptions=True)

        # Feed should have an error recorded
        db = app.state.db
        feed = db.get_feed("test-feed")
        assert feed["last_error"] == "boom"


# ------------------------------------------------------------------ #
# htmx integration tests
# ------------------------------------------------------------------ #

HTMX_HEADERS = {"HX-Request": "true"}


class TestHtmxNavigation:
    async def test_stats_htmx_returns_content_only(self, client):
        resp = await client.get("/ui/", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "Dashboard" in resp.text
        # Should NOT contain the full page shell
        assert "<html" not in resp.text
        assert "activity-footer" not in resp.text

    async def test_feeds_htmx_returns_content_only(self, client):
        resp = await client.get("/ui/feeds", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "test-feed" in resp.text
        assert "<html" not in resp.text

    async def test_add_page_htmx_returns_content_only(self, client):
        resp = await client.get("/ui/add", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "Add Feed" in resp.text
        assert "<html" not in resp.text

    async def test_import_page_htmx_returns_content_only(self, client):
        resp = await client.get("/ui/import", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "<html" not in resp.text

    async def test_feeds_search_htmx(self, client):
        resp = await client.get("/ui/feeds?q=test", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "test-feed" in resp.text
        assert "<html" not in resp.text


class TestHtmxActions:
    async def test_check_now_htmx_returns_204(self, client):
        resp = await client.post("/ui/check-now", headers=HTMX_HEADERS)
        assert resp.status_code == 204

    async def test_add_feed_htmx_returns_hx_redirect(self, client):
        resp = await client.post("/ui/add", data={
            "url": "https://example.com/podcast/rss",
            "name": "htmx-test",
            "mode": "",
            "quality": "",
            "llm_trim": "",
            "date_cutoff": "",
            "sponsorblock_delay_minutes": "",
            "title_exclude": "",
            "claude_prompt_extra": "",
        }, headers=HTMX_HEADERS, follow_redirects=False)
        assert resp.headers.get("HX-Redirect") == "/ui/feeds"

    async def test_feed_update_htmx_returns_card(self, client):
        resp = await client.post("/ui/feed-action", data={
            "feed_name": "test-feed",
            "action": "update",
            "mode": "video",
            "quality": "1440",
            "sponsorblock": "true",
            "llm_trim": "false",
            "block_shorts": "true",
            "min_duration_seconds": "60",
            "date_cutoff": "",
            "sponsorblock_delay_minutes": "0",
            "title_exclude": "",
            "claude_prompt_extra": "",
        }, headers=HTMX_HEADERS, follow_redirects=False)
        assert resp.status_code == 200
        assert "test-feed" in resp.text
        assert "card-test-feed" in resp.text
        assert "<html" not in resp.text

    async def test_delete_feed_htmx_removes_card(self, client, config):
        # Add a feed to delete
        await client.post("/ui/add", data={
            "url": "https://www.youtube.com/@ToDelete",
            "name": "to-delete",
            "mode": "", "quality": "", "llm_trim": "",
            "date_cutoff": "", "sponsorblock_delay_minutes": "",
            "title_exclude": "", "claude_prompt_extra": "",
        }, follow_redirects=False)

        resp = await client.post("/ui/feed-action", data={
            "feed_name": "to-delete",
            "action": "delete",
        }, headers=HTMX_HEADERS, follow_redirects=False)
        assert resp.status_code == 200
        assert resp.text == ""


class TestUIAccessByClientIP:
    """The UI is open to localhost and Tailnet devices, closed to everyone else."""

    @pytest.fixture
    def make_client(self, config, tmp_path):
        @asynccontextmanager
        async def _make(client_ip):
            old_cwd = os.getcwd()
            os.chdir(str(tmp_path))
            try:
                with patch.dict("sys.modules", {"apscheduler": None, "apscheduler.schedulers": None, "apscheduler.schedulers.asyncio": None}):
                    app = create_app(config)
                    async with app.router.lifespan_context(app):
                        transport = httpx.ASGITransport(app=app, client=(client_ip, 51234))
                        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                            yield c
            finally:
                os.chdir(old_cwd)
        return _make

    @pytest.mark.parametrize("ip", ["100.101.102.103", "fd7a:115c:a1e0::1"])
    async def test_tailnet_device_allowed(self, make_client, ip):
        async with make_client(ip) as c:
            resp = await c.get("/ui/")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text

    @pytest.mark.parametrize("ip", ["203.0.113.9", "192.168.1.50"])
    async def test_non_tailnet_device_forbidden(self, make_client, ip):
        async with make_client(ip) as c:
            resp = await c.get("/ui/")
        assert resp.status_code == 403
