"""Tests for the Episodes pages: list, per-episode inspector, and the audio they stream."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest

from siphon.app import create_app
from siphon.config import (
    AuthConfig,
    CookiesConfig,
    FeedConfig,
    FeedDefaults,
    ScheduleConfig,
    ServerConfig,
    SiphonConfig,
    StorageConfig,
    YouTubeConfig,
)
from siphon.episode_files import original_path, transcript_path

AUDIT = {
    "segments": [
        {"start": 10, "end": 40, "label": "Acme", "confidence": 0.95},
        {"start": 40, "end": 70, "label": "Globex", "confidence": 0.9},
        {"start": 200, "end": 230, "label": "maybe-an-ad", "confidence": 0.6},
    ],
    "high_confidence": [{"start": 10, "end": 70, "label": "Acme + Globex", "confidence": 0.95, "merged_from": 2}],
    "marginal": [{"start": 200, "end": 230, "label": "maybe-an-ad", "confidence": 0.6}],
    "applied_cuts": [{"start": 9.4, "end": 70.6}],
    "original_duration": 600.0,
    "final_duration": 538.8,
}
TRANSCRIPT = {"segments": [
    {"start": 0, "end": 5, "text": "Welcome to the show"},
    {"start": 20, "end": 25, "text": "This episode is brought to you by Acme"},
    {"start": 100, "end": 105, "text": "regular content here"},
    {"start": 200, "end": 205, "text": "check out my other thing"},
], "words": [], "language": "en", "duration": 600.0,
   "text": "Welcome to the show This episode is brought to you by Acme regular content here check out my other thing"}


@pytest.fixture
def config(tmp_path):
    return SiphonConfig(
        server=ServerConfig(host="127.0.0.1", port=8585, base_url="https://test.example.com"),
        auth=AuthConfig(username="u", password="p"),
        youtube=YouTubeConfig(api_key="k"),
        storage=StorageConfig(
            download_dir=str(tmp_path / "media"), database=str(tmp_path / "t.db"),
            max_disk_gb=10, keep_inspection_data=True,
        ),
        schedule=ScheduleConfig(check_interval_minutes=30),
        cookies=CookiesConfig(),
        defaults=FeedDefaults(sponsorblock_delay_minutes=0),
        feeds=[FeedConfig(name="show", url="https://example.com/rss", type="podcast", llm_trim=True)],
    )


@pytest.fixture
async def app(config, tmp_path):
    old = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        with patch.dict("sys.modules", {"apscheduler": None, "apscheduler.schedulers": None, "apscheduler.schedulers.asyncio": None}):
            application = create_app(config)
            async with application.router.lifespan_context(application):
                yield application
    finally:
        os.chdir(old)


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _write(path, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def seed(app, config, video_id="ep1", title="The Big Episode", *, audit=AUDIT, transcript=TRANSCRIPT,
         original=True, clean=True, ext=".mp3"):
    """Insert a finished episode plus whichever files it should have."""
    feed_dir = os.path.join(config.storage.download_dir, "show")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_id)
    file_path = os.path.join(feed_dir, f"{safe}{ext}")
    if clean:
        _write(file_path, b"CLEANAUDIO")
    if original and clean:
        _write(original_path(file_path), b"ORIGINALAUDIO")
    if transcript is not None:
        _write(transcript_path(config.storage.download_dir, "show", video_id), json.dumps(transcript).encode())
    db = app.state.db
    db.insert_episode(video_id=video_id, feed_name="show", title=title, status="done")
    db.update_episode_status(
        video_id, "show", "done", file_path=file_path, llm_trim_status="done",
        llm_cuts_applied=len(audit.get("high_confidence", [])), llm_segments_json=json.dumps(audit),
    )
    return file_path


class TestEpisodesList:
    async def test_nav_link_present(self, client):
        assert "/ui/episodes" in (await client.get("/ui/")).text

    async def test_lists_processed_episodes_with_link(self, app, config, client):
        seed(app, config)
        resp = await client.get("/ui/episodes")
        assert resp.status_code == 200
        assert "The Big Episode" in resp.text
        assert "/ui/episode?feed=show&amp;id=ep1" in resp.text

    async def test_empty_state(self, client):
        assert "No episodes have finished ad detection yet" in (await client.get("/ui/episodes")).text

    async def test_feed_filter(self, app, config, client):
        seed(app, config)
        assert "The Big Episode" in (await client.get("/ui/episodes?feed=show")).text
        assert "The Big Episode" not in (await client.get("/ui/episodes?feed=other")).text

    async def test_unfinished_episodes_are_not_listed(self, app, config, client):
        app.state.db.insert_episode(video_id="p1", feed_name="show", title="Still processing", status="pending_claude")
        assert "Still processing" not in (await client.get("/ui/episodes")).text

    async def test_hint_shown_when_data_is_not_kept(self, app, config, client):
        config.storage.keep_inspection_data = False
        assert "keep_inspection_data" in (await client.get("/ui/episodes")).text


class TestEpisodePage:
    async def test_shows_players_cuts_summary_and_transcript(self, app, config, client):
        seed(app, config)
        resp = await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})
        assert resp.status_code == 200
        text = resp.text
        assert "The Big Episode" in text
        assert 'id="player-original"' in text and 'id="player-clean"' in text
        assert "which=original" in text and "which=clean" in text
        assert "Acme + Globex" in text                       # the cut
        assert "10:00" in text and "8:59" in text             # original and cleaned lengths
        assert "Flagged but not cut" in text and "maybe-an-ad" in text
        assert 'class="seg seg-cut"' in text                  # the ad line is highlighted as cut
        assert 'class="seg seg-possible"' in text
        assert 'class="seg seg-content"' in text
        assert "brought to you by Acme" in text

    async def test_transcript_text_is_escaped(self, app, config, client):
        transcript = {"segments": [{"start": 0, "end": 5, "text": "<script>alert(1)</script>"}]}
        seed(app, config, transcript=transcript)
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text

    async def test_missing_transcript_shows_a_notice(self, app, config, client):
        seed(app, config, transcript=None)
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert "transcript was not kept" in text
        assert 'id="transcript"' not in text
        assert "Acme + Globex" in text   # the cut list still works from the stored record

    async def test_missing_original_explains_how_to_get_it(self, app, config, client):
        seed(app, config, original=False)
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert 'id="player-original"' not in text
        assert 'id="player-clean"' in text
        assert "processed before originals were kept" in text

    async def test_missing_original_when_setting_off(self, app, config, client):
        seed(app, config, original=False)
        config.storage.keep_inspection_data = False
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert "storage.keep_inspection_data" in text

    async def test_no_cuts_uses_one_file_for_both_players(self, app, config, client):
        seed(app, config, audit={"segments": [], "high_confidence": [], "marginal": []}, original=False)
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert 'id="player-original"' in text and 'id="player-clean"' in text
        assert "same file" in text
        assert "No ads were cut" in text

    async def test_video_episodes_get_video_players(self, app, config, client):
        seed(app, config, ext=".mp4")
        text = (await client.get("/ui/episode", params={"feed": "show", "id": "ep1"})).text
        assert '<video id="player-original"' in text

    async def test_htmx_partial_has_no_page_title_text(self, app, config, client):
        seed(app, config)
        params = {"feed": "show", "id": "ep1"}
        partial = (await client.get("/ui/episode", params=params, headers={"HX-Request": "true"})).text
        assert "<title>" not in partial and "<html" not in partial
        assert not partial.lstrip().startswith("- ")          # the title suffix must not leak into the body
        full = (await client.get("/ui/episode", params=params)).text
        assert "<title>Siphon - The Big Episode</title>" in full
        listing = (await client.get("/ui/episodes", headers={"HX-Request": "true"})).text
        assert not listing.lstrip().startswith("- ")

    async def test_unknown_episode_is_404(self, client):
        assert (await client.get("/ui/episode", params={"feed": "show", "id": "nope"})).status_code == 404

    async def test_podcast_ids_with_slashes_and_colons_work(self, app, config, client):
        video_id = "https://feeds.example.com/ep/123?x=1"
        seed(app, config, video_id=video_id, title="URL id episode")
        listing = await client.get("/ui/episodes")
        assert "URL id episode" in listing.text
        page = await client.get("/ui/episode", params={"feed": "show", "id": video_id})
        assert page.status_code == 200 and "URL id episode" in page.text
        audio = await client.get("/ui/episode/audio", params={"feed": "show", "id": video_id, "which": "clean"})
        assert audio.status_code == 200 and audio.content == b"CLEANAUDIO"


class TestEpisodeAudio:
    async def test_serves_cleaned_and_original(self, app, config, client):
        seed(app, config)
        clean = await client.get("/ui/episode/audio", params={"feed": "show", "id": "ep1", "which": "clean"})
        original = await client.get("/ui/episode/audio", params={"feed": "show", "id": "ep1", "which": "original"})
        assert (clean.status_code, clean.content) == (200, b"CLEANAUDIO")
        assert (original.status_code, original.content) == (200, b"ORIGINALAUDIO")
        assert clean.headers["content-type"] == "audio/mpeg"

    async def test_range_requests_work_for_seeking(self, app, config, client):
        seed(app, config)
        resp = await client.get(
            "/ui/episode/audio", params={"feed": "show", "id": "ep1", "which": "original"},
            headers={"Range": "bytes=0-3"},
        )
        assert resp.status_code == 206 and resp.content == b"ORIG"

    async def test_no_cuts_original_is_the_clean_file(self, app, config, client):
        seed(app, config, audit={"segments": [], "high_confidence": [], "marginal": []}, original=False)
        resp = await client.get("/ui/episode/audio", params={"feed": "show", "id": "ep1", "which": "original"})
        assert resp.status_code == 200 and resp.content == b"CLEANAUDIO"

    async def test_missing_original_is_404(self, app, config, client):
        seed(app, config, original=False)
        resp = await client.get("/ui/episode/audio", params={"feed": "show", "id": "ep1", "which": "original"})
        assert resp.status_code == 404

    async def test_missing_clean_file_is_404(self, app, config, client):
        seed(app, config, clean=False)
        resp = await client.get("/ui/episode/audio", params={"feed": "show", "id": "ep1", "which": "clean"})
        assert resp.status_code == 404

    async def test_bad_which_and_unknown_episode(self, app, config, client):
        seed(app, config)
        assert (await client.get("/ui/episode/audio", params={"feed": "show", "id": "ep1", "which": "../../etc/passwd"})).status_code == 400
        assert (await client.get("/ui/episode/audio", params={"feed": "show", "id": "nope", "which": "clean"})).status_code == 404

    async def test_paths_come_from_the_database_not_the_request(self, app, config, client):
        seed(app, config)
        resp = await client.get("/ui/episode/audio", params={"feed": "../..", "id": "ep1", "which": "clean"})
        assert resp.status_code == 404


class TestLocalhostOnly:
    async def test_remote_clients_are_refused(self, app, config):
        seed(app, config)
        transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 4321))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as remote:
            for url in ("/ui/episodes", "/ui/episode?feed=show&id=ep1", "/ui/episode/audio?feed=show&id=ep1&which=clean"):
                assert (await remote.get(url)).status_code == 403, url
