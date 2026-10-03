"""Tests for downloading an episode again: database flag, staged download and the rules."""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from siphon.config import (
    AuthConfig, CookiesConfig, FeedConfig, FeedDefaults, LLMConfig, ScheduleConfig,
    ServerConfig, SiphonConfig, StorageConfig, YouTubeConfig,
)
from siphon.db import Database
from siphon.episode_files import original_path, staging_path, transcript_path
from siphon.pipeline import _redownload_pass, process_downloads
from siphon.reprocess import ReprocessError, check_reprocess, reprocess_episode

GUID = "guid-1"
FEED_XML_EPISODES = [{"guid": GUID, "audio_url": "https://cdn.example.com/ep.mp3"}]


@pytest.fixture()
def db():
    d = Database(":memory:")
    d.upsert_feed("pod", "https://example.com/rss", "podcast")
    d.upsert_feed("yt", "https://www.youtube.com/@x", "youtube")
    yield d
    d.close()


@pytest.fixture()
def config(tmp_path):
    return SiphonConfig(
        server=ServerConfig(host="127.0.0.1", port=8585, base_url="http://localhost"),
        auth=AuthConfig(username="u", password="p"),
        youtube=YouTubeConfig(api_key="k"),
        storage=StorageConfig(download_dir=str(tmp_path / "media"), database=":memory:", max_disk_gb=100),
        schedule=ScheduleConfig(podcast_download_delay_seconds=0, youtube_download_delay_seconds=0),
        cookies=CookiesConfig(),
        defaults=FeedDefaults(min_duration_seconds=60, block_shorts=False),
        llm=LLMConfig(),
        feeds=[
            FeedConfig(name="pod", url="https://example.com/rss", type="podcast", llm_trim=True),
            FeedConfig(name="yt", url="https://www.youtube.com/@x", type="youtube"),
        ],
    )


def make_episode(db, config, *, status="done", feed="pod", files=True, ext=".mp3"):
    feed_dir = os.path.join(config.storage.download_dir, feed)
    os.makedirs(feed_dir, exist_ok=True)
    path = os.path.join(feed_dir, "guid-1" + ext) if files else None
    if files:
        open(path, "wb").write(b"OLD-CLEANED")
        open(original_path(path), "wb").write(b"OLD-ORIGINAL")
        open(transcript_path(config.storage.download_dir, feed, GUID), "w").write("{}")
    db.insert_episode(video_id=GUID, feed_name=feed, title="Ep", upload_date="20260929", status="done")
    db.update_episode_status(
        GUID, feed, status, file_path=path, file_size=11, llm_trim_status="done" if status == "done" else None,
        llm_cuts_applied=2, llm_segments_json='{"high_confidence": []}', redownload_requested=1,
    )
    return path


def downloads(content=b"NEW-AUDIO", episodes=None, url="https://cdn.example.com/ep.mp3", duration=1800,
              valid=True, download_error=None):
    """Patch everything a re-download touches outside the database and the filesystem."""
    def fake_download(audio_url, output_path, **kwargs):
        if download_error:
            open(output_path, "wb").write(b"PARTIAL")      # a half-written file, as a real failure leaves
            raise download_error
        open(output_path, "wb").write(content)
        return len(content)

    feed_episodes = [{"guid": GUID, "audio_url": url}] if episodes is None else episodes
    return [
        patch("siphon.podcast.fetch_podcast_rss", return_value=b"<rss/>"),
        patch("siphon.podcast.parse_podcast_feed", return_value={"episodes": feed_episodes}),
        patch("siphon.podcast.download_podcast_audio", side_effect=fake_download),
        patch("siphon.cutter.validate_file", return_value=valid),
        patch("siphon.cutter.normalize_timestamps", return_value=False),
        patch("siphon.pipeline._get_file_duration", new=AsyncMock(return_value=duration)),
    ]


async def run_pass(config, db, **kw):
    patches = downloads(**kw)
    for p in patches:
        p.start()
    try:
        await _redownload_pass(config, db)
    finally:
        for p in patches:
            p.stop()


def row(db):
    return db.get_episode(GUID, "pod")


# ------------------------------------------------------------------ database
class TestDatabase:
    def test_new_column_exists_and_defaults_to_zero(self, db):
        db.insert_episode(video_id="a", feed_name="pod", title="A")
        assert db.get_episode("a", "pod")["redownload_requested"] == 0

    def test_old_databases_are_upgraded(self, tmp_path):
        path = str(tmp_path / "old.db")
        first = Database(path)
        first.upsert_feed("f", "u", "podcast")
        first.insert_episode(video_id="v", feed_name="f", title="t")
        first.close()
        # make it look like a database from before this column existed
        conn = sqlite3.connect(path)
        conn.execute("ALTER TABLE episodes DROP COLUMN redownload_requested")
        conn.commit()
        assert "redownload_requested" not in [r[1] for r in conn.execute("PRAGMA table_info(episodes)")]
        conn.close()

        upgraded = Database(path)
        assert upgraded.get_episode("v", "f")["redownload_requested"] == 0
        upgraded.close()

    def test_done_episode_is_only_flagged(self, db, config):
        make_episode(db, config)
        db.update_episode_status(GUID, "pod", "done", redownload_requested=0)
        db.request_redownload(GUID, "pod", "done")
        r = row(db)
        assert (r["status"], r["redownload_requested"]) == ("done", 1)
        assert r["llm_cuts_applied"] == 2          # nothing about the current copy changes yet

    @pytest.mark.parametrize("status", ["filtered", "pruned", "failed"])
    def test_other_statuses_go_back_into_the_normal_queue(self, db, config, status):
        make_episode(db, config, status=status, files=False)
        db.update_episode_status(GUID, "pod", status, filter_reason="too_short", error="boom", retry_count=3)
        db.request_redownload(GUID, "pod", status)
        r = row(db)
        assert (r["status"], r["redownload_requested"], r["retry_count"]) == ("eligible", 0, 0)
        assert (r["filter_reason"], r["error"], r["llm_segments_json"]) == (None, None, None)

    def test_only_flagged_done_episodes_are_returned(self, db, config):
        make_episode(db, config)
        db.insert_episode(video_id="other", feed_name="pod", title="x", status="done")
        assert [e["video_id"] for e in db.get_redownload_requests()] == [GUID]
        db.update_episode_status(GUID, "pod", "pending_claude")
        assert db.get_redownload_requests() == []

    def test_recent_episodes_lists_everything_newest_published_first(self, db):
        for vid, date, status in (("a", "20260101", "done"), ("b", "20260301", "filtered"), ("c", "20260201", "pruned")):
            db.insert_episode(video_id=vid, feed_name="pod", title=vid, upload_date=date, status=status)
        assert [e["video_id"] for e in db.get_recent_episodes()] == ["b", "c", "a"]
        assert [e["video_id"] for e in db.get_recent_episodes(feed_name="nope")] == []
        assert len(db.get_recent_episodes(limit=2)) == 2


# ------------------------------------------------------------------ the download itself
class TestRedownload:
    async def test_new_audio_replaces_the_old_and_the_episode_is_reprocessed(self, db, config):
        path = make_episode(db, config)
        await run_pass(config, db)
        r = row(db)
        assert open(path, "rb").read() == b"NEW-AUDIO"
        assert (r["status"], r["redownload_requested"], r["error"]) == ("pending_whisper", 0, None)
        assert (r["llm_trim_status"], r["llm_cuts_applied"], r["llm_segments_json"]) == ("pending", 0, None)
        assert (r["file_size"], r["duration"], r["file_path"]) == (len(b"NEW-AUDIO"), 1800, path)
        # what belonged to the old audio is gone: its original, its transcript and any staging file
        assert not os.path.exists(original_path(path))
        assert not os.path.exists(transcript_path(config.storage.download_dir, "pod", GUID))
        assert not os.path.exists(staging_path(path))

    async def test_without_ad_removal_the_episode_is_simply_done(self, db, config):
        config.feeds[0].llm_trim = False
        make_episode(db, config)
        await run_pass(config, db)
        r = row(db)
        assert r["status"] == "done" and r["llm_cuts_applied"] == 2   # llm fields are not touched

    async def test_a_changed_file_extension_moves_the_episode_to_the_new_file(self, db, config):
        old = make_episode(db, config, ext=".mp3")
        await run_pass(config, db, url="https://cdn.example.com/ep.m4a")
        new = row(db)["file_path"]
        assert new.endswith(".m4a") and open(new, "rb").read() == b"NEW-AUDIO"
        assert not os.path.exists(old) and not os.path.exists(original_path(old))

    @pytest.mark.parametrize("kwargs,expected", [
        ({"episodes": []}, "no longer lists this episode"),
        ({"valid": False}, "corrupt"),
        ({"duration": 20}, "length filter"),
        ({"download_error": httpx.ConnectError("no route")}, "The download failed"),
    ])
    async def test_a_failed_attempt_leaves_the_existing_copy_exactly_as_it_was(self, db, config, kwargs, expected):
        path = make_episode(db, config)
        await run_pass(config, db, **kwargs)
        r = row(db)
        assert expected in r["error"] and "left as it was" in r["error"]
        assert (r["status"], r["redownload_requested"], r["llm_cuts_applied"]) == ("done", 0, 2)
        assert open(path, "rb").read() == b"OLD-CLEANED"
        assert open(original_path(path), "rb").read() == b"OLD-ORIGINAL"
        assert os.path.exists(transcript_path(config.storage.download_dir, "pod", GUID))
        assert not os.path.exists(staging_path(path))                      # no half-written file left behind

    async def test_a_file_that_cannot_be_replaced_is_left_alone(self, db, config):
        path = make_episode(db, config)
        with patch("siphon.pipeline.os.replace", side_effect=PermissionError("in use")):
            await run_pass(config, db)
        r = row(db)
        assert "could not replace" in r["error"] and r["status"] == "done"
        assert open(path, "rb").read() == b"OLD-CLEANED"
        assert not os.path.exists(staging_path(path))

    async def test_youtube_episodes_are_refused(self, db, config):
        db.insert_episode(video_id="v1", feed_name="yt", title="V", status="done")
        db.update_episode_status("v1", "yt", "done", redownload_requested=1)
        await run_pass(config, db)
        r = db.get_episode("v1", "yt")
        assert "only available for podcast feeds" in r["error"] and r["redownload_requested"] == 0

    async def test_an_episode_whose_feed_was_removed_is_refused(self, db, config):
        make_episode(db, config)
        config.feeds.pop(0)
        await run_pass(config, db)
        assert "no longer in the config" in row(db)["error"]

    async def test_unflagged_episodes_are_ignored(self, db, config):
        path = make_episode(db, config)
        db.update_episode_status(GUID, "pod", "done", redownload_requested=0)
        await run_pass(config, db)
        assert open(path, "rb").read() == b"OLD-CLEANED" and row(db)["status"] == "done"

    async def test_the_download_worker_does_it_as_part_of_its_run(self, db, config):
        path = make_episode(db, config)
        patches = downloads()
        for p in patches:
            p.start()
        try:
            await process_downloads(config, db)
        finally:
            for p in patches:
                p.stop()
        assert open(path, "rb").read() == b"NEW-AUDIO" and row(db)["status"] == "pending_whisper"

    async def test_staging_files_survive_the_temp_file_cleanup(self, db, config):
        from siphon.pipeline import _cleanup_temp_files
        path = make_episode(db, config)
        open(staging_path(path), "wb").write(b"in progress")
        _cleanup_temp_files(config)
        assert os.path.exists(staging_path(path))


# ------------------------------------------------------------------ rules
class TestRules:
    @pytest.mark.parametrize("status", ["done", "filtered", "pruned", "failed"])
    def test_allowed_statuses(self, db, config, status):
        make_episode(db, config, status=status, files=False)
        db.update_episode_status(GUID, "pod", status, redownload_requested=0)
        assert check_reprocess(config, row(db), "redownload") is None

    @pytest.mark.parametrize("status", ["pending", "eligible", "downloading", "pending_whisper", "pending_claude"])
    def test_refused_while_in_the_pipeline(self, db, config, status):
        make_episode(db, config, status=status, files=False)
        db.update_episode_status(GUID, "pod", status, redownload_requested=0)
        assert "still being processed" in check_reprocess(config, row(db), "redownload")

    def test_refused_when_already_queued(self, db, config):
        make_episode(db, config)                       # make_episode leaves redownload_requested=1
        assert "already waiting" in check_reprocess(config, row(db), "redownload")

    def test_refused_for_youtube_and_for_unknown_feeds(self, db, config):
        db.insert_episode(video_id="v", feed_name="yt", title="V", status="done")
        assert "only available for podcast" in check_reprocess(config, db.get_episode("v", "yt"), "redownload")
        config.feeds.clear()
        assert "no longer in the config" in check_reprocess(config, db.get_episode("v", "yt"), "redownload")

    def test_queueing_a_done_episode_touches_nothing_else(self, db, config):
        path = make_episode(db, config)
        db.update_episode_status(GUID, "pod", "done", redownload_requested=0)
        message = reprocess_episode(config, db, row(db), "redownload")
        assert "downloaded again" in message
        assert (row(db)["status"], row(db)["redownload_requested"]) == ("done", 1)
        assert open(path, "rb").read() == b"OLD-CLEANED" and os.path.exists(original_path(path))

    def test_queueing_a_filtered_episode_puts_it_in_the_download_queue(self, db, config):
        make_episode(db, config, status="filtered", files=False)
        db.update_episode_status(GUID, "pod", "filtered", redownload_requested=0, filter_reason="too_short")
        reprocess_episode(config, db, row(db), "redownload")
        assert row(db)["status"] == "eligible" and row(db)["filter_reason"] is None

    def test_refusal_raises_with_the_reason(self, db, config):
        make_episode(db, config)
        with pytest.raises(ReprocessError, match="already waiting"):
            reprocess_episode(config, db, row(db), "redownload")

    def test_detect_on_a_filtered_episode_points_to_download_and_process(self, db, config):
        make_episode(db, config, status="filtered", files=False)
        problem = check_reprocess(config, row(db), "detect")
        assert "is filtered" in problem and "Download and process" in problem
