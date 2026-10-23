"""Tests for siphon.pipeline – feed checking, downloading, and pruning."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

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
from siphon.db import Database
from siphon.pipeline import _prune_disk, check_feeds, process_downloads


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def db(tmp_path):
    """In-memory database with a pre-registered feed."""
    d = Database(":memory:")
    d.upsert_feed("test-feed", "https://www.youtube.com/@TestChannel", "youtube")
    yield d
    d.close()


@pytest.fixture()
def download_dir(tmp_path):
    """Temporary download directory."""
    d = tmp_path / "media"
    d.mkdir()
    return str(d)


@pytest.fixture()
def config(download_dir):
    """Minimal SiphonConfig for testing."""
    return SiphonConfig(
        server=ServerConfig(host="127.0.0.1", port=8080, base_url="http://localhost"),
        auth=AuthConfig(username="user", password="pass"),
        youtube=YouTubeConfig(api_key="test-key"),
        storage=StorageConfig(
            download_dir=download_dir,
            database=":memory:",
            max_disk_gb=100,
            youtube_keep_per_feed=50,
            podcast_keep_per_feed=200,
        ),
        schedule=ScheduleConfig(
            youtube_feeds_per_check=10,
            podcast_feeds_per_check=30,
            youtube_download_workers=2,
            youtube_download_delay_seconds=0,
            youtube_max_downloads_per_hour=100,
            podcast_download_workers=10,
            podcast_download_delay_seconds=0,
            podcast_max_downloads_per_hour=100,
        ),
        cookies=CookiesConfig(source="browser", browser="firefox"),
        defaults=FeedDefaults(
            sponsorblock_delay_minutes=60,
            block_shorts=True,
            min_duration_seconds=60,
        ),
        feeds=[
            FeedConfig(name="test-feed", url="https://www.youtube.com/@TestChannel"),
        ],
    )


def _sample_entries():
    """Return a list of sample yt-dlp flat-playlist entries."""
    return [
        {
            "id": "vid001",
            "title": "Great Video",
            "url": "https://www.youtube.com/watch?v=vid001",
            "duration": 600,
            "upload_date": "20250601",
            "description": "A great video",
            "channel": "TestChannel",
            "thumbnail": "https://img.youtube.com/vi/vid001/0.jpg",
        },
        {
            "id": "vid002",
            "title": "Another Video",
            "url": "https://www.youtube.com/watch?v=vid002",
            "duration": 900,
            "upload_date": "20250602",
            "description": "Another video",
            "channel": "TestChannel",
            "thumbnail": "https://img.youtube.com/vi/vid002/0.jpg",
        },
    ]


# ------------------------------------------------------------------ #
# check_feeds
# ------------------------------------------------------------------ #


class TestCheckFeeds:
    @patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST")
    @patch("siphon.youtube.get_channel_metadata", return_value={})
    @patch("siphon.youtube.list_videos")
    async def test_check_feeds_discovers_new_episodes(
        self, mock_list, mock_meta, mock_resolve, config, db
    ):
        mock_list.return_value = _sample_entries()

        await check_feeds(config, db)

        ep1 = db.get_episode("vid001", "test-feed")
        ep2 = db.get_episode("vid002", "test-feed")

        assert ep1 is not None
        assert ep1["title"] == "Great Video"
        assert ep1["status"] == "pending"

        assert ep2 is not None
        assert ep2["title"] == "Another Video"
        assert ep2["status"] == "pending"

    @patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST")
    @patch("siphon.youtube.get_channel_metadata", return_value={})
    @patch("siphon.youtube.list_videos")
    async def test_check_feeds_filters_shorts(self, mock_list, mock_meta, mock_resolve, config, db):
        entries = [
            {
                "id": "short01",
                "title": "Quick Clip",
                "url": "https://www.youtube.com/shorts/short01",
                "duration": 30,
                "upload_date": "20250601",
                "description": "A short",
                "channel": "TestChannel",
                "thumbnail": "https://img.youtube.com/vi/short01/0.jpg",
            },
        ]
        mock_list.return_value = entries

        await check_feeds(config, db)

        ep = db.get_episode("short01", "test-feed")
        assert ep is not None
        assert ep["status"] == "filtered"

    @patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST")
    @patch("siphon.youtube.get_channel_metadata", return_value={})
    @patch("siphon.youtube.list_videos")
    async def test_check_feeds_skips_existing(self, mock_list, mock_meta, mock_resolve, config, db):
        # Pre-insert the episode
        db.insert_episode(
            video_id="vid001",
            feed_name="test-feed",
            title="Already Here",
            status="done",
        )

        mock_list.return_value = _sample_entries()

        await check_feeds(config, db)

        ep = db.get_episode("vid001", "test-feed")
        # Should still be the original – not overwritten
        assert ep["title"] == "Already Here"
        assert ep["status"] == "done"

    @patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST")
    @patch("siphon.youtube.get_channel_metadata", return_value={})
    @patch("siphon.youtube.list_videos")
    async def test_check_feeds_sets_eligible_at_old_video(self, mock_list, mock_meta, mock_resolve, config, db):
        """Old videos (publish + delay already passed) should be eligible immediately."""
        mock_list.return_value = [_sample_entries()[0]]  # upload_date="20250601"

        before = datetime.now(timezone.utc)
        await check_feeds(config, db)
        after = datetime.now(timezone.utc)

        ep = db.get_episode("vid001", "test-feed")
        assert ep is not None
        assert ep["eligible_at"] is not None

        eligible_at = datetime.fromisoformat(ep["eligible_at"]).replace(tzinfo=timezone.utc)
        # Old video: publish + delay is in the past, so eligible_at ≈ now
        assert eligible_at >= before - timedelta(seconds=5)
        assert eligible_at <= after + timedelta(seconds=5)

    @patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST")
    @patch("siphon.youtube.get_channel_metadata", return_value={})
    @patch("siphon.youtube.list_videos")
    async def test_check_feeds_sets_eligible_at_fresh_video(self, mock_list, mock_meta, mock_resolve, config, db):
        """Fresh videos should be eligible at publish_date + sponsorblock_delay."""
        # Use a large delay so publish_date + delay is guaranteed to be in the future
        config.defaults.sponsorblock_delay_minutes = 99999
        now = datetime.now(timezone.utc)
        published_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        today = now.strftime("%Y%m%d")
        entry = {
            **_sample_entries()[0],
            "id": "vid_fresh",
            "upload_date": today,
            "published_at": published_at,
        }
        mock_list.return_value = [entry]

        await check_feeds(config, db)

        ep = db.get_episode("vid_fresh", "test-feed")
        assert ep is not None
        assert ep["eligible_at"] is not None

        eligible_at = datetime.fromisoformat(ep["eligible_at"]).replace(tzinfo=timezone.utc)
        delay = timedelta(minutes=config.defaults.sponsorblock_delay_minutes)
        # eligible_at should be approximately published_at + delay (using full timestamp)
        assert eligible_at >= now + delay - timedelta(seconds=5)
        assert eligible_at <= now + delay + timedelta(seconds=5)

    @patch("siphon.youtube.resolve_channel_id", return_value="UC_TEST")
    @patch("siphon.youtube.get_channel_metadata", return_value={})
    @patch("siphon.youtube.list_videos")
    async def test_check_feeds_handles_error(self, mock_list, mock_meta, mock_resolve, config, db):
        mock_list.side_effect = Exception("network timeout")

        await check_feeds(config, db)

        feed = db.get_feed("test-feed")
        assert feed["last_error"] == "network timeout"
        assert feed["last_checked_at"] is not None


class TestReevaluateConfigFiltered:
    """Episodes filtered under an old config get re-checked on feed check."""

    @pytest.fixture()
    def podcast_db(self, db):
        db.upsert_feed("pod", "https://example.com/rss", "podcast")
        return db

    def _rss(self, *eps):
        return {"title": "Pod", "image_url": None, "episodes": [
            {"guid": g, "title": t, "audio_url": f"https://example.com/{g}.mp3",
             "pub_date": d, "duration": 1800, "description": "", "thumbnail_url": None}
            for g, t, d in eps
        ]}

    @patch("siphon.podcast.fetch_podcast_rss", return_value=b"")
    @patch("siphon.podcast.parse_podcast_feed")
    async def test_earlier_date_cutoff_unfilters_old_podcast_episodes(
        self, mock_parse, mock_fetch, config, podcast_db
    ):
        from siphon.config import resolve_feed
        from siphon.pipeline import _check_podcast_feed

        mock_parse.return_value = self._rss(
            ("new", "Newest", "20261023"),
            ("old", "Older", "20261016"),
            ("ancient", "Ancient", "20251201"),
        )
        fc = FeedConfig(name="pod", url="https://example.com/rss", type="podcast", date_cutoff="20261020")
        await _check_podcast_feed(resolve_feed(fc, config.defaults), podcast_db)
        assert podcast_db.get_episode("new", "pod")["status"] == "pending"
        assert podcast_db.get_episode("old", "pod")["status"] == "filtered"
        assert podcast_db.get_episode("old", "pod")["filter_reason"] == "too_old"

        # User moves the start date back to 1/1/26
        fc = FeedConfig(name="pod", url="https://example.com/rss", type="podcast", date_cutoff="1/1/26")
        await _check_podcast_feed(resolve_feed(fc, config.defaults), podcast_db)

        old = podcast_db.get_episode("old", "pod")
        assert old["status"] == "pending"
        assert old["filter_reason"] is None
        assert old["eligible_at"] is not None
        assert podcast_db.get_episode("ancient", "pod")["status"] == "filtered"

    def test_title_match_cleared_when_exclude_removed(self, config, db):
        from siphon.config import resolve_feed
        from siphon.pipeline import _reevaluate_config_filtered

        db.insert_episode(video_id="v1", feed_name="test-feed", title="Bonus: Q&A",
                          duration=600, upload_date="20260301", status="filtered",
                          filter_reason="title_match:bonus")
        resolved = resolve_feed(config.feeds[0], config.defaults)
        assert _reevaluate_config_filtered(resolved, db) == 1
        assert db.get_episode("v1", "test-feed")["status"] == "pending"

    def test_non_config_reasons_untouched(self, config, db):
        from siphon.config import resolve_feed
        from siphon.pipeline import _reevaluate_config_filtered

        for vid, reason in (("s1", "short"), ("s2", "duration_unknown"), ("s3", "stale_livestream")):
            db.insert_episode(video_id=vid, feed_name="test-feed", title="x", duration=600,
                              upload_date="20260301", status="filtered", filter_reason=reason)
        resolved = resolve_feed(config.feeds[0], config.defaults)
        assert _reevaluate_config_filtered(resolved, db) == 0
        for vid in ("s1", "s2", "s3"):
            assert db.get_episode(vid, "test-feed")["status"] == "filtered"

    def test_still_filtered_reason_updated(self, config, db):
        from siphon.config import resolve_feed
        from siphon.pipeline import _reevaluate_config_filtered

        config.feeds[0] = FeedConfig(name="test-feed", url="https://www.youtube.com/@TestChannel",
                                     min_duration_seconds=900)
        db.insert_episode(video_id="v1", feed_name="test-feed", title="x", duration=600,
                          upload_date="20250101", status="filtered", filter_reason="too_old")
        resolved = resolve_feed(config.feeds[0], config.defaults)
        assert _reevaluate_config_filtered(resolved, db) == 0
        ep = db.get_episode("v1", "test-feed")
        assert ep["status"] == "filtered"
        assert ep["filter_reason"] == "too_short"


# ------------------------------------------------------------------ #
# process_downloads
# ------------------------------------------------------------------ #


class TestProcessDownloads:
    @patch("siphon.cutter.has_real_video_stream", return_value=True)
    @patch("siphon.cutter.validate_file", return_value=True)
    @patch("siphon.pipeline.find_downloaded_file")
    @patch("siphon.pipeline.download_video")
    async def test_process_downloads_downloads_eligible(
        self, mock_download, mock_find, mock_validate, mock_has_video, config, db, download_dir
    ):
        # Insert an eligible episode
        db.insert_episode(
            video_id="vid001",
            feed_name="test-feed",
            title="Great Video",
            status="eligible",
            eligible_at="2020-01-01T00:00:00",
        )

        mock_download.return_value = {}
        mock_find.return_value = (
            os.path.join(download_dir, "test-feed", "vid001.mp4"),
            50_000_000,
        )

        await process_downloads(config, db)

        ep = db.get_episode("vid001", "test-feed")
        assert ep["status"] == "done"
        assert ep["file_size"] == 50_000_000
        assert ep["mime_type"] == "video/mp4"
        assert "vid001.mp4" in ep["file_path"]

    @patch("siphon.pipeline.find_downloaded_file")
    @patch("siphon.pipeline.download_video")
    async def test_process_downloads_skips_when_hourly_cap_reached(
        self, mock_download, mock_find, config, db, download_dir
    ):
        config.schedule.youtube_max_downloads_per_hour = 0  # cap at zero

        db.insert_episode(
            video_id="vid001",
            feed_name="test-feed",
            title="Great Video",
            status="eligible",
            eligible_at="2020-01-01T00:00:00",
        )

        await process_downloads(config, db)

        # Episode should remain eligible — download was skipped
        ep = db.get_episode("vid001", "test-feed")
        assert ep["status"] == "eligible"
        mock_download.assert_not_called()

    @patch("siphon.pipeline.find_downloaded_file")
    @patch("siphon.pipeline.download_video")
    async def test_process_downloads_handles_download_failure(
        self, mock_download, mock_find, config, db
    ):
        db.insert_episode(
            video_id="vid001",
            feed_name="test-feed",
            title="Great Video",
            status="eligible",
            eligible_at="2020-01-01T00:00:00",
        )

        mock_download.side_effect = Exception("yt-dlp crashed")

        await process_downloads(config, db)

        ep = db.get_episode("vid001", "test-feed")
        assert ep["status"] == "failed"
        assert ep["error"] == "yt-dlp crashed"
        assert ep["retry_count"] == 1


# ------------------------------------------------------------------ #
# _prune_disk
# ------------------------------------------------------------------ #


class TestPruneDisk:
    async def test_prune_disk_removes_old_files(self, config, db, tmp_path):
        # Override youtube_keep_per_feed to a small number
        config.storage.youtube_keep_per_feed = 2

        # Create 4 "done" episodes with files
        feed_dir = tmp_path / "media" / "test-feed"
        feed_dir.mkdir(parents=True, exist_ok=True)

        for i in range(4):
            vid = f"vid{i:03d}"
            fpath = str(feed_dir / f"{vid}.mp4")
            with open(fpath, "wb") as f:
                f.write(b"x" * 1000)

            db.insert_episode(
                video_id=vid,
                feed_name="test-feed",
                title=f"Episode {i}",
                upload_date=f"2025-0{i + 1}-01",
                status="done",
            )
            db.update_episode_status(
                vid,
                "test-feed",
                "done",
                file_path=fpath,
                file_size=1000,
            )

        assert db.get_feed_episode_count("test-feed") == 4

        await _prune_disk(config, db)

        # After pruning, only youtube_keep_per_feed (2) should remain as 'done'
        assert db.get_feed_episode_count("test-feed") == 2

    async def test_prune_disk_global_limit(self, config, db, tmp_path):
        # Set a very small global limit
        config.storage.max_disk_gb = 0  # 0 bytes allowed

        feed_dir = tmp_path / "media" / "test-feed"
        feed_dir.mkdir(parents=True, exist_ok=True)

        fpath = str(feed_dir / "vid001.mp4")
        with open(fpath, "wb") as f:
            f.write(b"x" * 1000)

        db.insert_episode(
            video_id="vid001",
            feed_name="test-feed",
            title="Episode 1",
            upload_date="2025-01-01",
            status="done",
        )
        db.update_episode_status(
            "vid001",
            "test-feed",
            "done",
            file_path=fpath,
            file_size=1000,
        )

        assert db.get_disk_usage() == 1000

        await _prune_disk(config, db)

        ep = db.get_episode("vid001", "test-feed")
        assert ep["status"] == "pruned"

    async def test_prune_disk_handles_missing_file(self, config, db):
        """Pruning should not crash when the file is already gone."""
        config.storage.max_disk_gb = 0

        db.insert_episode(
            video_id="vid001",
            feed_name="test-feed",
            title="Episode 1",
            upload_date="2025-01-01",
            status="done",
        )
        db.update_episode_status(
            "vid001",
            "test-feed",
            "done",
            file_path="/nonexistent/vid001.mp4",
            file_size=1000,
        )

        # Should not raise
        await _prune_disk(config, db)

        ep = db.get_episode("vid001", "test-feed")
        assert ep["status"] == "pruned"
