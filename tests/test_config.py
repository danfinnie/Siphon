"""Tests for siphon.config — loading, validation, and default merging."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from siphon.config import (
    FeedConfig,
    FeedDefaults,
    LLMConfig,
    ResolvedFeed,
    SiphonConfig,
    load_config,
    resolve_feed,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CONFIG = FIXTURES / "sample_config.yaml"


# ---------------------------------------------------------------------------
# Loading from the sample fixture
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_sample_config(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert isinstance(cfg, SiphonConfig)

    def test_server_fields(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.port == 8585
        assert cfg.server.base_url == "https://test.example.com"
        assert cfg.server.media_base_url == "http://test-media.example.com:8585"

    def test_auth_fields(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert cfg.auth.username == "testuser"
        assert cfg.auth.password == "testpass"

    def test_storage_fields(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert cfg.storage.download_dir == "./test_media"
        assert cfg.storage.database == ":memory:"
        assert cfg.storage.max_disk_gb == 10
        assert cfg.storage.youtube_keep_per_feed == 5
        assert cfg.storage.podcast_keep_per_feed == 20

    def test_schedule_fields(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert cfg.schedule.check_interval_minutes == 30
        assert cfg.schedule.youtube_feeds_per_check == 5
        assert cfg.schedule.podcast_feeds_per_check == 15
        assert cfg.schedule.youtube_download_workers == 1
        assert cfg.schedule.youtube_download_delay_seconds == 120
        assert cfg.schedule.youtube_max_downloads_per_hour == 10
        assert cfg.schedule.podcast_download_workers == 10
        assert cfg.schedule.podcast_download_delay_seconds == 2
        assert cfg.schedule.podcast_max_downloads_per_hour == 120

    def test_cookies_fields(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert cfg.cookies.source == "browser"
        assert cfg.cookies.browser == "firefox"

    def test_defaults_fields(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert cfg.defaults.mode == "video"
        assert cfg.defaults.quality == 1080
        assert cfg.defaults.sponsorblock is True
        assert cfg.defaults.sponsorblock_categories == ["sponsor", "selfpromo"]
        assert cfg.defaults.sponsorblock_delay_minutes == 0
        assert cfg.defaults.block_shorts is True
        assert cfg.defaults.min_duration_seconds == 60
        assert cfg.defaults.date_cutoff is None
        assert cfg.defaults.title_exclude == []
        assert cfg.defaults.llm_trim is False

    def test_llm_defaults(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert cfg.llm.whisper_model == "base"
        assert cfg.llm.whisper_device == "cpu"
        assert cfg.llm.claude_model == "claude-sonnet-4-6"
        assert cfg.llm.confidence_threshold == 0.75
        assert cfg.llm.min_segment_duration == 7
        assert cfg.llm.max_segment_duration == 300

    def test_feeds_count(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        assert len(cfg.feeds) == 3

    def test_feed_without_overrides(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        feed = cfg.feeds[0]
        assert feed.name == "test-channel"
        assert feed.url == "https://www.youtube.com/@TestChannel"
        assert feed.type == "youtube"
        # All override fields should be None
        for field_name in FeedDefaults.model_fields:
            assert getattr(feed, field_name) is None

    def test_feed_with_overrides(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        feed = cfg.feeds[1]
        assert feed.name == "test-channel-2"
        assert feed.sponsorblock is False
        assert feed.title_exclude == ["#shorts", "live"]

    def test_podcast_feed(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        feed = cfg.feeds[2]
        assert feed.name == "test-podcast"
        assert feed.type == "podcast"
        assert feed.mode == "audio"
        assert feed.llm_trim is True
        assert feed.claude_prompt_extra == "Also remove Discord promotions."


# ---------------------------------------------------------------------------
# Quality field — int or "max"
# ---------------------------------------------------------------------------


class TestQuality:
    def test_quality_int(self):
        d = FeedDefaults(quality=1080)
        assert d.quality == 1080

    def test_quality_max_string(self):
        d = FeedDefaults(quality="max")
        assert d.quality == "max"

    def test_quality_max_case_insensitive(self):
        d = FeedDefaults(quality="MAX")
        assert d.quality == "max"

    def test_quality_numeric_string(self):
        d = FeedDefaults(quality="720")
        assert d.quality == 720

    def test_quality_invalid_string(self):
        with pytest.raises(Exception):
            FeedDefaults(quality="potato")

    def test_feed_quality_max(self):
        feed = FeedConfig(name="f", url="http://x", quality="max")
        assert feed.quality == "max"

    def test_feed_quality_none(self):
        feed = FeedConfig(name="f", url="http://x")
        assert feed.quality is None

    def test_resolved_quality_max(self):
        defaults = FeedDefaults(quality="max")
        feed = FeedConfig(name="f", url="http://x")
        resolved = resolve_feed(feed, defaults)
        assert resolved.quality == "max"


# ---------------------------------------------------------------------------
# Default merging via resolve_feed
# ---------------------------------------------------------------------------


class TestResolveFeed:
    def test_feed_with_no_overrides_inherits_all_defaults(self):
        defaults = FeedDefaults(quality=1080)
        feed = FeedConfig(name="ch", url="https://example.com")
        resolved = resolve_feed(feed, defaults)

        assert isinstance(resolved, ResolvedFeed)
        assert resolved.name == "ch"
        assert resolved.url == "https://example.com"
        assert resolved.type == "youtube"
        assert resolved.quality == 1080
        assert resolved.mode == "video"
        assert resolved.sponsorblock is True
        assert resolved.sponsorblock_delay_minutes == 1440
        assert resolved.block_shorts is True
        assert resolved.min_duration_seconds == 60
        assert resolved.date_cutoff is None
        assert resolved.title_exclude == []
        assert resolved.llm_trim is False

    def test_feed_override_takes_precedence(self):
        defaults = FeedDefaults()
        feed = FeedConfig(
            name="music",
            url="https://example.com",
            mode="audio",
            quality=720,
            sponsorblock=False,
        )
        resolved = resolve_feed(feed, defaults)

        assert resolved.mode == "audio"
        assert resolved.quality == 720
        assert resolved.sponsorblock is False
        # Non-overridden fields still come from defaults
        assert resolved.block_shorts is True
        assert resolved.min_duration_seconds == 60

    def test_override_sponsorblock_categories(self):
        defaults = FeedDefaults()
        feed = FeedConfig(
            name="ch",
            url="https://example.com",
            sponsorblock_categories=["sponsor"],
        )
        resolved = resolve_feed(feed, defaults)
        assert resolved.sponsorblock_categories == ["sponsor"]

    def test_override_title_exclude(self):
        defaults = FeedDefaults()
        feed = FeedConfig(
            name="ch",
            url="https://example.com",
            title_exclude=["#shorts", "live"],
        )
        resolved = resolve_feed(feed, defaults)
        assert resolved.title_exclude == ["#shorts", "live"]

    def test_override_date_cutoff(self):
        defaults = FeedDefaults()
        feed = FeedConfig(
            name="ch",
            url="https://example.com",
            date_cutoff="20240601",
        )
        resolved = resolve_feed(feed, defaults)
        assert resolved.date_cutoff == "20240601"

    def test_resolve_from_loaded_config(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        # First feed: no overrides
        resolved = resolve_feed(cfg.feeds[0], cfg.defaults)
        assert resolved.quality == 1080
        assert resolved.sponsorblock is True
        assert resolved.title_exclude == []
        assert resolved.type == "youtube"

        # Second feed: has overrides
        resolved2 = resolve_feed(cfg.feeds[1], cfg.defaults)
        assert resolved2.sponsorblock is False
        assert resolved2.title_exclude == ["#shorts", "live"]
        # Inherits the rest from defaults
        assert resolved2.quality == 1080
        assert resolved2.mode == "video"

    def test_resolve_podcast_feed(self):
        cfg = load_config(str(SAMPLE_CONFIG))
        resolved = resolve_feed(cfg.feeds[2], cfg.defaults)
        assert resolved.type == "podcast"
        assert resolved.mode == "audio"
        assert resolved.llm_trim is True
        assert resolved.claude_prompt_extra == "Also remove Discord promotions."
        assert resolved.claude_prompt_override is None

    def test_all_fields_overridden(self):
        defaults = FeedDefaults()
        feed = FeedConfig(
            name="full",
            url="https://example.com",
            mode="audio",
            quality=480,
            sponsorblock=False,
            sponsorblock_categories=["sponsor"],
            sponsorblock_delay_minutes=0,
            block_shorts=False,
            min_duration_seconds=0,
            date_cutoff="20230101",
            title_exclude=["test"],
            llm_trim=True,
        )
        resolved = resolve_feed(feed, defaults)
        assert resolved.mode == "audio"
        assert resolved.quality == 480
        assert resolved.sponsorblock is False
        assert resolved.sponsorblock_categories == ["sponsor"]
        assert resolved.sponsorblock_delay_minutes == 0
        assert resolved.block_shorts is False
        assert resolved.min_duration_seconds == 0
        assert resolved.date_cutoff == "20230101"
        assert resolved.title_exclude == ["test"]
        assert resolved.llm_trim is True

    def test_prompt_fields_not_in_defaults(self):
        """claude_prompt_extra and claude_prompt_override are per-feed only."""
        defaults = FeedDefaults()
        feed = FeedConfig(
            name="ch",
            url="http://x",
            claude_prompt_extra="extra stuff",
            claude_prompt_override="full override",
        )
        resolved = resolve_feed(feed, defaults)
        assert resolved.claude_prompt_extra == "extra stuff"
        assert resolved.claude_prompt_override == "full override"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_server(self, tmp_path):
        data = {
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"name": "f", "url": "http://x"}],
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            load_config(str(p))

    def test_missing_auth(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"name": "f", "url": "http://x"}],
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            load_config(str(p))

    def test_missing_storage(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "feeds": [{"name": "f", "url": "http://x"}],
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            load_config(str(p))

    def test_missing_feeds(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            load_config(str(p))

    def test_feed_missing_name(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"url": "http://x"}],
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            load_config(str(p))

    def test_feed_missing_url(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"name": "f"}],
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            load_config(str(p))

    def test_invalid_mode(self):
        with pytest.raises(Exception):
            FeedDefaults(mode="invalid")

    def test_invalid_feed_mode(self):
        with pytest.raises(Exception):
            FeedConfig(name="f", url="http://x", mode="invalid")

    def test_invalid_feed_type(self):
        with pytest.raises(Exception):
            FeedConfig(name="f", url="http://x", type="invalid")


# ---------------------------------------------------------------------------
# Defaults for optional sections
# ---------------------------------------------------------------------------


class TestOptionalSectionDefaults:
    def test_schedule_defaults(self, tmp_path):
        """When schedule is omitted from YAML, ScheduleConfig defaults apply."""
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"name": "f", "url": "http://x"}],
        }
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(str(p))
        assert cfg.schedule.check_interval_minutes == 30
        assert cfg.schedule.youtube_download_workers == 2
        assert cfg.schedule.podcast_download_workers == 10

    def test_cookies_defaults(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"name": "f", "url": "http://x"}],
        }
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(str(p))
        assert cfg.cookies.source == "browser"
        assert cfg.cookies.browser == "firefox"

    def test_feed_defaults_section_defaults(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"name": "f", "url": "http://x"}],
        }
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(str(p))
        assert cfg.defaults.mode == "video"
        assert cfg.defaults.quality == 1440
        assert cfg.defaults.sponsorblock is True
        assert cfg.defaults.sponsorblock_categories == [
            "sponsor",
            "selfpromo",
            "interaction",
            "intro",
            "outro",
        ]
        assert cfg.defaults.llm_trim is False

    def test_llm_defaults(self, tmp_path):
        data = {
            "server": {"host": "0.0.0.0", "port": 8080, "base_url": "http://x"},
            "auth": {"username": "u", "password": "p"},
            "storage": {"download_dir": ".", "database": "db"}, "youtube": {"api_key": "test"}, "youtube": {"api_key": "test"},
            "feeds": [{"name": "f", "url": "http://x"}],
        }
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(str(p))
        assert isinstance(cfg.llm, LLMConfig)
        assert cfg.llm.whisper_model == "base"
        assert cfg.llm.confidence_threshold == 0.75


class TestMaxAgeDays:
    """max_age_days is a rolling date_cutoff that moves forward every day."""

    @staticmethod
    def _resolve(defaults=None, **feed_kwargs):
        feed = FeedConfig(name="show", url="https://example.com/rss", type="podcast", **feed_kwargs)
        return resolve_feed(feed, defaults or FeedDefaults())

    def test_off_by_default(self):
        assert self._resolve().date_cutoff is None
        assert self._resolve().max_age_days is None

    def test_becomes_a_date_cutoff(self):
        from freezegun import freeze_time
        with freeze_time("2026-10-10 12:00:00"):
            resolved = self._resolve(FeedDefaults(max_age_days=4))
        assert resolved.date_cutoff == "20261006"

    def test_cutoff_moves_forward_as_days_pass(self):
        from freezegun import freeze_time
        defaults = FeedDefaults(max_age_days=4)
        with freeze_time("2026-10-10"):
            first = self._resolve(defaults).date_cutoff
        with freeze_time("2026-10-15"):
            later = self._resolve(defaults).date_cutoff
        assert (first, later) == ("20261006", "20261011")

    def test_later_of_explicit_cutoff_and_age_wins(self):
        from freezegun import freeze_time
        with freeze_time("2026-10-10"):
            # explicit cutoff is more recent than the 4-day window
            assert self._resolve(FeedDefaults(max_age_days=4, date_cutoff="20261009")).date_cutoff == "20261009"
            # the 4-day window is more recent than the explicit cutoff
            assert self._resolve(FeedDefaults(max_age_days=4, date_cutoff="20260101")).date_cutoff == "20261006"

    def test_per_feed_value_overrides_default(self):
        from freezegun import freeze_time
        with freeze_time("2026-10-10"):
            resolved = self._resolve(FeedDefaults(max_age_days=4), max_age_days=1)
        assert resolved.date_cutoff == "20261009"


class TestQualityOptionDefaults:
    def test_new_llm_options_default_to_existing_behaviour(self):
        llm = LLMConfig()
        assert llm.whisper_vad_filter is False
        assert llm.snap_cuts_to_silence is False
        assert llm.snap_window_seconds == 3.0
