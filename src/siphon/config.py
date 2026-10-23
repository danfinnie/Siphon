"""Configuration models for Siphon, validated with Pydantic v2."""

from __future__ import annotations

import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, field_validator

# Quality type: an integer (e.g. 1080, 1440) or the literal "max"
Quality = Union[int, Literal["max"]]

DEFAULT_AD_PROMPT = textwrap.dedent("""\
    You are analyzing a podcast/video transcript to identify non-content segments
    that should be removed for personal use. Identify any of the following:
    - Paid sponsor reads or product promotions
    - Patreon, membership, or donation pitches
    - Merchandise promotion
    - Social media follow requests
    - Newsletter or mailing list signups
    - App review/subscribe/like requests
    - Live show or event ticket promotions
    - Cross-promotion of other shows or networks
    - Detailed credits or listing of all the individual contributors
    - The host's own book, course, or consulting pitch

    The transcript is provided in two formats:
    1. SEGMENTS — coarse time-stamped chunks for understanding context and flow
    2. WORD TIMESTAMPS — per-word timing for precise cut points

    Use the SEGMENTS to identify which parts of the content are ads vs real content.
    Then use the WORD TIMESTAMPS to pick the exact start and end times for each cut.
    Start each cut at the first word of the ad segment. End each cut at the last word
    before real content resumes. Be precise — do not clip real content.

    Return ONLY JSON. No explanation, no markdown fences. Format:
    {
      "segments": [
        {
          "start": 0.0,
          "end": 0.0,
          "type": "ad",
          "label": "brief human-readable label",
          "confidence": 0.0
        }
      ]
    }
    Confidence is 0.0-1.0. Only include segments with confidence >= 0.5.
    If no ad segments found, return {"segments": []}.
""")


class ServerConfig(BaseModel):
    host: str
    port: int
    base_url: str
    media_base_url: str = ""
    timezone: str = "America/Los_Angeles"


class AuthConfig(BaseModel):
    username: str
    password: str


class StorageConfig(BaseModel):
    download_dir: str
    database: str
    max_disk_gb: int = 100
    youtube_keep_per_feed: int = 50
    podcast_keep_per_feed: int = 200


class ScheduleConfig(BaseModel):
    check_interval_minutes: int = 30
    youtube_feeds_per_check: int = 10
    podcast_feeds_per_check: int = 30

    # YouTube — gentle on a single platform
    youtube_download_interval_minutes: int = 5
    youtube_download_workers: int = 2
    youtube_download_delay_seconds: int = 120
    youtube_max_downloads_per_hour: int = 10

    # Podcast — many different hosts, small files
    podcast_download_interval_minutes: int = 5
    podcast_download_workers: int = 10
    podcast_download_delay_seconds: int = 2
    podcast_max_downloads_per_hour: int = 120


class YouTubeConfig(BaseModel):
    api_key: str
    quota_cooldown_hours: int = 4
    country: str = "US"
    # yt-dlp player clients for cookie-authenticated downloads.  None uses
    # Siphon's built-in default (siphon.downloader.DEFAULT_PLAYER_CLIENTS);
    # ["default"] hands the choice back to yt-dlp.
    player_clients: list[str] | None = None


class PocketCastsConfig(BaseModel):
    email: str = ""
    password: str = ""
    auto_prune: bool = False
    feeds_per_check: int = 5
    auto_prune_interval_hours: int = 24


class CookiesConfig(BaseModel):
    source: str = "browser"
    browser: str = "firefox"


class LLMConfig(BaseModel):
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_workers: int = 1
    whisper_word_timestamps: bool = True
    word_timestamps_max_minutes: int = 45
    claude_model: str = "claude-sonnet-4-6"
    claude_effort: str = "medium"
    default_ad_prompt: str = DEFAULT_AD_PROMPT
    confidence_threshold: float = 0.75
    min_segment_duration: int = 7
    max_segment_duration: int = 300
    claude_concurrency: int = 3


def normalize_date_cutoff(v: object) -> str | None:
    """Coerce a date cutoff to ``YYYYMMDD``.

    Filters compare cutoffs to ``YYYYMMDD`` upload dates as strings, so any
    other format (``2026-01-01``, ``1/1/26``, a YAML date) silently breaks
    the comparison. Unrecognized strings are passed through unchanged.
    """
    if v is None or v == "":
        return None
    if hasattr(v, "strftime"):  # YAML parses bare 2026-01-01 as a date
        return v.strftime("%Y%m%d")
    s = str(v).strip()
    if re.fullmatch(r"\d{8}", s):
        return s
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return s


class FeedDefaults(BaseModel):
    mode: Literal["video", "audio"] = "video"
    quality: Quality = 1440
    sponsorblock: bool = True
    sponsorblock_categories: list[str] = [
        "sponsor",
        "selfpromo",
        "interaction",
        "intro",
        "outro",
    ]
    sponsorblock_delay_minutes: int = 1440
    block_shorts: bool = True
    min_duration_seconds: int = 60
    date_cutoff: str | None = None
    title_exclude: list[str] = []
    llm_trim: bool = False

    @field_validator("date_cutoff", mode="before")
    @classmethod
    def _validate_date_cutoff(cls, v: object) -> str | None:
        return normalize_date_cutoff(v)

    @field_validator("quality", mode="before")
    @classmethod
    def _validate_quality(cls, v: object) -> Quality:
        if isinstance(v, str):
            if v.lower() == "max":
                return "max"
            # Allow numeric strings like "1080"
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"quality must be an integer or 'max', got {v!r}")
        return v  # type: ignore[return-value]


class FeedConfig(BaseModel):
    """Per-feed configuration. All FeedDefaults fields are optional so that
    omitted values inherit from the top-level defaults."""

    name: str
    url: str

    @field_validator("url", mode="before")
    @classmethod
    def _strip_url(cls, v: object) -> str:
        return v.strip() if isinstance(v, str) else v  # type: ignore[return-value]
    type: Literal["youtube", "podcast"] = "youtube"

    mode: Optional[Literal["video", "audio"]] = None
    quality: Optional[Quality] = None
    sponsorblock: Optional[bool] = None
    sponsorblock_categories: Optional[list[str]] = None
    sponsorblock_delay_minutes: Optional[int] = None
    block_shorts: Optional[bool] = None
    min_duration_seconds: Optional[int] = None
    date_cutoff: Optional[str] = None
    title_exclude: Optional[list[str]] = None
    llm_trim: Optional[bool] = None

    # Per-feed prompt customization (not in FeedDefaults)
    claude_prompt_extra: Optional[str] = None
    claude_prompt_override: Optional[str] = None

    # Display name for RSS/UI (not in FeedDefaults)
    display_name: Optional[str] = None

    # Pocket Casts private URL (not in FeedDefaults)
    pc_url: Optional[str] = None

    @field_validator("date_cutoff", mode="before")
    @classmethod
    def _validate_date_cutoff(cls, v: object) -> str | None:
        return normalize_date_cutoff(v)

    @field_validator("quality", mode="before")
    @classmethod
    def _validate_quality(cls, v: object) -> Quality | None:
        if v is None:
            return None
        if isinstance(v, str):
            if v.lower() == "max":
                return "max"
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"quality must be an integer or 'max', got {v!r}")
        return v  # type: ignore[return-value]


class ResolvedFeed(BaseModel):
    """A feed with all defaults resolved — no Optional fields."""

    name: str
    url: str
    type: Literal["youtube", "podcast"]
    mode: Literal["video", "audio"]
    quality: Quality
    sponsorblock: bool
    sponsorblock_categories: list[str]
    sponsorblock_delay_minutes: int
    block_shorts: bool
    min_duration_seconds: int
    date_cutoff: str | None
    title_exclude: list[str]
    llm_trim: bool

    # Per-feed prompt customization (no default fallback)
    claude_prompt_extra: str | None = None
    claude_prompt_override: str | None = None

    # Display name for RSS/UI (no default fallback)
    display_name: str | None = None

    # Pocket Casts private URL (no default fallback)
    pc_url: str | None = None


class SiphonConfig(BaseModel):
    server: ServerConfig
    auth: AuthConfig
    storage: StorageConfig
    youtube: YouTubeConfig
    schedule: ScheduleConfig = ScheduleConfig()
    cookies: CookiesConfig = CookiesConfig()
    pocketcasts: PocketCastsConfig = PocketCastsConfig()
    defaults: FeedDefaults = FeedDefaults()
    llm: LLMConfig = LLMConfig()
    feeds: list[FeedConfig]


def resolve_feed(feed: FeedConfig, defaults: FeedDefaults) -> ResolvedFeed:
    """Merge a FeedConfig onto FeedDefaults.

    For every field defined in FeedDefaults, use the feed-level value when it
    is not None; otherwise fall back to the corresponding default value.
    """
    merged: dict = {"name": feed.name, "url": feed.url, "type": feed.type}
    for field_name in FeedDefaults.model_fields:
        feed_value = getattr(feed, field_name)
        if feed_value is not None:
            merged[field_name] = feed_value
        else:
            merged[field_name] = getattr(defaults, field_name)
    # Per-feed-only fields (not in FeedDefaults)
    merged["claude_prompt_extra"] = feed.claude_prompt_extra
    merged["claude_prompt_override"] = feed.claude_prompt_override
    merged["display_name"] = feed.display_name
    merged["pc_url"] = feed.pc_url
    return ResolvedFeed(**merged)


def load_config(path: str) -> SiphonConfig:
    """Read a YAML file and return a validated SiphonConfig."""
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    cfg = SiphonConfig(**data)
    # Store the source path so the UI can save back to it
    cfg._config_path = str(Path(path).resolve())  # type: ignore[attr-defined]
    return cfg
