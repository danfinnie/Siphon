<p align="center">
  <img src="img/hero.png" alt="Siphon" width="600">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-380+-brightgreen?logo=pytest&logoColor=white" alt="Tests 380+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/platform-Windows-0078D4?logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/htmx-2.0-blue?logo=htmx&logoColor=white" alt="htmx 2.0">
</p>

# Siphon

Self-hosted podcast pipeline that downloads YouTube channels and podcast feeds, filters garbage (youtube shorts, minimum length, title filter), strips ads (or any content - intros, credits, etc.) using SponsorBlock + LLM analysis (Whisper + Claude), and serves clean RSS feeds to your podcast app over Tailscale. Configurable per feed.

Built for a very specific stack: **Windows, Tailscale Funnel, Pocket Casts, Claude Code (Max subscription), Firefox cookies for YouTube Premium, Youtube API key**. It works great for that. If your setup is different, expect to adapt.

<table>
<tr>
<td width="50%"><a href="img/dashboard.png"><img src="img/dashboard.png" alt="Dashboard"></a></td>
<td width="50%"><a href="img/feeds.png"><img src="img/feeds.png" alt="Feeds"></a></td>
</tr>
<tr>
<td align="center"><sub>Dashboard</sub></td>
<td align="center"><sub>Feeds</sub></td>
</tr>
</table>

## What it does

- **YouTube channels** &rarr; discovers videos via YouTube Data API v3, downloads via yt-dlp with your Firefox YouTube Premium session, applies SponsorBlock cuts, runs Whisper transcription + Claude ad detection, serves as a podcast feed
- **Podcast feeds** &rarr; downloads audio from RSS, runs Whisper + Claude to detect and cut sponsor reads/promos/self-promotion, serves a clean RSS feed
- **Three-queue pipeline** &rarr; Download &rarr; Whisper &rarr; Claude, each running independently with their own scheduling and concurrency
- **Web UI** at `localhost:8585/ui/` for feed management, OPML import, live activity monitoring, stats dashboard with insights
- **System tray icon** for pause/resume/quit with adjustable Whisper workers (runs at below-normal CPU priority)
- **Tailscale Funnel** for HTTPS RSS serving to Pocket Casts, media served over Tailnet only (no auth needed on your network)
- **Per-feed control** &rarr; toggle SponsorBlock, LLM trim, short blocking per feed; filter by title, duration, date; supplement or fully replace the Claude prompt per feed
- **Auto-prune** &rarr; episodes you finish in Pocket Casts are automatically cleaned up via the Pocket Casts API
- **Auto-ban** for vulnerability scanners (fail2ban-style IP blocking)

## Web UI

Available at `http://localhost:8585/ui/` (localhost only, no auth). Three themes: Light, Dark, Black. SPA-like navigation via htmx.

- **Dashboard** with system stats, lifetime metrics, and insights (most stale feeds, disk hogs, longest processing, cut stats)
- **Feed management** — add/edit/delete feeds with all per-feed config options, type-aware forms (YouTube vs podcast)
- **OPML import** for bulk podcast migration from Pocket Casts or other apps
- **Live activity log** in sticky footer with queue status, worker activity, and timing
- **Sort & filter** — bidirectional sort by name, date, LLM cuts, SB cuts, size, time saved; filter by YouTube, Podcast, LLM, SponsorBlock
- **Search** — instant feed search from the header
- **Per-feed stats** — In RSS count, queue depths, SB/LLM cut totals
- **Mark as Caught Up** — trims to 1 episode, sets date cutoff, cleans disk

## System Tray

- **Pause/Resume** — three-state (Running / Pending Pause / Paused) with graceful drain
- **Whisper workers** — adjustable (1-5, for live system use vs overnight processing)
- **Test YouTube Login** — verify Firefox cookie status
- **Open Config** — launch web UI

## Auto-Prune

Siphon integrates with the Pocket Casts API to automatically clean up episodes you've finished listening to or archived. Runs in the background — no manual maintenance needed.

- Episodes marked as **completed** or **archived** in Pocket Casts are pruned from disk
- Always keeps at least 1 episode per feed (Pocket Casts requires it for private feeds)
- Checks a few feeds per cycle with configurable interval (default: every 24 hours per feed)
- Requires Pocket Casts email/password login (no SSO)

## Setup

### Prerequisites

- Python 3.11+
- ffmpeg on PATH
- Deno 2.3+ on PATH (for yt-dlp's YouTube challenge solver)
- yt-dlp kept current (`pip install -U yt-dlp`) — YouTube-side changes routinely break releases more than a couple of months old
- [Tailscale](https://tailscale.com/) with Funnel enabled and MagicDNS + HTTPS certs
- [YouTube Data API v3](https://console.cloud.google.com/apis/api/youtube.googleapis.com) key
- Firefox with YouTube Premium logged in (for cookies)
- Claude Code CLI on PATH (Max subscription)
- NVIDIA GPU (optional, for CUDA-accelerated Whisper — requires `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`)

### Install

```bash
git clone https://github.com/cwilliams5/Siphon.git
cd Siphon
pip install -e .

# For CUDA Whisper acceleration (optional):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

### Configure

Copy `config.example.yaml` to your data directory (outside the repo — keep secrets out of git):

```bash
mkdir /path/to/siphon-data
cp config.example.yaml /path/to/siphon-data/config.yaml
```

Edit the config with your Tailscale hostname, auth credentials, YouTube API key, and feed list.

### Run

```bash
python -m siphon -c "/path/to/siphon-data/config.yaml"
```

Or create a batch file for windowless operation (Windows):

```batch
@echo off
set PATH=%PATH%;C:\Users\you\.deno\bin
cd /d "path\to\Siphon"
pythonw -m siphon -c "path\to\siphon-data\config.yaml"
```

Use `--verbose` flag for console output. Use `--no-tray` to disable the system tray icon.

### Tailscale Funnel

```bash
tailscale funnel --bg 8585
```

RSS feeds are served over HTTPS with Basic Auth. Media files are served over Tailnet only (no auth, requires Tailscale on your phone).

### Pocket Casts

1. Copy the RSS URL from the web UI (includes embedded auth credentials)
2. Submit at [pocketcasts.com/submit](https://pocketcasts.com/submit) as a private feed
3. Save the `pca.st/private/...` URL back in the feed's settings

### Tests

380+ tests covering config, DB, pipeline, filters, feed generation, ad detection, SponsorBlock, YouTube API, UI routes, and htmx integration.

```bash
python -m pytest tests/
```

## Key config options

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `youtube` | `api_key` | required | YouTube Data API v3 key |
| `youtube` | `quota_cooldown_hours` | `4` | Hours to pause after API 403 |
| `youtube` | `country` | `US` | ISO country code for filtering region-blocked videos |
| `server` | `timezone` | `America/Los_Angeles` | Timezone for activity log timestamps |
| `server` | `media_base_url` | `""` | Tailnet-internal URL for media files |
| `schedule` | `check_interval_minutes` | `30` | How often to check for new episodes |
| `schedule` | `youtube_max_downloads_per_hour` | `10` | YouTube download rate limit |
| `schedule` | `podcast_max_downloads_per_hour` | `120` | Podcast download rate limit |
| `pocketcasts` | `email` | `""` | Pocket Casts login email (for auto-prune) |
| `pocketcasts` | `password` | `""` | Pocket Casts login password |
| `pocketcasts` | `auto_prune` | `false` | Auto-prune episodes completed/archived in Pocket Casts |
| `pocketcasts` | `feeds_per_check` | `5` | Max feeds to check per cycle (20s delay between each) |
| `pocketcasts` | `auto_prune_interval_hours` | `24` | Hours between rechecking each feed |
| `defaults` | `sponsorblock_delay_minutes` | `1440` | Wait for SB segments to be crowdsourced |
| `defaults` | `llm_trim` | `false` | Enable Whisper + Claude ad detection |
| `llm` | `whisper_model` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `llm` | `whisper_device` | `cpu` | Whisper device (`cpu` or `cuda`) |
| `llm` | `whisper_workers` | `1` | Concurrent Whisper workers (CPU only, CUDA forced to 1) |
| `llm` | `claude_concurrency` | `3` | Parallel Claude CLI invocations |
| `llm` | `claude_model` | `claude-sonnet-4-6` | Claude model for ad detection |
| `llm` | `claude_effort` | `medium` | Claude thinking depth (low/medium/high/max) |
| `llm` | `word_timestamps_max_minutes` | `45` | Max episode length for word-level timestamps |
| `llm` | `confidence_threshold` | `0.75` | Minimum confidence to cut a detected segment |
| `llm` | `merge_gap_seconds` | `3.0` | Detected ads closer together than this are cut as one chunk (back-to-back ads are removed whole) |
| `llm` | `whisper_vad_filter` | `false` | Skip non-speech (music, jingles) when transcribing; timestamps stay aligned to the original audio |
| `llm` | `snap_cuts_to_silence` | `false` | Move each cut point to the nearest silence so cuts land in a pause |
| `llm` | `snap_window_seconds` | `3.0` | How far (seconds) a cut point may move to find a silence |
| `storage` | `max_disk_gb` | `1000` | Auto-prune oldest episodes when exceeded |

### Per-feed overrides

Every feed can override the defaults above. Set these in `config.yaml` under each feed entry, or edit them in the web UI.

| Key | Default | Description |
|-----|---------|-------------|
| `sponsorblock` | `true` | Enable/disable SponsorBlock segment removal (YouTube only) |
| `sponsorblock_delay_minutes` | `1440` | Wait time after publish for SB segments to be crowdsourced |
| `llm_trim` | `false` | Enable Whisper + Claude ad detection for this feed |
| `max_age_days` | none | Ignore episodes older than this many days (a rolling alternative to `date_cutoff`; the later of the two applies) |
| `quality` | `1440` | YouTube video quality (`1440`, `1080`, or `max`) |
| `block_shorts` | `true` | Filter out YouTube Shorts (< 60s) |
| `min_duration_seconds` | `0` | Skip episodes shorter than this |
| `date_cutoff` | none | Ignore episodes published before this date (YYYYMMDD) |
| `title_exclude` | `[]` | Skip episodes whose title contains any of these strings |
| `claude_prompt_extra` | none | Append additional instructions to the default Claude ad-detection prompt |
| `claude_prompt_override` | none | Replace the default prompt entirely with a custom one |

`claude_prompt_extra` is useful for feed-specific tuning — e.g. *"This is an interview podcast, a guest discussing their work is not an ad."*. `claude_prompt_override` replaces the entire prompt, giving full control over what Claude looks for.

See `config.example.yaml` for the full schema.

## Architecture

```mermaid
flowchart TB
    subgraph Discovery
        YT_API[YouTube Data API v3<br/>playlistItems.list<br/>1 unit per 50 videos]
        POD_RSS[Podcast RSS Feed<br/>httpx with browser UA]
    end

    subgraph Acquisition
        YT_DL[yt-dlp Download<br/>Firefox cookies / YT Premium<br/>Rate: 10/hr, 120s delay]
        SB[SponsorBlock<br/>Auto-cut known segments<br/>Stream copy, no re-encode]
        POD_DL[HTTP Download<br/>Rate: 120/hr, 2s delay<br/>10 parallel workers]
        YT_DL --> SB
    end

    subgraph Transcription
        WHISPER[faster-whisper<br/>CUDA: 1 worker / CPU: 1-5 workers<br/>Word-level timestamps<br/>Singleton model, shared weights]
    end

    subgraph Analysis
        CLAUDE1[Claude CLI #1<br/>+ ffmpeg cut]
        CLAUDE2[Claude CLI #2<br/>+ ffmpeg cut]
        CLAUDE3[Claude CLI #3<br/>+ ffmpeg cut]
    end

    subgraph Storage
        DB[SQLite Database<br/>WAL mode, episode state<br/>metrics, feed metadata]
        MEDIA[Media Files<br/>MP4 / MP3 on disk]
    end

    subgraph Serving
        RSS_SRV[RSS Generator<br/>FastAPI + Jinja2 XML<br/>iTunes namespace]
        MEDIA_SRV[Media Server<br/>FastAPI static files]
        WEB_SRV[Web UI Server<br/>FastAPI + Jinja2 + htmx]
    end

    subgraph Network
        FUNNEL[Tailscale Funnel<br/>HTTPS + Basic Auth]
        TAILNET[Tailscale Network<br/>HTTP, no auth]
        LOCALHOST[Localhost Only<br/>No auth]
    end

    subgraph Clients
        PC_SCRAPER[Pocket Casts<br/>Feed Scraper]
        PC_APP[Pocket Casts<br/>App Playback]
        WEBUI[Web UI<br/>Dashboard + Feed Management]
    end

    YT_API --> YT_DL
    POD_RSS --> POD_DL
    SB --> WHISPER
    POD_DL --> WHISPER
    WHISPER --> CLAUDE1
    WHISPER --> CLAUDE2
    WHISPER --> CLAUDE3
    CLAUDE1 --> DB
    CLAUDE2 --> DB
    CLAUDE3 --> DB
    CLAUDE1 --> MEDIA
    CLAUDE2 --> MEDIA
    CLAUDE3 --> MEDIA
    DB --> RSS_SRV
    DB --> WEB_SRV
    MEDIA --> MEDIA_SRV
    RSS_SRV --> FUNNEL
    MEDIA_SRV --> TAILNET
    WEB_SRV --> LOCALHOST
    FUNNEL --> PC_SCRAPER
    TAILNET --> PC_APP
    LOCALHOST --> WEBUI

    style YT_API fill:#c0392b,color:#fff
    style WHISPER fill:#2c3e50,color:#fff
    style CLAUDE1 fill:#8e44ad,color:#fff
    style CLAUDE2 fill:#8e44ad,color:#fff
    style CLAUDE3 fill:#8e44ad,color:#fff
    style SB fill:#2980b9,color:#fff
    style DB fill:#2c3e50,color:#fff
    style MEDIA fill:#2c3e50,color:#fff
    style RSS_SRV fill:#16a085,color:#fff
    style MEDIA_SRV fill:#16a085,color:#fff
    style WEB_SRV fill:#16a085,color:#fff
    style FUNNEL fill:#27ae60,color:#fff
    style TAILNET fill:#27ae60,color:#fff
    style LOCALHOST fill:#27ae60,color:#fff
    style PC_SCRAPER fill:#f39c12,color:#fff
    style PC_APP fill:#f39c12,color:#fff
    style WEBUI fill:#f39c12,color:#fff
```

### Three-queue pipeline

The pipeline uses three independent workers, each running on their own schedule:

| Worker | Interval | Concurrency | What it does |
|--------|----------|-------------|-------------|
| **Download** | 5 min | Sequential (rate limited) | Downloads media, applies SponsorBlock for YouTube |
| **Whisper** | 30 sec | CUDA: 1 / CPU: 1-5 (configurable) | Transcribes audio with word-level timestamps. Singleton model, shared weights. |
| **Claude** | 30 sec | 3 concurrent (configurable) | Detects ad segments, applies ffmpeg cuts |

Episodes flow through: `eligible` &rarr; `downloading` &rarr; `pending_whisper` &rarr; `pending_claude` &rarr; `done`

Episodes only appear in RSS after the full pipeline completes. Feeds without LLM trim skip directly to `done`.

### How Claude processes ads

1. **Whisper** transcribes the audio with word-level timestamps (CUDA: ~30s, CPU: ~5min per episode)
2. Claude receives a dual-format transcript:
   - **Segments** (coarse, for understanding context): `[0:00-0:45] Welcome to the show...`
   - **Word timestamps** (precise, for cut points): `0.00 Welcome  0.31 to  0.45 the...`
3. Claude identifies ad segments with start/end times and confidence scores
4. Segments above the confidence threshold are cut via ffmpeg stream copy
5. Per-episode metrics recorded: whisper time, claude time, ffmpeg time, word count, device used

For episodes longer than 45 minutes, word timestamps are omitted to stay within context limits (configurable).

### How ffmpeg cuts work

Claude returns all ad segments with timestamps referencing the **original file**. Rather than cutting sequentially (which would shift timestamps after each cut), ffmpeg inverts the cut list into keep-ranges:

1. Sort all ad segments by start time and merge overlaps
2. Invert to get the **keep ranges** — the gaps between ads
3. Extract each keep range from the original file with `-ss`/`-to` and `-c copy` (stream copy, no re-encode)
4. Concatenate all kept pieces via ffmpeg concat demuxer

Every extraction reads from the untouched original, so timestamps never shift. The entire operation is stream-copy — no audio/video re-encoding, so it's fast regardless of file size.

### YouTube integration

- **Discovery**: YouTube Data API v3 `playlistItems.list` at 1 unit per 50 videos (500,000 videos/day capacity)
- **First check**: pages backwards through entire channel until `date_cutoff` — one-time cost
- **Subsequent checks**: pages backwards until hitting a known video — typically 1-2 API calls
- **Quota cooldown**: on 403, all YouTube API calls pause for configurable hours (default 4)
- **Downloads**: yt-dlp with Firefox cookie integration for YouTube Premium, rate limited at 10/hr
- **SponsorBlock**: segments counted and tracked per episode for insights
- **Region filtering**: videos blocked in your country are silently skipped during discovery

### Podcast integration

- **Downloads**: 120/hour, 2-second delay, 10 parallel workers
- **30 feeds checked per cycle** (vs 10 for YouTube)
- **Browser User-Agent** for hosts that block default Python agents
- **Artwork**: pulled from RSS `<itunes:image>` and served in generated feeds

## Metrics & observability

Per-episode metrics tracked in SQLite:
- `whisper_duration_seconds`, `claude_duration_seconds`, `ffmpeg_duration_seconds`
- `whisper_word_count`, `whisper_segment_count`, `transcript_size_bytes`
- `whisper_model`, `whisper_device` (CPU vs CUDA comparison)
- `llm_cuts_applied`, `sb_cuts_applied`, `sb_seconds_removed`
- `filter_reason` (too_old, too_short, short, title_match)

Dashboard insights computed from these metrics: time saved, most stale feeds, disk usage by feed, longest processing episodes, highest cut rates, highest filter rates, most active feeds, queue backlog, feed errors.
