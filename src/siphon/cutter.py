"""ffmpeg-based segment cutting for ad removal."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str) -> None:
    """Extract audio track from a video file for Whisper transcription."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # WAV format for Whisper
        "-ar", "16000",           # 16kHz sample rate (Whisper's native rate)
        "-ac", "1",               # mono
        output_path,
    ]
    logger.info("Extracting audio from %s to %s", video_path, output_path)
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:300]}")


def get_duration(file_path: str) -> float:
    """Get the duration of a media file in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")
    return float(result.stdout.strip())


def validate_file(file_path: str) -> bool:
    """Validate a media file using ffprobe. Returns True if the file is valid."""
    if not os.path.exists(file_path):
        return False
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode != 0:
        logger.warning("File validation failed for %s: %s", file_path, result.stderr[:200])
        return False
    try:
        dur = float(result.stdout.strip())
        if dur <= 0:
            logger.warning("File validation failed for %s: duration is %s", file_path, dur)
            return False
    except (ValueError, TypeError):
        logger.warning("File validation failed for %s: could not parse duration", file_path)
        return False
    return True


_THUMBNAIL_CODECS = {"png", "mjpeg", "bmp", "gif", "tiff", "webp"}


def has_real_video_stream(file_path: str) -> bool:
    """Return True if the file has a real (non-thumbnail) video stream.

    Detects the audio-only-with-embedded-thumbnail shape that yt-dlp's
    `/best` fallback can produce: an mp4 container with opus audio and
    a PNG cover image masquerading as a video stream (codec=png,
    nb_frames typically 1-10).
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=codec_name,nb_frames",
        "-of", "csv=p=0",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode != 0:
        return False
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        codec = parts[0].strip().lower()
        if codec in _THUMBNAIL_CODECS:
            continue
        try:
            nb_frames = int(parts[1]) if len(parts) > 1 and parts[1].strip() not in ("", "N/A") else -1
        except ValueError:
            nb_frames = -1
        # Real video: known video codec + either unknown frame count (streaming) or > 10 frames
        if nb_frames == -1 or nb_frames > 10:
            return True
    return False


def normalize_timestamps(file_path: str) -> bool:
    """Normalize a media file's timestamps to start at zero.

    Fixes audio/video desync caused by SponsorBlock stream-copy cuts
    leaving non-zero start times. Stream copy — no re-encoding.
    Returns True if normalization was applied, False if not needed.
    """
    # Check if start times are non-zero
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=start_time",
        "-of", "csv=p=0",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode != 0:
        return False

    needs_fix = False
    for line in result.stdout.strip().split("\n"):
        try:
            start = float(line.strip())
            if start > 0.5:  # More than 500ms offset
                needs_fix = True
                break
        except (ValueError, TypeError):
            continue

    if not needs_fix:
        return False

    logger.info("Normalizing timestamps for %s (non-zero start detected)", file_path)

    ext = os.path.splitext(file_path)[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output = os.path.join(tmpdir, f"normalized{ext}")
        cmd = [
            "ffmpeg", "-y",
            "-i", file_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            temp_output,
        ]
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        if result.returncode != 0:
            logger.warning("Timestamp normalization failed: %s", result.stderr[:200])
            return False

        if not validate_file(temp_output):
            logger.warning("Normalized file failed validation, keeping original")
            return False

        import shutil
        shutil.move(temp_output, file_path)
        logger.info("Timestamps normalized for %s", file_path)
        return True


def find_nearest_silence(
    file_path: str,
    target: float,
    window: float = 3.0,
    noise_db: int = -30,
    min_silence: float = 0.15,
) -> float:
    """Return the middle of the silence closest to ``target`` (within +/- ``window`` seconds).

    Uses ffmpeg's silencedetect on a short slice around the target. If no silence is found,
    or ffmpeg fails, ``target`` is returned unchanged.
    """
    start_search = max(0.0, target - window)
    duration = (target + window) - start_search
    cmd = [
        "ffmpeg", "-v", "info",
        "-ss", str(start_search),
        "-t", str(duration),
        "-i", file_path,
        "-vn",
        "-af", f"silencedetect=noise={noise_db}dB:duration={min_silence}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Silence detection failed near %.1fs: %s", target, exc)
        return target

    output = result.stderr or ""
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", output)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", output)]
    # A silence that runs past the end of the slice has a start but no end
    if len(ends) < len(starts):
        ends.append(duration)

    best = target
    best_distance = float("inf")
    for silence_start, silence_end in zip(starts, ends):
        middle = start_search + (max(silence_start, 0.0) + silence_end) / 2.0
        distance = abs(middle - target)
        if distance < best_distance:
            best, best_distance = middle, distance

    if best_distance <= window:
        logger.debug("Snapped cut point %.2fs to silence at %.2fs", target, best)
        return best
    return target


def _merge_ranges(ranges: list[dict], merge_gap: float) -> list[dict]:
    """Sort ranges and merge those that overlap or are within ``merge_gap`` of each other."""
    merged: list[dict] = []
    for seg in sorted(ranges, key=lambda s: s["start"]):
        if merged and seg["start"] <= merged[-1]["end"] + merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append({"start": seg["start"], "end": seg["end"]})
    return merged


# Chime marking where an ad was cut: two soft bell notes (A5 then E6) with a short lead-in.
CHIME_SECONDS = 1.7
_CHIME_ENCODERS = {"mp3": ("libmp3lame", "128k"), "aac": ("aac", "96k")}


def _chime_expression(volume: float) -> str:
    """ffmpeg aevalsrc expression for the chime: decaying bell notes with a 3 ms attack."""
    def note(freq: float, start: float) -> str:
        u = f"(t-{start})"
        partials = f"(sin(2*PI*{freq}*{u})+0.35*sin(4*PI*{freq}*{u})+0.12*sin(6*PI*{freq}*{u}))"
        return f"if(gte(t\\,{start})\\,min(1\\,{u}*300)*exp(-4.5*{u})*{partials}\\,0)"
    return f"{volume}*0.45*({note(880, 0.15)}+{note(1318.51, 0.40)})"


def audio_stream_info(file_path: str) -> dict | None:
    """Codec, sample rate and channels of the first audio stream, and whether there is real video."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,sample_rate,channels",
        "-of", "json", file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        streams = json.loads(result.stdout or "{}").get("streams", [])
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    audio = next((st for st in streams if st.get("codec_type") == "audio"), None)
    if audio is None:
        return None
    return {
        "codec": audio.get("codec_name"),
        "sample_rate": int(audio.get("sample_rate") or 0),
        "channels": int(audio.get("channels") or 0),
        "has_video": has_real_video_stream(file_path),
    }


def make_chime(output_path: str, info: dict, volume: float = 0.5) -> float | None:
    """Write the chime to ``output_path`` in the episode's codec, sample rate and channel count.

    Matching the episode means it can be joined to the cut audio without re-encoding anything.
    Returns the chime's length in seconds, or None if it could not be made.
    """
    encoder, bitrate = _CHIME_ENCODERS[info["codec"]]
    volume = min(max(float(volume), 0.05), 0.9)   # 0.9 keeps the two overlapping notes from clipping
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"aevalsrc={_chime_expression(volume)}:s={info['sample_rate']}:d={CHIME_SECONDS}",
        "-ac", str(info["channels"]), "-ar", str(info["sample_rate"]),
        "-c:a", encoder, "-b:a", bitrate,
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
        if result.returncode != 0:
            logger.warning("Could not make the chime: %s", result.stderr[-300:])
            return None
        return get_duration(output_path)
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
        logger.warning("Could not make the chime: %s", exc)
        return None


def cut_segments(
    input_path: str,
    segments: list[dict],
    output_path: str | None = None,
    merge_gap: float = 1.0,
    snap_to_silence: bool = False,
    snap_window: float = 3.0,
    applied: list | None = None,
    chime: bool = False,
    chime_volume: float = 0.5,
    report: dict | None = None,
) -> str:
    """Remove ad segments from a media file using ffmpeg concat demuxer.

    Segments are the parts to REMOVE. This function inverts them to get
    keep-ranges, then concatenates those ranges.

    Writes to a temp file first and validates before replacing the original.
    If validation fails, the original file is preserved.

    Segments that overlap, touch, or are less than ``merge_gap`` seconds apart are
    merged into one cut, so no sliver of an ad is left between two removed segments.

    With ``snap_to_silence`` each cut point is moved to the nearest silence within
    ``snap_window`` seconds, so cuts land in a pause rather than mid-sound. Cuts that start
    or end within 2 seconds of the file's edge are extended to the edge.

    If ``applied`` is a list, it is filled with the ranges that were actually removed
    (``{"start", "end"}`` in seconds on the original timeline) after merging and snapping.

    With ``chime`` a short chime is inserted at every cut (including one at the very start or end
    of the episode) so a listener can hear where an ad was removed. Ads that were merged into one
    cut get one chime. It is only added to audio-only mp3 and aac files; for anything else the
    cut is made without it. Embedded cover art is left out of the result when a chime is added.
    If ``report`` is a dict it receives ``chime_seconds`` and ``chime_count``
    when chimes were inserted.

    If output_path is None, overwrites the input file.
    Returns the final output path.
    """
    if not segments:
        logger.info("No segments to cut, skipping")
        return input_path

    # Get total duration
    total_duration = get_duration(input_path)

    # Sort segments by start time and merge overlaps and near-touching neighbours
    merged = _merge_ranges(segments, merge_gap)

    if snap_to_silence:
        snapped = []
        for seg in merged:
            start = 0.0 if seg["start"] <= 2.0 else find_nearest_silence(input_path, seg["start"], snap_window)
            end = (
                total_duration
                if seg["end"] >= total_duration - 2.0
                else find_nearest_silence(input_path, seg["end"], snap_window)
            )
            if end > start:
                snapped.append({"start": start, "end": end})
        merged = _merge_ranges(snapped, merge_gap)

    # Invert to get keep ranges, noting where each cut falls so a chime can go there
    plan: list[tuple] = []   # ("keep", start, end) or ("cut",)
    prev_end = 0.0
    for seg in merged:
        if seg["start"] > prev_end:
            plan.append(("keep", prev_end, seg["start"]))
        plan.append(("cut",))
        prev_end = seg["end"]
    if prev_end < total_duration:
        plan.append(("keep", prev_end, total_duration))
    keep_ranges = [(item[1], item[2]) for item in plan if item[0] == "keep"]

    if not keep_ranges:
        logger.warning("All content would be cut — skipping")
        return input_path

    chime_info = None
    if chime:
        info = audio_stream_info(input_path)
        if info and not info["has_video"] and info["codec"] in _CHIME_ENCODERS and info["sample_rate"] > 0:
            chime_info = info
        else:
            logger.info("Not adding a chime to %s: only audio-only mp3 and aac files are supported", input_path)

    if applied is not None:
        applied[:] = [{"start": m["start"], "end": m["end"]} for m in merged]

    logger.info(
        "Cutting %d segments from %s, keeping %d ranges",
        len(merged), input_path, len(keep_ranges),
    )

    ext = os.path.splitext(input_path)[1]

    # Use temp dir for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create individual segment files
        segment_files = []
        for i, (start, end) in enumerate(keep_ranges):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}{ext}")
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(start),
                "-to", str(end),
                # With a chime, keep only the audio: the chime file has no embedded cover art, and
                # pieces with different streams cannot be joined. (Feeds serve the artwork anyway.)
                *(["-map", "0:a:0"] if chime_info is not None else []),
                "-c", "copy",
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=300)
            if result.returncode != 0:
                logger.error("ffmpeg segment extraction failed: %s", result.stderr[:300])
                raise RuntimeError(f"ffmpeg segment extraction failed for range {start}-{end}")
            segment_files.append(seg_path)

        chime_path = None
        chime_length = None
        if chime_info is not None:
            chime_path = os.path.join(tmpdir, f"chime{ext}")
            chime_length = make_chime(chime_path, chime_info, chime_volume)
            if chime_length is None:
                chime_path = None   # carry on without it rather than fail the cut

        # Create concat list file: the kept pieces in order, with a chime at each cut
        def _concat_line(path: str) -> str:
            # ffmpeg concat requires forward slashes and escaped single quotes
            escaped = path.replace("\\", "/").replace("'", "'\\''")
            return f"file '{escaped}'\n"

        list_path = os.path.join(tmpdir, "concat.txt")
        chimes_added = 0
        pieces = iter(segment_files)
        with open(list_path, "w", encoding="utf-8") as f:
            for item in plan:
                if item[0] == "keep":
                    f.write(_concat_line(next(pieces)))
                elif chime_path is not None:
                    f.write(_concat_line(chime_path))
                    chimes_added += 1
        if report is not None and chimes_added:
            report["chime_seconds"] = chime_length
            report["chime_count"] = chimes_added

        # Concat to temp file (not directly to output)
        temp_output = os.path.join(tmpdir, f"output{ext}")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            temp_output,
        ]
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        if result.returncode != 0:
            logger.error("ffmpeg concat failed: %s", result.stderr[:300])
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr[:300]}")

        # Validate the output before replacing the original
        if not validate_file(temp_output):
            raise RuntimeError(f"Cut output failed validation for {input_path}")

        # Move validated output to final destination
        import shutil
        final = output_path or input_path
        shutil.move(temp_output, final)
        return final
