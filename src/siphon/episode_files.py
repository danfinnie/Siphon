"""Where the extra per-episode files for the episode inspector live.

When ``storage.keep_inspection_data`` is on, Siphon keeps two things next to each episode:

* the original audio from before the ads were cut (``<name>.orig.<ext>``), and
* the Whisper transcript (``<id>_transcript.json``).

Both are removed whenever the episode itself is pruned or deleted.
"""

from __future__ import annotations

import os
import re
import shutil


def original_path(file_path: str) -> str:
    """Path of the pre-cut original that sits beside an episode file."""
    stem, ext = os.path.splitext(file_path)
    return f"{stem}.orig{ext}"


def transcript_path(download_dir: str, feed_name: str, video_id: str) -> str:
    """Path of the transcript JSON for an episode."""
    safe_id = re.sub(r"[^\w-]", "_", video_id)[:80]
    return os.path.join(download_dir, feed_name, f"{safe_id}_transcript.json")


def remove_extras(download_dir: str, episode: dict) -> None:
    """Delete an episode's kept original and transcript, ignoring files that are not there."""
    paths = []
    if episode.get("file_path"):
        paths.append(original_path(episode["file_path"]))
    feed_name, video_id = episode["feed_name"], episode["video_id"]
    paths.append(transcript_path(download_dir, feed_name, video_id))
    # older code paths used the raw (unsanitised) video id
    paths.append(os.path.join(download_dir, feed_name, f"{video_id}_transcript.json"))
    for path in dict.fromkeys(paths):
        try:
            os.remove(path)
        except OSError:
            pass


def restore_original(file_path: str) -> bool:
    """Put the kept original back as the episode file (the kept copy stays in place).

    Returns False when there is no kept original, i.e. the episode file already is the original.
    Raises OSError if the file cannot be replaced (for example while it is being streamed).
    """
    kept = original_path(file_path)
    if not os.path.isfile(kept):
        return False
    staging = f"{file_path}.restoring"
    try:
        shutil.copy2(kept, staging)
        os.replace(staging, file_path)
    finally:
        if os.path.exists(staging):
            os.remove(staging)
    return True
