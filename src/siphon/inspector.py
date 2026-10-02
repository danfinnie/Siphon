"""Turn a stored transcript and ad-detection record into what the Episodes page shows.

Everything here is a pure function of the data Siphon saves for an episode:

* the ad-detection audit (``episodes.llm_segments_json``): every segment Claude returned
  (``segments``), the ones that were cut (``high_confidence``), the near misses
  (``marginal``) and, for episodes cut by newer versions, the ranges actually removed after
  merging and snapping (``applied_cuts``) plus the before/after durations;
* the Whisper transcript JSON.

All times are seconds on the *original* audio's timeline unless a name says ``clean``.
"""

from __future__ import annotations

import json

# A transcript line counts as cut when at least this share of it lies inside a cut.
FULL_OVERLAP = 0.5
# Below FULL_OVERLAP but above this share (or MIN_EDGE_SECONDS) it is flagged as a boundary line.
EDGE_SHARE = 0.05
MIN_EDGE_SECONDS = 0.15


def parse_audit(raw: str | None) -> dict:
    """Parse ``llm_segments_json``; anything unreadable gives an empty audit."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _as_range(seg: dict) -> tuple[float, float] | None:
    try:
        start, end = float(seg["start"]), float(seg["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def removed_ranges(audit: dict) -> list[tuple[float, float]]:
    """The ranges taken out of the audio, sorted and merged.

    Uses ``applied_cuts`` (what ffmpeg really removed) when it was recorded, otherwise the
    ``high_confidence`` segments.
    """
    source = audit.get("applied_cuts") or audit.get("high_confidence") or []
    ranges = sorted(r for r in (_as_range(s) for s in source if isinstance(s, dict)) if r)
    merged: list[list[float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def chime_seconds(audit: dict) -> float:
    """Length of the chime inserted at each cut (0 when no chimes were added)."""
    try:
        return max(0.0, float(audit.get("chime_seconds") or 0))
    except (TypeError, ValueError):
        return 0.0


def build_cut_list(audit: dict) -> list[dict]:
    """One entry per removed chunk, with its label, confidence and splice position.

    ``clean_at`` is where the splice falls on the *cleaned* audio's timeline.
    """
    detected = [
        (rng, seg) for seg in audit.get("high_confidence") or []
        if isinstance(seg, dict) and (rng := _as_range(seg))
    ]
    chime = chime_seconds(audit)
    cuts = []
    removed_before = 0.0
    for start, end in removed_ranges(audit):
        matches = [seg for (rs, re_), seg in detected if _overlap(start, end, rs, re_) > 0]
        labels: list[str] = []
        for seg in matches:
            label = str(seg.get("label") or "").strip()
            if label and label not in labels:
                labels.append(label)
        confidences = [seg.get("confidence") for seg in matches if isinstance(seg.get("confidence"), (int, float))]
        cuts.append({
            "index": len(cuts) + 1,
            "start": start,
            "end": end,
            "duration": end - start,
            "label": " + ".join(labels) if labels else "cut",
            "confidence": max(confidences) if confidences else None,
            "parts": len(matches),
            # where the splice falls on the cleaned timeline; each earlier cut left a chime behind
            "clean_at": start - removed_before + chime * len(cuts),
            "chime": chime,
        })
        removed_before += end - start
    return cuts


def classify_segments(transcript_segments: list[dict], audit: dict) -> list[dict]:
    """Label each transcript line.

    status is one of:
      ``cut``       at least half the line was removed
      ``edge``      the line touches a cut boundary (part of it was removed)
      ``possible``  Claude flagged it with middling confidence, but it was not cut
      ``detected``  Claude flagged it, but it was ignored (too short, too long, ...)
      ``content``   nothing flagged it
    """
    cuts = build_cut_list(audit)
    marginal = [
        (rng, seg) for seg in audit.get("marginal") or []
        if isinstance(seg, dict) and (rng := _as_range(seg))
    ]
    detected = [
        (rng, seg) for seg in audit.get("segments") or []
        if isinstance(seg, dict) and (rng := _as_range(seg))
    ]

    result = []
    for seg in transcript_segments:
        rng = _as_range(seg)
        if rng is None:
            continue
        start, end = rng
        length = end - start
        item = {
            "start": start, "end": end, "text": str(seg.get("text", "")).strip(),
            "status": "content", "label": "", "cut_index": None,
        }

        in_cut = 0.0
        best_cut = None
        for cut in cuts:
            amount = _overlap(start, end, cut["start"], cut["end"])
            in_cut += amount
            if best_cut is None or amount > best_cut[0]:
                best_cut = (amount, cut)

        if in_cut >= FULL_OVERLAP * length:
            item.update(status="cut", label=best_cut[1]["label"], cut_index=best_cut[1]["index"])
        elif in_cut > max(EDGE_SHARE * length, MIN_EDGE_SECONDS) or (in_cut > 0 and length <= MIN_EDGE_SECONDS):
            item.update(status="edge", label=best_cut[1]["label"], cut_index=best_cut[1]["index"])
        else:
            for status, pool in (("possible", marginal), ("detected", detected)):
                hit = next(
                    (s for (rs, re_), s in pool if _overlap(start, end, rs, re_) >= FULL_OVERLAP * length),
                    None,
                )
                if hit is not None:
                    item.update(status=status, label=str(hit.get("label") or "").strip())
                    break
        result.append(item)
    return result


def flagged_not_cut(audit: dict) -> list[dict]:
    """Segments Claude returned that were not cut, with the reason when it can be worked out."""
    cut_ranges = removed_ranges(audit)
    marginal_ranges = {
        r for r in (_as_range(s) for s in audit.get("marginal") or [] if isinstance(s, dict)) if r
    }
    rows = []
    for seg in audit.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        rng = _as_range(seg)
        if rng is None:
            continue
        start, end = rng
        covered = sum(_overlap(start, end, a, b) for a, b in cut_ranges)
        if covered >= FULL_OVERLAP * (end - start):
            continue
        confidence = seg.get("confidence")
        if rng in marginal_ranges:
            reason = "confidence below the cut threshold"
        elif isinstance(confidence, (int, float)) and confidence < 0.5:
            reason = "confidence too low"
        elif end - start < 7:
            reason = "shorter than the minimum length"
        else:
            reason = "not cut (length or confidence limits)"
        rows.append({
            "start": start, "end": end, "duration": end - start,
            "label": str(seg.get("label") or "").strip() or "flagged",
            "confidence": confidence if isinstance(confidence, (int, float)) else None,
            "reason": reason,
        })
    return sorted(rows, key=lambda r: r["start"])


def summarize(audit: dict, episode: dict, cuts: list[dict]) -> dict:
    """Headline numbers: durations before/after, time removed, number of cuts."""
    removed = sum(c["duration"] for c in cuts)
    original = audit.get("original_duration")
    final = audit.get("final_duration")
    if original is None and final is not None and removed:
        original = final + removed
    if final is None and original is not None and removed:
        final = max(0.0, original - removed)
    percent = (100.0 * removed / original) if original else None
    return {
        "original_duration": original,
        "final_duration": final,
        "removed": removed,
        "percent": percent,
        "cut_count": len(cuts),
    }


def clean_time_map(cuts: list[dict]) -> list[dict]:
    """Compact data the page script uses to convert between the two timelines."""
    return [
        {"start": c["start"], "end": c["end"], "clean_at": c["clean_at"], "chime": c.get("chime", 0.0)}
        for c in cuts
    ]


def format_clock(seconds) -> str:
    """Seconds -> ``M:SS`` or ``H:MM:SS``."""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "-"
    total = max(0, total)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def load_transcript_segments(path: str) -> list[dict] | None:
    """Read the saved transcript's segments, or None if there is no usable transcript."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    segments = data.get("segments") if isinstance(data, dict) else None
    return segments if isinstance(segments, list) and segments else None
