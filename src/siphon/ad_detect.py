"""Ad segment detection using Claude CLI with structured output."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from siphon.config import LLMConfig, ResolvedFeed

logger = logging.getLogger(__name__)

# JSON Schema for Claude's structured output
AD_SEGMENTS_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "type": {"type": "string"},
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["start", "end", "label", "confidence"],
            },
        },
    },
    "required": ["segments"],
})

CLI_TIMEOUT_SECONDS = 300


class ClaudeCLIError(RuntimeError):
    """The Claude CLI run failed for an episode-specific or unknown reason.

    The pipeline counts these against the episode (three strikes → skipped).
    """


class ClaudeAuthError(ClaudeCLIError):
    """The CLI is logged out or its OAuth session could not be refreshed.

    Systemic: every run fails until someone runs ``claude /login`` on the
    host, so the pipeline pauses the worker and raises an alert instead of
    burning each episode's retries.
    """


class ClaudeTransientError(ClaudeCLIError):
    """Server overload, usage limit, or the CLI's stdin race.

    Expected to clear on its own; the pipeline backs off and retries later
    without penalising the episode.
    """


# Matched against the lower-cased CLI error text.  Auth is checked first so
# "OAuth session expired" can never be read as transient.  Numeric HTTP codes
# need word boundaries: "150000 tokens" must not look like a 500.
_AUTH_RE = re.compile(
    r"failed to authenticate|oauth|not logged in|/login|authentication_error"
    r"|invalid api key|invalid x-api-key|unauthorized|\b401\b"
)
_TRANSIENT_RE = re.compile(
    r"overloaded|\b5(?:00|02|03|04|29)\b|rate.?limit|usage limit|limit reached|resets at"
    r"|no stdin data received|internal server error|econnreset|etimedout|econnrefused"
    r"|network error|fetch failed|connection error"
)
_STDIN_RACE_MARKER = "no stdin data received"


def resolve_prompt(feed: ResolvedFeed, llm_config: LLMConfig) -> str:
    """Build the final prompt for ad detection.

    If claude_prompt_override is set, use it exclusively.
    Otherwise, use the global default_ad_prompt + any claude_prompt_extra.
    """
    if feed.claude_prompt_override:
        return feed.claude_prompt_override
    prompt = llm_config.default_ad_prompt
    if feed.claude_prompt_extra:
        prompt += "\n\nAdditional instructions:\n" + feed.claude_prompt_extra
    return prompt


def build_transcript_for_claude(
    transcript_text: str,
    segments: list,
    words: list | None,
) -> str:
    """Build a dual-format transcript for Claude with segments and word timestamps.

    When word timestamps are available, produces:
        SEGMENTS (for understanding content):
        [0:00-0:45] Welcome to the show...
        ...

        WORD TIMESTAMPS (for precise cut points):
        0.00 Welcome
        0.31 to
        ...

    When words are not available, falls back to segment-only format.
    """
    parts = []

    # Format segments as [M:SS-M:SS] text
    if segments:
        parts.append("SEGMENTS (for understanding content):")
        for seg in segments:
            start_m, start_s = divmod(int(seg["start"]), 60)
            end_m, end_s = divmod(int(seg["end"]), 60)
            parts.append(f"[{start_m}:{start_s:02d}-{end_m}:{end_s:02d}] {seg['text']}")
        parts.append("")

    # Format word-level timestamps if available
    if words:
        parts.append("WORD TIMESTAMPS (for precise cut points):")
        for w in words:
            parts.append(f"{w['start']:.2f} {w['word']}")
        parts.append("")

    if parts:
        return "\n".join(parts)

    # Ultimate fallback: just use the raw text
    return transcript_text


def _cli_creationflags() -> int:
    """Process-creation flags for the CLI subprocess.

    Siphon itself runs at below-normal priority (see ``__main__``) and children
    inherit that.  The CLI must be spawned at NORMAL explicitly: its CPU cost
    is a few seconds of startup, but it guards stdin with a hard-coded 3 s
    timer, and a below-normal process starved by ffmpeg, Whisper or a game
    loses that race even though the prompt is already sitting in the pipe.
    """
    return 0x00000020 if sys.platform == "win32" else 0  # NORMAL_PRIORITY_CLASS


def cli_failure_reason(result: subprocess.CompletedProcess) -> str:
    """Best human-readable reason for a non-zero CLI exit.

    With ``--output-format json`` the CLI reports auth, overload and API errors
    inside the stdout envelope (``is_error: true``, ``result: "..."``) and
    leaves stderr empty, so stdout is consulted first.
    """
    text = ""
    try:
        payload = json.loads(result.stdout or "")
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        text = str(payload.get("result") or payload.get("error") or payload.get("message") or "")
    if not text.strip():
        text = result.stderr or ""
    if not text.strip():
        text = f"exit code {result.returncode} with no output"
    return " ".join(text.split())


def classify_cli_failure(reason: str) -> type[ClaudeCLIError]:
    """Map a CLI failure message to the exception class the pipeline should see."""
    low = reason.lower()
    if _AUTH_RE.search(low):
        return ClaudeAuthError
    if _TRANSIENT_RE.search(low):
        return ClaudeTransientError
    return ClaudeCLIError


def detect_ads(
    transcript_text: str,
    prompt: str,
    model: str = "claude-sonnet-4-6",
    effort: str = "medium",
    words: list | None = None,
    segments: list | None = None,
    title: str | None = None,
    feed_name: str | None = None,
) -> dict[str, Any]:
    """Invoke Claude CLI to detect ad segments in a transcript.

    Returns the structured output dict: {"segments": [...]}

    Raises :class:`ClaudeAuthError` / :class:`ClaudeTransientError` /
    :class:`ClaudeCLIError` so the pipeline can tell a logged-out CLI or an
    overloaded API apart from a failure specific to this episode.
    """
    formatted = build_transcript_for_claude(
        transcript_text, segments or [], words,
    )
    context = ""
    if feed_name or title:
        context = "EPISODE CONTEXT:\n"
        if feed_name:
            context += f"Feed: {feed_name}\n"
        if title:
            context += f"Title: {title}\n"
        context += "\n"
    full_prompt = f"{prompt}\n\n{context}TRANSCRIPT:\n{formatted}"

    cmd = [
        "claude",
        "-p",
        "--model", model,
        "--output-format", "json",
        "--json-schema", AD_SEGMENTS_SCHEMA,
        "--effort", effort,
    ]

    logger.info("Running Claude CLI for ad detection (model=%s, effort=%s, prompt_len=%d)",
                model, effort, len(full_prompt))

    # Hand the CLI a real file on stdin, not a pipe.  The CLI guards stdin with
    # a hard 3 s timer that starts when it begins reading; feeding it through a
    # pipe (input=) makes that a race against Python's writer thread being
    # scheduled to fill the pipe — which loses under concurrency + CUDA Whisper
    # load, especially for 100 KB+ prompts that overflow the OS pipe buffer and
    # block the writer mid-send.  A file's bytes are available immediately, so
    # the CLI's first read always beats the timer.  Transcripts are also far
    # past the Windows 32 KB argv limit, which rules out a prompt argument.
    fd, prompt_path = tempfile.mkstemp(prefix="siphon-claude-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(full_prompt)

        result = None
        for attempt in (1, 2):
            with open(prompt_path, "r", encoding="utf-8") as stdin_f:
                result = subprocess.run(
                    cmd,
                    stdin=stdin_f,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=CLI_TIMEOUT_SECONDS,
                    creationflags=_cli_creationflags(),
                )
            if result.returncode == 0:
                break
            reason = cli_failure_reason(result)
            if attempt == 1 and _STDIN_RACE_MARKER in reason.lower():
                # A file on stdin should never trip the CLI's 3 s guard; keep one
                # retry as a cheap safety net for any residual startup hiccup.
                logger.warning("Claude CLI missed its stdin window, retrying once: %s", reason[:160])
                time.sleep(2)
                continue
            err_cls = classify_cli_failure(reason)
            logger.error("Claude CLI failed (rc=%d, %s): %s", result.returncode, err_cls.__name__, reason[:500])
            raise err_cls(f"Claude CLI failed (exit code {result.returncode}): {reason[:300]}")
    finally:
        try:
            os.remove(prompt_path)
        except OSError:
            pass

    # Parse the JSON output — claude --output-format json wraps in a result envelope
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.error("Claude CLI output is not valid JSON: %s", result.stdout[:500])
        raise ClaudeCLIError(f"Claude CLI output is not valid JSON: {e}") from e

    # Extract structured_output from the envelope
    if "structured_output" in output:
        return output["structured_output"]

    # If the output is already the segments dict
    if "segments" in output:
        return output

    logger.warning("Unexpected Claude CLI output structure: %s", list(output.keys()))
    return {"segments": []}


def _merge_group(members: list[dict]) -> dict:
    """Collapse a group of back-to-back segments into one segment spanning all of them."""
    if len(members) == 1:
        return members[0]
    labels: list[str] = []
    for m in members:
        label = m.get("label", "")
        if label and label not in labels:
            labels.append(label)
    merged = dict(members[0])
    merged.update({
        "start": min(m["start"] for m in members),
        "end": max(m["end"] for m in members),
        "label": " + ".join(labels),
        "confidence": max(m.get("confidence", 0) for m in members),
        "merged_from": len(members),
    })
    return merged


def filter_segments(
    segments: list[dict],
    confidence_threshold: float = 0.75,
    min_duration: int = 7,
    max_duration: int = 300,
    merge_gap: float = 3.0,
) -> tuple[list[dict], list[dict]]:
    """Decide which detected segments to cut.

    Returns (high_confidence, marginal) where:
    - high_confidence: segments to cut. Segments that overlap, touch, or sit within
      ``merge_gap`` seconds of each other are merged into one chunk, so back-to-back ads
      are removed as a single piece (0:00-0:30 plus 0:30-1:00 becomes 0:00-1:00).
    - marginal: segments between 0.5 and the threshold (logged but not cut)

    Neighbouring segments are judged as a group, not one by one:
    - a group is cut when at least one member reaches ``confidence_threshold``, so a
      lower-confidence segment sitting right next to a confident ad is cut with it
      instead of being left behind;
    - ``min_duration`` applies to the merged group, so a short tag-line next to an ad is
      cut with it, while an isolated short segment is still ignored;
    - ``max_duration`` applies to each detected segment (a guard against a runaway
      detection), not to the merged group, so two long ads in a row are still removed.
    """
    candidates = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start
        confidence = seg.get("confidence", 0)

        # Skip segments outside duration bounds
        if duration <= 0 or duration > max_duration:
            logger.debug(
                "Skipping segment %.1f-%.1f (duration=%.1fs, over the %ds limit)",
                start, end, duration, max_duration,
            )
            continue
        if confidence < 0.5:
            continue
        candidates.append(seg)

    candidates.sort(key=lambda s: s["start"])

    groups: list[list[dict]] = []
    group_end = 0.0
    for seg in candidates:
        if groups and seg["start"] - group_end <= merge_gap:
            groups[-1].append(seg)
            group_end = max(group_end, seg["end"])
        else:
            groups.append([seg])
            group_end = seg["end"]

    high_confidence = []
    marginal = []
    for members in groups:
        span = max(m["end"] for m in members) - min(m["start"] for m in members)
        if span < min_duration:
            logger.debug("Skipping segment group %.1f-%.1f (duration=%.1fs, under %ds)",
                         members[0]["start"], members[-1]["end"], span, min_duration)
            continue

        if max(m.get("confidence", 0) for m in members) >= confidence_threshold:
            merged = _merge_group(members)
            if len(members) > 1:
                logger.info(
                    "Merged %d adjacent segments into one cut: %.1f-%.1f (%s)",
                    len(members), merged["start"], merged["end"], merged["label"],
                )
            high_confidence.append(merged)
        else:
            for m in members:
                marginal.append(m)
                logger.info(
                    "Marginal detection: %.1f-%.1f '%s' (confidence=%.2f)",
                    m.get("start", 0), m.get("end", 0), m.get("label", ""), m.get("confidence", 0),
                )

    return high_confidence, marginal
