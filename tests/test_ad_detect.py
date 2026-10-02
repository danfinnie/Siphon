"""Tests for siphon.ad_detect — Claude CLI ad detection."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from siphon import ad_detect
from siphon.ad_detect import (
    ClaudeAuthError,
    ClaudeCLIError,
    ClaudeTransientError,
    classify_cli_failure,
    cli_failure_reason,
    detect_ads,
    filter_segments,
    resolve_prompt,
)
from siphon.config import FeedConfig, FeedDefaults, LLMConfig, resolve_feed


# ------------------------------------------------------------------ #
# Prompt resolution
# ------------------------------------------------------------------ #

class TestResolvePrompt:
    def test_default_prompt_used(self):
        llm = LLMConfig()
        defaults = FeedDefaults()
        feed_cfg = FeedConfig(name="f", url="http://x")
        feed = resolve_feed(feed_cfg, defaults)
        prompt = resolve_prompt(feed, llm)
        assert "sponsor" in prompt.lower()

    def test_prompt_extra_appended(self):
        llm = LLMConfig()
        defaults = FeedDefaults()
        feed_cfg = FeedConfig(name="f", url="http://x", claude_prompt_extra="Also remove Discord promos.")
        feed = resolve_feed(feed_cfg, defaults)
        prompt = resolve_prompt(feed, llm)
        assert "Discord promos" in prompt
        assert "sponsor" in prompt.lower()  # default is still there

    def test_prompt_override_replaces(self):
        llm = LLMConfig()
        defaults = FeedDefaults()
        feed_cfg = FeedConfig(name="f", url="http://x", claude_prompt_override="Only remove sponsor reads.")
        feed = resolve_feed(feed_cfg, defaults)
        prompt = resolve_prompt(feed, llm)
        assert prompt == "Only remove sponsor reads."


# ------------------------------------------------------------------ #
# Segment filtering
# ------------------------------------------------------------------ #

class TestFilterSegments:
    def test_high_confidence_passes(self):
        segments = [{"start": 0, "end": 30, "label": "ad", "confidence": 0.9}]
        high, marginal = filter_segments(segments, confidence_threshold=0.75)
        assert len(high) == 1
        assert len(marginal) == 0

    def test_marginal_confidence(self):
        segments = [{"start": 0, "end": 30, "label": "ad", "confidence": 0.6}]
        high, marginal = filter_segments(segments, confidence_threshold=0.75)
        assert len(high) == 0
        assert len(marginal) == 1

    def test_below_minimum_confidence_dropped(self):
        segments = [{"start": 0, "end": 30, "label": "ad", "confidence": 0.3}]
        high, marginal = filter_segments(segments, confidence_threshold=0.75)
        assert len(high) == 0
        assert len(marginal) == 0

    def test_too_short_filtered(self):
        segments = [{"start": 0, "end": 3, "label": "ad", "confidence": 0.9}]
        high, marginal = filter_segments(segments, min_duration=7)
        assert len(high) == 0

    def test_too_long_filtered(self):
        segments = [{"start": 0, "end": 600, "label": "ad", "confidence": 0.9}]
        high, marginal = filter_segments(segments, max_duration=300)
        assert len(high) == 0

    def test_mixed_segments(self):
        segments = [
            {"start": 0, "end": 30, "label": "sponsor", "confidence": 0.95},
            {"start": 100, "end": 120, "label": "self-promo", "confidence": 0.6},
            {"start": 200, "end": 202, "label": "tiny", "confidence": 0.99},
            {"start": 300, "end": 310, "label": "low-conf", "confidence": 0.3},
        ]
        high, marginal = filter_segments(
            segments, confidence_threshold=0.75, min_duration=7, max_duration=300
        )
        assert len(high) == 1  # sponsor
        assert high[0]["label"] == "sponsor"
        assert len(marginal) == 1  # self-promo
        assert marginal[0]["label"] == "self-promo"

    def test_empty_segments(self):
        high, marginal = filter_segments([])
        assert high == []
        assert marginal == []


class TestAdjacentSegmentMerging:
    """Back-to-back ads must be removed whole, as a single chunk."""

    @staticmethod
    def _ad(start, end, label="ad", confidence=0.9):
        return {"start": start, "end": end, "label": label, "confidence": confidence}

    def test_exactly_back_to_back_ads_become_one_chunk(self):
        high, marginal = filter_segments([self._ad(0, 30, "A"), self._ad(30, 60, "B")])
        assert len(high) == 1
        assert (high[0]["start"], high[0]["end"]) == (0, 60)
        assert high[0]["label"] == "A + B"
        assert high[0]["merged_from"] == 2
        assert marginal == []

    def test_small_gap_between_ads_is_merged(self):
        # the sliver of music/silence between two ads is removed with them
        high, _ = filter_segments([self._ad(100, 130), self._ad(131.5, 160)])
        assert [(h["start"], h["end"]) for h in high] == [(100, 160)]

    def test_gap_larger_than_merge_gap_keeps_ads_separate(self):
        high, _ = filter_segments([self._ad(100, 130), self._ad(140, 170)], merge_gap=3.0)
        assert [(h["start"], h["end"]) for h in high] == [(100, 130), (140, 170)]

    def test_merge_gap_is_configurable(self):
        segments = [self._ad(100, 130), self._ad(140, 170)]
        high, _ = filter_segments(segments, merge_gap=10.0)
        assert [(h["start"], h["end"]) for h in high] == [(100, 170)]
        high, _ = filter_segments(segments, merge_gap=0)
        assert len(high) == 2

    def test_three_ads_in_a_row_become_one_chunk(self):
        high, _ = filter_segments([
            self._ad(0, 20, "A"), self._ad(20, 45, "B"), self._ad(46, 80, "C"),
        ])
        assert [(h["start"], h["end"]) for h in high] == [(0, 80)]
        assert high[0]["merged_from"] == 3

    def test_unsorted_and_overlapping_input(self):
        high, _ = filter_segments([self._ad(50, 90), self._ad(0, 60), self._ad(200, 230)])
        assert [(h["start"], h["end"]) for h in high] == [(0, 90), (200, 230)]

    def test_chunk_takes_highest_confidence(self):
        high, _ = filter_segments([
            self._ad(0, 30, confidence=0.8), self._ad(30, 60, confidence=0.95),
        ])
        assert high[0]["confidence"] == 0.95

    def test_marginal_neighbour_of_confident_ad_is_cut_with_it(self):
        # 0.6 is below the 0.75 threshold, but it is back-to-back with a sure ad
        high, marginal = filter_segments([
            self._ad(0, 30, "sure", 0.95), self._ad(30, 60, "unsure", 0.6),
        ])
        assert [(h["start"], h["end"]) for h in high] == [(0, 60)]
        assert marginal == []

    def test_isolated_marginal_segment_stays_marginal(self):
        high, marginal = filter_segments([
            self._ad(0, 30, "sure", 0.95), self._ad(100, 130, "unsure", 0.6),
        ])
        assert [(h["start"], h["end"]) for h in high] == [(0, 30)]
        assert [m["label"] for m in marginal] == ["unsure"]

    def test_marginal_neighbours_alone_stay_marginal(self):
        high, marginal = filter_segments([
            self._ad(0, 30, confidence=0.6), self._ad(30, 60, confidence=0.6),
        ])
        assert high == []
        assert len(marginal) == 2

    def test_short_tagline_next_to_an_ad_is_cut_with_it(self):
        # a 4 second tag-line is under min_duration alone, but it belongs to the ad
        high, _ = filter_segments([self._ad(0, 30, "ad"), self._ad(30, 34, "tagline")], min_duration=7)
        assert [(h["start"], h["end"]) for h in high] == [(0, 34)]

    def test_isolated_short_segment_is_still_ignored(self):
        high, _ = filter_segments([self._ad(0, 4), self._ad(100, 130)], min_duration=7)
        assert [(h["start"], h["end"]) for h in high] == [(100, 130)]

    def test_two_long_ads_in_a_row_are_not_dropped_by_max_duration(self):
        # each is within the 300s limit; together they are 400s and must still be cut
        high, _ = filter_segments([self._ad(0, 200), self._ad(200, 400)], max_duration=300)
        assert [(h["start"], h["end"]) for h in high] == [(0, 400)]

    def test_single_segment_over_max_duration_is_still_dropped(self):
        high, _ = filter_segments([self._ad(0, 600)], max_duration=300)
        assert high == []

    def test_single_segment_is_returned_unchanged(self):
        seg = self._ad(10, 40, "solo", 0.9)
        high, _ = filter_segments([seg])
        assert high == [seg]
        assert "merged_from" not in high[0]

    def test_input_segments_are_not_mutated(self):
        a, b = self._ad(0, 30, "A"), self._ad(30, 60, "B")
        filter_segments([a, b])
        assert (a["start"], a["end"], a["label"]) == (0, 30, "A")
        assert (b["start"], b["end"], b["label"]) == (30, 60, "B")


# ------------------------------------------------------------------ #
# detect_ads (mocked CLI)
# ------------------------------------------------------------------ #

class TestDetectAds:
    @patch("siphon.ad_detect.subprocess.run")
    def test_detect_ads_parses_structured_output(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "structured_output": {
                    "segments": [
                        {"start": 10.0, "end": 45.0, "label": "sponsor", "confidence": 0.9}
                    ]
                }
            }),
            stderr="",
        )

        result = detect_ads("some transcript", "detect ads prompt")
        assert len(result["segments"]) == 1
        assert result["segments"][0]["label"] == "sponsor"

    @patch("siphon.ad_detect.subprocess.run")
    def test_detect_ads_handles_direct_output(self, mock_run):
        """If CLI returns the segments dict directly (no envelope)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "segments": [
                    {"start": 5.0, "end": 35.0, "label": "ad", "confidence": 0.8}
                ]
            }),
            stderr="",
        )

        result = detect_ads("some transcript", "detect ads prompt")
        assert len(result["segments"]) == 1

    @patch("siphon.ad_detect.subprocess.run")
    def test_detect_ads_cli_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: model not found",
        )

        with pytest.raises(RuntimeError, match="exit code 1"):
            detect_ads("transcript", "prompt")

    @patch("siphon.ad_detect.subprocess.run")
    def test_detect_ads_bad_json_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not json",
            stderr="",
        )

        with pytest.raises(RuntimeError, match="not valid JSON"):
            detect_ads("transcript", "prompt")

    @patch("siphon.ad_detect.subprocess.run")
    def test_detect_ads_empty_result(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"structured_output": {"segments": []}}),
            stderr="",
        )

        result = detect_ads("transcript", "prompt")
        assert result["segments"] == []

    @patch("siphon.ad_detect.subprocess.run")
    def test_detect_ads_passes_model_and_effort(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"structured_output": {"segments": []}}),
            stderr="",
        )

        detect_ads("transcript", "prompt", model="claude-haiku-4-5-20251001", effort="low")

        call_args = mock_run.call_args[0][0]
        assert "--model" in call_args
        model_idx = call_args.index("--model")
        assert call_args[model_idx + 1] == "claude-haiku-4-5-20251001"
        assert "--effort" in call_args
        effort_idx = call_args.index("--effort")
        assert call_args[effort_idx + 1] == "low"


# ------------------------------------------------------------------ #
# Failure classification — auth / transient / other
# ------------------------------------------------------------------ #

def _cli_result(returncode=1, stdout="", stderr=""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def _error_envelope(text):
    """What `claude -p --output-format json` prints on a failed run (stderr stays empty)."""
    return json.dumps({"type": "result", "subtype": "success", "is_error": True, "result": text})


class TestFailureClassification:
    def test_reason_prefers_stdout_result_over_empty_stderr(self):
        r = _cli_result(
            stdout=_error_envelope("Failed to authenticate: OAuth session expired and could not be refreshed"),
            stderr="",
        )
        assert cli_failure_reason(r) == "Failed to authenticate: OAuth session expired and could not be refreshed"

    def test_reason_falls_back_to_stderr_then_exit_code(self):
        assert cli_failure_reason(_cli_result(stdout="", stderr="  Error: boom\n")) == "Error: boom"
        assert cli_failure_reason(_cli_result(returncode=3)) == "exit code 3 with no output"

    @pytest.mark.parametrize("text", [
        "Failed to authenticate: OAuth session expired and could not be refreshed",
        "Not logged in. Please run /login",
        "API Error: 401 authentication_error",
    ])
    def test_auth_errors(self, text):
        assert classify_cli_failure(text) is ClaudeAuthError

    @pytest.mark.parametrize("text", [
        "API Error: 529 Overloaded. This is a server-side issue, usually temporary",
        "Claude AI usage limit reached|1755990000",
        "You've hit your limit. Your limit resets at 3pm",
        "Warning: no stdin data received in 3s, proceeding without it. Error: Input must be provided",
        "API Error: 503 Service Unavailable",
        "fetch failed: ECONNRESET",
    ])
    def test_transient_errors(self, text):
        assert classify_cli_failure(text) is ClaudeTransientError

    @pytest.mark.parametrize("text", [
        "Error: model not found",
        "Prompt is too long: 150000 tokens > 100000 maximum",  # '500' inside a number is not HTTP 500
        "exit code 1 with no output",
    ])
    def test_other_errors_count_as_strikes(self, text):
        assert classify_cli_failure(text) is ClaudeCLIError


class TestDetectAdsFailureHandling:
    @patch("siphon.ad_detect.subprocess.run")
    def test_auth_failure_raises_auth_error_with_real_reason(self, mock_run):
        mock_run.return_value = _cli_result(
            stdout=_error_envelope("Failed to authenticate: OAuth session expired and could not be refreshed"),
        )
        with pytest.raises(ClaudeAuthError, match="OAuth session expired"):
            detect_ads("transcript", "prompt")

    @patch("siphon.ad_detect.subprocess.run")
    def test_overloaded_raises_transient_error(self, mock_run):
        mock_run.return_value = _cli_result(stdout=_error_envelope("API Error: 529 Overloaded."))
        with pytest.raises(ClaudeTransientError, match="529"):
            detect_ads("transcript", "prompt")

    @patch("siphon.ad_detect.time.sleep")
    @patch("siphon.ad_detect.subprocess.run")
    def test_stdin_race_is_retried_once(self, mock_run, mock_sleep):
        stdin_race = _cli_result(
            stderr="Warning: no stdin data received in 3s, proceeding without it.\n"
                   "Error: Input must be provided either through stdin or as a prompt argument when using --print",
        )
        ok = _cli_result(returncode=0, stdout=json.dumps({"structured_output": {"segments": []}}))
        mock_run.side_effect = [stdin_race, ok]

        assert detect_ads("transcript", "prompt") == {"segments": []}
        assert mock_run.call_count == 2

    @patch("siphon.ad_detect.time.sleep")
    @patch("siphon.ad_detect.subprocess.run")
    def test_stdin_race_twice_is_transient(self, mock_run, mock_sleep):
        stdin_race = _cli_result(stderr="Warning: no stdin data received in 3s, proceeding without it.")
        mock_run.side_effect = [stdin_race, stdin_race]

        with pytest.raises(ClaudeTransientError):
            detect_ads("transcript", "prompt")
        assert mock_run.call_count == 2

    @patch("siphon.ad_detect.subprocess.run")
    def test_unknown_failure_is_plain_cli_error_without_retry(self, mock_run):
        mock_run.return_value = _cli_result(stderr="Error: model not found")

        with pytest.raises(ClaudeCLIError) as excinfo:
            detect_ads("transcript", "prompt")
        assert type(excinfo.value) is ClaudeCLIError
        assert mock_run.call_count == 1

    @patch("siphon.ad_detect.subprocess.run")
    def test_prompt_is_fed_from_a_file_not_a_pipe(self, mock_run):
        """A real file on stdin makes the prompt bytes instantly available, so the
        CLI's hard-coded 3 s stdin guard cannot lose a scheduling race under load
        (which is what kept happening with input= piping on 100 KB+ prompts)."""
        seen = {}

        def fake_run(cmd, **kwargs):
            assert "input" not in kwargs
            seen["stdin_content"] = kwargs["stdin"].read()
            seen["path"] = kwargs["stdin"].name
            return _cli_result(returncode=0, stdout=json.dumps({"structured_output": {"segments": []}}))

        mock_run.side_effect = fake_run

        detect_ads("some transcript text", "detect ads prompt")

        assert "detect ads prompt" in seen["stdin_content"]
        assert "some transcript text" in seen["stdin_content"]
        # The temp file is cleaned up afterwards
        import os
        assert not os.path.exists(seen["path"])

    @patch("siphon.ad_detect.subprocess.run")
    def test_cli_spawned_at_normal_priority_on_windows(self, mock_run):
        mock_run.return_value = _cli_result(
            returncode=0, stdout=json.dumps({"structured_output": {"segments": []}}),
        )
        with patch.object(ad_detect.sys, "platform", "win32"):
            detect_ads("transcript", "prompt")
        # NORMAL_PRIORITY_CLASS — not BELOW_NORMAL, which loses the CLI's 3 s stdin race under load
        assert mock_run.call_args.kwargs["creationflags"] == 0x00000020
