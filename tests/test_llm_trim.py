"""Tests for siphon.llm_trim — full pipeline orchestrator."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from siphon.config import FeedConfig, FeedDefaults, LLMConfig, resolve_feed
from siphon.llm_trim import run_llm_trim


def _make_feed(**overrides) -> tuple:
    defaults = FeedDefaults()
    feed_data = {"name": "test", "url": "http://x", "llm_trim": True}
    feed_data.update(overrides)
    feed_cfg = FeedConfig(**feed_data)
    return resolve_feed(feed_cfg, defaults), LLMConfig()


class TestRunLLMTrim:
    @patch("siphon.llm_trim.cut_segments")
    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    @patch("siphon.llm_trim.extract_audio")
    def test_full_pipeline_video(self, mock_extract, mock_transcribe, mock_detect, mock_cut):
        feed, llm = _make_feed()

        mock_transcribe.return_value = {
            "text": "Hello and welcome. This video is sponsored by Acme Corp.",
            "segments": [],
            "language": "en",
            "duration": 300.0,
        }
        mock_detect.return_value = {
            "segments": [
                {"start": 10.0, "end": 45.0, "type": "ad", "label": "sponsor", "confidence": 0.95}
            ]
        }

        result = run_llm_trim("/video.mp4", feed, llm)

        assert result["llm_trim_status"] == "done"
        assert result["llm_cuts_applied"] == 1
        mock_extract.assert_called_once()  # extracted audio from video
        mock_transcribe.assert_called_once()
        mock_detect.assert_called_once()
        mock_cut.assert_called_once()

    @patch("siphon.llm_trim.cut_segments")
    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    def test_full_pipeline_audio(self, mock_transcribe, mock_detect, mock_cut):
        feed, llm = _make_feed()

        mock_transcribe.return_value = {
            "text": "Welcome to the podcast. Squarespace!",
            "segments": [],
            "language": "en",
            "duration": 300.0,
        }
        mock_detect.return_value = {
            "segments": [
                {"start": 20.0, "end": 80.0, "type": "ad", "label": "squarespace", "confidence": 0.9}
            ]
        }

        # Audio file — no extract_audio step
        result = run_llm_trim("/podcast.mp3", feed, llm)

        assert result["llm_trim_status"] == "done"
        assert result["llm_cuts_applied"] == 1
        mock_cut.assert_called_once()

    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    def test_no_ads_detected(self, mock_transcribe, mock_detect):
        feed, llm = _make_feed()

        mock_transcribe.return_value = {
            "text": "Pure content, no ads here.",
            "segments": [],
            "language": "en",
            "duration": 300.0,
        }
        mock_detect.return_value = {"segments": []}

        result = run_llm_trim("/clean.mp3", feed, llm)

        assert result["llm_trim_status"] == "done"
        assert result["llm_cuts_applied"] == 0

    @patch("siphon.llm_trim.transcribe")
    def test_empty_transcript_skips_detection(self, mock_transcribe):
        feed, llm = _make_feed()

        mock_transcribe.return_value = {
            "text": "",
            "segments": [],
            "language": "en",
            "duration": 0.0,
        }

        result = run_llm_trim("/silent.mp3", feed, llm)

        assert result["llm_trim_status"] == "done"
        assert result["llm_cuts_applied"] == 0

    @patch("siphon.llm_trim.transcribe")
    def test_transcription_error_handled(self, mock_transcribe):
        feed, llm = _make_feed()
        mock_transcribe.side_effect = RuntimeError("Whisper crashed")

        result = run_llm_trim("/bad.mp3", feed, llm)

        assert result["llm_trim_status"] == "error"
        assert "Whisper crashed" in result.get("error", "")

    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    def test_claude_error_handled(self, mock_transcribe, mock_detect):
        feed, llm = _make_feed()
        mock_transcribe.return_value = {
            "text": "Some transcript",
            "segments": [],
            "language": "en",
            "duration": 300.0,
        }
        mock_detect.side_effect = RuntimeError("Claude CLI failed")

        result = run_llm_trim("/test.mp3", feed, llm)

        assert result["llm_trim_status"] == "error"
        assert "Claude CLI failed" in result.get("error", "")

    @patch("siphon.llm_trim.cut_segments")
    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    def test_low_confidence_not_cut(self, mock_transcribe, mock_detect, mock_cut):
        feed, llm = _make_feed()

        mock_transcribe.return_value = {
            "text": "Some content",
            "segments": [],
            "language": "en",
            "duration": 300.0,
        }
        mock_detect.return_value = {
            "segments": [
                {"start": 10.0, "end": 40.0, "type": "ad", "label": "maybe-ad", "confidence": 0.6}
            ]
        }

        result = run_llm_trim("/test.mp3", feed, llm)

        assert result["llm_trim_status"] == "done"
        assert result["llm_cuts_applied"] == 0  # below threshold
        mock_cut.assert_not_called()

    @patch("siphon.llm_trim.cut_segments")
    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    def test_audit_data_stored(self, mock_transcribe, mock_detect, mock_cut):
        feed, llm = _make_feed()

        mock_transcribe.return_value = {
            "text": "content",
            "segments": [],
            "language": "en",
            "duration": 300.0,
        }
        mock_detect.return_value = {
            "segments": [
                {"start": 10.0, "end": 40.0, "type": "ad", "label": "sponsor", "confidence": 0.95},
                {"start": 100.0, "end": 120.0, "type": "ad", "label": "marginal", "confidence": 0.6},
            ]
        }

        result = run_llm_trim("/test.mp3", feed, llm)

        audit = json.loads(result["llm_segments_json"])
        assert len(audit["segments"]) == 2
        assert len(audit["high_confidence"]) == 1
        assert len(audit["marginal"]) == 1


class TestQualityOptionsReachTheirTargets:
    @patch("siphon.llm_trim.cut_segments")
    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    def test_options_are_passed_to_transcribe_and_cut(self, mock_transcribe, mock_detect, mock_cut):
        feed, _ = _make_feed()
        llm = LLMConfig(whisper_vad_filter=True, snap_cuts_to_silence=True, snap_window_seconds=2.0)
        mock_transcribe.return_value = {
            "text": "Welcome. Squarespace!", "segments": [], "language": "en", "duration": 300.0,
        }
        mock_detect.return_value = {
            "segments": [{"start": 20.0, "end": 80.0, "type": "ad", "label": "x", "confidence": 0.9}]
        }

        run_llm_trim("/podcast.mp3", feed, llm)

        assert mock_transcribe.call_args.kwargs["vad_filter"] is True
        assert mock_cut.call_args.kwargs["snap_to_silence"] is True
        assert mock_cut.call_args.kwargs["snap_window"] == 2.0

    @patch("siphon.llm_trim.cut_segments")
    @patch("siphon.llm_trim.detect_ads")
    @patch("siphon.llm_trim.transcribe")
    def test_defaults_keep_existing_behaviour(self, mock_transcribe, mock_detect, mock_cut):
        feed, llm = _make_feed()
        mock_transcribe.return_value = {
            "text": "Welcome. Squarespace!", "segments": [], "language": "en", "duration": 300.0,
        }
        mock_detect.return_value = {
            "segments": [{"start": 20.0, "end": 80.0, "type": "ad", "label": "x", "confidence": 0.9}]
        }

        run_llm_trim("/podcast.mp3", feed, llm)

        assert mock_transcribe.call_args.kwargs["vad_filter"] is False
        assert mock_cut.call_args.kwargs["snap_to_silence"] is False
