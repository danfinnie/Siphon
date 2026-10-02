"""Tests for siphon.transcribe option passing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from siphon.transcribe import transcribe


def _fake_model():
    model = MagicMock()
    model.transcribe.return_value = ([], SimpleNamespace(duration=10.0, language="en"))
    return model


class TestVadFilter:
    @patch("siphon.transcribe._get_model")
    def test_vad_filter_off_by_default(self, mock_get_model):
        model = mock_get_model.return_value = _fake_model()
        transcribe("/a.mp3")
        assert model.transcribe.call_args.kwargs["vad_filter"] is False

    @patch("siphon.transcribe._get_model")
    def test_vad_filter_passed_to_whisper(self, mock_get_model):
        model = mock_get_model.return_value = _fake_model()
        transcribe("/a.mp3", vad_filter=True)
        assert model.transcribe.call_args.kwargs["vad_filter"] is True
        assert model.transcribe.call_args.kwargs["word_timestamps"] is True
